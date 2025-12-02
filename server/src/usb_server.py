import argparse
from collections import defaultdict
import socket
import struct
import subprocess
from threading import Thread, Event, Lock
from queue import Empty, Queue
from dataclasses import dataclass
from loguru import logger
import cv2
import numpy as np
from typing import Any
import sys

# import av
from PIL import Image
import io
from PyQt5 import QtWidgets, QtGui
import time
from enum import Enum
import pyautogui as pg

from src.gesture_classifier_model import (
    get_mediapipe_model,
    predict,
    Gesture,
    Handedness,
    Landmark,
    GestureRecognizerCustomResult,
)

HOST = "0.0.0.0"
WINDOWS_IPROXY_PATH = "C:\\Users\\hunta\\Documents\\msys64\\mingw64\\bin"


# ### Receive USB Data ###


class Orientation(Enum):
    PORTRAIT = 0
    LANDSCAPE_LEFT = 1
    PORTRAIT_UPSIDE_DOWN = 2
    LANDSCAPE_RIGHT = 3

    @staticmethod
    def from_int(value: int) -> "Orientation":
        if value == 0:
            return Orientation.PORTRAIT
        elif value == 1:
            return Orientation.LANDSCAPE_LEFT
        elif value == 2:
            return Orientation.PORTRAIT_UPSIDE_DOWN
        elif value == 3:
            return Orientation.LANDSCAPE_RIGHT
        else:
            raise ValueError(f"Invalid orientation value: {value}")


@dataclass
class Frame:
    orientation: Orientation
    depth_data_size: int
    video_data_size: int
    depth_data: bytes
    video_data: bytes


class SingleItemQueue:
    def __init__(self):
        self.lock = Lock()
        self.frame = None

    def set(self, frame):
        with self.lock:
            self.frame = frame

    def get(self):
        with self.lock:
            return self.frame


def recv_exact(sock: socket.socket, length: int):
    """Receive an exact number of bytes or raise."""
    data = b""
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            raise ConnectionError("Socket closed")
        data += packet
    return data


def run_websocket_thread_udp(port: int, queue: Queue[Frame], event: Event) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 1024)
    server.bind((HOST, port))
    # server.listen(1)

    frames: dict[int, dict[str, dict | int | Any]] = defaultdict(lambda: {"chunks": {}, "total": None})
    latest_complete_frame = None

    # while True:
    #     conn, addr = server.accept()
    #     logger.info(f"Connection from {addr} has been established!")

    #     try:
    while True:
        if event.is_set():
            logger.info("End event set, closing websocket connection thread")
            # conn.close()
            return

        # Read from USB
        try:
            packet, addr = server.recvfrom(65535)
        except Exception as e:
            logger.info(f"Error receiving UDP packet: {e}")
            continue

        frame_id = int.from_bytes(packet[0:8], byteorder="little")
        # frame_id = struct.unpack("d", packet[0:8])[0]
        packet_id = int.from_bytes(packet[8:10], byteorder="little")
        total_packets = int.from_bytes(packet[10:12], byteorder="little")
        payload = packet[12:]

        f = frames[frame_id]
        f["chunks"][packet_id] = payload  # type: ignore
        f["total"] = total_packets

        # If the frame is complete
        if len(f["chunks"]) == f["total"]:  # type: ignore
            # print(f"Complete frame {frame_id}")
            chunks = [f["chunks"][i] for i in range(f["total"])]  # type: ignore
            latest_complete_frame = b"".join(chunks)

            del frames[frame_id]
            # Dropping old frames automatically
            for old_id in list(frames.keys()):
                if old_id < frame_id:
                    del frames[old_id]

        if latest_complete_frame is None:
            continue

        orientation, depth_size, video_size = struct.unpack("<III", latest_complete_frame[0:12])
        depth_data = latest_complete_frame[12 : 12 + depth_size]
        video_data = latest_complete_frame[12 + depth_size : 12 + depth_size + video_size]
        # print("Received frame")
        latest_complete_frame = None

        # if queue.full():
        #     queue.get_nowait()
        frame = Frame(
            orientation=Orientation.from_int(orientation),
            depth_data=depth_data,
            video_data=video_data,
            depth_data_size=depth_size,
            video_data_size=video_size,
        )
        # queue.put_nowait(frame)
        try:
            queue.get_nowait()
            # print("Dropped frame")
        except Empty:
            pass
        queue.put_nowait(frame)

        # except Exception as e:
        #     logger.info(f"Server connection ended: {e}")

        # conn.close()
        # logger.info("Server waiting for next connection...")


def run_websocket_thread_tcp(port: int, queue: Queue[Frame], event: Event) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, port))
    server.listen(1)

    while True:
        conn, addr = server.accept()
        logger.info(f"Connection from {addr} has been established!")

        try:
            while True:
                # Read from USB
                header = recv_exact(conn, 12)
                orientation, depth_size, video_size = struct.unpack("<III", header)
                depth_data = recv_exact(conn, depth_size)
                video_data = recv_exact(conn, video_size)
                # print("Received frame")

                # if queue.full():
                #     queue.get_nowait()
                frame = Frame(
                    orientation=Orientation.from_int(orientation),
                    depth_data=depth_data,
                    video_data=video_data,
                    depth_data_size=depth_size,
                    video_data_size=video_size,
                )
                # queue.put_nowait(frame)
                try:
                    queue.get_nowait()
                    # print("Dropped frame")
                except Empty:
                    pass
                queue.put_nowait(frame)

                if event.is_set():
                    logger.info("End event set, closing websocket connection thread")
                    conn.close()
                    return

        except Exception as e:
            logger.info(f"Server connection ended: {e}")

        conn.close()
        logger.info("Server waiting for next connection...")


# ### Handle USB Data ###


def display_frame(frame: Frame) -> None:
    depth_map = np.frombuffer(frame.depth_data, np.float16).copy()
    depth_map = depth_map.reshape((480, 640))
    depth_map[np.isnan(depth_map)] = 0.0
    depth_map = np.maximum(np.minimum(depth_map, 1.5), 0.01)
    depth_map = (depth_map - 0.01) / (1.5 - 0.01)  # Normalize to 0-1
    depth_map = (depth_map * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_TURBO)
    cv2.imshow("Depth Map", depth_colored)
    cv2.waitKey(1)

    color_frame = np.frombuffer(frame.video_data, np.uint8).reshape((480, 640, 3))
    open_cv_image = cv2.cvtColor(np.array(color_frame), cv2.COLOR_RGB2BGR)
    cv2.imshow("iPhone Camera", open_cv_image)
    cv2.waitKey(1)


def display_image(color_image, depth_map) -> None:
    depth_map = np.maximum(np.minimum(depth_map, 1.5), 0.01)
    depth_map = (depth_map - 0.01) / (1.5 - 0.01)  # Normalize to 0-1
    depth_map = (depth_map * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_TURBO)
    cv2.imshow("Depth Map", depth_colored)
    cv2.waitKey(1)

    cv2.imshow("iPhone Camera", color_image)
    cv2.waitKey(1)


def get_image(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    depth_map = np.frombuffer(frame.depth_data, np.float16).copy()
    depth_map = depth_map.reshape((480, 640))
    depth_map[np.isnan(depth_map)] = 100.0

    color_frame = np.frombuffer(frame.video_data, np.uint8).reshape((480, 640, 3))
    open_cv_image = cv2.cvtColor(np.array(color_frame), cv2.COLOR_RGB2BGR)

    depth_map = depth_map[:, 80:560].copy()  # Crop to center 480 width
    open_cv_image = open_cv_image[:, 80:560, :].copy()

    match frame.orientation:
        case Orientation.PORTRAIT:
            depth_map = cv2.rotate(depth_map, cv2.ROTATE_90_CLOCKWISE)  # type: ignore
            open_cv_image = cv2.rotate(open_cv_image, cv2.ROTATE_90_CLOCKWISE)
        case Orientation.LANDSCAPE_LEFT:
            depth_map = cv2.rotate(depth_map, cv2.ROTATE_180)  # type: ignore
            open_cv_image = cv2.rotate(open_cv_image, cv2.ROTATE_180)
        case Orientation.PORTRAIT_UPSIDE_DOWN:
            depth_map = cv2.rotate(depth_map, cv2.ROTATE_90_COUNTERCLOCKWISE)  # type: ignore
            open_cv_image = cv2.rotate(open_cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        case Orientation.LANDSCAPE_RIGHT:
            pass

    depth_map = cv2.flip(depth_map, 1)  # type: ignore
    open_cv_image = cv2.flip(open_cv_image, 1)

    return open_cv_image, depth_map


def run_computer_control_thread(queue: Queue[Frame], event: Event) -> None:
    detector = get_mediapipe_model()
    pg.PAUSE = 0

    cursor_velocity: tuple[float, float] = (0, 0)
    are_dragging = False
    # last_10_records = defaultdict(list)
    last_10_left_click: list[bool] = []
    last_10_right_click: list[bool] = []
    last_10_gestures: list[Gesture] = []

    # LEFT_CLICK_RECORD = "left_click"
    # RIGHT_CLICK_RECORD = "right_click"
    # GESTURE_RECORD = "gesture"

    REPEATED_FRAMES_BEFORE_ACTION = 2
    CURSOR_VELOCITY_DECAY_RATE = 0.8

    CLICK_DEPTH_THRESHOLD = 0.3
    MOVE_DEPTH_THRESHOLD = 0.5
    CLICK_DEPTH_LIMIT = MOVE_DEPTH_THRESHOLD - 0.5  # Limits are for visuals only
    MOVE_DEPTH_LIMIT = 1.1  # Limits are for visuals only
    THRESHOLDS = (CLICK_DEPTH_THRESHOLD, MOVE_DEPTH_THRESHOLD, CLICK_DEPTH_LIMIT, MOVE_DEPTH_LIMIT)

    MOUSE_SCALING = 1500
    SWIPE_SCALING = 1000

    prev_index_location = None

    while True:
        if event.is_set():
            logger.info("End event set, closing computer control thread")
            cv2.destroyAllWindows()
            return

        try:
            frame = queue.get_nowait()
        except Empty:
            continue

        color_image, depth_map = get_image(frame)
        detection_result, annotated_image = predict(color_image, detector, depth_map, THRESHOLDS)

        display_image(color_image, depth_map)
        cv2.imshow("Annotated Image", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Quit key entered. Exiting...")
            break

        if not detection_result.hand_detected:
            prev_index_location = None
            are_dragging = False
            pg.mouseUp(button="left")
            continue

        last_10_left_click = last_10_left_click[-10:]
        last_10_right_click = last_10_right_click[-10:]
        last_10_gestures = last_10_gestures[-5:]
        last_10_gestures.append(detection_result.gesture)

        # When we see a palm, cancel all actions
        if detection_result.gesture in (Gesture.palm, Gesture.stop, Gesture.stop_inverted) and last_10_gestures.count(detection_result.gesture) >= REPEATED_FRAMES_BEFORE_ACTION:
            logger.info("Palm detected, cancelling actions")
            cursor_velocity = (0, 0)
            are_dragging = False
            pg.mouseUp(button="left")
            last_10_left_click.append(False)
            last_10_right_click.append(False)
            prev_index_location = detection_result.index_finger_tip
            continue

        # Left Click
        # Conditions:
        # - Gesture must be index finger - one
        # - Gesture must be poking through click pane
        if (
            detection_result.gesture == Gesture.one
            and detection_result.index_finger_tip.z < CLICK_DEPTH_THRESHOLD
            and last_10_gestures.count(Gesture.one) >= REPEATED_FRAMES_BEFORE_ACTION
            and not any(last_10_left_click)
        ):
            logger.info("Left click")
            pg.click(button="left")
            last_10_left_click.append(True)

        # Right Click
        # Same conditions as left click but for peace gesture
        if (
            detection_result.gesture == Gesture.peace
            and detection_result.index_finger_tip.z < CLICK_DEPTH_THRESHOLD
            and last_10_gestures.count(Gesture.peace) >= REPEATED_FRAMES_BEFORE_ACTION
            and not any(last_10_right_click)
        ):
            logger.info("Right click")
            pg.click(button="right")
            last_10_right_click.append(True)

        # Click and hold for dragging
        if (
            detection_result.gesture in (Gesture.ok, Gesture.fist)
            and detection_result.index_finger_tip.z < MOVE_DEPTH_THRESHOLD
            and last_10_gestures.count(detection_result.gesture) >= REPEATED_FRAMES_BEFORE_ACTION
            and not are_dragging
        ):
            logger.info("Starting left click drag")
            pg.mouseDown(button="left")
            are_dragging = True

        if detection_result.index_finger_tip.z >= MOVE_DEPTH_THRESHOLD and are_dragging:
            logger.info("Ending left click drag")
            pg.mouseUp(button="left")
            are_dragging = False

        # Small, absolute movements
        # Conditions:
        # - Gesture must be index finger - one
        # - Gesture must be within move pane
        if detection_result.gesture == Gesture.one and detection_result.index_finger_tip.z < MOVE_DEPTH_THRESHOLD and prev_index_location is not None:
            logger.info("Moving cursor with small movements")
            move_x_relative = prev_index_location.x - detection_result.index_finger_tip.x
            move_y_relative = prev_index_location.y - detection_result.index_finger_tip.y
            move_x_relative *= MOUSE_SCALING
            move_y_relative *= MOUSE_SCALING
            pg.moveRel(-move_x_relative, -move_y_relative, duration=0.1)

        # Sweeping, general movements
        if detection_result.gesture in (Gesture.two_up, Gesture.two_up_inverted) and detection_result.index_finger_tip.z < MOVE_DEPTH_THRESHOLD and prev_index_location is not None:
            logger.info("Moving cursor with velocity")
            cursor_velocity = (
                cursor_velocity[0] + -(prev_index_location.x - detection_result.index_finger_tip.x) * SWIPE_SCALING,
                cursor_velocity[1] + (prev_index_location.y - detection_result.index_finger_tip.y) * SWIPE_SCALING,
            )

        prev_index_location = detection_result.index_finger_tip
        pg.moveRel(cursor_velocity[0], -cursor_velocity[1], duration=0.1)
        cursor_velocity = (cursor_velocity[0] * CURSOR_VELOCITY_DECAY_RATE, cursor_velocity[1] * CURSOR_VELOCITY_DECAY_RATE)


def run(args: argparse.Namespace):
    queue: Queue[Frame] = Queue(maxsize=1)
    # queue = SingleItemQueue()

    END_EVENT = Event()
    END_EVENT.clear()

    if args.tcp_mode:
        websocket_thread = Thread(target=run_websocket_thread_tcp, args=(args.port, queue, END_EVENT))
    else:
        websocket_thread = Thread(target=run_websocket_thread_udp, args=(args.port, queue, END_EVENT))
    websocket_thread.start()
    computer_control_thread = Thread(target=run_computer_control_thread, args=(queue, END_EVENT))
    computer_control_thread.start()

    while True:
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            END_EVENT.set()
            logger.info("Server shutting down...")
            sys.exit()
            websocket_thread.join(2)
            computer_control_thread.join(2)
            break
