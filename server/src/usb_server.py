import argparse
import socket
import struct
import subprocess
from threading import Thread, Event
from queue import Queue
from dataclasses import dataclass
from loguru import logger
import cv2
import numpy as np

# import av
from PIL import Image
import io
from PyQt5 import QtWidgets, QtGui
import time
from enum import Enum

from src.gesture_classifier_model import get_mediapipe_model, predict

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


def recv_exact(sock: socket.socket, length: int):
    """Receive an exact number of bytes or raise."""
    data = b""
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            raise ConnectionError("Socket closed")
        data += packet
    return data


def run_websocket_thread(port: int, queue: Queue[Frame], event: Event) -> None:
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

                if queue.full():
                    queue.get_nowait()
                frame = Frame(
                    orientation=Orientation.from_int(orientation),
                    depth_data=depth_data,
                    video_data=video_data,
                    depth_data_size=depth_size,
                    video_data_size=video_size,
                )
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
    depth_map = np.maximum(np.minimum(depth_map, 1.5), 0.1)
    depth_map = (depth_map - 0.1) / (1.5 - 0.1)  # Normalize to 0-1
    depth_map = (depth_map * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_TURBO)
    cv2.imshow("Depth Map", depth_colored)
    cv2.waitKey(1)

    color_frame = np.frombuffer(frame.video_data, np.uint8).reshape((480, 640, 3))
    open_cv_image = cv2.cvtColor(np.array(color_frame), cv2.COLOR_RGB2BGR)
    cv2.imshow("iPhone Camera", open_cv_image)
    cv2.waitKey(1)


def get_image(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    depth_map = np.frombuffer(frame.depth_data, np.float16).copy()
    depth_map = depth_map.reshape((480, 640))
    depth_map[np.isnan(depth_map)] = 0.0
    depth_map = np.maximum(np.minimum(depth_map, 1.5), 0.1)
    depth_map = (depth_map - 0.1) / (1.5 - 0.1)  # Normalize to 0-1
    # depth_map = (depth_map * 255).astype(np.uint8)

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

    return depth_map, open_cv_image


def run_computer_control_thread(queue: Queue[Frame], event: Event) -> None:
    detector = get_mediapipe_model()

    while True:
        if event.is_set():
            logger.info("End event set, closing computer control thread")
            cv2.destroyAllWindows()
            return

        frame = queue.get()

        depth_map, color_image = get_image(frame)
        queue.task_done()

        detection_result, annotated_image = predict(color_image, detector, depth_map)

        cv2.imshow("Annotated Image", annotated_image)
        cv2.waitKey(1)


def run(args: argparse.Namespace):
    queue: Queue[Frame] = Queue(maxsize=3)

    END_EVENT = Event()
    END_EVENT.clear()

    websocket_thread = Thread(target=run_websocket_thread, args=(args.port, queue, END_EVENT))
    websocket_thread.start()
    computer_control_thread = Thread(target=run_computer_control_thread, args=(queue, END_EVENT))
    computer_control_thread.start()

    while True:
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            END_EVENT.set()
            logger.info("Server shutting down...")
            websocket_thread.join(2)
            computer_control_thread.join(2)
            break
