import argparse
import socket
import struct
import subprocess
from threading import Thread
from queue import Queue
from dataclasses import dataclass
from loguru import logger
import cv2
import numpy as np
import av
from PIL import Image
import io
from PyQt5 import QtWidgets, QtGui

HOST = "0.0.0.0"

WINDOWS_IPROXY_PATH = "C:\\Users\\hunta\\Documents\\msys64\\mingw64\\bin"


def recv_exact(sock: socket.socket, length: int):
    """Receive an exact number of bytes or raise."""
    data = b""
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            raise ConnectionError("Socket closed")
        data += packet
    return data


@dataclass
class Frame:
    depth_data: bytes
    video_data: bytes
    depth_data_size: int
    video_data_size: int


def run_websocket_thread(port: int, queue: Queue[Frame]) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, port))
    server.listen(1)

    while True:
        conn, addr = server.accept()
        print(f"Connection from {addr} has been established!")

        try:
            while True:
                # Read from USB
                header = recv_exact(conn, 8)
                depth_size, video_size = struct.unpack("<II", header)
                depth_data = recv_exact(conn, depth_size)
                video_data = recv_exact(conn, video_size)
                # print(f"[Server] Received frame → depth={depth_size} bytes, video={video_size} bytes")

                if queue.full():
                    queue.get_nowait()
                frame = Frame(
                    depth_data=depth_data,
                    video_data=video_data,
                    depth_data_size=depth_size,
                    video_data_size=video_size,
                )
                queue.put_nowait(frame)

        except Exception as e:
            print(f"[Server] Connection ended: {e}")

        conn.close()
        print("[Server] Waiting for next connection...")


def handle_frame(frame: Frame) -> None:

    # Trying to figure out if depth data just contains nothing lol
    # total_ones = 0

    # for b in frame.depth_data:
    #     total_ones += bin(b).count("1")
    # print(f"Depth data total 1 bits: {total_ones}")

    depth_map = np.frombuffer(frame.depth_data, np.float16).copy()
    depth_map = depth_map.reshape((480, 640))
    # depth_map.setflags(write=True)
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


def run_computer_control_thread(queue: Queue[Frame]) -> None:
    # app = QtWidgets.QApplication([])
    # label = QtWidgets.QLabel()
    # label.resize(640, 480)
    # label.show()
    while True:
        frame = queue.get()
        handle_frame(frame)
        queue.task_done()


def run(args: argparse.Namespace):
    queue: Queue[Frame] = Queue(maxsize=3)

    websocket_thread = Thread(target=run_websocket_thread, args=(args.port, queue))
    websocket_thread.start()
    computer_control_thread = Thread(target=run_computer_control_thread, args=(queue,))
    computer_control_thread.start()

    while True:
        try:
            websocket_thread.join(3)
            # computer_control_thread.join(3)
        except KeyboardInterrupt:
            print("[Server] Shutting down...")
            websocket_thread.join(1)
            # computer_control_thread.join(1)
            break
