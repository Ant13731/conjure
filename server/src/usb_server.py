import argparse
import socket
import struct
import subprocess
from threading import Thread
from queue import Queue
from dataclasses import dataclass

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


def handle_frame(frame: Frame):
    print("Frame handled")
    return None


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
                print(f"[Server] Received frame → depth={depth_size} bytes, video={video_size} bytes")

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


def run_computer_control_thread(queue: Queue[Frame]) -> None:
    while True:
        frame = queue.get()
        handle_frame(frame)
        queue.task_done()


def run(args: argparse.Namespace):
    try:
        iproxy_process = subprocess.Popen(["iproxy", str(args.port), str(args.port)])
    except FileNotFoundError:
        try:
            iproxy_process = subprocess.Popen([f"{WINDOWS_IPROXY_PATH}\\iproxy.exe", str(args.port), str(args.port)])
        except FileNotFoundError:
            print("[Server] iproxy not found. Ensure it is installed and in your PATH. Exiting...")
            return

    queue: Queue[Frame] = Queue(maxsize=3)
    websocket_thread = Thread(target=run_websocket_thread, args=(args.port, queue))
    websocket_thread.start()

    if iproxy_process and iproxy_process.poll() is not None:
        print("[Server] iproxy process terminated unexpectedly.")
        print(f"Process exited with {iproxy_process.returncode} STDERR:", iproxy_process.stderr)
        return

    while True:
        try:
            websocket_thread.join(3)
        except KeyboardInterrupt:
            print("[Server] Shutting down...")
            if iproxy_process:
                iproxy_process.terminate()
            websocket_thread.join(1)
            break
