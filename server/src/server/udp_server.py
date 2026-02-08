import asyncio
from dataclasses import dataclass, field
import json
import socket
from threading import Thread, Event

from loguru import logger

from src.schema import LandmarkedFrame, Settings
from src.computer_control import ComputerControl


@dataclass
class UDPServer:
    server_port: int
    computer_control: ComputerControl | None = None
    stop_event: Event = Event()

    def start(self) -> None:
        """Starts the http server to initiate WebRTC connections. Blocking call."""
        print("Starting computer control...")
        self.computer_control = ComputerControl(end_event=self.stop_event)
        self.computer_control.start()

        # Init UDP socket
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 300)
        server.bind(("0.0.0.0", self.server_port))
        server.settimeout(1.0)

        while True:
            if self.stop_event.is_set():
                logger.info("End event set, closing websocket connection thread")
                return

            try:
                packet, addr = server.recvfrom(65535)
            except socket.timeout:
                continue
            except Exception as e:
                logger.info(f"Error receiving UDP packet: {e}")
                continue

            packet_type = packet[0]
            payload = packet[1:]
            logger.info(f"Received UDP packet of type {packet_type} and length {len(payload)} from {addr}")

            if packet_type == 0:
                self.stream_message(payload)
            elif packet_type == 1:
                self.settings_message(payload)
                # Send ack
                server.sendto(b"\x02" + b"Ack: Settings updated", addr)

    def stream_message(self, message: bytes) -> None:
        assert self.computer_control is not None, "ComputerControl should be initialized in start()"
        frame_data = json.loads(message)
        frame = LandmarkedFrame.from_(frame_data)
        self.computer_control.receive_frame(frame)

    def settings_message(self, message: bytes) -> None:
        assert self.computer_control is not None, "ComputerControl should be initialized in start()"
        settings_data = json.loads(message)
        settings = Settings.from_(settings_data)
        self.computer_control.update_settings(settings)
