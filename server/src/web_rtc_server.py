from dataclasses import dataclass, field
from queue import Queue, Empty
import asyncio
from typing import NoReturn

import cv2
from aiohttp import web
from aiortc import RTCSessionDescription, RTCPeerConnection, RTCDataChannel
from aiortc.contrib.media import MediaRelay

from loguru import logger
from threading import Thread, Event
from src.usb_server import Frame, run_computer_control_thread
from src.gesture_classifier_model import GestureRecognizerCustomResult


@dataclass
class WebRTCServer:
    server_port: int

    peer_connections: set[RTCPeerConnection] = field(default_factory=set)
    queue: Queue[GestureRecognizerCustomResult] = field(default_factory=lambda: Queue(maxsize=1))
    computer_control_thread: Thread | None = None

    server = web.Application()
    server_thread: Thread | None = None
    stop_event: Event = Event()

    def start(self) -> None:
        """Starts the http server to initiate WebRTC connections. Blocking call."""
        self.server.on_shutdown.append(self.clear_connections)
        self.server.router.add_post("/offer", self.accept_offer)
        web.run_app(self.server, port=self.server_port)
        # Separate thread doesnt work on macos...
        # self.server_thread = Thread(target=web.run_app, args=(self.server,), kwargs={"port": self.server_port})
        # self.server_thread.start()

    def block(self) -> None:
        """Blocks the main thread until the server is stopped."""

        while True:
            try:
                self.stop_event.wait(timeout=3)
                if self.stop_event.is_set():
                    break
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, stopping server...")
                self.stop_event.set()
                break

        asyncio.run(self.server.shutdown())

        if self.server_thread is not None:
            self.server_thread.join(5)

    async def accept_offer(self, request: web.Request) -> web.Response:
        data = await request.json()

        logger.info(f"Received offer: {data}. Checking for validity...")
        if not data:
            logger.warning("No offer data provided")
            return web.json_response({"message": "No offer data provided", "data": data}, status=400)
        if data.get("sdp") is None or data.get("type") is None:
            logger.warning("Invalid offer data: require 'sdp' and 'type' fields")
            return web.json_response({"message": "Invalid offer data: require 'sdp' and 'type' fields", "data": data}, status=400)

        logger.info(f"Handling RTC connection offer...")
        rtc_offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        logger.info("RTC Offer SDP:", rtc_offer.sdp)

        peer_connection = RTCPeerConnection()
        self.peer_connections.add(peer_connection)

        @peer_connection.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info(f"Connection state changed to {peer_connection.connectionState}")
            if peer_connection.connectionState == "failed":
                await peer_connection.close()
                self.peer_connections.discard(peer_connection)

        @peer_connection.on("datachannel")
        async def on_data_channel(channel: RTCDataChannel):
            logger.info(f"Data channel received: {channel.label}")

            @channel.on("message")
            async def on_message(message: bytes):
                logger.info(f"Message received on data channel: {message}, {channel.label}")

                if channel.label == "stream":
                    pass
                elif channel.label == "settings":
                    pass
                else:
                    logger.warning(f"Unknown data channel label: {channel.label}")

        logger.info("Setting remote peer connection description...")
        await peer_connection.setRemoteDescription(rtc_offer)

        logger.info(f"Creating RTC answer...")
        answer = await peer_connection.createAnswer()

        logger.info("Setting local peer connection description...")
        await peer_connection.setLocalDescription(answer)

        logger.info(f"Sending RTC answer...")
        return web.json_response(
            {
                "message": "Offer successful",
                "data": {
                    "sdp": peer_connection.localDescription.sdp,
                    "type": peer_connection.localDescription.type,
                },
            },
            status=200,
        )

    async def clear_connections(self, app: web.Application) -> None:
        """Cleans up peer connections on server shutdown."""
        coros = [peer_connection.close() for peer_connection in self.peer_connections]
        await asyncio.gather(*coros)
        self.peer_connections.clear()
