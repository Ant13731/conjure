from aiohttp import web
from aiortc import RTCSessionDescription, RTCPeerConnection, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from queue import Queue
import asyncio
import cv2

from loguru import logger

# Code adapted from https://github.com/aiortc/aiortc/blob/main/examples/server/server.py

pcs = set()
video_relay = MediaRelay()


# class VideoReceiverTrack(MediaStreamTrack):
#     kind = "video"

#     def __init__(self, track, params):
#         super().__init__()
#         self.track = track
#         self.params = params

#     async def recv(self):
#         frame = await self.track.recv()
#         img = frame.to_ndarray(format="bgr24")
#         cv2.imshow("iPhone Camera", img)


#         # Here you can process the frame as needed, e.g., resizing, filtering, etc.
#         return frame


async def offer(request):
    data = await request.json()

    logger.info(f"Received offer: {data}. Checking for validity...")
    if not data:
        return web.json_response({"message": "No offer data provided", "data": data}, status=400)
    elif data.get("sdp") is None or data.get("type") is None:
        return web.json_response({"message": "Invalid offer data: require 'sdp' and 'type' fields", "data": data}, status=400)

    logger.info(f"Handling RTC connection offer...")
    rtc_offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        logger.info(f"Connection state changed to {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    async def on_track(track):
        logger.info(f"Track received: {track.kind}")
        if track.kind == "video":
            logger.info(f"Video track received")
            while True:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                cv2.imshow("iPhone Camera", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # pc.addTrack(
            #     VideoReceiverTrack(video_relay.subscribe(track), params=data["video"]),
            # )
        else:
            logger.warning(f"Unknown track kind received: {track.kind}")

        @track.on("ended")
        async def on_ended():
            logger.info(f"Track ended: {track.kind}")

    await pc.setRemoteDescription(rtc_offer)

    logger.info(f"Creating RTC answer...")
    answer = await pc.createAnswer()

    logger.info(f"Sending RTC answer...")
    await pc.setLocalDescription(answer)
    logger.info(f"RTC answer sent")

    return web.json_response({"message": "Offer successful", "data": {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}}, status=200)


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


handshake_server = web.Application()
handshake_server.on_shutdown.append(on_shutdown)
handshake_server.router.add_post("/offer", offer)
