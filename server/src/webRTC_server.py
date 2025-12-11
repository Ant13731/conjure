from queue import Queue, Empty
import asyncio

import cv2
from aiohttp import web
from aiortc import RTCSessionDescription, RTCPeerConnection, MediaStreamTrack
from aiortc.contrib.media import MediaRelay

from loguru import logger
from threading import Thread, Event
from src.usb_server import Frame, run_computer_control_thread
from src.gesture_classifier_model import GestureRecognizerCustomResult

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

import numpy as np
import matplotlib.pyplot as plt


# def decode_rgba_to_depth(rgba_image):
#     # Combine R and G into 16-bit unsigned integers
#     R = rgba_image[..., 0].astype(np.uint16)
#     G = rgba_image[..., 1].astype(np.uint16)
#     depth_uint16 = (R << 8) | G

#     # Convert to float16 (proper IEEE 754 decoding)
#     depth_float16 = depth_uint16.view(np.float16)

#     # Convert to float32 for processing
#     depth_float32 = depth_float16.astype(np.float32)
#     return depth_float32


# def depth_to_heatmap(depth_float32, min_depth=0.0, max_depth=5.0):
#     # Clip to expected depth range
#     depth_clipped = np.clip(depth_float32, min_depth, max_depth)

#     # Normalize to 0–1
#     normalized = (depth_clipped - min_depth) / (max_depth - min_depth)

#     # Apply a colormap
#     heatmap = plt.cm.jet(normalized)[:, :, :3]  # RGB only
#     heatmap = (heatmap * 255).astype(np.uint8)
#     return heatmap


async def offer(request):
    data = await request.json()

    logger.info(f"Received offer: {data}. Checking for validity...")
    if not data:
        return web.json_response({"message": "No offer data provided", "data": data}, status=400)
    elif data.get("sdp") is None or data.get("type") is None:
        return web.json_response({"message": "Invalid offer data: require 'sdp' and 'type' fields", "data": data}, status=400)

    logger.info(f"Handling RTC connection offer...")
    rtc_offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

    print("RTC Offer SDP:", rtc_offer.sdp)

    # Increase bandwidth:
    new_lines = rtc_offer.sdp.splitlines()

    for l in range(len(rtc_offer.sdp.splitlines())):
        if new_lines[l].startswith("m=video"):
            found_m = l
            break
    else:
        found_m = len(rtc_offer.sdp.splitlines()) - 1
    found_m += 1
    while new_lines[found_m].startswith("i=") or new_lines[found_m].startswith("c="):
        found_m += 1
    if new_lines[found_m].startswith("b="):
        new_lines[found_m] = "b=AS:300000"
    else:
        # Insert bandwidth line after m=video
        new_lines.insert(found_m, "b=AS:300000")

    # Patch H264 fmtp line
    # if line.startswith("a=fmtp:") and "H264" in rtc_offer.sdp:
    #     if "profile-level-id" in line:
    #         # Example: add ultra-high bitrate and frame size settings
    #         new_line = line + ";max-fs=12288;max-fr=60;max-br=5000000;max-mbps=5000000"
    #         new_lines[-1] = new_line

    # # Patch VP8/VP9
    # if line.startswith("a=rtpmap:") and "VP8" in line:
    #     new_lines.append("a=fmtp:{} max-fs=12288;max-fr=60".format(line.split(":")[1].split(" ")[0]))
    rtc_offer = RTCSessionDescription(sdp="\n".join(new_lines), type=rtc_offer.type)

    # Prefer USB connection if available
    # if "172.20.10" in rtc_offer.sdp:
    #     logger.info("USB connection found in SDP, filtering out non-USB candidates...")
    #     filtered_sdp_lines = []
    #     for line in rtc_offer.sdp.splitlines():
    #         if line.startswith("a=candidate:"):
    #             if "172.20.10" not in line:
    #                 continue
    #             filtered_sdp_lines.append(line)
    #         else:
    #             filtered_sdp_lines.append(line)
    #     filtered_sdp = "\n".join(filtered_sdp_lines)
    #     rtc_offer = RTCSessionDescription(sdp=filtered_sdp, type=rtc_offer.type)

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        logger.info(f"Connection state changed to {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # @pc.on("track")
    # async def on_track(track):
    #     logger.info(f"Track received: {track.kind}, id={track.id}")
    #     if track.kind == "video":
    #         logger.info(f"Video track received")

    #         # if track.id == "video0":
    #         #     while True:
    #         #         frame = await track.recv()
    #         #         logger.info(f"{track.id}: Received video frame {frame}")
    #         #         img = frame.to_ndarray(format="bgr24")
    #         #         cv2.imshow("iPhone Camera", img)
    #         #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #         #             break
    #         # else:
    #         #     while True:

    #         #         try:
    #         #             frame = await track.recv()
    #         #             logger.info(f"{track.id}: Received video frame {frame}")
    #         #             # img = frame.to_ndarray(format="bgr24")
    #         #             img = frame.to_ndarray(format="bgra")
    #         #             cv2.imshow("iPhone Camera Depth", img)
    #         #             if cv2.waitKey(1) & 0xFF == ord("q"):
    #         #                 break
    #         #         except Exception as e:
    #         #             logger.error(f"Error receiving depth frame: {e}")
    #         #             break
    #     else:
    #         logger.warning(f"Unknown track kind received: {track.kind}")

    #     @track.on("ended")
    #     async def on_ended():
    #         logger.info(f"Track ended: {track.kind}")

    @pc.on("datachannel")
    async def on_data_channel(channel):
        logger.info(f"Data channel received: {vars(channel)}")

        @channel.on("message")
        def on_message(message):
            logger.info(f"Message received on data channel: {message}")
            try:
                result = GestureRecognizerCustomResult.from_webrtc_result(message)
                global queue

                try:
                    queue.get_nowait()
                    # print("Dropped frame")
                except Empty:
                    pass
                queue.put_nowait(result)

            except Exception as e:
                logger.error(f"Error processing message on data channel: {e}")

        @channel.on("open")
        async def on_depth_channel(depth_data):
            logger.info("Data channel opened")

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


queue: Queue[GestureRecognizerCustomResult] = Queue(maxsize=1)
stop_event = Event()
computer_control_thread = Thread(target=run_computer_control_thread, args=(queue, stop_event))
computer_control_thread.start()

handshake_server = web.Application()
handshake_server.on_shutdown.append(on_shutdown)
handshake_server.router.add_post("/offer", offer)
