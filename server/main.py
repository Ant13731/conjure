import argparse
from threading import Thread

from loguru import logger
from aiohttp import web

from src.webRTC_server import handshake_server
from src.usb_server import run

# NOTE in order to get this to run on mac
# - connect mac to personal hotspot on phone
# - use ifconfig | grep -C3 172 to find out the IP address the phone should target - should be the second one (172.20.10.15)


def main():
    parser = argparse.ArgumentParser(description="Conjure server - pair with client")
    parser.add_argument("--tcp_mode", action="store_true", help="Use TCP over USB instead of UDP")
    parser.add_argument("--webrtc_mode", action="store_true", help="Enable WebRTC mode for video streaming")
    # parser.add_argument("--force_usb_mode", action="store_false", help="Enable USB mode for video streaming")
    parser.add_argument("--port", type=int, default=5001, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    # parser.add_argument("--host", type=str, default="100.95.197.55", help="Host to run the server on")
    # parser.add_argument("--host", type=str, default="100.115.181.103", help="Host to run the server on")
    args = parser.parse_args()

    if args.webrtc_mode:
        logger.info(f"Starting WebRTC server on {args.host}:{args.port}")
        web.run_app(handshake_server, host=args.host, port=args.port)
    else:
        logger.info(f"Starting USB receiver on port {args.port}")
        run(args)


if __name__ == "__main__":
    main()
