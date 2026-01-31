import argparse

from loguru import logger
from aiohttp import web

from src.usb_server import run
from src.web_rtc_server import WebRTCServer

# NOTE in order to get this to run on mac
# - connect mac to personal hotspot on phone
# - use ifconfig | grep -C3 172 to find out the IP address the phone should target - should be the second one (172.20.10.15)


def main():
    parser = argparse.ArgumentParser(description="Conjure server - pair with ios-client")
    parser.add_argument("--tcp", action="store_true", help="Use TCP instead of UDP. Only works with --usb_mode")
    parser.add_argument("--usb_mode", action="store_true", help="Enable USB mode for receiving")
    parser.add_argument("--port", type=int, default=5001, help="Port to run the server on")
    args = parser.parse_args()

    if args.usb_mode:
        logger.info(f"Starting USB receiver on port {args.port}")
        run(args)
    else:
        logger.info(f"Starting WebRTC server on port {args.port}")
        webrtc_server = WebRTCServer(server_port=args.port)
        webrtc_server.start()
        # webrtc_server.block()


if __name__ == "__main__":
    main()
