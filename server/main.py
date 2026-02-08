import argparse
import signal
import sys

from loguru import logger

from src.server.web_rtc_server import WebRTCServer
from src.server.udp_server import UDPServer

# NOTE in order to get this to run on usb with a macbook
# - connect mac to personal hotspot on phone
# - use ifconfig | grep -C3 172 to find out the IP address the phone should target - should be the second one (172.20.10.15)


def main():
    parser = argparse.ArgumentParser(description="Conjure server - pair with ios-client")
    parser.add_argument("--port", type=int, default=5001, help="Port to run the server on")
    parser.add_argument("--use_webrtc", action="store_true", help="Whether to use WebRTC server or UDP server")
    args = parser.parse_args()

    if args.use_webrtc:
        logger.info(f"Starting WebRTC server on port {args.port}")
        server = WebRTCServer(server_port=args.port)
    else:
        logger.info(f"Starting UDP server on port {args.port}")
        server = UDPServer(server_port=args.port)

    def signal_handler(sig, frame):
        logger.info("Keyboard interrupt received, shutting down gracefully...")
        server.stop_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Server interrupted")
        server.stop_event.set()


if __name__ == "__main__":
    main()
