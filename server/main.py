import argparse
import signal
import sys

from loguru import logger

from src.web_rtc_server import WebRTCServer

# NOTE in order to get this to run on mac
# - connect mac to personal hotspot on phone
# - use ifconfig | grep -C3 172 to find out the IP address the phone should target - should be the second one (172.20.10.15)


def main():
    parser = argparse.ArgumentParser(description="Conjure server - pair with ios-client")
    parser.add_argument("--port", type=int, default=5001, help="Port to run the server on")
    args = parser.parse_args()

    logger.info(f"Starting WebRTC server on port {args.port}")
    webrtc_server = WebRTCServer(server_port=args.port)

    def signal_handler(sig, frame):
        logger.info("Keyboard interrupt received, shutting down gracefully...")
        webrtc_server.stop_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        webrtc_server.start()
    except KeyboardInterrupt:
        logger.info("Server interrupted")
        webrtc_server.stop_event.set()


if __name__ == "__main__":
    main()
