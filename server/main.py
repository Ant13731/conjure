import argparse
from threading import Thread

from loguru import logger
from aiohttp import web

from src.webRTC_server import handshake_server
from src.usb_server import run


def main():
    parser = argparse.ArgumentParser(description="Conjure server - pair with client")
    parser.add_argument("--usb_mode", action="store_true", help="Enable WebRTC mode for video streaming")
    parser.add_argument("--force_usb_mode", action="store_false", help="Enable USB mode for video streaming")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    # parser.add_argument("--host", type=str, default="100.95.197.55", help="Host to run the server on")
    # parser.add_argument("--host", type=str, default="100.115.181.103", help="Host to run the server on")
    args = parser.parse_args()

    if args.usb_mode:
        logger.info(f"Starting USB receiver on port {args.port}")
        run(args)
    else:
        logger.info(f"Starting WebRTC server on {args.host}:{args.port}")
        web.run_app(handshake_server, host=args.host, port=args.port)

    # Start the handshake server on the current computer
    # handshake_server_thread = Thread(target=handshake_server.run, kwargs={"host": "127.0.0.1", "port": args.port})
    # handshake_server_thread.start()
    # logger.info(f"Handshake server running on port {args.port}")
    # # Keep the server running in case of disconnect

    # offer = handshake_handler()
    # process_offer(offer)


if __name__ == "__main__":
    main()
