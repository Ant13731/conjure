import argparse
from threading import Thread

from loguru import logger
from aiohttp import web

from src.handshake import handshake_server


def main():
    parser = argparse.ArgumentParser(description="Conjure server - pair with client")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="100.115.181.103", help="Host to run the server on")
    args = parser.parse_args()

    logger.info(f"Starting server on {args.host}:{args.port}")

    handshake_server_runner = web.run_app(handshake_server, host=args.host, port=args.port)

    # Start the handshake server on the current computer
    # handshake_server_thread = Thread(target=handshake_server.run, kwargs={"host": "127.0.0.1", "port": args.port})
    # handshake_server_thread.start()
    # logger.info(f"Handshake server running on port {args.port}")
    # # Keep the server running in case of disconnect

    # offer = handshake_handler()
    # process_offer(offer)


if __name__ == "__main__":
    main()
