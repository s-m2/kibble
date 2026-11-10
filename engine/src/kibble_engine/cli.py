"""Command line entry point. Plain argparse on purpose."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys

from kibble_engine import __version__


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from kibble_engine.api.app import create_app

    token = None if args.no_auth else os.environ.get("KIBBLE_TOKEN")
    if token is None and not args.no_auth:
        print(
            "KIBBLE_TOKEN is not set; refusing to start (use --no-auth for local dev).",
            file=sys.stderr,
        )
        return 2

    port = args.port or _pick_free_port("127.0.0.1")
    # Single handshake line on stdout so a launcher can discover the port.
    # Everything else (logs, diagnostics) goes to stderr.
    print(json.dumps({"event": "listening", "host": "127.0.0.1", "port": port}), flush=True)
    uvicorn.run(create_app(token), host="127.0.0.1", port=port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kibble-engine")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print the engine version")

    serve = sub.add_parser("serve", help="run the local HTTP engine (127.0.0.1 only)")
    serve.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    serve.add_argument("--no-auth", action="store_true", help="disable the token check (dev only)")

    args = parser.parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    return _serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
