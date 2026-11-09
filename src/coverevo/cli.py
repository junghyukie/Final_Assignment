"""Command-line entry point for optional real LLM heuristic discovery."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coverevo",
        description="COVER-EVO: heuristic discovery for minimum weighted dominating set.",
    )
    parser.add_argument("--version", action="version", version=f"COVER-EVO {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    discovery = subparsers.add_parser(
        "discovery", help="Run optional real LLM heuristic discovery."
    )
    discovery.add_argument(
        "--config", type=Path, default=Path("configs/protocol.live.yaml"),
        help="Live discovery configuration (default: configs/protocol.live.yaml).",
    )
    discovery.add_argument("--output", type=Path, required=True)
    discovery.add_argument("--env-file", type=Path, default=Path(".env"))
    discovery.add_argument(
        "--allow-live", action="store_true",
        help="Explicitly authorize provider requests, which may incur charges.",
    )
    discovery.add_argument(
        "--skip-pilot", action="store_true",
        help="Explicitly skip the pilot, as required by the direct discovery runner.",
    )
    discovery.add_argument(
        "--check-config", action="store_true",
        help="Validate configuration without reading credentials or making requests.",
    )
    discovery.add_argument(
        "--max-attempts", type=int,
        help="Stop after this many attempts in this invocation; zero is allowed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    from .live import read_live_config, run_live_discovery
    from .storage import encode_lossless

    try:
        if args.check_config:
            config = read_live_config(args.config)
            result = {
                "status": "CONFIG_VALID", "network_calls": 0,
                "model_id": config.model["id"],
                "local_cap_usd": config.budget["local_cap_usd"],
                "runs": config.search["runs"],
                "slots_per_run": config.search["slots_per_run"],
            }
        else:
            def progress(record):
                print(
                    "COVER-EVO LIVE progress: " + json.dumps(record, sort_keys=True),
                    flush=True,
                )

            result = run_live_discovery(
                args.config, args.output, env_path=args.env_file,
                allow_live=args.allow_live, skip_pilot=args.skip_pilot,
                max_attempts=args.max_attempts, progress=progress,
            )
        print("COVER-EVO discovery: " + json.dumps(encode_lossless(result), sort_keys=True))
        return 1 if result.get("status") == "STOPPED" else 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"COVER-EVO discovery: ERROR: {error}", file=sys.stderr)
        return 1
