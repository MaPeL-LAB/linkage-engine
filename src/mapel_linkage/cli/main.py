from __future__ import annotations

import argparse
from collections.abc import Sequence

from mapel_linkage import __version__

TARGET_COMMANDS = (
    "validate-config",
    "generate-candidates",
    "train",
    "predict",
    "assign",
    "evaluate",
    "run",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapel-linkage",
        description="Linkage Engine pre-alpha documentation scaffold.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    status = subparsers.add_parser("status", help="Show implementation status.")
    status.set_defaults(handler=_status)
    for command in TARGET_COMMANDS:
        subparser = subparsers.add_parser(
            command,
            help="Target interface; not implemented yet.",
        )
        subparser.add_argument("--config", metavar="CONFIG")
        subparser.set_defaults(handler=_not_implemented)
    return parser


def _status(_: argparse.Namespace) -> int:
    print("Linkage Engine is a documentation-first pre-alpha scaffold.")
    print("No production linkage model or real-data validation is implemented.")
    return 0


def _not_implemented(namespace: argparse.Namespace) -> int:
    print(
        f"ERROR ML-PREALPHA-001: '{namespace.command}' is a target interface "
        "and is not implemented."
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    handler = getattr(namespace, "handler", None)
    if handler is None:
        build_parser().print_help()
        return 0
    return int(handler(namespace))
