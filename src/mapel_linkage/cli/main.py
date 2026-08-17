from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mapel_linkage import __version__
from mapel_linkage.configuration import (
    compile_config,
    load_config,
    write_configuration_json_schema,
)
from mapel_linkage.governance.errors import SafeError, SafeErrorCode

UNIMPLEMENTED_COMMANDS = (
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
        description="Privacy-bounded probabilistic record linkage engine.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    status = subparsers.add_parser("status", help="Show implementation status.")
    status.set_defaults(handler=_status)

    validate = subparsers.add_parser(
        "validate-config",
        help="Validate and compile a local YAML or JSON project configuration.",
    )
    validate.add_argument("--config", metavar="CONFIG", required=True)
    validate.add_argument("--project-root", metavar="ROOT", default=".")
    validate.set_defaults(handler=_validate_config)

    schema = subparsers.add_parser(
        "emit-config-schema",
        help="Write the machine-readable configuration JSON Schema.",
    )
    schema.add_argument("--output", metavar="OUTPUT", required=True)
    schema.set_defaults(handler=_emit_schema)

    for command in UNIMPLEMENTED_COMMANDS:
        subparser = subparsers.add_parser(
            command,
            help="Target interface; not yet implemented end to end.",
        )
        subparser.add_argument("--config", metavar="CONFIG")
        subparser.set_defaults(handler=_not_implemented)
    return parser


def _status(_: argparse.Namespace) -> int:
    print("Linkage Engine M1 through M2C are implemented; M2D is an evidence-only candidate.")
    print("Calibration, assignment, decisions, and end-to-end execution remain pre-alpha.")
    return 0


def _validate_config(namespace: argparse.Namespace) -> int:
    try:
        loaded = load_config(Path(namespace.config))
        plan = compile_config(
            loaded.config,
            project_root=Path(namespace.project_root),
        )
    except SafeError as error:
        print(error.render(), file=sys.stderr)
        return 2
    summary = plan.safe_summary()
    print(
        "Configuration valid: "
        f"digest={str(summary['configuration_digest'])[:12]} "
        f"datasets={summary['dataset_count']} "
        f"variables={summary['variable_count']}"
    )
    return 0


def _emit_schema(namespace: argparse.Namespace) -> int:
    try:
        write_configuration_json_schema(Path(namespace.output))
    except OSError:
        error = SafeError(
            SafeErrorCode.CONFIG_SCHEMA_WRITE,
            "Configuration schema could not be written.",
        )
        print(error.render(), file=sys.stderr)
        return 2
    print("Configuration JSON Schema written.")
    return 0


def _not_implemented(namespace: argparse.Namespace) -> int:
    print(
        f"ERROR ML-PREALPHA-001: '{namespace.command}' is a target interface "
        "and is not implemented.",
        file=sys.stderr,
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    handler = getattr(namespace, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(namespace))
