from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mapel_linkage import __version__
from mapel_linkage.configuration import (
    compile_config,
    load_config,
    write_configuration_json_schema,
)
from mapel_linkage.domain.errors import LinkageRuntimeError
from mapel_linkage.governance.errors import SafeError, SafeErrorCode
from mapel_linkage.pipeline import SyntheticVerticalSliceRunner
from mapel_linkage.pipeline.local_workspace import initialise_local_project, run_doctor
from mapel_linkage.synthetic import SyntheticGenerationConfig

_PIPELINE_COMMANDS = (
    "generate-candidates",
    "train",
    "predict",
    "assign",
    "evaluate",
    "run",
)
_STAGE_BY_COMMAND = {
    "generate-candidates": "candidate_generation",
    "train": "pair_model_training_and_scoring",
    "predict": "champion_selection_and_calibration",
    "assign": "candidate_ranking_and_assignment",
    "evaluate": "synthetic_evaluation",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapel-linkage",
        description="Privacy-bounded probabilistic record linkage engine.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    status = subparsers.add_parser("status", help="Show implementation status.")
    status.set_defaults(handler=_status)

    doctor = subparsers.add_parser("doctor", help="Check the local execution environment safely.")
    doctor.add_argument("--project-root", metavar="ROOT", default=".")
    doctor.set_defaults(handler=_doctor)

    initialise = subparsers.add_parser(
        "init-local-project",
        help="Create ignored local operational directories without row-level examples.",
    )
    initialise.add_argument("--directory", metavar="DIRECTORY", required=True)
    initialise.set_defaults(handler=_initialise_local_project)

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

    for command in _PIPELINE_COMMANDS:
        subparser = subparsers.add_parser(
            command,
            help="Execute the installed synthetic vertical slice through the requested stage.",
        )
        subparser.add_argument("--config", metavar="CONFIG", required=True)
        subparser.add_argument("--project-root", metavar="ROOT", default=".")
        subparser.add_argument(
            "--synthetic-demo",
            action="store_true",
            help="Generate and use synthetic inputs only.",
        )
        subparser.add_argument(
            "--entity-count",
            type=int,
            default=120,
            help="Synthetic entity count; minimum 100 for protected split coverage.",
        )
        subparser.add_argument(
            "--scipy-reference",
            action="store_true",
            help="Use the dense SciPy reference assignment solver.",
        )
        subparser.set_defaults(handler=_run_pipeline_command)
    return parser


def _status(_: argparse.Namespace) -> int:
    print(
        "Linkage Engine M1 through M2E are merged; the complete synthetic MVP "
        "development candidate adds calibration, ranking, one-to-one no-match assignment, "
        "four-status decisions, restricted review export, aggregate evaluation, and "
        "orchestration."
    )
    print(
        "Synthetic testing establishes software behaviour only; real-data validation "
        "and operational approval remain local and not established."
    )
    return 0


def _doctor(namespace: argparse.Namespace) -> int:
    try:
        report = run_doctor(Path(namespace.project_root))
    except LinkageRuntimeError as error:
        print(f"ERROR {error.code}: {error.public_message}", file=sys.stderr)
        return 2
    print(json.dumps(report.safe_summary(), sort_keys=True))
    return 0 if report.ready_for_synthetic_run else 2


def _initialise_local_project(namespace: argparse.Namespace) -> int:
    try:
        created = initialise_local_project(Path(namespace.directory))
    except LinkageRuntimeError as error:
        print(f"ERROR {error.code}: {error.public_message}", file=sys.stderr)
        return 2
    print(f"Local project workspace initialized: entries={len(created)}")
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


def _generation_spec(namespace: argparse.Namespace, seed: int) -> SyntheticGenerationConfig:
    count = int(namespace.entity_count)
    if count < 100:
        raise LinkageRuntimeError(
            "ML-CLI-001",
            "The complete synthetic benchmark requires at least 100 generated entities.",
        )
    if count > 100_000:
        raise LinkageRuntimeError(
            "ML-CLI-003",
            "The complete synthetic benchmark permits at most 100000 generated entities.",
        )
    return SyntheticGenerationConfig(
        seed=seed,
        entity_count=count,
        left_only_count=max(4, count // 15),
        right_only_count=max(4, count // 15),
        duplicate_count=max(4, count // 15),
        competing_candidate_count=max(8, count // 6),
        source_a_missing_rate=0.05,
        source_b_missing_rate=0.20,
        source_b_typo_rate=0.35,
        source_b_date_shift_rate=0.20,
    )


def _run_pipeline_command(namespace: argparse.Namespace) -> int:
    if not namespace.synthetic_demo:
        print(
            "ERROR ML-CLI-002: This repository build executes row-level examples only "
            "with --synthetic-demo. Operational records must remain local.",
            file=sys.stderr,
        )
        return 2
    try:
        loaded = load_config(Path(namespace.config))
        plan = compile_config(
            loaded.config,
            project_root=Path(namespace.project_root),
        )
        result = SyntheticVerticalSliceRunner.run(
            plan,
            generation=_generation_spec(namespace, plan.random_seed),
            prefer_ortools=not bool(namespace.scipy_reference),
        )
    except SafeError as error:
        print(error.render(), file=sys.stderr)
        return 2
    except LinkageRuntimeError as error:
        print(f"ERROR {error.code}: {error.public_message}", file=sys.stderr)
        return 2

    if namespace.command == "run":
        print(json.dumps(result.safe_summary(), sort_keys=True))
        return 0
    stage_name = _STAGE_BY_COMMAND[namespace.command]
    stage = next(item for item in result.stage_summaries if item.stage == stage_name)
    print(json.dumps(stage.safe_summary(), sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    handler = getattr(namespace, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(namespace))
