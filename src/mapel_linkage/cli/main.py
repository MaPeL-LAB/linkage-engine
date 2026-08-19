from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from mapel_linkage import __version__
from mapel_linkage.capabilities import WorkflowStatus, capabilities, capability_summary
from mapel_linkage.configuration import (
    compile_config,
    load_config,
    write_configuration_json_schema,
)
from mapel_linkage.configuration.compiler import ExecutionPlan
from mapel_linkage.domain.errors import LinkageRuntimeError
from mapel_linkage.governance.errors import SafeError, SafeErrorCode
from mapel_linkage.pipeline import SyntheticVerticalSliceRunner
from mapel_linkage.pipeline.local_workspace import initialise_local_project, run_doctor
from mapel_linkage.profiling import build_preflight_task_profile
from mapel_linkage.recommendation import (
    AdvisorContext,
    RecommendationIntent,
    RuntimeDependency,
    recommend_pipeline,
)
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

    status = subparsers.add_parser("status", help="Show auditable implementation status.")
    status_format = status.add_mutually_exclusive_group()
    status_format.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the complete machine-readable capability matrix.",
    )
    status_format.add_argument(
        "--details",
        action="store_true",
        help="Emit one privacy-safe line per capability.",
    )
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

    profile = subparsers.add_parser(
        "profile-job",
        help="Emit a privacy-safe configuration-only preflight task profile.",
    )
    profile.add_argument("--config", metavar="CONFIG", required=True)
    profile.add_argument("--project-root", metavar="ROOT", default=".")
    profile.set_defaults(handler=_profile_job)

    recommend = subparsers.add_parser(
        "recommend-pipeline",
        help="Produce an advisory Stage-1 structural pipeline shortlist.",
    )
    recommend.add_argument("--config", metavar="CONFIG", required=True)
    recommend.add_argument("--project-root", metavar="ROOT", default=".")
    recommend.add_argument(
        "--intent",
        choices=tuple(item.value for item in RecommendationIntent),
        default=RecommendationIntent.DEVELOP_NEW_RECIPE.value,
    )
    recommend.set_defaults(handler=_recommend_pipeline)

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


def _status(namespace: argparse.Namespace) -> int:
    registered = capabilities()
    summary = capability_summary()
    if namespace.as_json:
        print(
            json.dumps(
                {
                    "engine_version": __version__,
                    "summary": summary,
                    "capabilities": [item.safe_summary() for item in registered],
                },
                sort_keys=True,
            )
        )
        return 0

    if namespace.details:
        for item in registered:
            print(
                f"{item.capability_id}\tcomponent={item.component_status.value}\t"
                f"workflow={item.workflow_status.value}\t"
                f"runtime={item.runtime_verification.value}\t"
                "operational_validation=not_established"
            )
        return 0

    integrated_count = sum(item.workflow_status is WorkflowStatus.INTEGRATED for item in registered)
    component_only_count = sum(
        item.workflow_status is WorkflowStatus.COMPONENT_ONLY for item in registered
    )
    print(
        "Linkage Engine capability status: "
        f"workflow_integrated={integrated_count} "
        f"component_only={component_only_count} "
        f"total={len(registered)}."
    )
    print(
        "Complete orchestrated workflow: generated-synthetic two-source link_only "
        "with one-to-one assignment."
    )
    print(
        "The Stage-1 Linkage Strategy Advisor performs structural eligibility, Pareto "
        "shortlisting, and abstention without empirical performance claims."
    )
    print(
        "M3 through M7 contain implemented components whose general configuration and "
        "CLI orchestration is still pending."
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


def _compile_plan(namespace: argparse.Namespace) -> ExecutionPlan:
    loaded = load_config(Path(namespace.config))
    return compile_config(
        loaded.config,
        project_root=Path(namespace.project_root),
    )


def _validate_config(namespace: argparse.Namespace) -> int:
    try:
        plan = _compile_plan(namespace)
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


def _profile_job(namespace: argparse.Namespace) -> int:
    try:
        profile = build_preflight_task_profile(_compile_plan(namespace))
    except SafeError as error:
        print(error.render(), file=sys.stderr)
        return 2
    except LinkageRuntimeError as error:
        print(f"ERROR {error.code}: {error.public_message}", file=sys.stderr)
        return 2
    except ValidationError:
        print(
            "ERROR ML-PROFILE-001: The preflight task profile could not be constructed.",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(profile.safe_summary(), sort_keys=True))
    return 0


def _available_runtimes() -> tuple[RuntimeDependency, ...]:
    runtimes = [RuntimeDependency.CORE]
    if importlib.util.find_spec("lightgbm") is not None:
        runtimes.append(RuntimeDependency.LIGHTGBM)
    if importlib.util.find_spec("torch") is not None:
        runtimes.append(RuntimeDependency.PYTORCH)
    return tuple(runtimes)


def _verified_labels_available(plan: ExecutionPlan) -> bool:
    labels = plan.config.labels
    return labels is not None and labels.source.kind in {
        "synthetic_truth",
        "verified_human_adjudication",
        "verified_gold_standard",
    }


def _recommend_pipeline(namespace: argparse.Namespace) -> int:
    try:
        plan = _compile_plan(namespace)
        profile = build_preflight_task_profile(plan)
        context = AdvisorContext(
            intent=RecommendationIntent(namespace.intent),
            verified_labels_available=_verified_labels_available(plan),
            approved_recipe_available=False,
            protected_out_of_fold_predictions_available=False,
            available_runtimes=_available_runtimes(),
            approved_artifact_model_ids=(),
            benchmark_family_count=0,
        )
        recommendation = recommend_pipeline(plan, context=context, profile=profile)
    except SafeError as error:
        print(error.render(), file=sys.stderr)
        return 2
    except LinkageRuntimeError as error:
        print(f"ERROR {error.code}: {error.public_message}", file=sys.stderr)
        return 2
    except ValidationError:
        print(
            "ERROR ML-ADVISOR-002: The structural recommendation could not be constructed.",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(recommendation.safe_summary(), sort_keys=True))
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
        plan = _compile_plan(namespace)
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
