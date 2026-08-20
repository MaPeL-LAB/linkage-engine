from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from mapel_linkage import __version__
from mapel_linkage.adjudication import (
    AdjudicationWorkflowRunner,
    ReviewQueueEntry,
    import_adjudications_from_csv,
    import_adjudications_from_jsonl,
    sample_active_learning_queue,
)
from mapel_linkage.benchmarking import (
    BenchmarkPortfolioRunner,
    BenchmarkRegistry,
    BenchmarkScenarioGenerator,
    generate_and_run_seed_corpus,
)
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
    MetaRankingLinkageAdvisor,
    RecommendationIntent,
    RuntimeDependency,
    SimilarityLinkageAdvisor,
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
        help="Produce an advisory Stage-1 or Stage-2 structural pipeline shortlist.",
    )
    recommend.add_argument("--config", metavar="CONFIG", required=True)
    recommend.add_argument("--project-root", metavar="ROOT", default=".")
    recommend.add_argument(
        "--intent",
        choices=tuple(item.value for item in RecommendationIntent),
        default=RecommendationIntent.DEVELOP_NEW_RECIPE.value,
    )
    recommend.add_argument(
        "--method",
        choices=("structural", "similarity", "meta-ranker"),
        default="structural",
        help="Advisor recommendation method.",
    )
    recommend.add_argument(
        "--registry-dir",
        metavar="DIR",
        default=None,
        help="Path to BenchmarkRegistry directory for similarity/meta-ranker advisor.",
    )
    recommend.set_defaults(handler=_recommend_pipeline)

    sample_queue = subparsers.add_parser(
        "sample-review-queue",
        help="Prioritize and sample a bounded review queue using active learning.",
    )
    sample_queue.add_argument("--input-queue", metavar="FILE", required=True)
    sample_queue.add_argument("--output", metavar="OUTPUT", required=True)
    sample_queue.add_argument(
        "--strategy",
        choices=("uncertainty", "margin", "committee", "hybrid"),
        default="uncertainty",
    )
    sample_queue.add_argument("--budget", type=int, default=50)
    sample_queue.set_defaults(handler=_sample_review_queue)

    benchmark = subparsers.add_parser(
        "run-benchmark",
        help="Execute synthetic benchmark portfolio and persist results in the registry.",
    )
    benchmark.add_argument("--output-dir", metavar="DIR", required=True)
    benchmark.add_argument(
        "--families",
        metavar="FAMILIES",
        default=None,
        help="Comma-separated scenario families to benchmark.",
    )
    benchmark.add_argument(
        "--replicates",
        type=int,
        default=1,
        metavar="N",
        help="Number of replicates per instance.",
    )
    benchmark.set_defaults(handler=_run_benchmark)

    seed_bm = subparsers.add_parser(
        "seed-benchmarks",
        help="Generate standard benchmark scenario families and populate the registry.",
    )
    seed_bm.add_argument("--registry-dir", metavar="DIR", required=True)
    seed_bm.add_argument(
        "--replicates",
        type=int,
        default=2,
        metavar="N",
        help="Number of replicates per instance.",
    )
    seed_bm.set_defaults(handler=_seed_benchmarks)

    import_rev = subparsers.add_parser(
        "import-reviews",
        help="Import reviewer decision batches into the append-only adjudication ledger.",
    )
    import_rev.add_argument("--reviews", metavar="FILE", required=True)
    import_rev.add_argument("--ledger-path", metavar="LEDGER", default=None)
    import_rev.add_argument("--ledger-id", metavar="ID", default="adjudication-ledger")
    import_rev.set_defaults(handler=_import_reviews)

    consensus = subparsers.add_parser(
        "resolve-consensus",
        help="Resolve multi-reviewer adjudication consensus and identify disagreements.",
    )
    consensus.add_argument("--reviews", metavar="FILE", required=True)
    consensus.add_argument(
        "--policy",
        choices=(
            "majority_vote",
            "unanimous_only",
            "senior_reviewer_override",
            "strict_double_review",
        ),
        default="majority_vote",
    )
    consensus.add_argument("--threshold", type=float, default=0.5)
    consensus.set_defaults(handler=_resolve_consensus)

    promote = subparsers.add_parser(
        "promote-labels",
        help="Promote consensus decisions into an immutable verified label batch.",
    )
    promote.add_argument("--reviews", metavar="FILE", required=True)
    promote.add_argument("--output", metavar="OUTPUT", required=True)
    promote.add_argument("--label-source-kind", default="verified_human_adjudication")
    promote.add_argument(
        "--partition",
        choices=("training", "validation", "calibration"),
        default="training",
    )
    promote.set_defaults(handler=_promote_labels)

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
        method = getattr(namespace, "method", "structural")
        if method == "meta-ranker":
            registry = (
                BenchmarkRegistry(Path(namespace.registry_dir))
                if getattr(namespace, "registry_dir", None)
                else None
            )
            advisor_meta = MetaRankingLinkageAdvisor(registry=registry)
            report_meta = advisor_meta.advise(plan, context=context, profile=profile)
            print(json.dumps(report_meta.safe_summary(), sort_keys=True))
            return 0
        elif method == "similarity":
            registry = (
                BenchmarkRegistry(Path(namespace.registry_dir))
                if getattr(namespace, "registry_dir", None)
                else None
            )
            advisor = SimilarityLinkageAdvisor(registry=registry)
            report = advisor.recommend(plan, context=context, profile=profile)
            print(json.dumps(report.safe_summary(), sort_keys=True))
            return 0
        else:
            recommendation = recommend_pipeline(plan, context=context, profile=profile)
            print(json.dumps(recommendation.safe_summary(), sort_keys=True))
            return 0
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


def _sample_review_queue(namespace: argparse.Namespace) -> int:
    in_path = Path(namespace.input_queue)
    out_path = Path(namespace.output)
    budget = max(0, int(namespace.budget))
    strategy = namespace.strategy

    entries: list[ReviewQueueEntry] = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        entries.append(
            ReviewQueueEntry(
                relationship_id=data["relationship_id"],
                source_record_ref=data.get("source_record_ref", ""),
                target_record_ref=data.get("target_record_ref"),
                relationship_status=data.get("relationship_status", "review_required"),
                calibrated_probability=data.get("calibrated_probability"),
                candidate_rank=data.get("candidate_rank"),
                probability_margin=data.get("probability_margin", 0.0),
                review_reason_codes=tuple(data.get("review_reason_codes", ("review_required",))),
                model_version=data.get("model_version", "v1.0"),
                decision_rule_id=data.get("decision_rule_id", "rule"),
                assignment_method=data.get("assignment_method", "ortools"),
                run_id=data.get("run_id", "run_01"),
            )
        )

    sampled = sample_active_learning_queue(entries, budget=budget, strategy=strategy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_lines = [
        json.dumps(
            {
                "relationship_id": e.relationship_id,
                "relationship_status": e.relationship_status,
                "candidate_rank": e.candidate_rank,
                "calibrated_probability": e.calibrated_probability,
                "probability_margin": e.probability_margin,
                "review_reason_codes": list(e.review_reason_codes),
                "model_version": e.model_version,
                "decision_rule_id": e.decision_rule_id,
                "assignment_method": e.assignment_method,
                "run_id": e.run_id,
            },
            sort_keys=True,
        )
        + "\n"
        for e in sampled.entries
    ]
    out_path.write_text("".join(out_lines), encoding="utf-8")
    print(json.dumps(sampled.safe_summary(), sort_keys=True))
    return 0


def _seed_benchmarks(namespace: argparse.Namespace) -> int:
    reg_dir = Path(namespace.registry_dir)
    replicates = max(1, int(namespace.replicates))
    registry = generate_and_run_seed_corpus(
        registry_directory=reg_dir,
        replicates=replicates,
    )
    report = registry.generate_coverage_report()
    print(
        json.dumps(
            {
                "seed_corpus_report": report.safe_summary(),
                "registry_dir": str(reg_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def _import_reviews(namespace: argparse.Namespace) -> int:
    reviews_path = Path(namespace.reviews)
    ledger_path = Path(namespace.ledger_path) if namespace.ledger_path else None
    res = AdjudicationWorkflowRunner.import_reviews(
        reviews_source=reviews_path,
        ledger_path=ledger_path,
        ledger_id=namespace.ledger_id,
        strict_candidate_check=False,
    )
    print(json.dumps(res.safe_summary(), sort_keys=True))
    return 0


def _resolve_consensus(namespace: argparse.Namespace) -> int:
    reviews_path = Path(namespace.reviews)
    if reviews_path.suffix.lower() == ".csv":
        imported = import_adjudications_from_csv(reviews_path)
    else:
        imported = import_adjudications_from_jsonl(reviews_path)

    report = AdjudicationWorkflowRunner.resolve_consensus(
        reviews=imported.records,
        policy=namespace.policy,
        agreement_threshold=float(namespace.threshold),
    )
    print(json.dumps(report.safe_summary(), sort_keys=True))
    return 0


def _promote_labels(namespace: argparse.Namespace) -> int:
    reviews_path = Path(namespace.reviews)
    output_path = Path(namespace.output)
    if reviews_path.suffix.lower() == ".csv":
        imported = import_adjudications_from_csv(reviews_path)
    else:
        imported = import_adjudications_from_jsonl(reviews_path)

    res = AdjudicationWorkflowRunner.promote_to_verified_labels(
        consensus_items=imported.records,
        target_partition=namespace.partition,
        output_manifest_path=output_path,
    )
    print(json.dumps(res.safe_summary(), sort_keys=True))
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


def _run_benchmark(namespace: argparse.Namespace) -> int:
    output_dir = Path(namespace.output_dir)
    replicates = max(1, int(namespace.replicates))
    families_filter = (
        [f.strip() for f in namespace.families.split(",") if f.strip()]
        if namespace.families
        else None
    )

    generator = BenchmarkScenarioGenerator()
    runner = BenchmarkPortfolioRunner()
    registry = BenchmarkRegistry(output_dir)

    for fam in generator.list_families():
        registry.save_family(fam)
    for inst in generator.list_instances():
        registry.save_instance(inst)

    results = runner.run_portfolio(
        generator,
        families=families_filter,
        replicates=replicates,
    )

    for res in results:
        registry.save_run_record(res.record, metrics=res.metrics, failure=res.failure)

    report = registry.generate_coverage_report()
    print(
        json.dumps(
            {
                "benchmark_report": report.safe_summary(),
                "output_directory": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    handler = getattr(namespace, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(namespace))
