from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest
from pydantic import ValidationError

from mapel_linkage.benchmarking.advisor_catalogue import (
    BenchmarkShardPlan,
    build_advisor_corpus_design,
    build_benchmark_shard_plan,
)
from mapel_linkage.benchmarking.advisor_execution import (
    CorpusExecutionApproval,
    CorpusShardExecutionReport,
    audit_advisor_corpus,
    execute_advisor_corpus_shard,
)
from mapel_linkage.benchmarking.contracts import (
    BenchmarkAggregateMetrics,
    BenchmarkFailureRecord,
    BenchmarkRunRecord,
    BenchmarkRunStatus,
)
from mapel_linkage.benchmarking.generator import BenchmarkScenarioGenerator
from mapel_linkage.benchmarking.registry import BenchmarkRegistry
from mapel_linkage.benchmarking.runner import (
    BenchmarkPortfolioRunner,
    BenchmarkRecipe,
    BenchmarkRunResult,
    benchmark_replicate_seed,
    benchmark_run_id,
)

_DIGEST = "c" * 64


class _AggregateOnlyFakeRunner(BenchmarkPortfolioRunner):
    def run_portfolio(
        self,
        generator: BenchmarkScenarioGenerator,
        *,
        families: Iterable[str] | None = None,
        instances: Iterable[str] | None = None,
        recipes: Iterable[BenchmarkRecipe] | None = None,
        replicates: int = 1,
        base_seed: int = 20260816,
        replicate_start: int = 0,
    ) -> tuple[BenchmarkRunResult, ...]:
        del families
        assert replicates == 1
        target_instances = tuple(instances or ())
        target_recipes = tuple(recipes or self.list_recipes())
        provenance = self.provenance_summary()
        results: list[BenchmarkRunResult] = []
        for instance_id in target_instances:
            instance = generator.get_instance(instance_id)
            profile = generator.build_task_profile(instance_id)
            replicate_id = f"replicate.{replicate_start:07d}"
            seed = benchmark_replicate_seed(
                instance_id=instance_id,
                replicate_number=replicate_start,
                base_seed=base_seed,
            )
            for index, recipe in enumerate(target_recipes):
                run_id = benchmark_run_id(
                    instance_id=instance_id,
                    recipe_id=recipe.recipe_id,
                    replicate_id=replicate_id,
                )
                if self.adapter_statuses()[recipe.recipe_id] == "success_capable":
                    metrics = BenchmarkAggregateMetrics(
                        candidate_recall=0.80 + 0.01 * index,
                        candidate_recall_at_k={"1": 0.70 + 0.01 * index, "5": 0.9},
                        sensitivity=0.75,
                        positive_predictive_value=0.72,
                        brier_score=0.18,
                        calibration_intercept=0.0,
                        calibration_slope=1.0,
                        mean_reciprocal_rank=0.78,
                        runtime_ms=1,
                        peak_memory_mb=1,
                    )
                    record = BenchmarkRunRecord(
                        run_id=run_id,
                        family_id=instance.family_id,
                        instance_id=instance_id,
                        replicate_id=replicate_id,
                        task_profile_digest=profile.profile_digest,
                        pipeline_recipe_digest=recipe.recipe_digest,
                        engine_commit=provenance["engine_commit"],
                        dependency_lock_digest=provenance["dependency_lock_digest"],
                        environment_digest=provenance["environment_digest"],
                        random_seed=seed,
                        status=BenchmarkRunStatus.SUCCESS,
                        aggregate_metrics_digest=metrics.metrics_digest,
                        stage_artifact_manifest_digest=_DIGEST,
                        runtime_ms=1,
                        peak_memory_mb=1,
                    )
                    results.append(BenchmarkRunResult(record=record, metrics=metrics))
                else:
                    failure = BenchmarkFailureRecord(
                        run_id=run_id,
                        family_id=instance.family_id,
                        instance_id=instance_id,
                        replicate_id=replicate_id,
                        recipe_id=recipe.recipe_id,
                        status=BenchmarkRunStatus.INELIGIBLE,
                        failure_code="ML-BENCH-INELIGIBLE-ADAPTER",
                        error_message="No truth-safe package benchmark adapter is registered.",
                    )
                    record = BenchmarkRunRecord(
                        run_id=run_id,
                        family_id=instance.family_id,
                        instance_id=instance_id,
                        replicate_id=replicate_id,
                        task_profile_digest=profile.profile_digest,
                        pipeline_recipe_digest=recipe.recipe_digest,
                        engine_commit=provenance["engine_commit"],
                        dependency_lock_digest=provenance["dependency_lock_digest"],
                        environment_digest=provenance["environment_digest"],
                        random_seed=seed,
                        status=BenchmarkRunStatus.INELIGIBLE,
                        failure_code=failure.failure_code,
                        runtime_ms=0,
                        peak_memory_mb=0,
                    )
                    results.append(BenchmarkRunResult(record=record, failure=failure))
        return tuple(results)


def _approval(plan: BenchmarkShardPlan) -> CorpusExecutionApproval:
    return CorpusExecutionApproval(
        approval_reference="synthetic-corpus-owner-approval",
        human_approved=True,
        design_digest=build_advisor_corpus_design().design_digest,
        shard_plan_digest=plan.plan_digest,
        replicates=1,
    )


def _execute(
    registry: BenchmarkRegistry,
) -> tuple[BenchmarkShardPlan, CorpusShardExecutionReport]:
    plan = build_benchmark_shard_plan(shard_count=256)
    report = execute_advisor_corpus_shard(
        registry=registry,
        shard_plan=plan,
        shard_index=0,
        approval=_approval(plan),
        replicates=1,
        runner=_AggregateOnlyFakeRunner(),
    )
    return plan, report


def test_execution_requires_literal_human_approval() -> None:
    plan = build_benchmark_shard_plan(shard_count=256)
    with pytest.raises(ValidationError):
        CorpusExecutionApproval.model_validate(
            {
                "approval_reference": "synthetic-owner",
                "human_approved": False,
                "design_digest": build_advisor_corpus_design().design_digest,
                "shard_plan_digest": plan.plan_digest,
                "replicates": 1,
            }
        )


def test_shard_execution_is_append_only_idempotent_and_privacy_safe(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path / "registry")
    plan, first = _execute(registry)
    second = execute_advisor_corpus_shard(
        registry=registry,
        shard_plan=plan,
        shard_index=0,
        approval=_approval(plan),
        replicates=1,
        runner=_AggregateOnlyFakeRunner(),
    )

    assert first.newly_persisted_run_count == 14
    assert first.successful_run_count == 6
    assert first.retained_non_success_run_count == 8
    assert second.newly_persisted_run_count == 0
    assert second.resumed_run_count == 14
    rendered = json.dumps(second.safe_summary(), sort_keys=True)
    assert "synthetic-corpus-owner-approval" not in rendered
    assert str(tmp_path) not in rendered
    assert "instance.advisor_v2" not in rendered
    assert second.contains_record_values is False
    assert second.contains_identifiers is False
    assert second.contains_candidate_pairs is False
    assert second.decision_authority == "none"
    assert second.assignment_authority == "none"
    assert second.merge_authority == "none"
    assert second.automatic_promotion == "prohibited"

    readiness = audit_advisor_corpus(
        registry=registry,
        shard_plan=plan,
        replicates=1,
        runner=_AggregateOnlyFakeRunner(),
    )
    assert readiness.execution_status == "partial"
    assert readiness.expected_run_count == 1960
    assert readiness.completed_run_count == 14
    assert readiness.planned_replicates_per_instance == 1
    assert readiness.required_evidence_cell_count == 280
    assert readiness.successful_evidence_cell_count == 2
    assert readiness.expected_required_adapter_run_count == 840
    assert readiness.successful_required_adapter_run_count == 6
    assert readiness.failed_required_adapter_run_count == 0
    assert readiness.missing_required_adapter_run_count == 834
    assert readiness.successful_overlap_family_count == 2
    assert readiness.advisor_evidence_ready is False


def test_resume_rejects_tampered_metrics_without_overwrite(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path / "registry")
    plan, _ = _execute(registry)
    metrics_path = next(registry.metrics_dir.glob("*.json"))
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["candidate_recall"] = 0.01
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity"):
        execute_advisor_corpus_shard(
            registry=registry,
            shard_plan=plan,
            shard_index=0,
            approval=_approval(plan),
            replicates=1,
            runner=_AggregateOnlyFakeRunner(),
        )
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["candidate_recall"] == 0.01


def test_resume_rejects_stale_environment_and_governance_collision(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path / "registry")
    plan, _ = _execute(registry)
    run_path = next(registry.runs_dir.glob("*.json"))
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["environment_digest"] = "d" * 64
    run_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FileExistsError, match="stale"):
        execute_advisor_corpus_shard(
            registry=registry,
            shard_plan=plan,
            shard_index=0,
            approval=_approval(plan),
            replicates=1,
            runner=_AggregateOnlyFakeRunner(),
        )

    approval_path = next((registry.root_directory / "governance").glob("approval.*.json"))
    approval_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="conflicting"):
        execute_advisor_corpus_shard(
            registry=registry,
            shard_plan=plan,
            shard_index=0,
            approval=_approval(plan),
            replicates=1,
            runner=_AggregateOnlyFakeRunner(),
        )
