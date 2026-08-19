"""Tests for the file-backed benchmark registry and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mapel_linkage.benchmarking.contracts import (
    BenchmarkAggregateMetrics,
    BenchmarkFailureRecord,
    BenchmarkRunRecord,
    BenchmarkRunStatus,
    ScenarioFamilyManifest,
    ScenarioInstanceManifest,
)
from mapel_linkage.benchmarking.generator import BenchmarkScenarioGenerator
from mapel_linkage.benchmarking.registry import BenchmarkRegistry
from mapel_linkage.benchmarking.runner import BenchmarkPortfolioRunner
from mapel_linkage.cli.main import main

_DIGEST = "a" * 64
_OTHER_DIGEST = "b" * 64


def test_registry_save_and_load_manifests(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path)

    family = ScenarioFamilyManifest(
        family_id="family.typo",
        mechanism_tags=("character_substitution", "token_transposition"),
        latent_scenario_manifest_digest=_DIGEST,
        prospectively_held_out=True,
    )
    instance = ScenarioInstanceManifest(
        family_id=family.family_id,
        instance_id="instance.low",
        family_digest=family.family_digest,
        latent_parameter_manifest_digest=_OTHER_DIGEST,
        observable_profile_digest=_DIGEST,
        planned_replicates=5,
    )

    registry.save_family(family)
    registry.save_instance(instance)

    loaded_family = registry.load_family("family.typo")
    loaded_instance = registry.load_instance("instance.low")

    assert loaded_family.family_id == family.family_id
    assert loaded_family.family_digest == family.family_digest
    assert loaded_instance.instance_id == instance.instance_id
    assert loaded_instance.instance_digest == instance.instance_digest

    assert len(registry.list_families()) == 1
    assert len(registry.list_instances()) == 1


def test_registry_save_run_record_metrics_and_failure(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path)

    metrics = BenchmarkAggregateMetrics(
        candidate_recall=0.95,
        candidate_recall_at_k={"1": 0.8, "5": 0.95},
        sensitivity=0.90,
        positive_predictive_value=0.85,
        brier_score=0.08,
        calibration_intercept=0.01,
        calibration_slope=1.02,
        mean_reciprocal_rank=0.88,
        runtime_ms=150,
        peak_memory_mb=64,
    )

    run_record = BenchmarkRunRecord(
        run_id="run.test.001",
        family_id="family.typo",
        instance_id="instance.low",
        replicate_id="replicate.001",
        task_profile_digest=_DIGEST,
        pipeline_recipe_digest=_OTHER_DIGEST,
        engine_commit="1" * 40,
        dependency_lock_digest=_DIGEST,
        environment_digest=_OTHER_DIGEST,
        random_seed=42,
        status=BenchmarkRunStatus.SUCCESS,
        aggregate_metrics_digest=metrics.metrics_digest,
        stage_artifact_manifest_digest=_OTHER_DIGEST,
        runtime_ms=150,
        peak_memory_mb=64,
    )

    registry.save_run_record(run_record, metrics=metrics)

    loaded_record = registry.load_run_record("run.test.001")
    loaded_metrics = registry.load_metrics("run.test.001")

    assert loaded_record.run_id == run_record.run_id
    assert loaded_record.aggregate_metrics_digest == metrics.metrics_digest
    assert loaded_metrics is not None
    assert loaded_metrics.candidate_recall == 0.95

    # Failure record test
    fail_record = BenchmarkRunRecord(
        run_id="run.test.failed",
        family_id="family.typo",
        instance_id="instance.high",
        replicate_id="replicate.002",
        task_profile_digest=_DIGEST,
        pipeline_recipe_digest=_OTHER_DIGEST,
        engine_commit="2" * 40,
        dependency_lock_digest=_DIGEST,
        environment_digest=_OTHER_DIGEST,
        random_seed=43,
        status=BenchmarkRunStatus.INELIGIBLE,
        failure_code="ML-BENCH-INELIGIBLE-LABELS",
    )
    failure_data = BenchmarkFailureRecord(
        run_id="run.test.failed",
        family_id="family.typo",
        instance_id="instance.high",
        replicate_id="replicate.002",
        recipe_id="recipe.xgboost",
        status=BenchmarkRunStatus.INELIGIBLE,
        failure_code="ML-BENCH-INELIGIBLE-LABELS",
        error_message="Labels unavailable in scenario",
    )
    registry.save_run_record(fail_record, failure=failure_data)

    loaded_fail = registry.load_failure_record("run.test.failed")
    assert loaded_fail is not None
    assert loaded_fail.failure_code == "ML-BENCH-INELIGIBLE-LABELS"


def test_registry_snapshot_and_coverage_report(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path)
    generator = BenchmarkScenarioGenerator()
    runner = BenchmarkPortfolioRunner()

    for fam in generator.list_families():
        registry.save_family(fam)
    for inst in generator.list_instances():
        registry.save_instance(inst)

    results = runner.run_portfolio(
        generator,
        families=("family.typo_stress",),
        replicates=1,
    )
    for res in results:
        registry.save_run_record(res.record, metrics=res.metrics, failure=res.failure)

    report = registry.generate_coverage_report()

    assert report.family_count == len(generator.list_families())
    assert report.instance_count == len(generator.list_instances())
    assert report.run_count == len(results)
    assert report.successful_run_count > 0
    assert report.held_out_mechanism_count >= 1
    assert isinstance(report.recipe_by_family_coverage, dict)
    assert isinstance(report.pairwise_comparison_counts, dict)

    summary = report.safe_summary()
    assert summary["contains_record_values"] is False


def test_cli_run_benchmark(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bench_dir = tmp_path / "benchmark_out"
    exit_code = main(
        [
            "run-benchmark",
            "--output-dir",
            str(bench_dir),
            "--families",
            "family.typo_stress",
            "--replicates",
            "1",
        ]
    )
    assert exit_code == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "benchmark_report" in payload
    assert payload["benchmark_report"]["successful_run_count"] > 0
    assert (bench_dir / "families").exists()
    assert (bench_dir / "instances").exists()
    assert (bench_dir / "runs").exists()
    assert (bench_dir / "reports").exists()
