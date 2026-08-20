"""Tests for the synthetic benchmark portfolio runner."""

from __future__ import annotations

import os
import subprocess
import sys

from mapel_linkage.benchmarking.contracts import BenchmarkRunStatus
from mapel_linkage.benchmarking.generator import BenchmarkScenarioGenerator
from mapel_linkage.benchmarking.runner import BenchmarkPortfolioRunner


def test_runner_executes_fellegi_sunter_and_xgboost_successfully() -> None:
    generator = BenchmarkScenarioGenerator()
    runner = BenchmarkPortfolioRunner()

    bundle = generator.generate("instance.typo_low", seed=42)

    # 1. Fellegi-Sunter baseline
    fs_recipe = next(
        r for r in runner.list_recipes() if r.recipe_id == "recipe.fellegi_sunter_reference"
    )
    res_fs = runner.run_single(
        bundle=bundle,
        recipe=fs_recipe,
        replicate_id="replicate.001",
        seed=42,
    )

    assert res_fs.record.status == BenchmarkRunStatus.SUCCESS
    assert res_fs.record.failure_code is None
    assert res_fs.metrics is not None
    assert 0.0 <= res_fs.metrics.candidate_recall <= 1.0
    assert 0.0 <= res_fs.metrics.sensitivity <= 1.0
    assert 0.0 <= res_fs.metrics.positive_predictive_value <= 1.0
    assert 0.0 <= res_fs.metrics.brier_score <= 1.0
    assert 0.0 <= res_fs.metrics.mean_reciprocal_rank <= 1.0
    assert res_fs.record.runtime_ms is not None and res_fs.record.runtime_ms >= 0
    assert res_fs.record.peak_memory_mb is not None

    # 2. XGBoost classifier
    xgb_recipe = next(
        r for r in runner.list_recipes() if r.recipe_id == "recipe.xgboost_classifier"
    )
    res_xgb = runner.run_single(
        bundle=bundle,
        recipe=xgb_recipe,
        replicate_id="replicate.001",
        seed=42,
    )

    assert res_xgb.record.status == BenchmarkRunStatus.SUCCESS
    assert res_xgb.metrics is not None
    assert res_xgb.metrics.candidate_recall > 0.5
    assert res_xgb.metrics.positive_predictive_value > 0.0


def test_runner_handles_ineligible_zero_label_gracefully() -> None:
    generator = BenchmarkScenarioGenerator()
    runner = BenchmarkPortfolioRunner()

    # Zero-label instance
    bundle_zero = generator.generate("instance.labels_zero", seed=99)
    xgb_recipe = next(
        r for r in runner.list_recipes() if r.recipe_id == "recipe.xgboost_classifier"
    )

    res = runner.run_single(
        bundle=bundle_zero,
        recipe=xgb_recipe,
        replicate_id="replicate.001",
        seed=99,
    )

    assert res.record.status == BenchmarkRunStatus.INELIGIBLE
    assert res.record.failure_code == "ML-BENCH-INELIGIBLE-LABELS"
    assert res.failure is not None
    assert "labels" in res.failure.error_message.lower()


def test_runner_handles_ineligible_mode_gracefully() -> None:
    generator = BenchmarkScenarioGenerator()
    runner = BenchmarkPortfolioRunner()

    # Dedupe-only instance with link_only recipe
    bundle_dedupe = generator.generate("instance.dedupe_standard", seed=101)
    fs_recipe = next(
        r for r in runner.list_recipes() if r.recipe_id == "recipe.fellegi_sunter_reference"
    )

    res = runner.run_single(
        bundle=bundle_dedupe,
        recipe=fs_recipe,
        replicate_id="replicate.001",
        seed=101,
    )

    assert res.record.status == BenchmarkRunStatus.INELIGIBLE
    assert res.record.failure_code == "ML-BENCH-INELIGIBLE-MODE"
    assert res.failure is not None


def test_runner_executes_ranking_recipe() -> None:
    generator = BenchmarkScenarioGenerator()
    runner = BenchmarkPortfolioRunner()

    bundle = generator.generate("instance.typo_low", seed=42)
    ranker_recipe = next(r for r in runner.list_recipes() if r.recipe_id == "recipe.xgboost_ranker")

    res = runner.run_single(
        bundle=bundle,
        recipe=ranker_recipe,
        replicate_id="replicate.001",
        seed=42,
    )

    assert res.record.status == BenchmarkRunStatus.SUCCESS
    assert res.metrics is not None
    assert "1" in res.metrics.candidate_recall_at_k
    assert "5" in res.metrics.candidate_recall_at_k
    assert "10" in res.metrics.candidate_recall_at_k


def test_portfolio_runner_multi_replicates() -> None:
    generator = BenchmarkScenarioGenerator()
    runner = BenchmarkPortfolioRunner()

    # Run only 1 family with 2 replicates to keep test fast
    results = runner.run_portfolio(
        generator,
        families=("family.typo_stress",),
        replicates=2,
    )

    instances_in_family = [
        inst for inst in generator.list_instances() if inst.family_id == "family.typo_stress"
    ]
    expected_runs = len(instances_in_family) * 2 * len(runner.list_recipes())
    assert len(results) == expected_runs

    # Check statuses
    success_runs = [r for r in results if r.record.status == BenchmarkRunStatus.SUCCESS]
    ineligible_runs = [r for r in results if r.record.status == BenchmarkRunStatus.INELIGIBLE]

    assert len(success_runs) > 0
    assert len(ineligible_runs) > 0  # Mode mismatch or missing runtime


def test_portfolio_seed_is_stable_across_python_hash_seeds() -> None:
    script = (
        "from mapel_linkage.benchmarking.generator import BenchmarkScenarioGenerator; "
        "from mapel_linkage.benchmarking.runner import BenchmarkPortfolioRunner; "
        "g=BenchmarkScenarioGenerator(); r=BenchmarkPortfolioRunner(); "
        "recipe=(r.list_recipes()[0],); "
        "result=r.run_portfolio(g, instances=('instance.typo_low',), recipes=recipe, "
        "replicates=1, base_seed=20260816); "
        "print(result[0].record.random_seed)"
    )
    observed: list[str] = []
    for python_hash_seed in ("1", "987654"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = python_hash_seed
        environment["MAPEL_TEST_DATA_POLICY"] = "synthetic_only"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        observed.append(completed.stdout.strip())

    assert len(set(observed)) == 1
