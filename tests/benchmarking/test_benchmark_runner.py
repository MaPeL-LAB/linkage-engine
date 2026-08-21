"""Tests for the synthetic benchmark portfolio runner."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np

from mapel_linkage.benchmarking.advisor_catalogue import build_advisor_v2_generator
from mapel_linkage.benchmarking.contracts import BenchmarkRunStatus
from mapel_linkage.benchmarking.generator import BenchmarkScenarioGenerator
from mapel_linkage.benchmarking.runner import BenchmarkPortfolioRunner
from mapel_linkage.models.boosted.training import (
    BoostedFeatureMatrix,
    BoostedLabelledMatrix,
)
from mapel_linkage.models.ranking.contracts import RankingFeatureMatrix
from mapel_linkage.models.ranking.training import build_ranking_scoring_matrix


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


def test_fellegi_sunter_scores_the_retained_upper_tail_regression_case() -> None:
    runner = BenchmarkPortfolioRunner()
    recipe = next(
        item
        for item in runner.list_recipes()
        if item.recipe_id == "recipe.fellegi_sunter_reference"
    )

    result = runner.run_portfolio(
        build_advisor_v2_generator(),
        instances=("instance.advisor_v2.f37.p03",),
        recipes=(recipe,),
        replicates=1,
        replicate_start=3,
    )[0]

    assert result.record.status is BenchmarkRunStatus.SUCCESS
    assert result.metrics is not None
    assert result.failure is None


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


def test_scoring_and_ranking_matrices_strip_truth_fields() -> None:
    pairs = (("left_1", "right_1"), ("left_2", "right_2"))
    labelled = BoostedLabelledMatrix(
        features=np.asarray([[0.1], [0.9]], dtype=np.float64),
        pair_references=pairs,
        pair_digests=tuple(
            hashlib.sha256(f"{left}\x00{right}".encode()).hexdigest() for left, right in pairs
        ),
        feature_names=("comparison_level",),
        feature_schema_digest="c" * 64,
        labels=np.asarray([0, 1], dtype=np.int8),
        partition="test",
        label_source_kind="synthetic_truth",
        label_authority_digest="d" * 64,
        selection_digest="e" * 64,
        positive_count=1,
        negative_count=1,
    )

    scoring = BenchmarkPortfolioRunner._as_scoring_matrix(labelled)
    ranking = build_ranking_scoring_matrix(scoring, query_side="source")

    assert type(scoring) is BoostedFeatureMatrix
    assert not hasattr(scoring, "labels")
    assert not hasattr(scoring, "partition")
    assert type(ranking) is RankingFeatureMatrix
    assert not hasattr(ranking, "relevance")


def test_metrics_are_mechanically_derived_instead_of_constant_placeholders() -> None:
    pairs = (("left_1", "right_1"), ("left_1", "right_2"))
    labels = np.asarray([1, 0], dtype=np.int8)
    correct = BenchmarkPortfolioRunner._mechanical_metrics(
        pair_references=pairs,
        labels=labels,
        scores=np.asarray([0.9, 0.1], dtype=np.float64),
        candidate_recall=1.0,
    )
    reversed_scores = BenchmarkPortfolioRunner._mechanical_metrics(
        pair_references=pairs,
        labels=labels,
        scores=np.asarray([0.1, 0.9], dtype=np.float64),
        candidate_recall=0.5,
    )

    assert correct.sensitivity == 1.0
    assert correct.positive_predictive_value == 1.0
    assert correct.mean_reciprocal_rank == 1.0
    assert reversed_scores.sensitivity == 0.0
    assert reversed_scores.positive_predictive_value == 0.0
    assert reversed_scores.mean_reciprocal_rank == 0.5
    assert correct.brier_score < reversed_scores.brier_score


def test_unimplemented_mode_adapters_abstain_without_metrics() -> None:
    generator = BenchmarkScenarioGenerator()
    runner = BenchmarkPortfolioRunner()
    cases = (
        ("instance.dedupe_standard", "recipe.single_source_dedupe"),
        ("instance.tri_source_standard", "recipe.multi_source_resolver"),
    )

    for instance_id, recipe_id in cases:
        bundle = generator.generate(instance_id, seed=20260816)
        recipe = next(item for item in runner.list_recipes() if item.recipe_id == recipe_id)
        result = runner.run_single(bundle=bundle, recipe=recipe, seed=20260816)
        assert result.record.status == BenchmarkRunStatus.INELIGIBLE
        assert result.record.failure_code == "ML-BENCH-INELIGIBLE-ADAPTER"
        assert result.metrics is None
        assert result.failure is not None


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
