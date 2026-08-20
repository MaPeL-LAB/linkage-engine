"""Unit tests for Stage-3 Learned Meta-Ranking Strategy Advisor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mapel_linkage.benchmarking.generator import BenchmarkScenarioGenerator
from mapel_linkage.benchmarking.registry import BenchmarkRegistry
from mapel_linkage.benchmarking.runner import BenchmarkPortfolioRunner
from mapel_linkage.benchmarking.seed_corpus import generate_and_run_seed_corpus
from mapel_linkage.configuration import compile_config, load_config
from mapel_linkage.configuration.compiler import ExecutionPlan
from mapel_linkage.domain.errors import AdvisorError
from mapel_linkage.profiling import build_preflight_task_profile
from mapel_linkage.recommendation.contracts import (
    MetaRankingAdvisoryReport,
    RecommendationIntent,
    RuntimeDependency,
)
from mapel_linkage.recommendation.eligibility import AdvisorContext
from mapel_linkage.recommendation.meta_ranker import (
    LearnedMetaRankerModel,
    MetaRankingLinkageAdvisor,
)
from tests.helpers import EXAMPLE_CONFIG, ROOT


def _plan() -> ExecutionPlan:
    loaded = load_config(EXAMPLE_CONFIG)
    return compile_config(loaded.config, project_root=ROOT)


@pytest.fixture(scope="module")
def populated_registry(tmp_path_factory: pytest.TempPathFactory) -> Path:
    reg_dir = tmp_path_factory.mktemp("meta_ranker_registry")
    generate_and_run_seed_corpus(
        registry_directory=reg_dir,
        generator=BenchmarkScenarioGenerator(),
        runner=BenchmarkPortfolioRunner(),
        families=("family.typo_stress", "family.missingness_regime", "family.date_variation"),
        instances=("instance.typo_low", "instance.missing_zero", "instance.date_shift_low"),
        replicates=1,
    )
    return reg_dir


def test_learned_meta_ranker_model_fit_predict() -> None:
    model = LearnedMetaRankerModel()
    np.random.seed(42)
    X = np.random.randn(20, 5)
    true_w = np.array([0.2, -0.1, 0.4, 0.05, -0.3])
    y = np.clip(0.6 + X @ true_w + np.random.randn(20) * 0.05, 0.0, 1.0)

    model.fit(X, y, coverage_level=0.90)

    assert model.weights is not None
    assert model.trained_run_count == 20
    assert 0.0 < model.conformal_residual_quantile < 0.5

    X_test = np.random.randn(5, 5)
    preds, lowers, uppers = model.predict_utility(X_test)

    assert len(preds) == 5
    assert all(0.0 <= p <= 1.0 for p in preds)
    assert all(low <= p <= up for p, low, up in zip(preds, lowers, uppers, strict=True))


def test_meta_ranker_fallback_on_empty_registry(tmp_path: Path) -> None:
    plan = _plan()
    profile = build_preflight_task_profile(plan)
    context = AdvisorContext(
        intent=RecommendationIntent.DEVELOP_NEW_RECIPE,
        verified_labels_available=True,
        approved_recipe_available=False,
        protected_out_of_fold_predictions_available=False,
        available_runtimes=(RuntimeDependency.CORE,),
        approved_artifact_model_ids=(),
    )

    advisor = MetaRankingLinkageAdvisor(registry=None)
    report = advisor.advise(plan, context=context, profile=profile)

    assert report.fallback_to_similarity is True
    assert report.recommendation_authority == "advisory_only"
    assert report.decision_authority == "none"
    assert report.assignment_authority == "none"
    assert report.merge_authority == "none"
    assert report.automatic_promotion == "prohibited"
    assert report.operational_validity == "not_established"


def test_meta_ranker_with_populated_registry(populated_registry: Path) -> None:
    registry = BenchmarkRegistry(populated_registry)
    plan = _plan()
    profile = build_preflight_task_profile(plan)
    context = AdvisorContext(
        intent=RecommendationIntent.DEVELOP_NEW_RECIPE,
        verified_labels_available=False,
        approved_recipe_available=False,
        protected_out_of_fold_predictions_available=False,
        available_runtimes=(RuntimeDependency.CORE,),
        approved_artifact_model_ids=(),
    )

    advisor = MetaRankingLinkageAdvisor(
        registry=registry,
        max_ood_distance=0.60,
    )
    report = advisor.advise(plan, context=context, profile=profile)

    assert isinstance(report, MetaRankingAdvisoryReport)
    assert report.recommendation.mandatory_baseline_candidate_id in {
        item.candidate_id for item in report.recommendation.shortlist
    }
    assert any(
        item.pair_model_family == "fellegi_sunter" for item in report.recommendation.shortlist
    )
    assert len(report.recommendation.shortlist) > 0

    # Safe summary check
    summary = report.safe_summary()
    assert summary["report_schema_version"] == "1"
    assert summary["recommendation_authority"] == "advisory_only"


def test_meta_ranker_test_partition_rejection() -> None:
    advisor = MetaRankingLinkageAdvisor(registry=None)
    plan = _plan()
    mock_context: Any = type("MockContext", (), {"test_partition_used": True})()

    with pytest.raises(AdvisorError, match="ML-ADVISOR-001"):
        advisor.advise(plan, context=mock_context)
