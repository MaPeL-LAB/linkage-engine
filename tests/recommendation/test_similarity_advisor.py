from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mapel_linkage.benchmarking.generator import BenchmarkScenarioGenerator
from mapel_linkage.benchmarking.runner import BenchmarkPortfolioRunner
from mapel_linkage.benchmarking.seed_corpus import generate_and_run_seed_corpus
from mapel_linkage.configuration import compile_config, load_config
from mapel_linkage.configuration.compiler import ExecutionPlan
from mapel_linkage.domain.errors import AdvisorError
from mapel_linkage.recommendation import (
    AbstentionReason,
    AdvisorContext,
    CoverageStatus,
    RecommendationIntent,
    RuntimeDependency,
    SimilarityLinkageAdvisor,
    recommend_with_similarity,
)
from tests.helpers import EXAMPLE_CONFIG, ROOT


def _plan() -> ExecutionPlan:
    loaded = load_config(EXAMPLE_CONFIG)
    return compile_config(loaded.config, project_root=ROOT)


@pytest.fixture(scope="module")
def populated_registry(tmp_path_factory: pytest.TempPathFactory) -> Path:
    reg_dir = tmp_path_factory.mktemp("benchmark_registry")
    generate_and_run_seed_corpus(
        registry_directory=reg_dir,
        generator=BenchmarkScenarioGenerator(),
        runner=BenchmarkPortfolioRunner(),
        families=("family.typo_stress", "family.missingness_regime", "family.date_variation"),
        instances=("instance.typo_low", "instance.missing_zero", "instance.date_shift_low"),
        replicates=1,
    )
    return reg_dir


def test_similarity_advisor_within_distribution(populated_registry: Path) -> None:
    from mapel_linkage.benchmarking.registry import BenchmarkRegistry

    registry = BenchmarkRegistry(populated_registry)
    advisor = SimilarityLinkageAdvisor(
        registry=registry,
        max_ood_distance=0.60,
        k_nearest_families=2,
    )

    plan = _plan()
    context = AdvisorContext(
        intent=RecommendationIntent.DEVELOP_NEW_RECIPE,
        verified_labels_available=True,
        available_runtimes=(RuntimeDependency.CORE,),
    )

    report = advisor.recommend(plan, context=context)

    # 1. Similarity Report structure
    assert report.synthetic_evidence_retrieved is True
    assert report.out_of_distribution is False
    assert report.out_of_distribution_score <= 0.60
    assert len(report.nearest_family_ids) == 2
    assert report.empirical_metric_distributions
    assert set(report.nearest_family_ids) <= {
        record.family_id for record in registry.list_run_records()
    }

    # Check metric distribution values
    for _cand_id, dist in report.empirical_metric_distributions.items():
        assert dist.sample_count >= 0
        assert 0.0 <= dist.mean_candidate_recall <= 1.0
        assert 0.0 <= dist.mean_positive_predictive_value <= 1.0
        assert 0.0 <= dist.mean_brier_score <= 1.0
        assert dist.operational_validity == "not_established"

    # 2. PipelineRecommendation checks
    rec = report.recommendation
    assert rec.coverage_status is CoverageStatus.WITHIN_BENCHMARK_ENVELOPE
    assert rec.out_of_distribution_score == report.out_of_distribution_score
    assert rec.registry_snapshot_digest is not None
    assert rec.abstained_from_empirical_ranking is False
    assert rec.empirical_performance_claims == "none"

    # Authority literal invariants
    assert rec.recommendation_authority == "advisory_only"
    assert rec.decision_authority == "none"
    assert rec.assignment_authority == "none"
    assert rec.merge_authority == "none"
    assert rec.automatic_promotion == "prohibited"
    assert rec.operational_validity == "not_established"

    # Baseline retention
    assert rec.mandatory_baseline_candidate_id in {c.candidate_id for c in rec.shortlist}
    assert any(c.pair_model_family == "fellegi_sunter" for c in rec.shortlist)


def test_similarity_advisor_out_of_distribution(populated_registry: Path) -> None:
    from mapel_linkage.benchmarking.registry import BenchmarkRegistry

    registry = BenchmarkRegistry(populated_registry)
    # Set impossible low OOD threshold
    advisor = SimilarityLinkageAdvisor(
        registry=registry,
        max_ood_distance=0.001,
        k_nearest_families=2,
    )

    plan = _plan()
    context = AdvisorContext(
        intent=RecommendationIntent.DEVELOP_NEW_RECIPE,
        verified_labels_available=True,
        available_runtimes=(RuntimeDependency.CORE,),
    )

    report = advisor.recommend(plan, context=context)

    assert report.synthetic_evidence_retrieved is False
    assert report.out_of_distribution is True
    assert report.recommendation.coverage_status is CoverageStatus.OUT_OF_DISTRIBUTION
    assert AbstentionReason.OUT_OF_DISTRIBUTION in report.recommendation.abstention_reasons
    assert report.recommendation.shortlist
    assert report.recommendation.mandatory_baseline_candidate_id in {
        c.candidate_id for c in report.recommendation.shortlist
    }


def test_similarity_advisor_empty_registry_fallback() -> None:
    advisor = SimilarityLinkageAdvisor(registry=None)

    plan = _plan()
    context = AdvisorContext(
        intent=RecommendationIntent.DEVELOP_NEW_RECIPE,
        verified_labels_available=True,
        available_runtimes=(RuntimeDependency.CORE,),
    )

    report = advisor.recommend(plan, context=context)

    assert report.synthetic_evidence_retrieved is False
    assert report.out_of_distribution is True
    assert report.recommendation.coverage_status is CoverageStatus.STRUCTURAL_ONLY
    assert AbstentionReason.NO_BENCHMARK_EVIDENCE in report.recommendation.abstention_reasons
    assert report.recommendation.shortlist


def test_similarity_advisor_locked_test_partition_guard() -> None:
    advisor = SimilarityLinkageAdvisor(registry=None)
    plan = _plan()

    # Direct AdvisorContext creation with test_partition_used=True fails validation
    with pytest.raises(ValidationError):
        AdvisorContext.model_validate(
            {
                "intent": RecommendationIntent.DEVELOP_NEW_RECIPE,
                "verified_labels_available": True,
                "test_partition_used": True,
            }
        )

    # Any object passing test_partition_used=True to recommend triggers AdvisorError
    mock_context: Any = type("MockContext", (), {"test_partition_used": True})()
    with pytest.raises(AdvisorError, match="ML-ADVISOR-001"):
        advisor.recommend(plan, context=mock_context)


def test_similarity_advisor_convenience_function(populated_registry: Path) -> None:
    from mapel_linkage.benchmarking.registry import BenchmarkRegistry

    registry = BenchmarkRegistry(populated_registry)
    plan = _plan()
    context = AdvisorContext(
        intent=RecommendationIntent.DEVELOP_NEW_RECIPE,
        verified_labels_available=False,
        available_runtimes=(RuntimeDependency.CORE,),
    )

    report = recommend_with_similarity(
        plan,
        context=context,
        registry=registry,
        max_ood_distance=0.60,
    )

    assert report.report_id.startswith("report.similarity.")
    assert report.recommendation.recommendation_id.startswith("advisor.similarity.")
    disqualified_cands = {c.candidate_id for c in report.recommendation.disqualified_candidates}
    assert disqualified_cands
