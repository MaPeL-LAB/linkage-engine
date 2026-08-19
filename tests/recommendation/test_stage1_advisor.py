from __future__ import annotations

import pytest
from pydantic import ValidationError

from mapel_linkage.configuration import compile_config, load_config
from mapel_linkage.configuration.compiler import ExecutionPlan
from mapel_linkage.profiling import build_preflight_task_profile
from mapel_linkage.recommendation import (
    AbstentionReason,
    AdvisorContext,
    CoverageStatus,
    PipelineRecommendation,
    RecommendationIntent,
    RuntimeDependency,
    recommend_pipeline,
)
from tests.helpers import EXAMPLE_CONFIG, ROOT


def _plan() -> ExecutionPlan:
    loaded = load_config(EXAMPLE_CONFIG)
    return compile_config(loaded.config, project_root=ROOT)


def test_stage1_returns_advisory_structural_shortlist_and_abstains_empirically() -> None:
    plan = _plan()
    context = AdvisorContext(
        intent=RecommendationIntent.DEVELOP_NEW_RECIPE,
        verified_labels_available=True,
        available_runtimes=(
            RuntimeDependency.CORE,
            RuntimeDependency.LIGHTGBM,
            RuntimeDependency.PYTORCH,
        ),
    )
    recommendation = recommend_pipeline(plan, context=context)

    assert recommendation.coverage_status is CoverageStatus.STRUCTURAL_ONLY
    assert recommendation.abstained_from_empirical_ranking is True
    assert AbstentionReason.NO_BENCHMARK_EVIDENCE in recommendation.abstention_reasons
    assert recommendation.empirical_performance_claims == "none"
    assert recommendation.recommendation_authority == "advisory_only"
    assert recommendation.decision_authority == "none"
    assert recommendation.assignment_authority == "none"
    assert recommendation.merge_authority == "none"
    assert recommendation.automatic_promotion == "prohibited"
    assert recommendation.operational_validity == "not_established"
    assert recommendation.shortlist
    assert recommendation.mandatory_baseline_candidate_id in {
        item.candidate_id for item in recommendation.shortlist
    }
    assert len(recommendation.shortlist) <= 4
    assert all(item.decision_authority == "none" for item in recommendation.shortlist)


def test_no_verified_labels_excludes_supervised_training_but_retains_baseline() -> None:
    plan = _plan()
    recommendation = recommend_pipeline(
        plan,
        context=AdvisorContext(
            intent=RecommendationIntent.DEVELOP_NEW_RECIPE,
            verified_labels_available=False,
            available_runtimes=(RuntimeDependency.CORE,),
        ),
    )

    assert any(item.pair_model_family == "fellegi_sunter" for item in recommendation.shortlist)
    disqualified_reasons = {
        reason for item in recommendation.disqualified_candidates for reason in item.reasons
    }
    assert "verified_labels_required" in disqualified_reasons


def test_inference_intent_requires_approved_recipe_and_artifact() -> None:
    recommendation = recommend_pipeline(
        _plan(),
        context=AdvisorContext(
            intent=RecommendationIntent.INFER_WITH_APPROVED_RECIPE,
            verified_labels_available=False,
            available_runtimes=(RuntimeDependency.CORE,),
        ),
    )

    assert recommendation.shortlist == ()
    assert AbstentionReason.APPROVED_RECIPE_REQUIRED in recommendation.abstention_reasons
    assert recommendation.mandatory_baseline_candidate_id is None


def test_profile_and_recommendation_do_not_include_project_or_source_fields() -> None:
    plan = _plan()
    profile = build_preflight_task_profile(plan)
    recommendation = recommend_pipeline(
        plan,
        profile=profile,
        context=AdvisorContext(
            intent=RecommendationIntent.DEVELOP_NEW_RECIPE,
            verified_labels_available=True,
            available_runtimes=(RuntimeDependency.CORE,),
        ),
    )
    rendered = repr(recommendation.safe_summary())

    assert plan.config.project.project_id not in rendered
    assert "record_key_a" not in rendered
    assert str(EXAMPLE_CONFIG) not in rendered
    assert "sensitivity" not in rendered
    assert "positive_predictive_value" not in rendered


def test_recommendation_authority_cannot_be_overridden() -> None:
    recommendation = recommend_pipeline(
        _plan(),
        context=AdvisorContext(
            intent=RecommendationIntent.DEVELOP_NEW_RECIPE,
            verified_labels_available=True,
            available_runtimes=(RuntimeDependency.CORE,),
        ),
    )
    payload = recommendation.model_dump(mode="json")
    payload["decision_authority"] = "identity_decision"

    with pytest.raises(ValidationError):
        PipelineRecommendation.model_validate(payload)
