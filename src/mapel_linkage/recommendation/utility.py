"""Shared aggregate utility policy for evidence-backed linkage strategy advice."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Final, Literal

from mapel_linkage.benchmarking.contracts import BenchmarkAggregateMetrics
from mapel_linkage.benchmarking.runner import BenchmarkPortfolioRunner
from mapel_linkage.recommendation.contracts import (
    EmpiricalMetricDistribution,
    StructuralPipelineCandidate,
)

AdvisorRecipeToken = Literal["fellegi_sunter", "xgboost_classifier", "xgboost_ranker"]

REQUIRED_ADVISOR_RECIPE_TOKENS: Final[tuple[AdvisorRecipeToken, ...]] = (
    "fellegi_sunter",
    "xgboost_classifier",
    "xgboost_ranker",
)
RECIPE_TOKEN_BY_ID: Final[Mapping[str, AdvisorRecipeToken]] = {
    "recipe.fellegi_sunter_reference": "fellegi_sunter",
    "recipe.xgboost_classifier": "xgboost_classifier",
    "recipe.xgboost_ranker": "xgboost_ranker",
}
ADVISOR_UTILITY_WEIGHTS: Final[Mapping[str, float]] = {
    "recall_at_1": 0.4,
    "positive_predictive_value": 0.4,
    "one_minus_brier_score": 0.2,
}


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


ADVISOR_UTILITY_POLICY_DIGEST: Final[str] = _digest(
    {
        "policy": "advisor-aggregate-utility-v1",
        "weights": ADVISOR_UTILITY_WEIGHTS,
        "range": [0.0, 1.0],
        "evidence_scope": "global_synthetic",
        "operational_validity": "not_established",
    }
)


def recipe_token_by_digest(
    runner: BenchmarkPortfolioRunner | None = None,
) -> dict[str, AdvisorRecipeToken]:
    """Bind exact package recipe digests to the three qualification adapters."""

    portfolio = runner or BenchmarkPortfolioRunner()
    return {
        recipe.recipe_digest: RECIPE_TOKEN_BY_ID[recipe.recipe_id]
        for recipe in portfolio.list_recipes()
        if recipe.recipe_id in RECIPE_TOKEN_BY_ID
    }


def candidate_recipe_token(
    candidate: StructuralPipelineCandidate,
) -> AdvisorRecipeToken | None:
    """Map an advisory candidate to a benchmark adapter without ID substrings."""

    if candidate.pair_model_family == "fellegi_sunter":
        return "fellegi_sunter"
    if candidate.pair_model_family != "xgboost":
        return None
    if candidate.ranking_strategy.value == "xgboost_ranker":
        return "xgboost_ranker"
    if candidate.ranking_strategy.value == "model_score":
        return "xgboost_classifier"
    return None


def benchmark_utility(metrics: BenchmarkAggregateMetrics) -> float:
    """Return the prespecified bounded utility for one aggregate benchmark run."""

    recall_at_1 = float(metrics.candidate_recall_at_k.get("1", metrics.candidate_recall))
    return min(
        1.0,
        max(
            0.0,
            ADVISOR_UTILITY_WEIGHTS["recall_at_1"] * recall_at_1
            + ADVISOR_UTILITY_WEIGHTS["positive_predictive_value"]
            * float(metrics.positive_predictive_value)
            + ADVISOR_UTILITY_WEIGHTS["one_minus_brier_score"] * (1.0 - float(metrics.brier_score)),
        ),
    )


def empirical_distribution_utility(distribution: EmpiricalMetricDistribution) -> float:
    """Apply the same utility estimand to a Stage-2 aggregate distribution."""

    if distribution.sample_count == 0:
        return -1.0
    return min(
        1.0,
        max(
            0.0,
            ADVISOR_UTILITY_WEIGHTS["recall_at_1"] * distribution.mean_recall_at_1
            + ADVISOR_UTILITY_WEIGHTS["positive_predictive_value"]
            * distribution.mean_positive_predictive_value
            + ADVISOR_UTILITY_WEIGHTS["one_minus_brier_score"]
            * (1.0 - distribution.mean_brier_score)
            - 0.2 * distribution.failure_rate,
        ),
    )


__all__ = [
    "ADVISOR_UTILITY_POLICY_DIGEST",
    "ADVISOR_UTILITY_WEIGHTS",
    "REQUIRED_ADVISOR_RECIPE_TOKENS",
    "AdvisorRecipeToken",
    "benchmark_utility",
    "candidate_recipe_token",
    "empirical_distribution_utility",
    "recipe_token_by_digest",
]
