from __future__ import annotations

import pytest
from pydantic import ValidationError

from mapel_linkage.configuration import load_config
from mapel_linkage.pipeline.model_portfolio import (
    ModelPortfolioDeclaration,
    PairModelCandidateDeclaration,
    RankingCandidateDeclaration,
    compile_model_portfolio,
)
from tests.helpers import EXAMPLE_CONFIG


def test_current_configuration_compiles_to_plural_portfolio() -> None:
    config = load_config(EXAMPLE_CONFIG).config
    portfolio = compile_model_portfolio(config)

    assert portfolio.mandatory_baseline_id == config.models.fellegi_sunter.model_id
    assert portfolio.pair_candidates[0].family == "fellegi_sunter"
    assert any(candidate.family == "xgboost" for candidate in portfolio.pair_candidates)
    assert any(candidate.family == "xgboost" for candidate in portfolio.ranking_candidates)
    assert portfolio.test_partition_may_select_portfolio is False
    assert portfolio.decision_authority == "none"
    assert portfolio.merge_authority == "none"
    assert len(portfolio.portfolio_digest) == 64


def test_explicit_portfolio_supports_multiple_challengers_and_stacking() -> None:
    portfolio = ModelPortfolioDeclaration(
        portfolio_id="portfolio_demo",
        pair_candidates=(
            PairModelCandidateDeclaration(
                model_id="fs_baseline",
                family="fellegi_sunter",
                implementation="mapel_reference_fellegi_sunter",
                role="baseline",
                require_verified_labels=False,
                artifact_format="package_json",
            ),
            PairModelCandidateDeclaration(
                model_id="xgb_candidate",
                family="xgboost",
                implementation="xgboost_classifier",
                role="challenger",
                require_verified_labels=True,
                artifact_format="xgboost_json",
            ),
            PairModelCandidateDeclaration(
                model_id="lgb_candidate",
                family="lightgbm",
                implementation="lightgbm_classifier",
                role="challenger",
                require_verified_labels=True,
                artifact_format="lightgbm_text",
            ),
            PairModelCandidateDeclaration(
                model_id="stacked_candidate",
                family="stacking",
                implementation="stacking_logistic",
                role="ensemble",
                require_verified_labels=True,
                artifact_format="package_json",
                base_model_ids=("xgb_candidate", "lgb_candidate"),
            ),
        ),
        ranking_candidates=(
            RankingCandidateDeclaration(
                model_id="xgb_ranker",
                family="xgboost",
                implementation="xgboost_ranker",
                query_side="source",
                top_k=5,
                artifact_format="xgboost_json",
            ),
            RankingCandidateDeclaration(
                model_id="lgb_ranker",
                family="lightgbm",
                implementation="lightgbm_ranker",
                query_side="source",
                top_k=5,
                artifact_format="lightgbm_text",
            ),
        ),
        mandatory_baseline_id="fs_baseline",
        maximum_challengers=3,
    )

    assert len(portfolio.pair_candidates) == 4
    assert len(portfolio.ranking_candidates) == 2
    assert portfolio.safe_summary()["pair_candidate_count"] == 4


def test_portfolio_rejects_duplicate_ids_and_invalid_stacking_references() -> None:
    baseline = PairModelCandidateDeclaration(
        model_id="fs_baseline",
        family="fellegi_sunter",
        implementation="mapel_reference_fellegi_sunter",
        role="baseline",
        require_verified_labels=False,
        artifact_format="package_json",
    )
    duplicate = PairModelCandidateDeclaration(
        model_id="fs_baseline",
        family="xgboost",
        implementation="xgboost_classifier",
        role="challenger",
        require_verified_labels=True,
        artifact_format="xgboost_json",
    )
    with pytest.raises(ValidationError):
        ModelPortfolioDeclaration(
            portfolio_id="duplicate_demo",
            pair_candidates=(baseline, duplicate),
            mandatory_baseline_id="fs_baseline",
        )

    invalid_stack = PairModelCandidateDeclaration(
        model_id="stacked_candidate",
        family="stacking",
        implementation="stacking_logistic",
        role="ensemble",
        require_verified_labels=True,
        artifact_format="package_json",
        base_model_ids=("missing_one", "missing_two"),
    )
    with pytest.raises(ValidationError):
        ModelPortfolioDeclaration(
            portfolio_id="invalid_stack_demo",
            pair_candidates=(baseline, invalid_stack),
            mandatory_baseline_id="fs_baseline",
        )
