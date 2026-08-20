from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from mapel_linkage.assignment.contracts import pair_digest
from mapel_linkage.calibration import (
    ChampionChallengerSelector,
    ChampionSelection,
    ModelEvaluationCandidate,
)
from mapel_linkage.configuration import compile_config, load_config
from mapel_linkage.configuration.models import ModelSelectionConfig
from mapel_linkage.pipeline import SyntheticPortfolioWorkflowRunner
from mapel_linkage.synthetic import SyntheticGenerationConfig
from tests.helpers import ROOT

ALL_MODEL_CONFIG = ROOT / "configs/examples/synthetic_all_models.yaml"


def _has_all_model_runtime() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("lightgbm", "torch", "splink", "xgboost")
    )


def _generation() -> SyntheticGenerationConfig:
    return SyntheticGenerationConfig(
        seed=20260816,
        entity_count=120,
        left_only_count=8,
        right_only_count=8,
        duplicate_count=8,
        competing_candidate_count=20,
        source_a_missing_rate=0.05,
        source_b_missing_rate=0.20,
        source_b_typo_rate=0.35,
        source_b_date_shift_rate=0.20,
    )


@pytest.mark.skipif(not _has_all_model_runtime(), reason="all-model runtime unavailable")
def test_all_model_configuration_runs_real_runtimes_and_reloaded_inference(
    tmp_path: Path,
) -> None:
    plan = compile_config(load_config(ALL_MODEL_CONFIG).config, project_root=tmp_path)
    result = SyntheticPortfolioWorkflowRunner.run(plan, generation=_generation(), k_folds=3)

    assert set(result.tournament.validation_reports) == {
        "fs_baseline",
        "xgb_pair_classifier",
        "lgb_pair_classifier",
        "pytorch_pair_mlp",
        "stacking_logistic",
    }
    assert set(result.tournament.ranking_validation_reports) == {
        "xgb_candidate_ranker",
        "lgb_candidate_ranker",
    }
    assert {manifest.model_id for manifest in result.tournament.oof_manifests} == {
        "xgb_pair_classifier",
        "lgb_pair_classifier",
        "pytorch_pair_mlp",
    }
    assert all(not manifest.test_partition_used for manifest in result.tournament.oof_manifests)
    assert result.inference_status == "replayed"
    assert result.inference.pair_count > 0
    assert (
        result.review_inference.synthetic_attestation_digest
        != result.inference.synthetic_attestation_digest
    )
    review_pairs = {
        pair_digest(item.source_record_ref, item.target_record_ref)
        for item in result.review_inference.decisions
        if item.target_record_ref is not None
    }
    replay_pairs = {
        pair_digest(item.source_record_ref, item.target_record_ref)
        for item in result.inference.decisions
        if item.target_record_ref is not None
    }
    assert review_pairs
    assert replay_pairs
    assert review_pairs.isdisjoint(replay_pairs)
    assert result.persisted_recipe == result.tournament.recipe
    assert (
        result.persisted_calibrator.calibrator_digest == result.persisted_recipe.calibrator_digest
    )
    assert result.persisted_ranker is not None
    assert result.persisted_ranker.query_side == "source"
    assert result.persisted_ranker.model_id == "xgb_candidate_ranker"
    assert (
        result.persisted_ranker.artifact_digest == result.persisted_recipe.ranking_artifact_digest
    )
    assert result.locked_test_report.pair_count > 0
    assert result.tournament.test_partition_used_for_selection is False
    assert result.tournament.test_partition_used_for_calibration is False
    assert result.operational_validity == "not_established"
    assert result.merge_authority == "none"


@pytest.mark.skipif(not _has_all_model_runtime(), reason="all-model runtime unavailable")
def test_native_splink_champion_reloads_and_replays_prepared_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_select = ChampionChallengerSelector.select

    def force_native(
        candidates: Sequence[ModelEvaluationCandidate],
        config: ModelSelectionConfig,
    ) -> ChampionSelection:
        adjusted = tuple(
            replace(
                candidate,
                average_precision=1.0 if candidate.model_family == "fellegi_sunter" else 0.0,
                brier_score=0.0 if candidate.model_family == "fellegi_sunter" else 1.0,
            )
            for candidate in candidates
        )
        return original_select(adjusted, config)

    monkeypatch.setattr(ChampionChallengerSelector, "select", force_native)
    plan = compile_config(load_config(ALL_MODEL_CONFIG).config, project_root=tmp_path)
    result = SyntheticPortfolioWorkflowRunner.run(plan, generation=_generation(), k_folds=3)

    assert result.tournament.champion_selection.selected_model_family == "fellegi_sunter"
    assert result.persisted_recipe.champion_model_id == "fs_baseline"
    assert result.inference_status == "replayed"
    assert result.inference.pair_count > 0
    assert result.persisted_recipe.operational_validation.value == "not_established"
