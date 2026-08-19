from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from mapel_linkage.calibration.calibrators import SigmoidCalibrator
from mapel_linkage.calibration.contracts import (
    CalibratorArtifact,
    ChampionSelection,
    PairScoreBatch,
)
from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.pipeline.inference_runner import (
    infer_with_approved_recipe,
)
from mapel_linkage.pipeline.recipes import (
    OperationalValidationStatus,
    PipelineRecipeArtifact,
    RecipeApprovalStatus,
    RecipeExecutionMode,
)


def _make_dummy_calibrator(
    champion_model_id: str = "xgb_champion",
    champion_model_version: str = "v1",
) -> CalibratorArtifact:
    pairs = tuple((f"l_{i}", f"r_{i}") for i in range(8))
    pair_digests = tuple(hashlib.sha256(f"l_{i}\x00r_{i}".encode()).hexdigest() for i in range(8))
    sel_obj = ChampionSelection(
        selected_model_family="xgboost",
        selected_model_id=champion_model_id,
        selected_model_version=champion_model_version,
        selected_evidence_digest="a" * 64,
        selected_feature_schema_digest="f" * 64,
        selected_training_label_authority_digest="a" * 64,
        validation_label_authority_digest="b" * 64,
        partition_manifest_digest="c" * 64,
        primary_metric="average_precision",
        secondary_metric="brier_score",
        selection_digest="d" * 64,
        candidate_summaries=(),
    )
    score_batch = PairScoreBatch(
        pair_references=pairs,
        pair_digests=pair_digests,
        scores=np.asarray([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], dtype=np.float64),
        labels=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8),
        partition="calibration",
        source_model_family="xgboost",
        source_model_id=champion_model_id,
        source_model_version=champion_model_version,
        source_evidence_digest="a" * 64,
        feature_schema_digest="f" * 64,
        label_authority_digest="e" * 64,
        partition_manifest_digest="c" * 64,
        champion_selection_digest=sel_obj.selection_digest,
    )
    return SigmoidCalibrator.fit(score_batch, sel_obj)


def test_inference_runner_approved_recipe(tmp_path: Path) -> None:
    cal_art = _make_dummy_calibrator()

    recipe = PipelineRecipeArtifact(
        recipe_id="approved_rec_1",
        recipe_version="v1.0.0",
        linkage_mode="link_only",
        assignment_constraint="one_to_one",
        configuration_digest="c" * 64,
        candidate_plan_digest="d" * 64,
        feature_schema_digest="f" * 64,
        champion_model_id="xgb_champion",
        champion_model_version="v1",
        champion_artifact_digest="a" * 64,
        calibrator_digest=cal_art.calibrator_digest,
        ranking_artifact_digest=None,
        decision_policy_digest="e" * 64,
        validation_evidence_digest="a" * 64,
        approval_status=RecipeApprovalStatus.APPROVED_FOR_INFERENCE,
        operational_validation=OperationalValidationStatus.LOCALLY_ESTABLISHED,
    )

    source_keys = ("src_1", "src_2", "src_3")
    pair_refs = (("src_1", "tgt_1"), ("src_2", "tgt_2"), ("src_3", "tgt_3"))
    raw_scores = [0.95, 0.85, 0.10]

    out_dest = tmp_path / "decisions.csv"

    result = infer_with_approved_recipe(
        recipe=recipe,
        source_record_keys=source_keys,
        pair_references=pair_refs,
        raw_scores=raw_scores,
        calibrator_artifact=cal_art,
        execution_mode=RecipeExecutionMode.INFERENCE,
        source_dataset_id="source",
        target_dataset_id="target",
        output_decisions_path=out_dest,
    )

    assert result.recipe_id == "approved_rec_1"
    assert result.execution_mode == RecipeExecutionMode.INFERENCE
    assert result.pair_count == 3
    assert out_dest.is_file()
    assert len(result.decisions) == 3
    assert result.assignment_result.real_assignment_count >= 1

    summary = result.safe_summary()
    assert summary["recipe_id"] == "approved_rec_1"
    assert summary["output_written"] is True
    assert summary["pair_count"] == 3


def test_inference_runner_draft_recipe_rejected_for_inference() -> None:
    cal_art = _make_dummy_calibrator()

    draft_recipe = PipelineRecipeArtifact(
        recipe_id="draft_rec_1",
        recipe_version="v1.0.0",
        linkage_mode="link_only",
        assignment_constraint="one_to_one",
        configuration_digest="c" * 64,
        candidate_plan_digest="d" * 64,
        feature_schema_digest="f" * 64,
        champion_model_id="xgb_champion",
        champion_model_version="v1",
        champion_artifact_digest="a" * 64,
        calibrator_digest=cal_art.calibrator_digest,
        ranking_artifact_digest=None,
        decision_policy_digest="e" * 64,
        validation_evidence_digest="a" * 64,
        approval_status=RecipeApprovalStatus.DRAFT,
        operational_validation=OperationalValidationStatus.NOT_ESTABLISHED,
    )

    source_keys = ("s1", "s2")
    pair_refs = (("s1", "t1"), ("s2", "t2"))
    raw_scores = [0.9, 0.8]

    # Draft recipes are rejected for INFERENCE execution mode
    with pytest.raises(PipelineError) as exc_info:
        infer_with_approved_recipe(
            recipe=draft_recipe,
            source_record_keys=source_keys,
            pair_references=pair_refs,
            raw_scores=raw_scores,
            calibrator_artifact=cal_art,
            execution_mode=RecipeExecutionMode.INFERENCE,
        )
    assert "ML-RECIPE-005" in str(exc_info.value)

    # Draft recipes succeed in DEVELOPMENT mode
    dev_result = infer_with_approved_recipe(
        recipe=draft_recipe,
        source_record_keys=source_keys,
        pair_references=pair_refs,
        raw_scores=raw_scores,
        calibrator_artifact=cal_art,
        execution_mode=RecipeExecutionMode.DEVELOPMENT,
    )
    assert dev_result.execution_mode == RecipeExecutionMode.DEVELOPMENT
