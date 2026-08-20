from __future__ import annotations

import hashlib
from dataclasses import replace
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
    _resolve_recipe,
    attest_generated_synthetic_inference,
    infer_with_approved_recipe,
)
from mapel_linkage.pipeline.recipe_io import serialize_pipeline_recipe
from mapel_linkage.pipeline.recipes import (
    OperationalValidationStatus,
    PipelineRecipeArtifact,
    RecipeApprovalStatus,
    RecipeExecutionMode,
    SyntheticInferenceAttestation,
)
from mapel_linkage.synthetic import (
    SyntheticBundle,
    SyntheticGenerationConfig,
    generate_synthetic_bundle,
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


def _synthetic_inference_inputs() -> tuple[
    SyntheticBundle,
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    tuple[float, ...],
]:
    bundle = generate_synthetic_bundle(SyntheticGenerationConfig(seed=20260816))
    source_record_keys = tuple(record.record_key for record in bundle.source_a[:2])
    pair_references = (
        (source_record_keys[0], bundle.source_b[0].record_key),
        (source_record_keys[1], bundle.source_b[1].record_key),
    )
    return bundle, source_record_keys, pair_references, (0.9, 0.1)


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


def test_synthetic_inference_preserves_non_operational_authority_boundary() -> None:
    calibrator = _make_dummy_calibrator()
    bundle, source_keys, pair_refs, raw_scores = _synthetic_inference_inputs()
    synthetic_recipe = PipelineRecipeArtifact(
        recipe_id="synthetic_rec_1",
        recipe_version="v1.0.0",
        linkage_mode="link_only",
        assignment_constraint="one_to_one",
        configuration_digest="c" * 64,
        candidate_plan_digest="d" * 64,
        feature_schema_digest="f" * 64,
        champion_model_id="xgb_champion",
        champion_model_version="v1",
        champion_artifact_digest="a" * 64,
        calibrator_digest=calibrator.calibrator_digest,
        ranking_artifact_digest=None,
        decision_policy_digest="e" * 64,
        validation_evidence_digest="a" * 64,
        approval_status=RecipeApprovalStatus.SYNTHETIC_VALIDATED,
        operational_validation=OperationalValidationStatus.NOT_ESTABLISHED,
    )
    attestation = attest_generated_synthetic_inference(
        bundle=bundle,
        source_record_keys=source_keys,
        pair_references=pair_refs,
        raw_scores=raw_scores,
    )

    result = infer_with_approved_recipe(
        recipe=synthetic_recipe,
        source_record_keys=source_keys,
        pair_references=pair_refs,
        raw_scores=raw_scores,
        calibrator_artifact=calibrator,
        execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
        synthetic_attestation=attestation,
        synthetic_bundle=bundle,
        source_dataset_id="source_a",
        target_dataset_id="source_b",
    )

    assert result.execution_mode is RecipeExecutionMode.SYNTHETIC_INFERENCE
    assert synthetic_recipe.operational_validation is OperationalValidationStatus.NOT_ESTABLISHED
    assert synthetic_recipe.decision_authority == "explicit_policy_only"
    assert synthetic_recipe.merge_authority == "none"
    assert all(
        decision.decision_authority == "policy_classification" for decision in result.decisions
    )
    assert all(decision.merge_authority == "none" for decision in result.decisions)
    assert result.assignment_result.decision_authority == "none"
    assert result.assignment_result.assignment_authority == "global_selection_only"
    assert result.synthetic_attestation_digest == attestation.attestation_digest
    assert attestation.safe_summary()["operational_validity"] == "not_established"
    assert attestation.safe_summary()["decision_authority"] == "none"
    assert attestation.safe_summary()["assignment_authority"] == "none"
    assert attestation.safe_summary()["merge_authority"] == "none"

    serialized_source_keys = (source_keys[0],)
    serialized_pair_refs = (pair_refs[0],)
    serialized_scores = (raw_scores[0],)
    serialized_attestation = attest_generated_synthetic_inference(
        bundle=bundle,
        source_record_keys=serialized_source_keys,
        pair_references=serialized_pair_refs,
        raw_scores=serialized_scores,
    )
    serialized_result = infer_with_approved_recipe(
        recipe=serialize_pipeline_recipe(synthetic_recipe),
        source_record_keys=serialized_source_keys,
        pair_references=serialized_pair_refs,
        raw_scores=serialized_scores,
        calibrator_artifact=calibrator,
        execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
        synthetic_attestation=serialized_attestation,
        synthetic_bundle=bundle,
        source_dataset_id="source_a",
        target_dataset_id="source_b",
    )
    assert serialized_result.recipe_digest == synthetic_recipe.recipe_digest

    with pytest.raises(PipelineError, match="ML-RECIPE-005"):
        infer_with_approved_recipe(
            recipe=synthetic_recipe,
            source_record_keys=serialized_source_keys,
            pair_references=serialized_pair_refs,
            raw_scores=serialized_scores,
            calibrator_artifact=calibrator,
            execution_mode=RecipeExecutionMode.INFERENCE,
            source_dataset_id="source_a",
            target_dataset_id="source_b",
        )

    with pytest.raises(PipelineError, match="ML-RECIPE-016"):
        infer_with_approved_recipe(
            recipe=synthetic_recipe,
            source_record_keys=serialized_source_keys,
            pair_references=serialized_pair_refs,
            raw_scores=serialized_scores,
            calibrator_artifact=calibrator,
            execution_mode=RecipeExecutionMode.INFERENCE,
            synthetic_attestation=serialized_attestation,
            synthetic_bundle=bundle,
            source_dataset_id="source_a",
            target_dataset_id="source_b",
        )


def test_synthetic_inference_requires_exact_package_attestation() -> None:
    calibrator = _make_dummy_calibrator()
    bundle, source_keys, pair_refs, raw_scores = _synthetic_inference_inputs()
    synthetic_recipe = PipelineRecipeArtifact(
        recipe_id="synthetic_attestation_recipe",
        recipe_version="v1",
        linkage_mode="link_only",
        assignment_constraint="one_to_one",
        configuration_digest="c" * 64,
        candidate_plan_digest="d" * 64,
        feature_schema_digest="f" * 64,
        champion_model_id="xgb_champion",
        champion_model_version="v1",
        champion_artifact_digest="a" * 64,
        calibrator_digest=calibrator.calibrator_digest,
        ranking_artifact_digest=None,
        decision_policy_digest="e" * 64,
        validation_evidence_digest="a" * 64,
        approval_status=RecipeApprovalStatus.SYNTHETIC_VALIDATED,
        operational_validation=OperationalValidationStatus.NOT_ESTABLISHED,
    )

    with pytest.raises(PipelineError, match="ML-PIPE-062"):
        infer_with_approved_recipe(
            recipe=synthetic_recipe,
            source_record_keys=source_keys,
            pair_references=pair_refs,
            raw_scores=raw_scores,
            calibrator_artifact=calibrator,
            execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
            source_dataset_id="source_a",
            target_dataset_id="source_b",
        )

    attestation = attest_generated_synthetic_inference(
        bundle=bundle,
        source_record_keys=source_keys,
        pair_references=pair_refs,
        raw_scores=raw_scores,
    )
    with pytest.raises(PipelineError, match="ML-RECIPE-015"):
        infer_with_approved_recipe(
            recipe=synthetic_recipe,
            source_record_keys=source_keys,
            pair_references=pair_refs,
            raw_scores=(0.8, 0.2),
            calibrator_artifact=calibrator,
            execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
            synthetic_attestation=attestation,
            synthetic_bundle=bundle,
            source_dataset_id="source_a",
            target_dataset_id="source_b",
        )

    forged_attestation = SyntheticInferenceAttestation._issue(
        synthetic_bundle_digest="0" * 64,
        inference_input_digest=attestation.inference_input_digest,
        source_record_count=len(source_keys),
        pair_count=len(pair_refs),
    )
    with pytest.raises(PipelineError, match="ML-RECIPE-015"):
        infer_with_approved_recipe(
            recipe=synthetic_recipe,
            source_record_keys=source_keys,
            pair_references=pair_refs,
            raw_scores=raw_scores,
            calibrator_artifact=calibrator,
            execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
            synthetic_attestation=forged_attestation,
            synthetic_bundle=bundle,
            source_dataset_id="source_a",
            target_dataset_id="source_b",
        )

    with pytest.raises(PipelineError, match="ML-PIPE-060"):
        attest_generated_synthetic_inference(
            bundle=bundle,
            source_record_keys=source_keys,
            pair_references=pair_refs,
            raw_scores=raw_scores,
            source_dataset_id="operational_source",
            target_dataset_id="source_b",
        )


def test_synthetic_attestation_rejects_modified_bundle() -> None:
    bundle, source_keys, pair_refs, raw_scores = _synthetic_inference_inputs()
    modified_first = replace(bundle.source_a[0], label_value="synthetic_modified_value")
    modified_bundle = replace(bundle, source_a=(modified_first, *bundle.source_a[1:]))

    with pytest.raises(PipelineError, match="ML-PIPE-059"):
        attest_generated_synthetic_inference(
            bundle=modified_bundle,
            source_record_keys=source_keys,
            pair_references=pair_refs,
            raw_scores=raw_scores,
        )


@pytest.mark.parametrize("path_operation", ["is_file", "read_text"])
def test_recipe_path_os_errors_are_value_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_operation: str,
) -> None:
    recipe_path = tmp_path / "synthetic-restricted-recipe.json"
    recipe_path.write_text("{}", encoding="utf-8")
    restricted_value = "synthetic-private-path-fragment"

    def fail_safely(*_args: object, **_kwargs: object) -> bool:
        raise OSError(restricted_value)

    monkeypatch.setattr(Path, path_operation, fail_safely)
    with pytest.raises(PipelineError) as error:
        _resolve_recipe(recipe_path)

    assert error.value.code == "ML-PIPE-058"
    assert restricted_value not in str(error.value)
