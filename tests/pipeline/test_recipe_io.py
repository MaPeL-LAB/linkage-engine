from __future__ import annotations

import json

import pytest

from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.pipeline import (
    OperationalValidationStatus,
    PipelineRecipeArtifact,
    RecipeApprovalStatus,
    RecipeExecutionMode,
)
from mapel_linkage.pipeline.recipe_io import (
    deserialize_pipeline_recipe,
    serialize_pipeline_recipe,
)


def recipe(*, approved: bool = False) -> PipelineRecipeArtifact:
    return PipelineRecipeArtifact(
        recipe_id="recipe_demo",
        recipe_version="v1",
        linkage_mode="link_only",
        assignment_constraint="one_to_one",
        configuration_digest="a" * 64,
        candidate_plan_digest="b" * 64,
        feature_schema_digest="c" * 64,
        champion_model_id="xgb_candidate",
        champion_model_version="m2-xgboost-v1",
        champion_artifact_digest="d" * 64,
        calibrator_digest="e" * 64,
        ranking_artifact_digest="f" * 64,
        decision_policy_digest="1" * 64,
        validation_evidence_digest="2" * 64,
        approval_status=(
            RecipeApprovalStatus.APPROVED_FOR_INFERENCE
            if approved
            else RecipeApprovalStatus.SYNTHETIC_VALIDATED
        ),
        operational_validation=(
            OperationalValidationStatus.LOCALLY_ESTABLISHED
            if approved
            else OperationalValidationStatus.NOT_ESTABLISHED
        ),
    )


def test_recipe_json_round_trip_is_canonical_and_value_safe() -> None:
    original = recipe()
    payload = serialize_pipeline_recipe(original)
    restored = deserialize_pipeline_recipe(payload)

    assert restored == original
    assert restored.recipe_digest == original.recipe_digest
    assert payload == serialize_pipeline_recipe(restored)
    assert "candidate pairs" not in payload.lower()
    assert "private/" not in payload


def test_recipe_json_rejects_tampering_unknown_and_duplicate_keys() -> None:
    payload = json.loads(serialize_pipeline_recipe(recipe()))
    payload["champion_artifact_digest"] = "9" * 64
    with pytest.raises(PipelineError, match="ML-RECIPE-012"):
        deserialize_pipeline_recipe(json.dumps(payload))

    payload = json.loads(serialize_pipeline_recipe(recipe()))
    payload["unexpected"] = True
    with pytest.raises(PipelineError, match="ML-RECIPE-010"):
        deserialize_pipeline_recipe(json.dumps(payload))

    with pytest.raises(PipelineError, match="ML-RECIPE-006"):
        deserialize_pipeline_recipe('{"schema_version":"1","schema_version":"1"}')


def test_only_locally_validated_approved_recipe_can_control_inference() -> None:
    synthetic = deserialize_pipeline_recipe(serialize_pipeline_recipe(recipe()))
    with pytest.raises(PipelineError, match="ML-RECIPE-005"):
        synthetic.assert_usable_for(RecipeExecutionMode.INFERENCE)

    approved = deserialize_pipeline_recipe(serialize_pipeline_recipe(recipe(approved=True)))
    approved.assert_usable_for(RecipeExecutionMode.INFERENCE)
    assert approved.merge_authority == "none"
