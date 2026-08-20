from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mapel_linkage.domain.errors import FellegiSunterError, PipelineError
from mapel_linkage.io import DuckDBStore
from mapel_linkage.models import (
    SplinkNativeModelArtifact,
    assert_splink_native_recipe_binding,
    deserialize_splink_native_model,
    serialize_splink_native_model,
)
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
from tests.models.fellegi_sunter.test_splink_native import (
    _CONFIGURATION_DIGEST,
    _SEED,
    _fit_once,
    _model,
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


@pytest.fixture(scope="module")
def native_artifact() -> SplinkNativeModelArtifact:
    with DuckDBStore() as store:
        trained, settings_plan, _ = _fit_once(store)
    return deserialize_splink_native_model(
        serialize_splink_native_model(trained),
        settings_plan=settings_plan,
        model=_model(),
        configuration_digest=_CONFIGURATION_DIGEST,
        feature_schema_digest=trained.feature_schema_digest,
        random_seed=_SEED,
    )


def native_recipe(artifact: SplinkNativeModelArtifact) -> PipelineRecipeArtifact:
    return PipelineRecipeArtifact(
        recipe_id="recipe_native_demo",
        recipe_version="v1",
        linkage_mode="link_only",
        assignment_constraint="one_to_one",
        configuration_digest=artifact.configuration_digest,
        candidate_plan_digest="b" * 64,
        feature_schema_digest=artifact.feature_schema_digest,
        champion_model_id=artifact.model_id,
        champion_model_version=artifact.model_version,
        champion_artifact_digest=artifact.artifact_digest,
        calibrator_digest="e" * 64,
        ranking_artifact_digest="f" * 64,
        decision_policy_digest="1" * 64,
        validation_evidence_digest="2" * 64,
        approval_status=RecipeApprovalStatus.SYNTHETIC_VALIDATED,
        operational_validation=OperationalValidationStatus.NOT_ESTABLISHED,
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


def test_recipe_json_round_trip_binds_exact_native_splink_artifact(
    native_artifact: SplinkNativeModelArtifact,
) -> None:
    artifact = native_artifact
    restored = deserialize_pipeline_recipe(serialize_pipeline_recipe(native_recipe(artifact)))

    assert_splink_native_recipe_binding(recipe=restored, artifact=artifact)
    assert restored.champion_model_id == artifact.model_id
    assert restored.champion_model_version == artifact.model_version
    assert restored.champion_artifact_digest == artifact.artifact_digest
    assert restored.configuration_digest == artifact.configuration_digest
    assert restored.feature_schema_digest == artifact.feature_schema_digest


def test_native_splink_recipe_binding_rejects_all_contract_drift(
    native_artifact: SplinkNativeModelArtifact,
) -> None:
    artifact = native_artifact
    restored = deserialize_pipeline_recipe(serialize_pipeline_recipe(native_recipe(artifact)))
    drifted_recipes = (
        replace(restored, champion_model_id="fs_native_challenger"),
        replace(restored, champion_model_version="i1-splink-native-v2"),
        replace(restored, champion_artifact_digest="7" * 64),
        replace(restored, configuration_digest="8" * 64),
        replace(restored, feature_schema_digest="9" * 64),
    )

    for drifted in drifted_recipes:
        with pytest.raises(FellegiSunterError, match="ML-FS-077"):
            assert_splink_native_recipe_binding(recipe=drifted, artifact=artifact)


def test_only_locally_validated_approved_recipe_can_control_inference() -> None:
    synthetic = deserialize_pipeline_recipe(serialize_pipeline_recipe(recipe()))
    with pytest.raises(PipelineError, match="ML-RECIPE-005"):
        synthetic.assert_usable_for(RecipeExecutionMode.INFERENCE)

    approved = deserialize_pipeline_recipe(serialize_pipeline_recipe(recipe(approved=True)))
    approved.assert_usable_for(RecipeExecutionMode.INFERENCE)
    assert approved.merge_authority == "none"
