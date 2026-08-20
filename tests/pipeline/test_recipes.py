from __future__ import annotations

from typing import Any

import pytest

from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.pipeline import (
    OperationalValidationStatus,
    PipelineRecipeArtifact,
    RecipeApprovalStatus,
    RecipeExecutionMode,
)
from mapel_linkage.pipeline.recipes import SyntheticInferenceAttestation


def digest(character: str) -> str:
    return character * 64


def recipe(
    *,
    approval_status: RecipeApprovalStatus = RecipeApprovalStatus.DRAFT,
    operational_validation: OperationalValidationStatus = (
        OperationalValidationStatus.NOT_ESTABLISHED
    ),
) -> PipelineRecipeArtifact:
    return PipelineRecipeArtifact(
        recipe_id="synthetic_recipe",
        recipe_version="v1",
        linkage_mode="link_only",
        assignment_constraint="one_to_one",
        configuration_digest=digest("a"),
        candidate_plan_digest=digest("b"),
        feature_schema_digest=digest("c"),
        champion_model_id="xgb_pair_classifier",
        champion_model_version="v1",
        champion_artifact_digest=digest("d"),
        calibrator_digest=digest("e"),
        ranking_artifact_digest=digest("f"),
        decision_policy_digest=digest("1"),
        validation_evidence_digest=digest("2"),
        approval_status=approval_status,
        operational_validation=operational_validation,
    )


def test_pipeline_recipe_digest_is_deterministic_and_safe() -> None:
    first = recipe()
    second = recipe()

    assert first.recipe_digest == second.recipe_digest
    summary = first.safe_summary()
    assert summary["decision_authority"] == "explicit_policy_only"
    assert summary["merge_authority"] == "none"
    assert summary["ranking_artifact"] == "present"
    assert digest("d") not in repr(summary)


def test_draft_recipe_is_development_only() -> None:
    draft = recipe()

    draft.assert_usable_for(RecipeExecutionMode.DEVELOPMENT)
    with pytest.raises(PipelineError) as shadow_error:
        draft.assert_usable_for(RecipeExecutionMode.SHADOW)
    assert shadow_error.value.code == "ML-RECIPE-004"
    with pytest.raises(PipelineError) as inference_error:
        draft.assert_usable_for(RecipeExecutionMode.INFERENCE)
    assert inference_error.value.code == "ML-RECIPE-005"
    with pytest.raises(PipelineError) as synthetic_error:
        draft.assert_usable_for(RecipeExecutionMode.SYNTHETIC_INFERENCE)
    assert synthetic_error.value.code == "ML-RECIPE-013"


def test_synthetic_validated_recipe_requires_package_attestation_for_synthetic_inference() -> None:
    validated = recipe(approval_status=RecipeApprovalStatus.SYNTHETIC_VALIDATED)

    validated.assert_usable_for(RecipeExecutionMode.SHADOW)
    with pytest.raises(PipelineError) as synthetic_error:
        validated.assert_usable_for(RecipeExecutionMode.SYNTHETIC_INFERENCE)
    assert synthetic_error.value.code == "ML-RECIPE-014"
    with pytest.raises(PipelineError) as inference_error:
        validated.assert_usable_for(RecipeExecutionMode.INFERENCE)
    assert inference_error.value.code == "ML-RECIPE-005"


def test_synthetic_attestation_has_no_public_constructor() -> None:
    constructor: Any = SyntheticInferenceAttestation

    with pytest.raises(TypeError, match="issued by the package inference API"):
        constructor()


def test_inference_approval_requires_local_operational_validation() -> None:
    with pytest.raises(PipelineError) as error:
        recipe(approval_status=RecipeApprovalStatus.APPROVED_FOR_INFERENCE)

    assert error.value.code == "ML-RECIPE-003"


def test_locally_validated_approved_recipe_may_run_inference() -> None:
    approved = recipe(
        approval_status=RecipeApprovalStatus.APPROVED_FOR_INFERENCE,
        operational_validation=OperationalValidationStatus.LOCALLY_ESTABLISHED,
    )

    approved.assert_usable_for(RecipeExecutionMode.INFERENCE)
    with pytest.raises(PipelineError) as synthetic_error:
        approved.assert_usable_for(RecipeExecutionMode.SYNTHETIC_INFERENCE)
    assert synthetic_error.value.code == "ML-RECIPE-013"
