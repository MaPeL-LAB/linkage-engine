"""Strict non-executable JSON serialization for approved pipeline recipes."""

from __future__ import annotations

import json
from collections.abc import Iterable

from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.pipeline.recipes import (
    OperationalValidationStatus,
    PipelineRecipeArtifact,
    RecipeApprovalStatus,
)

_SCHEMA_VERSION = "1"
_MAX_RECIPE_BYTES = 262_144
_EXPECTED_KEYS = {
    "schema_version",
    "recipe_id",
    "recipe_version",
    "linkage_mode",
    "assignment_constraint",
    "configuration_digest",
    "candidate_plan_digest",
    "feature_schema_digest",
    "champion_model_id",
    "champion_model_version",
    "champion_artifact_digest",
    "calibrator_digest",
    "ranking_artifact_digest",
    "decision_policy_digest",
    "validation_evidence_digest",
    "approval_status",
    "operational_validation",
    "decision_authority",
    "merge_authority",
    "recipe_digest",
}


def _reject_duplicate_keys(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PipelineError("ML-RECIPE-006", "A pipeline recipe contains duplicate keys.")
        result[key] = value
    return result


def pipeline_recipe_payload(recipe: PipelineRecipeArtifact) -> dict[str, object]:
    """Return the canonical aggregate-only recipe payload."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "recipe_id": recipe.recipe_id,
        "recipe_version": recipe.recipe_version,
        "linkage_mode": recipe.linkage_mode,
        "assignment_constraint": recipe.assignment_constraint,
        "configuration_digest": recipe.configuration_digest,
        "candidate_plan_digest": recipe.candidate_plan_digest,
        "feature_schema_digest": recipe.feature_schema_digest,
        "champion_model_id": recipe.champion_model_id,
        "champion_model_version": recipe.champion_model_version,
        "champion_artifact_digest": recipe.champion_artifact_digest,
        "calibrator_digest": recipe.calibrator_digest,
        "ranking_artifact_digest": recipe.ranking_artifact_digest,
        "decision_policy_digest": recipe.decision_policy_digest,
        "validation_evidence_digest": recipe.validation_evidence_digest,
        "approval_status": recipe.approval_status.value,
        "operational_validation": recipe.operational_validation.value,
        "decision_authority": recipe.decision_authority,
        "merge_authority": recipe.merge_authority,
        "recipe_digest": recipe.recipe_digest,
    }


def serialize_pipeline_recipe(recipe: PipelineRecipeArtifact) -> str:
    """Serialize without code objects, paths, rows, identifiers, or candidate pairs."""
    return json.dumps(
        pipeline_recipe_payload(recipe),
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def deserialize_pipeline_recipe(payload: str) -> PipelineRecipeArtifact:
    """Load a strict recipe and detect unknown keys, duplicates, and tampering."""
    if len(payload.encode("utf-8")) > _MAX_RECIPE_BYTES:
        raise PipelineError("ML-RECIPE-007", "The pipeline recipe exceeds its size limit.")
    try:
        raw = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except PipelineError:
        raise
    except (TypeError, ValueError):
        raise PipelineError("ML-RECIPE-008", "The pipeline recipe is not valid JSON.") from None
    if not isinstance(raw, dict):
        raise PipelineError("ML-RECIPE-009", "The pipeline recipe must be a JSON object.")
    if set(raw) != _EXPECTED_KEYS:
        raise PipelineError("ML-RECIPE-010", "The pipeline recipe schema is invalid.")
    if raw.get("schema_version") != _SCHEMA_VERSION:
        raise PipelineError("ML-RECIPE-011", "The pipeline recipe schema version is unsupported.")
    expected_digest = raw.get("recipe_digest")
    try:
        recipe = PipelineRecipeArtifact(
            recipe_id=str(raw["recipe_id"]),
            recipe_version=str(raw["recipe_version"]),
            linkage_mode=raw["linkage_mode"],
            assignment_constraint=raw["assignment_constraint"],
            configuration_digest=str(raw["configuration_digest"]),
            candidate_plan_digest=str(raw["candidate_plan_digest"]),
            feature_schema_digest=str(raw["feature_schema_digest"]),
            champion_model_id=str(raw["champion_model_id"]),
            champion_model_version=str(raw["champion_model_version"]),
            champion_artifact_digest=str(raw["champion_artifact_digest"]),
            calibrator_digest=str(raw["calibrator_digest"]),
            ranking_artifact_digest=(
                None
                if raw["ranking_artifact_digest"] is None
                else str(raw["ranking_artifact_digest"])
            ),
            decision_policy_digest=str(raw["decision_policy_digest"]),
            validation_evidence_digest=str(raw["validation_evidence_digest"]),
            approval_status=RecipeApprovalStatus(str(raw["approval_status"])),
            operational_validation=OperationalValidationStatus(
                str(raw["operational_validation"])
            ),
            decision_authority=raw["decision_authority"],
            merge_authority=raw["merge_authority"],
        )
    except (KeyError, TypeError, ValueError, PipelineError):
        raise PipelineError("ML-RECIPE-010", "The pipeline recipe schema is invalid.") from None
    if not isinstance(expected_digest, str) or recipe.recipe_digest != expected_digest:
        raise PipelineError("ML-RECIPE-012", "The pipeline recipe integrity check failed.")
    return recipe


__all__ = [
    "deserialize_pipeline_recipe",
    "pipeline_recipe_payload",
    "serialize_pipeline_recipe",
]
