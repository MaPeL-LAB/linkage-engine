"""Immutable approval contract separating model development from inference."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from mapel_linkage.domain.errors import PipelineError

_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")

type AssignmentConstraint = Literal[
    "one_to_one",
    "many_to_one",
    "one_to_many",
    "unconstrained",
]
type LinkageMode = Literal[
    "link_only",
    "dedupe_only",
    "link_and_dedupe",
    "multi_source",
]


class RecipeApprovalStatus(StrEnum):
    """Governance state of a versioned pipeline recipe."""

    DRAFT = "draft"
    SYNTHETIC_VALIDATED = "synthetic_validated"
    LOCAL_VALIDATION_COMPLETE = "local_validation_complete"
    APPROVED_FOR_INFERENCE = "approved_for_inference"


class RecipeExecutionMode(StrEnum):
    """Permitted execution contexts for a recipe."""

    DEVELOPMENT = "development"
    SHADOW = "shadow"
    INFERENCE = "inference"


class OperationalValidationStatus(StrEnum):
    """Whether approved local validation has established operational fitness."""

    NOT_ESTABLISHED = "not_established"
    LOCALLY_ESTABLISHED = "locally_established"


def _require_identifier(value: str) -> None:
    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise PipelineError("ML-RECIPE-001", "A pipeline recipe identifier is invalid.")


def _require_digest(value: str) -> None:
    if _DIGEST_PATTERN.fullmatch(value) is None:
        raise PipelineError("ML-RECIPE-002", "A pipeline recipe digest is invalid.")


@dataclass(frozen=True, slots=True)
class PipelineRecipeArtifact:
    """Aggregate-only recipe binding approved pipeline artifacts and policies."""

    recipe_id: str
    recipe_version: str
    linkage_mode: LinkageMode
    assignment_constraint: AssignmentConstraint
    configuration_digest: str
    candidate_plan_digest: str
    feature_schema_digest: str
    champion_model_id: str
    champion_model_version: str
    champion_artifact_digest: str
    calibrator_digest: str
    ranking_artifact_digest: str | None
    decision_policy_digest: str
    validation_evidence_digest: str
    approval_status: RecipeApprovalStatus = RecipeApprovalStatus.DRAFT
    operational_validation: OperationalValidationStatus = (
        OperationalValidationStatus.NOT_ESTABLISHED
    )
    decision_authority: Literal["explicit_policy_only"] = "explicit_policy_only"
    merge_authority: Literal["none"] = "none"

    def __post_init__(self) -> None:
        for identifier in (
            self.recipe_id,
            self.recipe_version,
            self.champion_model_id,
            self.champion_model_version,
        ):
            _require_identifier(identifier)
        for digest in (
            self.configuration_digest,
            self.candidate_plan_digest,
            self.feature_schema_digest,
            self.champion_artifact_digest,
            self.calibrator_digest,
            self.decision_policy_digest,
            self.validation_evidence_digest,
        ):
            _require_digest(digest)
        if self.ranking_artifact_digest is not None:
            _require_digest(self.ranking_artifact_digest)
        if (
            self.approval_status is RecipeApprovalStatus.APPROVED_FOR_INFERENCE
            and self.operational_validation is not OperationalValidationStatus.LOCALLY_ESTABLISHED
        ):
            raise PipelineError(
                "ML-RECIPE-003",
                "Inference approval requires completed local operational validation.",
            )

    @property
    def recipe_digest(self) -> str:
        """Return a stable digest over aggregate recipe provenance."""
        payload = {
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "linkage_mode": self.linkage_mode,
            "assignment_constraint": self.assignment_constraint,
            "configuration_digest": self.configuration_digest,
            "candidate_plan_digest": self.candidate_plan_digest,
            "feature_schema_digest": self.feature_schema_digest,
            "champion_model_id": self.champion_model_id,
            "champion_model_version": self.champion_model_version,
            "champion_artifact_digest": self.champion_artifact_digest,
            "calibrator_digest": self.calibrator_digest,
            "ranking_artifact_digest": self.ranking_artifact_digest,
            "decision_policy_digest": self.decision_policy_digest,
            "validation_evidence_digest": self.validation_evidence_digest,
            "approval_status": self.approval_status.value,
            "operational_validation": self.operational_validation.value,
            "decision_authority": self.decision_authority,
            "merge_authority": self.merge_authority,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def assert_usable_for(self, mode: RecipeExecutionMode) -> None:
        """Reject execution modes that exceed the recipe's approval authority."""
        if mode is RecipeExecutionMode.DEVELOPMENT:
            return
        if mode is RecipeExecutionMode.SHADOW:
            if self.approval_status is RecipeApprovalStatus.DRAFT:
                raise PipelineError(
                    "ML-RECIPE-004",
                    "A draft pipeline recipe cannot execute in shadow mode.",
                )
            return
        if (
            self.approval_status is not RecipeApprovalStatus.APPROVED_FOR_INFERENCE
            or self.operational_validation is not OperationalValidationStatus.LOCALLY_ESTABLISHED
        ):
            raise PipelineError(
                "ML-RECIPE-005",
                "The pipeline recipe is not approved for new-data inference.",
            )

    def safe_summary(self) -> dict[str, str | None]:
        """Return aggregate recipe metadata without paths, rows, or candidate pairs."""
        return {
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "recipe_digest": self.recipe_digest,
            "linkage_mode": self.linkage_mode,
            "assignment_constraint": self.assignment_constraint,
            "champion_model_id": self.champion_model_id,
            "champion_model_version": self.champion_model_version,
            "approval_status": self.approval_status.value,
            "operational_validation": self.operational_validation.value,
            "decision_authority": self.decision_authority,
            "merge_authority": self.merge_authority,
            "ranking_artifact": ("present" if self.ranking_artifact_digest is not None else None),
        }


__all__ = [
    "OperationalValidationStatus",
    "PipelineRecipeArtifact",
    "RecipeApprovalStatus",
    "RecipeExecutionMode",
]
