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
_SYNTHETIC_ATTESTATION_ISSUER = object()

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
    SYNTHETIC_INFERENCE = "synthetic_inference"
    INFERENCE = "inference"


class OperationalValidationStatus(StrEnum):
    """Whether approved local validation has established operational fitness."""

    NOT_ESTABLISHED = "not_established"
    LOCALLY_ESTABLISHED = "locally_established"


@dataclass(frozen=True, slots=True, init=False, repr=False)
class SyntheticInferenceAttestation:
    """Opaque, aggregate-only authorization for one package-generated synthetic input.

    Instances are issued by the package inference API after it has regenerated and verified a
    :class:`~mapel_linkage.synthetic.SyntheticBundle`. Recipe approval alone never issues this
    capability, and the capability carries no relationship, assignment, or merge authority.
    """

    schema_version: Literal["1"]
    data_origin: Literal["package_generated_synthetic"]
    data_policy: Literal["synthetic_only"]
    operational_validity: Literal["not_established"]
    decision_authority: Literal["none"]
    assignment_authority: Literal["none"]
    merge_authority: Literal["none"]
    synthetic_bundle_digest: str
    inference_input_digest: str
    source_record_count: int
    pair_count: int
    _issuer: object

    def __new__(cls) -> SyntheticInferenceAttestation:
        raise TypeError(
            "SyntheticInferenceAttestation instances are issued by the package inference API."
        )

    @classmethod
    def _issue(
        cls,
        *,
        synthetic_bundle_digest: str,
        inference_input_digest: str,
        source_record_count: int,
        pair_count: int,
    ) -> SyntheticInferenceAttestation:
        """Issue an attestation after package-owned synthetic provenance verification."""
        instance = object.__new__(cls)
        object.__setattr__(instance, "schema_version", "1")
        object.__setattr__(instance, "data_origin", "package_generated_synthetic")
        object.__setattr__(instance, "data_policy", "synthetic_only")
        object.__setattr__(instance, "operational_validity", "not_established")
        object.__setattr__(instance, "decision_authority", "none")
        object.__setattr__(instance, "assignment_authority", "none")
        object.__setattr__(instance, "merge_authority", "none")
        object.__setattr__(instance, "synthetic_bundle_digest", synthetic_bundle_digest)
        object.__setattr__(instance, "inference_input_digest", inference_input_digest)
        object.__setattr__(instance, "source_record_count", source_record_count)
        object.__setattr__(instance, "pair_count", pair_count)
        object.__setattr__(instance, "_issuer", _SYNTHETIC_ATTESTATION_ISSUER)
        instance.assert_valid_contract()
        return instance

    def assert_valid_contract(self) -> None:
        """Reject forged, malformed, or authority-bearing attestations without echoing values."""
        if (
            getattr(self, "_issuer", None) is not _SYNTHETIC_ATTESTATION_ISSUER
            or getattr(self, "schema_version", None) != "1"
            or getattr(self, "data_origin", None) != "package_generated_synthetic"
            or getattr(self, "data_policy", None) != "synthetic_only"
            or getattr(self, "operational_validity", None) != "not_established"
            or getattr(self, "decision_authority", None) != "none"
            or getattr(self, "assignment_authority", None) != "none"
            or getattr(self, "merge_authority", None) != "none"
            or getattr(self, "source_record_count", 0) < 1
            or getattr(self, "pair_count", 0) < 1
        ):
            raise PipelineError(
                "ML-RECIPE-014",
                "The synthetic inference attestation is invalid or exceeds its authority.",
            )
        try:
            _require_digest(self.synthetic_bundle_digest)
            _require_digest(self.inference_input_digest)
        except PipelineError:
            raise PipelineError(
                "ML-RECIPE-014",
                "The synthetic inference attestation is invalid or exceeds its authority.",
            ) from None

    def assert_authorizes(
        self,
        *,
        inference_input_digest: str,
        source_record_count: int,
        pair_count: int,
    ) -> None:
        """Require an exact, value-hidden binding to the current inference input."""
        self.assert_valid_contract()
        if (
            self.inference_input_digest != inference_input_digest
            or self.source_record_count != source_record_count
            or self.pair_count != pair_count
        ):
            raise PipelineError(
                "ML-RECIPE-015",
                "The synthetic inference attestation does not authorize this input.",
            )

    @property
    def attestation_digest(self) -> str:
        """Return a stable aggregate digest for audit linkage."""
        payload = {
            "schema_version": self.schema_version,
            "data_origin": self.data_origin,
            "data_policy": self.data_policy,
            "operational_validity": self.operational_validity,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "synthetic_bundle_digest": self.synthetic_bundle_digest,
            "inference_input_digest": self.inference_input_digest,
            "source_record_count": self.source_record_count,
            "pair_count": self.pair_count,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, str | int]:
        """Return only aggregate authority and provenance metadata."""
        return {
            "attestation_digest": self.attestation_digest,
            "data_origin": self.data_origin,
            "data_policy": self.data_policy,
            "operational_validity": self.operational_validity,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "source_record_count": self.source_record_count,
            "pair_count": self.pair_count,
        }

    def __repr__(self) -> str:
        return "<SyntheticInferenceAttestation aggregate-only>"


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

    def assert_usable_for(
        self,
        mode: RecipeExecutionMode,
        *,
        synthetic_attestation: SyntheticInferenceAttestation | None = None,
    ) -> None:
        """Reject modes exceeding recipe approval and synthetic input authority."""
        if (
            mode is not RecipeExecutionMode.SYNTHETIC_INFERENCE
            and synthetic_attestation is not None
        ):
            raise PipelineError(
                "ML-RECIPE-016",
                "A synthetic inference attestation cannot authorize this execution mode.",
            )
        if mode is RecipeExecutionMode.DEVELOPMENT:
            return
        if mode is RecipeExecutionMode.SHADOW:
            if self.approval_status is RecipeApprovalStatus.DRAFT:
                raise PipelineError(
                    "ML-RECIPE-004",
                    "A draft pipeline recipe cannot execute in shadow mode.",
                )
            return
        if mode is RecipeExecutionMode.SYNTHETIC_INFERENCE:
            if (
                self.approval_status is not RecipeApprovalStatus.SYNTHETIC_VALIDATED
                or self.operational_validation is not OperationalValidationStatus.NOT_ESTABLISHED
            ):
                raise PipelineError(
                    "ML-RECIPE-013",
                    "Synthetic inference requires synthetic validation and cannot claim "
                    "operational validity.",
                )
            if synthetic_attestation is None:
                raise PipelineError(
                    "ML-RECIPE-014",
                    "Synthetic inference requires a package-issued input attestation.",
                )
            synthetic_attestation.assert_valid_contract()
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
    "SyntheticInferenceAttestation",
]
