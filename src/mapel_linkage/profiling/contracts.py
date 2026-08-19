"""Privacy-safe staged task-profile contracts for the Linkage Strategy Advisor."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

Identifier = Annotated[
    StrictStr,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$"),
]
Digest = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]


class ProfileScope(StrEnum):
    """Where a task profile may be retained or exported."""

    GLOBAL_SYNTHETIC = "global_synthetic"
    LOCAL_RESTRICTED = "local_restricted"


class CountBand(StrEnum):
    """Coarse count bands used instead of unnecessary exact local quantities."""

    NOT_OBSERVED = "not_observed"
    VERY_SMALL = "very_small"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    VERY_LARGE = "very_large"


class RateBand(StrEnum):
    """Coarse rate bands used by post-candidate and post-evidence profiles."""

    NOT_OBSERVED = "not_observed"
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class LabelEvidenceClass(StrEnum):
    """Label evidence visible at the configuration boundary."""

    NONE = "none"
    UNVERIFIED_REFERENCE = "unverified_reference"
    SYNTHETIC_TRUTH = "synthetic_truth"
    VERIFIED_HUMAN_ADJUDICATION = "verified_human_adjudication"
    VERIFIED_GOLD_STANDARD = "verified_gold_standard"


class CandidateBudgetStatus(StrEnum):
    """Whether candidate generation remained inside its declared budget."""

    NOT_EVALUATED = "not_evaluated"
    WITHIN_BUDGET = "within_budget"
    NEAR_LIMIT = "near_limit"
    TRUNCATED = "truncated"
    FAILED = "failed"


class CandidateRecallStatus(StrEnum):
    """Whether candidate recall was measurable using eligible truth."""

    NOT_AVAILABLE = "not_available"
    ELIGIBLE_TRUTH_AVAILABLE = "eligible_truth_available"
    ESTIMATED = "estimated"
    FAILED = "failed"


class CalibrationEvidenceStatus(StrEnum):
    """Aggregate calibration evidence available after model scoring."""

    NOT_EVALUATED = "not_evaluated"
    UNCALIBRATED = "uncalibrated"
    CALIBRATED_SYNTHETIC = "calibrated_synthetic"
    CALIBRATED_LOCAL_VERIFIED = "calibrated_local_verified"
    INVALID = "invalid"


class ProfileNode(BaseModel):
    """Strict immutable base class for profile contracts."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        hide_input_in_errors=True,
    )


class VariableTypeCount(ProfileNode):
    """Aggregate count for one canonical variable type."""

    data_type: Literal["string", "date", "integer", "float", "boolean", "categorical"]
    count: Annotated[StrictInt, Field(ge=0, le=100_000)]


class PreflightTaskProfile(ProfileNode):
    """Observable aggregate task features available before row-level execution."""

    profile_schema_version: Literal["1"] = "1"
    profile_scope: ProfileScope
    linkage_mode: Literal["link_only", "dedupe_only", "link_and_dedupe", "multi_source"]
    assignment_constraint: Literal["one_to_one", "many_to_one", "one_to_many", "unconstrained"]
    dataset_count: Annotated[StrictInt, Field(ge=1, le=10_000)]
    source_count: Annotated[StrictInt, Field(ge=0, le=10_000)]
    target_count: Annotated[StrictInt, Field(ge=0, le=10_000)]
    reference_count: Annotated[StrictInt, Field(ge=0, le=10_000)]
    auxiliary_count: Annotated[StrictInt, Field(ge=0, le=10_000)]
    variable_count: Annotated[StrictInt, Field(ge=1, le=100_000)]
    variable_type_counts: Annotated[tuple[VariableTypeCount, ...], Field(min_length=1)]
    restricted_variable_count: Annotated[StrictInt, Field(ge=0, le=100_000)]
    transformation_count: Annotated[StrictInt, Field(ge=0, le=1_000_000)]
    blocking_rule_count: Annotated[StrictInt, Field(ge=1, le=100_000)]
    comparison_count: Annotated[StrictInt, Field(ge=1, le=100_000)]
    record_count_band: CountBand = CountBand.NOT_OBSERVED
    candidate_pair_budget_band: CountBand
    label_evidence_class: LabelEvidenceClass
    verified_labels_available: StrictBool
    remote_uri_access_enabled: Literal[False] = False
    network_access_enabled: Literal[False] = False
    profile_stage: Literal["preflight"] = "preflight"
    contains_record_values: Literal[False] = False
    contains_source_field_names: Literal[False] = False
    small_cell_policy: Literal["suppressed_or_binned"] = "suppressed_or_binned"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_aggregates(self) -> Self:
        role_total = (
            self.source_count + self.target_count + self.reference_count + self.auxiliary_count
        )
        if role_total != self.dataset_count:
            raise ValueError("Dataset role counts must sum to the dataset count.")
        if sum(item.count for item in self.variable_type_counts) != self.variable_count:
            raise ValueError("Variable-type counts must sum to the variable count.")
        if self.restricted_variable_count > self.variable_count:
            raise ValueError("Restricted-variable count cannot exceed the variable count.")
        eligible = self.label_evidence_class in {
            LabelEvidenceClass.SYNTHETIC_TRUTH,
            LabelEvidenceClass.VERIFIED_HUMAN_ADJUDICATION,
            LabelEvidenceClass.VERIFIED_GOLD_STANDARD,
        }
        if self.verified_labels_available != eligible:
            raise ValueError("Verified-label availability must agree with the evidence class.")
        return self

    @property
    def profile_digest(self) -> str:
        """Return a stable digest over approved aggregate profile fields."""
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, object]:
        """Return aggregate structural metadata without project or source identifiers."""
        return {
            "profile_schema_version": self.profile_schema_version,
            "profile_digest": self.profile_digest,
            "profile_scope": self.profile_scope.value,
            "profile_stage": self.profile_stage,
            "linkage_mode": self.linkage_mode,
            "assignment_constraint": self.assignment_constraint,
            "dataset_count": self.dataset_count,
            "variable_count": self.variable_count,
            "variable_type_counts": [
                item.model_dump(mode="json") for item in self.variable_type_counts
            ],
            "restricted_variable_count": self.restricted_variable_count,
            "transformation_count": self.transformation_count,
            "blocking_rule_count": self.blocking_rule_count,
            "comparison_count": self.comparison_count,
            "record_count_band": self.record_count_band.value,
            "candidate_pair_budget_band": self.candidate_pair_budget_band.value,
            "label_evidence_class": self.label_evidence_class.value,
            "verified_labels_available": self.verified_labels_available,
            "contains_record_values": self.contains_record_values,
            "contains_source_field_names": self.contains_source_field_names,
            "operational_validity": self.operational_validity,
        }


class CandidateGraphProfile(ProfileNode):
    """Aggregate candidate-graph features available after candidate generation."""

    profile_schema_version: Literal["1"] = "1"
    profile_scope: ProfileScope
    preflight_profile_digest: Digest
    candidate_pair_count_band: CountBand
    mean_candidate_set_size_band: CountBand
    p95_candidate_set_size_band: CountBand
    zero_candidate_rate_band: RateBand
    conflict_density_band: RateBand
    candidate_budget_status: CandidateBudgetStatus
    candidate_recall_status: CandidateRecallStatus
    profile_stage: Literal["candidate_graph"] = "candidate_graph"
    contains_record_values: Literal[False] = False
    contains_candidate_pairs: Literal[False] = False
    small_cell_policy: Literal["suppressed_or_binned"] = "suppressed_or_binned"
    operational_validity: Literal["not_established"] = "not_established"

    @property
    def profile_digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, object]:
        return {
            "profile_schema_version": self.profile_schema_version,
            "profile_digest": self.profile_digest,
            "profile_scope": self.profile_scope.value,
            "profile_stage": self.profile_stage,
            "candidate_pair_count_band": self.candidate_pair_count_band.value,
            "mean_candidate_set_size_band": self.mean_candidate_set_size_band.value,
            "p95_candidate_set_size_band": self.p95_candidate_set_size_band.value,
            "zero_candidate_rate_band": self.zero_candidate_rate_band.value,
            "conflict_density_band": self.conflict_density_band.value,
            "candidate_budget_status": self.candidate_budget_status.value,
            "candidate_recall_status": self.candidate_recall_status.value,
            "contains_record_values": self.contains_record_values,
            "contains_candidate_pairs": self.contains_candidate_pairs,
            "operational_validity": self.operational_validity,
        }


class EvidenceProfile(ProfileNode):
    """Aggregate post-scoring evidence used only for advisory refinement and monitoring."""

    profile_schema_version: Literal["1"] = "1"
    profile_scope: ProfileScope
    preflight_profile_digest: Digest
    candidate_graph_profile_digest: Digest | None = None
    pair_model_count: Annotated[StrictInt, Field(ge=0, le=100)]
    top_score_margin_band: RateBand
    model_disagreement_band: RateBand
    review_burden_band: RateBand
    assignment_change_band: RateBand
    calibration_status: CalibrationEvidenceStatus
    profile_stage: Literal["evidence"] = "evidence"
    contains_record_values: Literal[False] = False
    contains_candidate_pairs: Literal[False] = False
    contains_score_vectors: Literal[False] = False
    small_cell_policy: Literal["suppressed_or_binned"] = "suppressed_or_binned"
    operational_validity: Literal["not_established"] = "not_established"

    @property
    def profile_digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, object]:
        return {
            "profile_schema_version": self.profile_schema_version,
            "profile_digest": self.profile_digest,
            "profile_scope": self.profile_scope.value,
            "profile_stage": self.profile_stage,
            "pair_model_count": self.pair_model_count,
            "top_score_margin_band": self.top_score_margin_band.value,
            "model_disagreement_band": self.model_disagreement_band.value,
            "review_burden_band": self.review_burden_band.value,
            "assignment_change_band": self.assignment_change_band.value,
            "calibration_status": self.calibration_status.value,
            "contains_record_values": self.contains_record_values,
            "contains_candidate_pairs": self.contains_candidate_pairs,
            "contains_score_vectors": self.contains_score_vectors,
            "operational_validity": self.operational_validity,
        }


__all__ = [
    "CalibrationEvidenceStatus",
    "CandidateBudgetStatus",
    "CandidateGraphProfile",
    "CandidateRecallStatus",
    "CountBand",
    "EvidenceProfile",
    "LabelEvidenceClass",
    "PreflightTaskProfile",
    "ProfileScope",
    "RateBand",
    "VariableTypeCount",
]
