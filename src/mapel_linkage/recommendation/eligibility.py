"""Hard eligibility rules for Stage-1 structural pipeline shortlisting."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from mapel_linkage.recommendation.contracts import (
    CandidateRetrievalStatus,
    RecommendationIntent,
    RuntimeDependency,
    StructuralPipelineCandidate,
)


class EligibilityReason(StrEnum):
    """Stable transparent eligibility outcomes."""

    ELIGIBLE = "eligible"
    VERIFIED_LABELS_REQUIRED = "verified_labels_required"
    RUNTIME_DEPENDENCY_UNAVAILABLE = "runtime_dependency_unavailable"
    PROTECTED_OUT_OF_FOLD_PREDICTIONS_REQUIRED = "protected_out_of_fold_predictions_required"
    TWO_ELIGIBLE_BASE_MODELS_REQUIRED = "two_eligible_base_models_required"
    APPROVED_RECIPE_REQUIRED = "approved_recipe_required"
    APPROVED_ARTIFACT_REQUIRED = "approved_artifact_required"
    CANDIDATE_RETRIEVAL_FAILED = "candidate_retrieval_failed"


class EligibilityNode(BaseModel):
    """Strict immutable eligibility contract."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        hide_input_in_errors=True,
    )


class AdvisorContext(EligibilityNode):
    """Aggregate lifecycle and runtime context for advisory filtering."""

    intent: RecommendationIntent
    verified_labels_available: StrictBool
    approved_recipe_available: StrictBool = False
    protected_out_of_fold_predictions_available: StrictBool = False
    available_runtimes: Annotated[
        tuple[RuntimeDependency, ...], Field(min_length=1, max_length=3)
    ] = (RuntimeDependency.CORE,)
    approved_artifact_model_ids: Annotated[
        tuple[Annotated[StrictStr, Field(min_length=1, max_length=128)], ...],
        Field(max_length=64),
    ] = ()
    candidate_retrieval_status: CandidateRetrievalStatus = CandidateRetrievalStatus.UNKNOWN
    benchmark_family_count: Annotated[StrictInt, Field(ge=0, le=10_000_000)] = 0
    test_partition_used: Literal[False] = False

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        if len(self.available_runtimes) != len(set(self.available_runtimes)):
            raise ValueError("Available runtimes must be unique.")
        if self.available_runtimes[0] is not RuntimeDependency.CORE:
            raise ValueError("The core runtime must be available and listed first.")
        if len(self.approved_artifact_model_ids) != len(set(self.approved_artifact_model_ids)):
            raise ValueError("Approved artifact model IDs must be unique.")
        return self


class EligibilityDecision(EligibilityNode):
    """One hard-rule decision for a structural pipeline candidate."""

    candidate_id: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    eligible: StrictBool
    reasons: Annotated[tuple[EligibilityReason, ...], Field(min_length=1, max_length=16)]
    eligibility_policy_version: Literal["stage1-v1"] = "stage1-v1"
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"

    @model_validator(mode="after")
    def validate_reasons(self) -> Self:
        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("Eligibility reasons must be unique.")
        if self.eligible != (self.reasons == (EligibilityReason.ELIGIBLE,)):
            raise ValueError("Eligibility status must agree with its reason codes.")
        return self


_EMPTY_MODEL_IDS: frozenset[str] = frozenset()


def evaluate_candidate(
    candidate: StructuralPipelineCandidate,
    *,
    context: AdvisorContext,
    eligible_base_model_ids: frozenset[str] = _EMPTY_MODEL_IDS,
) -> EligibilityDecision:
    """Apply hard rules; no weighted score may override a failed rule."""

    reasons: list[EligibilityReason] = []
    if context.candidate_retrieval_status is CandidateRetrievalStatus.FAILED:
        reasons.append(EligibilityReason.CANDIDATE_RETRIEVAL_FAILED)

    if context.intent in {
        RecommendationIntent.INFER_WITH_APPROVED_RECIPE,
        RecommendationIntent.SHADOW_SCORE_CHALLENGER,
    }:
        if not context.approved_recipe_available:
            reasons.append(EligibilityReason.APPROVED_RECIPE_REQUIRED)
        if candidate.pair_model_id not in context.approved_artifact_model_ids:
            reasons.append(EligibilityReason.APPROVED_ARTIFACT_REQUIRED)
    else:
        labels_required = candidate.requires_verified_labels
        if context.intent is RecommendationIntent.FIT_OR_SELECT_CALIBRATION:
            labels_required = True
        if labels_required and not context.verified_labels_available:
            reasons.append(EligibilityReason.VERIFIED_LABELS_REQUIRED)

    if not set(candidate.required_runtimes).issubset(context.available_runtimes):
        reasons.append(EligibilityReason.RUNTIME_DEPENDENCY_UNAVAILABLE)

    if candidate.requires_protected_out_of_fold_predictions:
        if not context.protected_out_of_fold_predictions_available:
            reasons.append(EligibilityReason.PROTECTED_OUT_OF_FOLD_PREDICTIONS_REQUIRED)
        if len(set(candidate.base_model_ids).intersection(eligible_base_model_ids)) < 2:
            reasons.append(EligibilityReason.TWO_ELIGIBLE_BASE_MODELS_REQUIRED)

    if not reasons:
        return EligibilityDecision(
            candidate_id=candidate.candidate_id,
            eligible=True,
            reasons=(EligibilityReason.ELIGIBLE,),
        )
    return EligibilityDecision(
        candidate_id=candidate.candidate_id,
        eligible=False,
        reasons=tuple(dict.fromkeys(reasons)),
    )


__all__ = [
    "AdvisorContext",
    "EligibilityDecision",
    "EligibilityReason",
    "evaluate_candidate",
]
