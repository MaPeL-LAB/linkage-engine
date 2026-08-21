"""Immutable advisory contracts that cannot acquire linkage or approval authority."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

Identifier = Annotated[
    StrictStr,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$"),
]
Digest = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]


class RecommendationIntent(StrEnum):
    """Lifecycle intent changes what is eligible; it is not an identity decision."""

    DEVELOP_NEW_RECIPE = "develop_new_recipe"
    EVALUATE_CHALLENGERS = "evaluate_challengers"
    FIT_OR_SELECT_CALIBRATION = "fit_or_select_calibration"
    INFER_WITH_APPROVED_RECIPE = "infer_with_approved_recipe"
    SHADOW_SCORE_CHALLENGER = "shadow_score_challenger"
    PLAN_BENCHMARK = "plan_benchmark"


class EvidenceScope(StrEnum):
    """Evidence classes that must remain distinguishable in advisor output."""

    GLOBAL_SYNTHETIC = "global_synthetic"
    LOCAL_SCHEMA_MATCHED_SYNTHETIC = "local_schema_matched_synthetic"
    LOCAL_VERIFIED_VALIDATION = "local_verified_validation"
    LOCAL_OPERATIONAL_MONITORING = "local_operational_monitoring"


class CoverageStatus(StrEnum):
    """How much empirical support exists for the recommendation."""

    STRUCTURAL_ONLY = "structural_only"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    WITHIN_BENCHMARK_ENVELOPE = "within_benchmark_envelope"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    NOT_EVALUATED = "not_evaluated"


class AbstentionReason(StrEnum):
    """Stable reason codes for refusing empirical or operational claims."""

    NO_BENCHMARK_EVIDENCE = "no_benchmark_evidence"
    NO_ELIGIBLE_PIPELINES = "no_eligible_pipelines"
    APPROVED_RECIPE_REQUIRED = "approved_recipe_required"
    APPROVED_ARTIFACT_REQUIRED = "approved_artifact_required"
    CANDIDATE_RETRIEVAL_FAILED = "candidate_retrieval_failed"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    LOCAL_CONFIRMATION_REQUIRED = "local_confirmation_required"


class RuntimeDependency(StrEnum):
    """Runtime families considered by transparent eligibility rules."""

    CORE = "core"
    LIGHTGBM = "lightgbm"
    PYTORCH = "pytorch"


class RankingStrategy(StrEnum):
    """Candidate ordering method in a structural pipeline template."""

    MODEL_SCORE = "model_score"
    XGBOOST_RANKER = "xgboost_ranker"
    LIGHTGBM_RANKER = "lightgbm_ranker"


class CandidateRetrievalStatus(StrEnum):
    """Whether the configured candidate plan has been evaluated."""

    UNKNOWN = "unknown"
    ESTABLISHED = "established"
    FAILED = "failed"


class RecommendationNode(BaseModel):
    """Strict immutable advisor contract."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        hide_input_in_errors=True,
    )


class EvidenceContribution(RecommendationNode):
    """Aggregate contribution from one evidence class."""

    scope: EvidenceScope
    scenario_family_count: Annotated[StrictInt, Field(ge=0, le=10_000_000)] = 0
    approved_run_count: Annotated[StrictInt, Field(ge=0, le=10_000_000)] = 0
    current: StrictBool = True
    eligible: StrictBool = False
    evidence_digest: Digest | None = None

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        if (self.scenario_family_count or self.approved_run_count) and self.evidence_digest is None:
            raise ValueError(
                "Non-empty evidence contributions require an aggregate evidence digest."
            )
        return self


class StructuralPipelineCandidate(RecommendationNode):
    """A complete structural recipe template, not an approved pipeline recipe."""

    candidate_id: Identifier
    configuration_digest: Digest
    portfolio_digest: Digest
    pair_model_id: Identifier
    pair_model_family: Literal["fellegi_sunter", "xgboost", "lightgbm", "pytorch", "stacking"]
    pair_model_role: Literal["baseline", "challenger", "ensemble", "shadow"]
    base_model_ids: Annotated[tuple[Identifier, ...], Field(max_length=16)] = ()
    ranking_strategy: RankingStrategy
    ranking_model_id: Identifier | None = None
    calibration_method: Literal["sigmoid", "isotonic", "beta"]
    linkage_mode: Literal["link_only", "dedupe_only", "link_and_dedupe", "multi_source"]
    assignment_constraint: Literal["one_to_one", "many_to_one", "one_to_many", "unconstrained"]
    requires_verified_labels: StrictBool
    requires_protected_out_of_fold_predictions: StrictBool = False
    required_runtimes: Annotated[tuple[RuntimeDependency, ...], Field(min_length=1, max_length=3)]
    structural_complexity: Annotated[StrictInt, Field(ge=0, le=3)]
    interaction_capacity: Annotated[StrictInt, Field(ge=0, le=3)]
    interpretability_score: Annotated[StrictInt, Field(ge=0, le=3)]
    artifact_portability_score: Annotated[StrictInt, Field(ge=0, le=3)]
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"

    @model_validator(mode="after")
    def validate_ranking_and_runtime(self) -> Self:
        if len(self.required_runtimes) != len(set(self.required_runtimes)):
            raise ValueError("Required runtime dependencies must be unique.")
        if self.required_runtimes[0] is not RuntimeDependency.CORE:
            raise ValueError("Every structural candidate must include the core runtime first.")
        if self.ranking_strategy is RankingStrategy.MODEL_SCORE:
            if self.ranking_model_id is not None:
                raise ValueError("Model-score ranking cannot name a learned ranker.")
        elif self.ranking_model_id is None:
            raise ValueError("A learned ranking strategy requires a ranking-model ID.")
        if self.pair_model_family == "stacking" and len(set(self.base_model_ids)) < 2:
            raise ValueError("A stacking candidate requires at least two base models.")
        if self.pair_model_family != "stacking" and self.base_model_ids:
            raise ValueError("Only stacking candidates may name base models.")
        return self

    @property
    def candidate_digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_digest": self.candidate_digest,
            "pair_model_id": self.pair_model_id,
            "pair_model_family": self.pair_model_family,
            "pair_model_role": self.pair_model_role,
            "ranking_strategy": self.ranking_strategy.value,
            "ranking_model_id": self.ranking_model_id,
            "calibration_method": self.calibration_method,
            "linkage_mode": self.linkage_mode,
            "assignment_constraint": self.assignment_constraint,
            "requires_verified_labels": self.requires_verified_labels,
            "requires_protected_out_of_fold_predictions": (
                self.requires_protected_out_of_fold_predictions
            ),
            "required_runtimes": [item.value for item in self.required_runtimes],
            "structural_complexity": self.structural_complexity,
            "interaction_capacity": self.interaction_capacity,
            "interpretability_score": self.interpretability_score,
            "artifact_portability_score": self.artifact_portability_score,
            "recommendation_authority": self.recommendation_authority,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
        }


class DisqualifiedCandidate(RecommendationNode):
    """One structural candidate rejected by hard eligibility rules."""

    candidate_id: Identifier
    candidate_digest: Digest
    reasons: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=16)]

    @model_validator(mode="after")
    def validate_reasons(self) -> Self:
        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("Disqualification reasons must be unique.")
        return self


class CandidateExplanation(RecommendationNode):
    """Transparent applied-rule explanation without empirical performance claims."""

    candidate_id: Identifier
    rule_codes: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=32)]
    empirical_performance_claim: Literal[False] = False

    @model_validator(mode="after")
    def validate_rule_codes(self) -> Self:
        if len(self.rule_codes) != len(set(self.rule_codes)):
            raise ValueError("Explanation rule codes must be unique.")
        return self


class PipelineRecommendation(RecommendationNode):
    """Stage-1 advisory result kept separate from PipelineRecipeArtifact approval."""

    recommendation_schema_version: Literal["1"] = "1"
    recommendation_id: Identifier
    intent: RecommendationIntent
    task_profile_digest: Digest
    utility_policy_digest: Digest
    eligibility_policy_digest: Digest
    registry_snapshot_digest: Digest | None = None
    coverage_status: CoverageStatus
    out_of_distribution_score: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] | None = None
    abstained_from_empirical_ranking: StrictBool = True
    abstention_reasons: Annotated[tuple[AbstentionReason, ...], Field(min_length=1, max_length=16)]
    mandatory_baseline_candidate_id: Identifier | None
    shortlist: Annotated[tuple[StructuralPipelineCandidate, ...], Field(max_length=9)] = ()
    structural_pareto_candidate_ids: Annotated[tuple[Identifier, ...], Field(max_length=32)] = ()
    disqualified_candidates: Annotated[tuple[DisqualifiedCandidate, ...], Field(max_length=64)] = ()
    explanations: Annotated[tuple[CandidateExplanation, ...], Field(max_length=64)] = ()
    evidence_contributions: Annotated[
        tuple[EvidenceContribution, ...], Field(min_length=1, max_length=4)
    ]
    required_local_confirmation: Annotated[
        tuple[Identifier, ...], Field(min_length=1, max_length=16)
    ]
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    empirical_performance_claims: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_recommendation(self) -> Self:
        shortlist_ids = [candidate.candidate_id for candidate in self.shortlist]
        if len(shortlist_ids) != len(set(shortlist_ids)):
            raise ValueError("Recommendation shortlist candidate IDs must be unique.")
        if len(self.structural_pareto_candidate_ids) != len(
            set(self.structural_pareto_candidate_ids)
        ):
            raise ValueError("Structural Pareto candidate IDs must be unique.")
        if (
            self.mandatory_baseline_candidate_id is not None
            and self.shortlist
            and self.mandatory_baseline_candidate_id not in shortlist_ids
        ):
            raise ValueError("A non-empty shortlist must retain the mandatory baseline.")
        if self.registry_snapshot_digest is None and any(
            contribution.scenario_family_count or contribution.approved_run_count
            for contribution in self.evidence_contributions
        ):
            raise ValueError("Non-empty advisor evidence requires a registry snapshot digest.")
        if (
            self.coverage_status
            in {
                CoverageStatus.WITHIN_BENCHMARK_ENVELOPE,
                CoverageStatus.OUT_OF_DISTRIBUTION,
            }
            and self.out_of_distribution_score is None
        ):
            raise ValueError("Empirical coverage states require an out-of-distribution score.")
        if (
            self.coverage_status is CoverageStatus.STRUCTURAL_ONLY
            and AbstentionReason.NO_BENCHMARK_EVIDENCE not in self.abstention_reasons
        ):
            raise ValueError("Structural-only advice must disclose missing benchmark evidence.")
        if not self.abstained_from_empirical_ranking and (
            self.registry_snapshot_digest is None
            or self.coverage_status is not CoverageStatus.WITHIN_BENCHMARK_ENVELOPE
            or not any(contribution.eligible for contribution in self.evidence_contributions)
        ):
            raise ValueError(
                "Empirical ranking requires eligible evidence within the benchmark envelope."
            )
        return self

    @property
    def recommendation_digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, object]:
        return {
            "recommendation_schema_version": self.recommendation_schema_version,
            "recommendation_id": self.recommendation_id,
            "recommendation_digest": self.recommendation_digest,
            "intent": self.intent.value,
            "task_profile_digest": self.task_profile_digest,
            "coverage_status": self.coverage_status.value,
            "out_of_distribution_score": self.out_of_distribution_score,
            "abstained_from_empirical_ranking": self.abstained_from_empirical_ranking,
            "abstention_reasons": [item.value for item in self.abstention_reasons],
            "mandatory_baseline_candidate_id": self.mandatory_baseline_candidate_id,
            "shortlist": [candidate.safe_summary() for candidate in self.shortlist],
            "structural_pareto_candidate_ids": list(self.structural_pareto_candidate_ids),
            "disqualified_candidates": [
                item.model_dump(mode="json") for item in self.disqualified_candidates
            ],
            "explanations": [item.model_dump(mode="json") for item in self.explanations],
            "evidence_contributions": [
                item.model_dump(mode="json") for item in self.evidence_contributions
            ],
            "required_local_confirmation": list(self.required_local_confirmation),
            "recommendation_authority": self.recommendation_authority,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "automatic_promotion": self.automatic_promotion,
            "empirical_performance_claims": self.empirical_performance_claims,
            "operational_validity": self.operational_validity,
        }


class EmpiricalMetricDistribution(RecommendationNode):
    """Aggregated empirical benchmark metrics across nearest scenario runs."""

    sample_count: Annotated[StrictInt, Field(ge=0)]
    mean_candidate_recall: float = Field(ge=0.0, le=1.0)
    mean_recall_at_1: float = Field(ge=0.0, le=1.0)
    mean_recall_at_5: float = Field(ge=0.0, le=1.0)
    mean_positive_predictive_value: float = Field(ge=0.0, le=1.0)
    mean_brier_score: float = Field(ge=0.0, le=1.0)
    mean_runtime_ms: float = Field(ge=0.0)
    mean_peak_memory_mb: float = Field(ge=0.0)
    failure_rate: float = Field(ge=0.0, le=1.0)
    operational_validity: Literal["not_established"] = "not_established"


class SimilarityAdvisoryReport(RecommendationNode):
    """Stage-2 similarity-based advisory report with benchmark evidence."""

    report_schema_version: Literal["1"] = "1"
    report_id: Identifier
    recommendation: PipelineRecommendation
    target_task_profile_digest: Digest
    nearest_family_ids: Annotated[tuple[Identifier, ...], Field(max_length=16)] = ()
    nearest_family_distances: dict[str, float] = Field(default_factory=dict)
    synthetic_evidence_retrieved: StrictBool
    out_of_distribution: StrictBool
    out_of_distribution_score: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    empirical_metric_distributions: dict[str, EmpiricalMetricDistribution] = Field(
        default_factory=dict
    )
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    operational_validity: Literal["not_established"] = "not_established"

    @property
    def report_digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, object]:
        return {
            "report_schema_version": self.report_schema_version,
            "report_id": self.report_id,
            "report_digest": self.report_digest,
            "recommendation_id": self.recommendation.recommendation_id,
            "nearest_family_ids": list(self.nearest_family_ids),
            "nearest_family_distances": self.nearest_family_distances,
            "synthetic_evidence_retrieved": self.synthetic_evidence_retrieved,
            "out_of_distribution": self.out_of_distribution,
            "out_of_distribution_score": self.out_of_distribution_score,
            "empirical_metric_distributions": {
                k: v.model_dump(mode="json") for k, v in self.empirical_metric_distributions.items()
            },
            "recommendation": self.recommendation.safe_summary(),
            "recommendation_authority": self.recommendation_authority,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "operational_validity": self.operational_validity,
        }


class PredictedCandidateUtility(RecommendationNode):
    """Calibrated meta-model predicted utility with conformal interval bounds."""

    candidate_id: Identifier
    predicted_utility: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    uncertainty_lower_bound: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    uncertainty_upper_bound: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    conformal_coverage_level: Annotated[StrictFloat, Field(ge=0.5, le=0.99)] = 0.90


class MetaRankingAdvisoryReport(RecommendationNode):
    """Stage-3 Learned Meta-Ranking Strategy Advisory Report."""

    report_schema_version: Literal["1"] = "1"
    report_id: Identifier
    recommendation: PipelineRecommendation
    predicted_candidate_utilities: dict[str, PredictedCandidateUtility] = Field(
        default_factory=dict
    )
    meta_model_type: str = "ridge_meta_ranker_v1"
    meta_model_trained_runs: int = 0
    fallback_to_similarity: StrictBool = False
    fallback_reason: str | None = None
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    operational_validity: Literal["not_established"] = "not_established"

    @property
    def report_digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, object]:
        return {
            "report_schema_version": self.report_schema_version,
            "report_id": self.report_id,
            "report_digest": self.report_digest,
            "recommendation_id": self.recommendation.recommendation_id,
            "meta_model_type": self.meta_model_type,
            "meta_model_trained_runs": self.meta_model_trained_runs,
            "fallback_to_similarity": self.fallback_to_similarity,
            "fallback_reason": self.fallback_reason,
            "predicted_candidate_utilities": {
                k: v.model_dump(mode="json") for k, v in self.predicted_candidate_utilities.items()
            },
            "recommendation": self.recommendation.safe_summary(),
            "recommendation_authority": self.recommendation_authority,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "operational_validity": self.operational_validity,
        }


__all__ = [
    "AbstentionReason",
    "CandidateExplanation",
    "CandidateRetrievalStatus",
    "CoverageStatus",
    "DisqualifiedCandidate",
    "EmpiricalMetricDistribution",
    "EvidenceContribution",
    "EvidenceScope",
    "MetaRankingAdvisoryReport",
    "PipelineRecommendation",
    "PredictedCandidateUtility",
    "RankingStrategy",
    "RecommendationIntent",
    "RuntimeDependency",
    "SimilarityAdvisoryReport",
    "StructuralPipelineCandidate",
]
