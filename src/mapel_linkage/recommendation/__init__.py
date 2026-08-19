"""Stage-1 advisory pipeline recommendation without identity authority."""

from mapel_linkage.recommendation.advisor import (
    build_structural_pipeline_candidates,
    recommend_pipeline,
)
from mapel_linkage.recommendation.contracts import (
    AbstentionReason,
    CandidateExplanation,
    CandidateRetrievalStatus,
    CoverageStatus,
    DisqualifiedCandidate,
    EvidenceContribution,
    EvidenceScope,
    PipelineRecommendation,
    RankingStrategy,
    RecommendationIntent,
    RuntimeDependency,
    StructuralPipelineCandidate,
)
from mapel_linkage.recommendation.eligibility import (
    AdvisorContext,
    EligibilityDecision,
    EligibilityReason,
    evaluate_candidate,
)
from mapel_linkage.recommendation.structural_pareto import (
    build_diverse_shortlist,
    structural_pareto_frontier,
)

__all__ = [
    "AbstentionReason",
    "AdvisorContext",
    "CandidateExplanation",
    "CandidateRetrievalStatus",
    "CoverageStatus",
    "DisqualifiedCandidate",
    "EligibilityDecision",
    "EligibilityReason",
    "EvidenceContribution",
    "EvidenceScope",
    "PipelineRecommendation",
    "RankingStrategy",
    "RecommendationIntent",
    "RuntimeDependency",
    "StructuralPipelineCandidate",
    "build_diverse_shortlist",
    "build_structural_pipeline_candidates",
    "evaluate_candidate",
    "recommend_pipeline",
    "structural_pareto_frontier",
]
