"""Stage-1 and Stage-2 advisory pipeline recommendations without identity authority."""

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
    EmpiricalMetricDistribution,
    EvidenceContribution,
    EvidenceScope,
    MetaRankingAdvisoryReport,
    PipelineRecommendation,
    PredictedCandidateUtility,
    RankingStrategy,
    RecommendationIntent,
    RuntimeDependency,
    SimilarityAdvisoryReport,
    StructuralPipelineCandidate,
)
from mapel_linkage.recommendation.distance import (
    DEFAULT_META_FEATURE_WEIGHTS,
    MetaFeatureDistanceComputer,
    TaskMetaFeatureVector,
    extract_family_meta_features,
)
from mapel_linkage.recommendation.eligibility import (
    AdvisorContext,
    EligibilityDecision,
    EligibilityReason,
    evaluate_candidate,
)
from mapel_linkage.recommendation.meta_ranker import (
    LearnedMetaRankerModel,
    MetaRankingLinkageAdvisor,
)
from mapel_linkage.recommendation.similarity_advisor import (
    SimilarityLinkageAdvisor,
    recommend_with_similarity,
)
from mapel_linkage.recommendation.structural_pareto import (
    build_diverse_shortlist,
    structural_pareto_frontier,
)

__all__ = [
    "DEFAULT_META_FEATURE_WEIGHTS",
    "AbstentionReason",
    "AdvisorContext",
    "CandidateExplanation",
    "CandidateRetrievalStatus",
    "CoverageStatus",
    "DisqualifiedCandidate",
    "EligibilityDecision",
    "EligibilityReason",
    "EmpiricalMetricDistribution",
    "EvidenceContribution",
    "EvidenceScope",
    "LearnedMetaRankerModel",
    "MetaFeatureDistanceComputer",
    "MetaRankingAdvisoryReport",
    "MetaRankingLinkageAdvisor",
    "PipelineRecommendation",
    "PredictedCandidateUtility",
    "RankingStrategy",
    "RecommendationIntent",
    "RuntimeDependency",
    "SimilarityAdvisoryReport",
    "SimilarityLinkageAdvisor",
    "StructuralPipelineCandidate",
    "TaskMetaFeatureVector",
    "build_diverse_shortlist",
    "build_structural_pipeline_candidates",
    "evaluate_candidate",
    "extract_family_meta_features",
    "recommend_pipeline",
    "recommend_with_similarity",
    "structural_pareto_frontier",
]
