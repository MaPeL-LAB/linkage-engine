"""Package-owned model adapters and statistical evidence contracts."""

from mapel_linkage.models.fellegi_sunter import (
    DuckDBFellegiSunterMatcher,
    DuckDBRandomPairSampler,
    FellegiSunterComparisonParameters,
    FellegiSunterLevelParameters,
    FellegiSunterModelArtifact,
    FellegiSunterScoreResult,
    RandomPairSampleResult,
    SplinkSettingsPlan,
    SplinkSettingsPlanCompiler,
)
from mapel_linkage.models.ranking import (
    RankingMatrix,
    RankingScoreBatch,
    XGBoostCandidateRanker,
    XGBoostRankingArtifact,
    build_ranking_matrix,
)

__all__ = [
    "DuckDBFellegiSunterMatcher",
    "DuckDBRandomPairSampler",
    "FellegiSunterComparisonParameters",
    "FellegiSunterLevelParameters",
    "FellegiSunterModelArtifact",
    "FellegiSunterScoreResult",
    "RandomPairSampleResult",
    "RankingMatrix",
    "RankingScoreBatch",
    "SplinkSettingsPlan",
    "SplinkSettingsPlanCompiler",
    "XGBoostCandidateRanker",
    "XGBoostRankingArtifact",
    "build_ranking_matrix",
]
