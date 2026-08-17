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

__all__ = [
    "DuckDBFellegiSunterMatcher",
    "DuckDBRandomPairSampler",
    "FellegiSunterComparisonParameters",
    "FellegiSunterLevelParameters",
    "FellegiSunterModelArtifact",
    "FellegiSunterScoreResult",
    "RandomPairSampleResult",
    "SplinkSettingsPlan",
    "SplinkSettingsPlanCompiler",
]
