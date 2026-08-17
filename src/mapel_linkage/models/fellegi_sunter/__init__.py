"""Fellegi-Sunter evidence models and Splink adapter contracts."""

from mapel_linkage.models.fellegi_sunter.reference import (
    DuckDBFellegiSunterMatcher,
    DuckDBRandomPairSampler,
    FellegiSunterComparisonParameters,
    FellegiSunterLevelParameters,
    FellegiSunterModelArtifact,
    FellegiSunterScoreResult,
    RandomPairSampleResult,
)
from mapel_linkage.models.fellegi_sunter.splink_adapter import (
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
