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
    SplinkCandidateParityChecker,
    SplinkCandidateParityReport,
    SplinkSettingsPlan,
    SplinkSettingsPlanCompiler,
)
from mapel_linkage.models.fellegi_sunter.splink_native import (
    SUPPORTED_SPLINK_VERSION,
    SplinkNativeDuckDBMatcher,
    SplinkNativeModelArtifact,
    SplinkNativeScoreResult,
    assert_splink_native_recipe_binding,
    deserialize_splink_native_model,
    serialize_splink_native_model,
    splink_native_feature_schema_digest,
)

__all__ = [
    "SUPPORTED_SPLINK_VERSION",
    "DuckDBFellegiSunterMatcher",
    "DuckDBRandomPairSampler",
    "FellegiSunterComparisonParameters",
    "FellegiSunterLevelParameters",
    "FellegiSunterModelArtifact",
    "FellegiSunterScoreResult",
    "RandomPairSampleResult",
    "SplinkCandidateParityChecker",
    "SplinkCandidateParityReport",
    "SplinkNativeDuckDBMatcher",
    "SplinkNativeModelArtifact",
    "SplinkNativeScoreResult",
    "SplinkSettingsPlan",
    "SplinkSettingsPlanCompiler",
    "assert_splink_native_recipe_binding",
    "deserialize_splink_native_model",
    "serialize_splink_native_model",
    "splink_native_feature_schema_digest",
]
