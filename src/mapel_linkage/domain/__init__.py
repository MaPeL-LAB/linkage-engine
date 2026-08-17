"""Core domain contracts for Linkage Engine."""

from mapel_linkage.domain.errors import (
    AnchorBudgetExceeded,
    AnchorEvidenceError,
    CandidateBudgetExceeded,
    CandidateGenerationError,
    ComparisonFeatureError,
    DataPlaneError,
    FellegiSunterBudgetExceeded,
    FellegiSunterError,
    LinkageRuntimeError,
    PreprocessingError,
)
from mapel_linkage.domain.table_refs import TableRef

__all__ = [
    "AnchorBudgetExceeded",
    "AnchorEvidenceError",
    "CandidateBudgetExceeded",
    "CandidateGenerationError",
    "ComparisonFeatureError",
    "DataPlaneError",
    "FellegiSunterBudgetExceeded",
    "FellegiSunterError",
    "LinkageRuntimeError",
    "PreprocessingError",
    "TableRef",
]
