"""Core domain contracts for Linkage Engine."""

from mapel_linkage.domain.errors import (
    AnchorBudgetExceeded,
    AnchorEvidenceError,
    BoostedTreeBudgetExceeded,
    BoostedTreeError,
    CandidateBudgetExceeded,
    CandidateGenerationError,
    ComparisonFeatureError,
    DataPlaneError,
    FellegiSunterBudgetExceeded,
    FellegiSunterError,
    LabelProvenanceError,
    LinkageRuntimeError,
    PreprocessingError,
)
from mapel_linkage.domain.table_refs import TableRef

__all__ = [
    "AnchorBudgetExceeded",
    "AnchorEvidenceError",
    "BoostedTreeBudgetExceeded",
    "BoostedTreeError",
    "CandidateBudgetExceeded",
    "CandidateGenerationError",
    "ComparisonFeatureError",
    "DataPlaneError",
    "FellegiSunterBudgetExceeded",
    "FellegiSunterError",
    "LabelProvenanceError",
    "LinkageRuntimeError",
    "PreprocessingError",
    "TableRef",
]
