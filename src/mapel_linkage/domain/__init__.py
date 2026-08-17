"""Core domain contracts for Linkage Engine."""

from mapel_linkage.domain.errors import (
    CandidateBudgetExceeded,
    CandidateGenerationError,
    DataPlaneError,
    LinkageRuntimeError,
)
from mapel_linkage.domain.table_refs import TableRef

__all__ = [
    "CandidateBudgetExceeded",
    "CandidateGenerationError",
    "DataPlaneError",
    "LinkageRuntimeError",
    "TableRef",
]
