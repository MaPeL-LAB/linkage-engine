"""Typed, bounded candidate retrieval for Linkage Engine."""

from mapel_linkage.candidate_generation.duckdb_generator import (
    CandidateDiagnostics,
    CandidateGenerationResult,
    DuckDBCandidateGenerator,
)
from mapel_linkage.candidate_generation.predicates import (
    AllOf,
    AnyOf,
    BlockingRule,
    CandidatePredicate,
    DateWindow,
    Exact,
    PrefixEqual,
)

__all__ = [
    "AllOf",
    "AnyOf",
    "BlockingRule",
    "CandidateDiagnostics",
    "CandidateGenerationResult",
    "CandidatePredicate",
    "DateWindow",
    "DuckDBCandidateGenerator",
    "Exact",
    "PrefixEqual",
]
