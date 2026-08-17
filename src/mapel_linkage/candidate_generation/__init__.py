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
    Exact,
    PrefixEqual,
)

__all__ = [
    "AllOf",
    "AnyOf",
    "BlockingRule",
    "CandidateDiagnostics",
    "CandidateGenerationResult",
    "DuckDBCandidateGenerator",
    "Exact",
    "PrefixEqual",
]
