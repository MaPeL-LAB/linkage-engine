"""Typed candidate-retrieval predicates with no raw-SQL escape hatch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from mapel_linkage.domain.errors import CandidateGenerationError, DataPlaneError
from mapel_linkage.domain.sql_identifiers import quote_identifier, validate_identifier


@dataclass(frozen=True, slots=True)
class Exact:
    variable_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.variable_id)


@dataclass(frozen=True, slots=True)
class PrefixEqual:
    variable_id: str
    length: int

    def __post_init__(self) -> None:
        validate_identifier(self.variable_id)
        if not 1 <= self.length <= 64:
            raise CandidateGenerationError(
                "ML-CANDIDATE-002", "A prefix length must be between 1 and 64."
            )


@dataclass(frozen=True, slots=True)
class AllOf:
    clauses: tuple[CandidatePredicate, ...]

    def __post_init__(self) -> None:
        if not self.clauses:
            raise CandidateGenerationError(
                "ML-CANDIDATE-003", "A conjunction requires at least one predicate."
            )


@dataclass(frozen=True, slots=True)
class AnyOf:
    clauses: tuple[CandidatePredicate, ...]

    def __post_init__(self) -> None:
        if not self.clauses:
            raise CandidateGenerationError(
                "ML-CANDIDATE-004", "A disjunction requires at least one predicate."
            )


type CandidatePredicate = Exact | PrefixEqual | AllOf | AnyOf


@dataclass(frozen=True, slots=True)
class BlockingRule:
    rule_id: str
    predicate: CandidatePredicate

    def __post_init__(self) -> None:
        validate_identifier(self.rule_id)


def compile_predicate(
    predicate: CandidatePredicate,
    variable_columns: Mapping[str, str],
    *,
    left_alias: str = "l",
    right_alias: str = "r",
) -> str:
    """Compile a package-owned predicate tree into quoted DuckDB SQL."""

    validate_identifier(left_alias)
    validate_identifier(right_alias)

    def column_pair(variable_id: str) -> tuple[str, str]:
        try:
            column = variable_columns[variable_id]
        except KeyError:
            raise CandidateGenerationError(
                "ML-CANDIDATE-005", "A blocking rule references an unavailable variable."
            ) from None
        try:
            quoted = quote_identifier(column)
        except DataPlaneError:
            raise CandidateGenerationError(
                "ML-CANDIDATE-009",
                "An unsafe canonical column mapping was rejected.",
            ) from None
        return (f"{left_alias}.{quoted}", f"{right_alias}.{quoted}")

    if isinstance(predicate, Exact):
        left, right = column_pair(predicate.variable_id)
        return f"({left} IS NOT NULL AND {right} IS NOT NULL AND {left} = {right})"
    if isinstance(predicate, PrefixEqual):
        left, right = column_pair(predicate.variable_id)
        length = predicate.length
        return (
            f"({left} IS NOT NULL AND {right} IS NOT NULL AND "
            f"SUBSTR({left}, 1, {length}) = SUBSTR({right}, 1, {length}))"
        )
    if isinstance(predicate, AllOf):
        return "(" + " AND ".join(
            compile_predicate(
                clause,
                variable_columns,
                left_alias=left_alias,
                right_alias=right_alias,
            )
            for clause in predicate.clauses
        ) + ")"
    if isinstance(predicate, AnyOf):
        return "(" + " OR ".join(
            compile_predicate(
                clause,
                variable_columns,
                left_alias=left_alias,
                right_alias=right_alias,
            )
            for clause in predicate.clauses
        ) + ")"
    raise CandidateGenerationError(
        "ML-CANDIDATE-006", "An unsupported blocking predicate was rejected."
    )
