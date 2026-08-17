"""Configuration-driven comparison-feature construction in local DuckDB tables."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from mapel_linkage.configuration.models import (
    CategoricalComparison,
    ComparisonConfig,
    DamerauLevenshteinComparison,
    DateDifferenceComparison,
    ElseLevel,
    ExactComparison,
    ExactLevel,
    JaroWinklerComparison,
    LevenshteinComparison,
    MaximumDifferenceLevel,
    MissingLevel,
    NumericDifferenceComparison,
    QGramComparison,
    ThresholdLevel,
)
from mapel_linkage.domain.errors import ComparisonFeatureError, DataPlaneError
from mapel_linkage.domain.sql_identifiers import quote_identifier
from mapel_linkage.domain.table_refs import TableRef
from mapel_linkage.io.duckdb_store import DuckDBStore
from mapel_linkage.preprocessing.dataset_preparer import PreparedDataset

_PAIR_COLUMNS: tuple[str, ...] = (
    "left_record_key",
    "right_record_key",
    "retrieval_rule_ids",
    "retrieval_rule_count",
)


def _stable_suffix(identifier: str) -> str:
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16]


def _qualified(alias: str, column: str) -> str:
    return f"{alias}.{quote_identifier(column)}"


def _cast_text(expression: str) -> str:
    return f"CAST({expression} AS VARCHAR)"


@dataclass(frozen=True, slots=True)
class ComparisonFeatureColumns:
    """Internal feature-column names for one configured comparison."""

    value: str
    level: str
    exact: str
    missing_left: str
    missing_right: str
    missing_both: str
    missing_any: str


@dataclass(frozen=True, slots=True)
class ComparisonFeatureResult:
    """Structural feature-table output that contains no pair or field values."""

    table: TableRef
    candidate_pair_count: int
    configured_comparison_count: int
    columns: Mapping[str, ComparisonFeatureColumns] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", MappingProxyType(dict(self.columns)))

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "candidate_pair_count": self.candidate_pair_count,
            "configured_comparison_count": self.configured_comparison_count,
            "feature_column_count": len(self.columns) * 7,
            "schema_digest": self.table.schema_digest,
        }


class DuckDBComparisonFeatureBuilder:
    """Build comparison metrics and configured levels over bounded candidates."""

    __slots__ = ("_store",)

    def __init__(self, store: DuckDBStore) -> None:
        self._store = store

    def build(
        self,
        *,
        candidates: TableRef,
        left: PreparedDataset,
        right: PreparedDataset,
        comparisons: Sequence[ComparisonConfig],
        record_key_column: str = "__ml_record_key",
    ) -> ComparisonFeatureResult:
        """Materialise one row of comparison features per candidate pair."""

        if not comparisons:
            raise ComparisonFeatureError(
                "ML-COMP-001", "At least one configured comparison is required."
            )
        comparison_ids = [comparison.id for comparison in comparisons]
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ComparisonFeatureError(
                "ML-COMP-002", "Duplicate comparison identifiers were rejected."
            )

        column_sets = {comparison.id: self._column_set(comparison.id) for comparison in comparisons}
        left_value_aliases: dict[str, str] = {}
        right_value_aliases: dict[str, str] = {}
        joined_values: list[str] = []
        for comparison in comparisons:
            left_column, right_column = self._variable_columns(left, right, comparison.variable)
            suffix = _stable_suffix(comparison.id)
            left_alias = f"__ml_l_{suffix}"
            right_alias = f"__ml_r_{suffix}"
            left_value_aliases[comparison.id] = left_alias
            right_value_aliases[comparison.id] = right_alias
            joined_values.extend(
                (
                    f"{_qualified('l', left_column)} AS {quote_identifier(left_alias)}",
                    f"{_qualified('r', right_column)} AS {quote_identifier(right_alias)}",
                )
            )

        candidate_table = quote_identifier(candidates.table_name)
        left_table = quote_identifier(left.table.table_name)
        right_table = quote_identifier(right.table.table_name)
        key = quote_identifier(record_key_column)
        pair_projection = ", ".join(
            f"c.{quote_identifier(column)} AS {quote_identifier(column)}"
            for column in _PAIR_COLUMNS
        )
        joined_select = (
            f"SELECT {pair_projection}, {', '.join(joined_values)} "
            f"FROM {candidate_table} AS c "
            f"INNER JOIN {left_table} AS l "
            f"ON c.{quote_identifier('left_record_key')} = l.{key} "
            f"INNER JOIN {right_table} AS r "
            f"ON c.{quote_identifier('right_record_key')} = r.{key}"
        )
        joined_count = self._safe_scalar(
            f"SELECT COUNT(*) FROM ({joined_select}) AS joined_pairs",
            code="ML-COMP-006",
            message="Candidate keys did not resolve to the prepared dataset tables.",
        )
        if joined_count != candidates.row_count:
            raise ComparisonFeatureError(
                "ML-COMP-006",
                "Candidate keys did not resolve to the prepared dataset tables.",
            )

        metric_projections = [quote_identifier(column) for column in _PAIR_COLUMNS]
        final_projections = [f"m.{quote_identifier(column)}" for column in _PAIR_COLUMNS]
        for comparison in comparisons:
            left_value = quote_identifier(left_value_aliases[comparison.id])
            right_value = quote_identifier(right_value_aliases[comparison.id])
            columns = column_sets[comparison.id]
            any_missing = f"({left_value} IS NULL OR {right_value} IS NULL)"
            both_missing = f"({left_value} IS NULL AND {right_value} IS NULL)"
            exact = f"CASE WHEN {any_missing} THEN NULL ELSE {left_value} = {right_value} END"
            value = self._metric_expression(comparison, left_value, right_value)
            metric_projections.extend(
                (
                    f"{value} AS {quote_identifier(columns.value)}",
                    f"{exact} AS {quote_identifier(columns.exact)}",
                    f"({left_value} IS NULL) AS {quote_identifier(columns.missing_left)}",
                    f"({right_value} IS NULL) AS {quote_identifier(columns.missing_right)}",
                    f"{both_missing} AS {quote_identifier(columns.missing_both)}",
                    f"{any_missing} AS {quote_identifier(columns.missing_any)}",
                )
            )
            final_projections.extend(
                (
                    f"m.{quote_identifier(columns.value)}",
                    self._level_expression(comparison, columns),
                    f"m.{quote_identifier(columns.exact)}",
                    f"m.{quote_identifier(columns.missing_left)}",
                    f"m.{quote_identifier(columns.missing_right)}",
                    f"m.{quote_identifier(columns.missing_both)}",
                    f"m.{quote_identifier(columns.missing_any)}",
                )
            )

        digest_input = "|".join(
            (
                candidates.schema_digest,
                left.table.schema_digest,
                right.table.schema_digest,
                *comparison_ids,
            )
        )
        suffix = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:12]
        table_name = f"__ml_features_{suffix}"
        sql = (
            f"WITH joined AS ({joined_select}), "
            f"metrics AS (SELECT {', '.join(metric_projections)} FROM joined) "
            f"SELECT {', '.join(final_projections)} FROM metrics AS m "
            f"ORDER BY m.{quote_identifier('left_record_key')}, "
            f"m.{quote_identifier('right_record_key')}"
        )
        try:
            table = self._store._create_temp_table_as(table_name, sql)
        except DataPlaneError as error:
            raise ComparisonFeatureError(
                "ML-COMP-007", "Comparison features could not be materialised."
            ) from error
        return ComparisonFeatureResult(
            table=table,
            candidate_pair_count=table.row_count,
            configured_comparison_count=len(comparisons),
            columns=column_sets,
        )

    @staticmethod
    def _column_set(comparison_id: str) -> ComparisonFeatureColumns:
        suffix = _stable_suffix(comparison_id)
        return ComparisonFeatureColumns(
            value=f"__ml_cmp_{suffix}_value",
            level=f"__ml_cmp_{suffix}_level",
            exact=f"__ml_cmp_{suffix}_exact",
            missing_left=f"__ml_cmp_{suffix}_missing_left",
            missing_right=f"__ml_cmp_{suffix}_missing_right",
            missing_both=f"__ml_cmp_{suffix}_missing_both",
            missing_any=f"__ml_cmp_{suffix}_missing_any",
        )

    @staticmethod
    def _variable_columns(
        left: PreparedDataset,
        right: PreparedDataset,
        variable_id: str,
    ) -> tuple[str, str]:
        try:
            left_column = left.variable_columns[variable_id]
            right_column = right.variable_columns[variable_id]
            quote_identifier(left_column)
            quote_identifier(right_column)
        except (KeyError, DataPlaneError):
            raise ComparisonFeatureError(
                "ML-COMP-003",
                "A configured comparison references an unavailable canonical variable.",
            ) from None
        return (left_column, right_column)

    def _safe_scalar(self, sql: str, *, code: str, message: str) -> int:
        try:
            return self._store._scalar_int(sql)
        except DataPlaneError:
            raise ComparisonFeatureError(code, message) from None

    @classmethod
    def _metric_expression(
        cls,
        comparison: ComparisonConfig,
        left_value: str,
        right_value: str,
    ) -> str:
        any_missing = f"({left_value} IS NULL OR {right_value} IS NULL)"
        function = comparison.function
        if isinstance(function, (ExactComparison, CategoricalComparison)):
            nonmissing = f"CASE WHEN {left_value} = {right_value} THEN 1.0 ELSE 0.0 END"
        elif isinstance(function, JaroWinklerComparison):
            nonmissing = (
                f"jaro_winkler_similarity({_cast_text(left_value)}, {_cast_text(right_value)})"
            )
        elif isinstance(function, LevenshteinComparison):
            nonmissing = cls._normalised_edit_similarity(
                "levenshtein",
                left_value,
                right_value,
                function.maximum_distance,
            )
        elif isinstance(function, DamerauLevenshteinComparison):
            nonmissing = cls._normalised_edit_similarity(
                "damerau_levenshtein",
                left_value,
                right_value,
                function.maximum_distance,
            )
        elif isinstance(function, QGramComparison):
            nonmissing = cls._qgram_similarity(left_value, right_value, function.q)
        elif isinstance(function, DateDifferenceComparison):
            nonmissing = (
                f"CAST(ABS(date_diff('{function.unit}', {left_value}, {right_value})) AS DOUBLE)"
            )
        elif isinstance(function, NumericDifferenceComparison):
            nonmissing = (
                f"ABS(CAST({left_value} AS DOUBLE) - CAST({right_value} AS DOUBLE)) "
                f"/ {function.scale!r}"
            )
        else:  # pragma: no cover - the discriminated configuration union prevents this.
            raise ComparisonFeatureError(
                "ML-COMP-004", "An unsupported comparison function was rejected."
            )
        return f"CASE WHEN {any_missing} THEN NULL ELSE {nonmissing} END"

    @staticmethod
    def _normalised_edit_similarity(
        function_name: str,
        left_value: str,
        right_value: str,
        maximum_distance: int | None,
    ) -> str:
        left_text = _cast_text(left_value)
        right_text = _cast_text(right_value)
        distance = f"{function_name}({left_text}, {right_text})"
        maximum_length = f"GREATEST(LENGTH({left_text}), LENGTH({right_text}))"
        normalised = (
            f"CASE WHEN {maximum_length} = 0 THEN 1.0 ELSE "
            f"GREATEST(0.0, 1.0 - CAST({distance} AS DOUBLE) "
            f"/ CAST({maximum_length} AS DOUBLE)) END"
        )
        if maximum_distance is None:
            return normalised
        return f"CASE WHEN {distance} > {maximum_distance} THEN 0.0 ELSE {normalised} END"

    @staticmethod
    def _qgram_similarity(left_value: str, right_value: str, q: int) -> str:
        def grams(value: str) -> str:
            text = _cast_text(value)
            indices = f"range(1, GREATEST(LENGTH({text}) - {q} + 2, 2))"
            return f"list_distinct(list_transform({indices}, i -> SUBSTR({text}, i, {q})))"

        left_grams = grams(left_value)
        right_grams = grams(right_value)
        intersection = f"list_count(list_intersect({left_grams}, {right_grams}))"
        denominator = f"list_count({left_grams}) + list_count({right_grams})"
        return f"COALESCE(2.0 * {intersection} / NULLIF({denominator}, 0), 1.0)"

    @staticmethod
    def _level_expression(
        comparison: ComparisonConfig,
        columns: ComparisonFeatureColumns,
    ) -> str:
        value = f"m.{quote_identifier(columns.value)}"
        exact = f"m.{quote_identifier(columns.exact)}"
        missing_any = f"m.{quote_identifier(columns.missing_any)}"
        has_missing_level = any(isinstance(level, MissingLevel) for level in comparison.levels)
        clauses: list[str] = []
        if not has_missing_level:
            clauses.append(f"WHEN {missing_any} THEN NULL")
        else_index: int | None = None
        for index, level in enumerate(comparison.levels):
            if isinstance(level, MissingLevel):
                clauses.append(f"WHEN {missing_any} THEN {index}")
            elif isinstance(level, ExactLevel):
                clauses.append(f"WHEN {exact} IS TRUE THEN {index}")
            elif isinstance(level, ThresholdLevel):
                clauses.append(f"WHEN {value} >= {level.minimum!r} THEN {index}")
            elif isinstance(level, MaximumDifferenceLevel):
                clauses.append(f"WHEN {value} <= {float(level.value)!r} THEN {index}")
            elif isinstance(level, ElseLevel):
                else_index = index
        if else_index is None:  # pragma: no cover - Pydantic validation requires an else level.
            raise ComparisonFeatureError(
                "ML-COMP-005", "A configured comparison has no fallback level."
            )
        return (
            "CAST(CASE "
            + " ".join(clauses)
            + f" ELSE {else_index} END AS INTEGER) AS {quote_identifier(columns.level)}"
        )
