"""Deterministic-anchor evidence evaluation without identity decisions."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from mapel_linkage.configuration.models import (
    AllPredicate,
    AnyPredicate,
    BlockPredicate,
    DateWindowPredicate,
    DeterministicAnchorConfig,
    ExactPredicate,
    PrefixEqualPredicate,
)
from mapel_linkage.domain.errors import (
    AnchorBudgetExceeded,
    AnchorEvidenceError,
    DataPlaneError,
)
from mapel_linkage.domain.sql_identifiers import quote_identifier
from mapel_linkage.domain.table_refs import TableRef
from mapel_linkage.io.duckdb_store import DuckDBStore
from mapel_linkage.preprocessing.dataset_preparer import PreparedDataset


def _sql_text_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _boolean(value: bool) -> str:
    return "TRUE" if value else "FALSE"


@dataclass(frozen=True, slots=True)
class AnchorEvidenceResult:
    """Structural anchor evidence; it contains no candidate-pair values."""

    table: TableRef
    configured_anchor_count: int
    evidence_row_count: int
    distinct_pair_count: int
    uniqueness_pass_count: int

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "configured_anchor_count": self.configured_anchor_count,
            "evidence_row_count": self.evidence_row_count,
            "distinct_pair_count": self.distinct_pair_count,
            "uniqueness_pass_count": self.uniqueness_pass_count,
            "schema_digest": self.table.schema_digest,
        }


class DuckDBAnchorEvidenceEvaluator:
    """Evaluate deterministic anchor predicates as evidence-only relationships."""

    __slots__ = ("_store",)

    def __init__(self, store: DuckDBStore) -> None:
        self._store = store

    def evaluate(
        self,
        *,
        left: PreparedDataset,
        right: PreparedDataset,
        anchors: Sequence[DeterministicAnchorConfig],
        maximum_anchor_pairs: int,
        record_key_column: str = "__ml_record_key",
    ) -> AnchorEvidenceResult:
        """Materialise deterministic anchor evidence with explicit uniqueness checks."""

        if maximum_anchor_pairs < 1:
            raise AnchorEvidenceError("ML-ANCHOR-001", "The anchor-pair budget must be positive.")
        anchor_ids = [anchor.id for anchor in anchors]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise AnchorEvidenceError(
                "ML-ANCHOR-002", "Duplicate anchor identifiers were rejected."
            )

        left_table = quote_identifier(left.table.table_name)
        right_table = quote_identifier(right.table.table_name)
        key = quote_identifier(record_key_column)
        selects: list[str] = []
        for anchor in anchors:
            condition = self._compile_predicate(anchor.predicate, left, right)
            selects.append(
                "SELECT "
                f"l.{key} AS left_record_key, "
                f"r.{key} AS right_record_key, "
                f"{_sql_text_literal(anchor.id)} AS anchor_rule_id, "
                f"{_boolean(anchor.require_unique_left)} AS require_unique_left, "
                f"{_boolean(anchor.require_unique_right)} AS require_unique_right "
                f"FROM {left_table} AS l INNER JOIN {right_table} AS r ON {condition}"
            )

        if selects:
            raw_sql = " UNION ALL ".join(selects)
        else:
            raw_sql = (
                "SELECT CAST(NULL AS VARCHAR) AS left_record_key, "
                "CAST(NULL AS VARCHAR) AS right_record_key, "
                "CAST(NULL AS VARCHAR) AS anchor_rule_id, "
                "CAST(NULL AS BOOLEAN) AS require_unique_left, "
                "CAST(NULL AS BOOLEAN) AS require_unique_right WHERE FALSE"
            )

        pair_count_sql = (
            "SELECT COUNT(*) FROM ("
            "SELECT DISTINCT left_record_key, right_record_key "
            f"FROM ({raw_sql}) AS raw_evidence"
            ") AS distinct_pairs"
        )
        distinct_pair_count = self._safe_scalar(
            pair_count_sql,
            code="ML-ANCHOR-005",
            message="Anchor evidence could not be counted safely.",
        )
        if distinct_pair_count > maximum_anchor_pairs:
            raise AnchorBudgetExceeded(
                "ML-ANCHOR-006",
                "Anchor evidence exceeded the configured aggregate pair budget.",
            )

        digest_input = "|".join(
            (
                left.table.schema_digest,
                right.table.schema_digest,
                *anchor_ids,
            )
        )
        suffix = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:12]
        table_name = f"__ml_anchor_evidence_{suffix}"
        materialisation_sql = (
            f"WITH raw AS ({raw_sql}), "
            "counts AS ("
            "SELECT raw.*, "
            "COUNT(*) OVER (PARTITION BY anchor_rule_id, left_record_key) "
            "AS left_match_count, "
            "COUNT(*) OVER (PARTITION BY anchor_rule_id, right_record_key) "
            "AS right_match_count FROM raw"
            ") "
            "SELECT left_record_key, right_record_key, anchor_rule_id, "
            "CAST(left_match_count AS INTEGER) AS left_match_count, "
            "CAST(right_match_count AS INTEGER) AS right_match_count, "
            "left_match_count = 1 AS left_unique, "
            "right_match_count = 1 AS right_unique, "
            "((NOT require_unique_left OR left_match_count = 1) AND "
            "(NOT require_unique_right OR right_match_count = 1)) AS uniqueness_pass, "
            "'evidence_only' AS evidence_action, "
            "FALSE AS eligible_as_training_truth "
            "FROM counts ORDER BY anchor_rule_id, left_record_key, right_record_key"
        )
        try:
            table = self._store._create_temp_table_as(table_name, materialisation_sql)
        except DataPlaneError:
            raise AnchorEvidenceError(
                "ML-ANCHOR-007", "Anchor evidence could not be materialised."
            ) from None
        uniqueness_pass_count = self._safe_scalar(
            f"SELECT COUNT(*) FROM {quote_identifier(table_name)} WHERE uniqueness_pass IS TRUE",
            code="ML-ANCHOR-008",
            message="Anchor diagnostics could not be calculated safely.",
        )
        return AnchorEvidenceResult(
            table=table,
            configured_anchor_count=len(anchors),
            evidence_row_count=table.row_count,
            distinct_pair_count=distinct_pair_count,
            uniqueness_pass_count=uniqueness_pass_count,
        )

    def _compile_predicate(
        self,
        predicate: BlockPredicate,
        left: PreparedDataset,
        right: PreparedDataset,
        *,
        left_alias: str = "l",
        right_alias: str = "r",
    ) -> str:
        if isinstance(predicate, ExactPredicate):
            left_value, right_value = self._column_pair(
                predicate.variable, left, right, left_alias, right_alias
            )
            return (
                f"({left_value} IS NOT NULL AND {right_value} IS NOT NULL AND "
                f"{left_value} = {right_value})"
            )
        if isinstance(predicate, PrefixEqualPredicate):
            left_value, right_value = self._column_pair(
                predicate.variable, left, right, left_alias, right_alias
            )
            return (
                f"({left_value} IS NOT NULL AND {right_value} IS NOT NULL AND "
                f"SUBSTR(CAST({left_value} AS VARCHAR), 1, {predicate.length}) = "
                f"SUBSTR(CAST({right_value} AS VARCHAR), 1, {predicate.length}))"
            )
        if isinstance(predicate, DateWindowPredicate):
            left_value, right_value = self._column_pair(
                predicate.variable, left, right, left_alias, right_alias
            )
            return (
                f"({left_value} IS NOT NULL AND {right_value} IS NOT NULL AND "
                f"ABS(date_diff('day', {left_value}, {right_value})) "
                f"<= {predicate.maximum_days})"
            )
        if isinstance(predicate, AllPredicate):
            return (
                "("
                + " AND ".join(
                    self._compile_predicate(term, left, right) for term in predicate.terms
                )
                + ")"
            )
        if isinstance(predicate, AnyPredicate):
            return (
                "("
                + " OR ".join(
                    self._compile_predicate(term, left, right) for term in predicate.terms
                )
                + ")"
            )
        raise AnchorEvidenceError("ML-ANCHOR-003", "An unsupported anchor predicate was rejected.")

    @staticmethod
    def _column_pair(
        variable_id: str,
        left: PreparedDataset,
        right: PreparedDataset,
        left_alias: str,
        right_alias: str,
    ) -> tuple[str, str]:
        try:
            left_column = quote_identifier(left.variable_columns[variable_id])
            right_column = quote_identifier(right.variable_columns[variable_id])
        except (KeyError, DataPlaneError):
            raise AnchorEvidenceError(
                "ML-ANCHOR-004",
                "An anchor references an unavailable canonical variable.",
            ) from None
        return (f"{left_alias}.{left_column}", f"{right_alias}.{right_column}")

    def _safe_scalar(self, sql: str, *, code: str, message: str) -> int:
        try:
            return self._store._scalar_int(sql)
        except DataPlaneError:
            raise AnchorEvidenceError(code, message) from None
