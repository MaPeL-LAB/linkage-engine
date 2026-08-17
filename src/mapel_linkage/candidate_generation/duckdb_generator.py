"""Privacy-safe DuckDB candidate generation for the first M2 slice."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mapel_linkage.candidate_generation.predicates import BlockingRule, compile_predicate
from mapel_linkage.domain.errors import CandidateBudgetExceeded, CandidateGenerationError
from mapel_linkage.domain.sql_identifiers import quote_identifier, validate_identifier
from mapel_linkage.domain.table_refs import TableRef
from mapel_linkage.io.duckdb_store import DuckDBStore


@dataclass(frozen=True, slots=True)
class CandidateGenerationResult:
    """Structural candidate-generation output; it contains no pair values."""

    table: TableRef
    candidate_pair_count: int
    configured_rule_count: int


@dataclass(frozen=True, slots=True)
class CandidateDiagnostics:
    """Aggregate diagnostics that are safe to report."""

    candidate_pair_count: int
    multi_rule_pair_count: int
    maximum_rules_per_pair: int


class DuckDBCandidateGenerator:
    """Generate link-only candidate pairs from typed blocking predicates."""

    __slots__ = ("_store",)

    def __init__(self, store: DuckDBStore) -> None:
        self._store = store

    def generate(
        self,
        *,
        left: TableRef,
        right: TableRef,
        variable_columns: Mapping[str, str],
        rules: Sequence[BlockingRule],
        maximum_candidate_pairs: int,
        record_key_column: str = "__ml_record_key",
    ) -> CandidateGenerationResult:
        """Materialise deduplicated candidates without making match decisions."""

        if not rules:
            raise CandidateGenerationError(
                "ML-CANDIDATE-001", "At least one blocking rule is required."
            )
        if maximum_candidate_pairs < 1:
            raise CandidateGenerationError(
                "ML-CANDIDATE-007", "The candidate-pair budget must be positive."
            )
        validate_identifier(record_key_column)
        for variable_id in variable_columns:
            validate_identifier(variable_id)

        left_table = quote_identifier(left.table_name)
        right_table = quote_identifier(right.table_name)
        key = quote_identifier(record_key_column)
        selects: list[str] = []
        for rule in rules:
            condition = compile_predicate(rule.predicate, variable_columns)
            selects.append(
                "SELECT "
                f"l.{key} AS left_record_key, "
                f"r.{key} AS right_record_key, "
                f"'{rule.rule_id}' AS retrieval_rule_id "
                f"FROM {left_table} AS l INNER JOIN {right_table} AS r ON {condition}"
            )
        union_sql = " UNION ALL ".join(selects)
        distinct_evidence = (
            "SELECT DISTINCT left_record_key, right_record_key, retrieval_rule_id "
            f"FROM ({union_sql}) AS retrieved"
        )
        pair_count_sql = (
            "SELECT COUNT(*) FROM ("
            "SELECT DISTINCT left_record_key, right_record_key "
            f"FROM ({distinct_evidence}) AS evidence"
            ") AS unique_pairs"
        )
        candidate_count = self._store._scalar_int(pair_count_sql)
        if candidate_count > maximum_candidate_pairs:
            raise CandidateBudgetExceeded(
                "ML-CANDIDATE-008",
                "Candidate generation exceeded the configured aggregate pair budget.",
            )

        digest_input = "|".join(
            [left.table_name, right.table_name, *(rule.rule_id for rule in rules)]
        )
        suffix = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:12]
        table_name = f"__ml_candidates_{suffix}"
        materialisation_sql = (
            "SELECT left_record_key, right_record_key, "
            "STRING_AGG(retrieval_rule_id, ',' ORDER BY retrieval_rule_id) "
            "AS retrieval_rule_ids, "
            "COUNT(*)::INTEGER AS retrieval_rule_count "
            f"FROM ({distinct_evidence}) AS evidence "
            "GROUP BY left_record_key, right_record_key "
            "ORDER BY left_record_key, right_record_key"
        )
        table = self._store._create_temp_table_as(table_name, materialisation_sql)
        return CandidateGenerationResult(
            table=table,
            candidate_pair_count=candidate_count,
            configured_rule_count=len(rules),
        )

    def diagnostics(self, result: CandidateGenerationResult) -> CandidateDiagnostics:
        """Return aggregate retrieval diagnostics without exposing any pair."""

        pair_count, multi_rule_count, max_rules = self._store._candidate_diagnostics(
            result.table.table_name
        )
        return CandidateDiagnostics(
            candidate_pair_count=pair_count,
            multi_rule_pair_count=multi_rule_count,
            maximum_rules_per_pair=max_rules,
        )
