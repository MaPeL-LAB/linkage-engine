from __future__ import annotations

import pytest

from mapel_linkage.candidate_generation import (
    AllOf,
    BlockingRule,
    DuckDBCandidateGenerator,
    Exact,
    PrefixEqual,
)
from mapel_linkage.domain import (
    CandidateBudgetExceeded,
    CandidateGenerationError,
    TableRef,
)
from mapel_linkage.io import ColumnSpec, DuckDBStore

COLUMNS = (
    ColumnSpec("__ml_record_key", "VARCHAR"),
    ColumnSpec("v_name", "VARCHAR"),
    ColumnSpec("v_date", "VARCHAR"),
    ColumnSpec("v_location", "VARCHAR"),
)


def _tables(store: DuckDBStore) -> tuple[TableRef, TableRef]:
    left = store.create_table_from_rows(
        "synthetic_left",
        COLUMNS,
        (
            ("left-1", "alpha", "2000-01-01", "zone-1"),
            ("left-2", "beta", "2001-01-01", "zone-2"),
            ("left-3", "gamma", "2002-01-01", "zone-3"),
        ),
    )
    right = store.create_table_from_rows(
        "synthetic_right",
        COLUMNS,
        (
            ("right-1", "alpha", "2000-01-01", "zone-9"),
            ("right-2", "alpine", "1999-01-01", "zone-1"),
            ("right-3", "beta", "2001-01-01", "zone-2"),
            ("right-4", "delta", "2003-01-01", "zone-4"),
        ),
    )
    return left, right


def _rules() -> tuple[BlockingRule, ...]:
    return (
        BlockingRule(
            "date_name_prefix",
            AllOf((Exact("date"), PrefixEqual("name", 1))),
        ),
        BlockingRule(
            "location_name_prefix",
            AllOf((Exact("location"), PrefixEqual("name", 1))),
        ),
    )


def test_candidate_generation_is_deduplicated_and_aggregate_only() -> None:
    with DuckDBStore() as store:
        left, right = _tables(store)
        generator = DuckDBCandidateGenerator(store)
        result = generator.generate(
            left=left,
            right=right,
            variable_columns={
                "name": "v_name",
                "date": "v_date",
                "location": "v_location",
            },
            rules=_rules(),
            maximum_candidate_pairs=100,
        )
        diagnostics = generator.diagnostics(result)

    assert result.candidate_pair_count == 3
    assert result.table.row_count == 3
    assert diagnostics.candidate_pair_count == 3
    assert diagnostics.multi_rule_pair_count == 1
    assert diagnostics.maximum_rules_per_pair == 2
    assert "left-1" not in repr(result)
    assert "right-1" not in repr(result)


def test_candidate_budget_is_checked_before_materialisation() -> None:
    with DuckDBStore() as store:
        left, right = _tables(store)
        generator = DuckDBCandidateGenerator(store)
        with pytest.raises(CandidateBudgetExceeded) as exc_info:
            generator.generate(
                left=left,
                right=right,
                variable_columns={
                    "name": "v_name",
                    "date": "v_date",
                    "location": "v_location",
                },
                rules=_rules(),
                maximum_candidate_pairs=2,
            )

    assert exc_info.value.code == "ML-CANDIDATE-008"
    assert "left-" not in str(exc_info.value)
    assert "right-" not in str(exc_info.value)


def test_missing_variable_is_rejected_without_echoing_mapping_values() -> None:
    sentinel = "SYNTHETIC-SECRET-COLUMN"
    with DuckDBStore() as store:
        left, right = _tables(store)
        generator = DuckDBCandidateGenerator(store)
        with pytest.raises(CandidateGenerationError) as exc_info:
            generator.generate(
                left=left,
                right=right,
                variable_columns={"other": sentinel},
                rules=(BlockingRule("missing_variable", Exact("name")),),
                maximum_candidate_pairs=100,
            )

    assert sentinel not in str(exc_info.value)
    assert exc_info.value.code == "ML-CANDIDATE-005"


def test_empty_blocking_rules_are_rejected() -> None:
    with DuckDBStore() as store:
        left, right = _tables(store)
        generator = DuckDBCandidateGenerator(store)
        with pytest.raises(CandidateGenerationError) as exc_info:
            generator.generate(
                left=left,
                right=right,
                variable_columns={"name": "v_name"},
                rules=(),
                maximum_candidate_pairs=100,
            )

    assert exc_info.value.code == "ML-CANDIDATE-001"


def test_date_window_blocking_uses_the_typed_dsl() -> None:
    from mapel_linkage.candidate_generation import DateWindow

    with DuckDBStore() as store:
        left = store.create_table_from_rows(
            "date_left",
            (
                ColumnSpec("__ml_record_key", "VARCHAR"),
                ColumnSpec("v_date", "DATE"),
            ),
            (("left-1", "2000-01-01"),),
        )
        right = store.create_table_from_rows(
            "date_right",
            (
                ColumnSpec("__ml_record_key", "VARCHAR"),
                ColumnSpec("v_date", "DATE"),
            ),
            (("right-1", "2000-01-03"), ("right-2", "2000-02-01")),
        )
        result = DuckDBCandidateGenerator(store).generate(
            left=left,
            right=right,
            variable_columns={"date": "v_date"},
            rules=(BlockingRule("date_window", DateWindow("date", 3)),),
            maximum_candidate_pairs=10,
        )
    assert result.candidate_pair_count == 1


def test_same_table_candidates_are_canonical_and_exclude_self_pairs() -> None:
    with DuckDBStore() as store:
        table = store.create_table_from_rows(
            "synthetic_dedupe_source",
            COLUMNS,
            (
                ("record-1", "alpha", "2000-01-01", "zone-1"),
                ("record-2", "alpine", "2000-01-02", "zone-1"),
                ("record-3", "beta", "2001-01-01", "zone-2"),
            ),
        )
        result = DuckDBCandidateGenerator(store).generate_deduplication(
            dataset=table,
            variable_columns={"location": "v_location"},
            rules=(BlockingRule("location_exact", Exact("location")),),
            maximum_candidate_pairs=10,
        )
        rows = store._fetch_model_rows(
            f"SELECT left_record_key, right_record_key FROM {result.table.table_name}"
        )

    assert rows == [("record-1", "record-2")]
    assert result.candidate_pair_count == 1


def test_same_table_candidate_budget_counts_canonical_pairs_only() -> None:
    with DuckDBStore() as store:
        table = store.create_table_from_rows(
            "synthetic_dedupe_budget",
            COLUMNS,
            (
                ("record-1", "alpha", "2000-01-01", "zone-1"),
                ("record-2", "alpine", "2000-01-02", "zone-1"),
                ("record-3", "amber", "2000-01-03", "zone-1"),
            ),
        )
        generator = DuckDBCandidateGenerator(store)
        with pytest.raises(CandidateBudgetExceeded):
            generator.generate_deduplication(
                dataset=table,
                variable_columns={"location": "v_location"},
                rules=(BlockingRule("location_exact", Exact("location")),),
                maximum_candidate_pairs=2,
            )
