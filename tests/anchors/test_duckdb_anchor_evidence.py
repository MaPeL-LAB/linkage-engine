from __future__ import annotations

from datetime import date
from typing import cast

import pytest

from mapel_linkage.anchors import DuckDBAnchorEvidenceEvaluator
from mapel_linkage.configuration.models import DeterministicAnchorConfig
from mapel_linkage.domain import AnchorBudgetExceeded, AnchorEvidenceError
from mapel_linkage.io import ColumnSpec, DuckDBStore
from mapel_linkage.preprocessing import PreparedDataset

_COLUMNS = (
    ColumnSpec("__ml_record_key", "VARCHAR"),
    ColumnSpec("v_text", "VARCHAR"),
    ColumnSpec("m_text", "BOOLEAN"),
    ColumnSpec("v_date", "DATE"),
    ColumnSpec("m_date", "BOOLEAN"),
)


def _prepared(store: DuckDBStore) -> tuple[PreparedDataset, PreparedDataset]:
    left_table = store.create_table_from_rows(
        "anchor_left",
        _COLUMNS,
        (
            ("left-1", "alpha", False, date(2020, 1, 1), False),
            ("left-2", "alfa", False, date(2020, 1, 2), False),
            ("left-3", "beta", False, date(2020, 2, 1), False),
        ),
    )
    right_table = store.create_table_from_rows(
        "anchor_right",
        _COLUMNS,
        (
            ("right-1", "alpha", False, date(2020, 1, 1), False),
            ("right-2", "alpine", False, date(2020, 1, 3), False),
            ("right-3", "beta", False, date(2020, 2, 2), False),
        ),
    )
    values = {"text_value": "v_text", "date_value": "v_date"}
    missing = {"text_value": "m_text", "date_value": "m_date"}
    return (
        PreparedDataset("left", left_table, values, missing),
        PreparedDataset("right", right_table, values, missing),
    )


def _anchors() -> tuple[DeterministicAnchorConfig, ...]:
    return tuple(
        DeterministicAnchorConfig.model_validate(payload)
        for payload in (
            {
                "id": "exact_text_date",
                "predicate": {
                    "kind": "all",
                    "terms": [
                        {"kind": "exact", "variable": "text_value"},
                        {"kind": "exact", "variable": "date_value"},
                    ],
                },
            },
            {
                "id": "text_prefix",
                "predicate": {
                    "kind": "prefix_equal",
                    "variable": "text_value",
                    "length": 1,
                },
            },
            {
                "id": "date_window",
                "predicate": {
                    "kind": "date_window",
                    "variable": "date_value",
                    "maximum_days": 1,
                },
                "require_unique_left": False,
                "require_unique_right": False,
            },
        )
    )


def test_anchor_evidence_is_evidence_only_with_uniqueness_diagnostics() -> None:
    with DuckDBStore() as store:
        left, right = _prepared(store)
        result = DuckDBAnchorEvidenceEvaluator(store).evaluate(
            left=left,
            right=right,
            anchors=_anchors(),
            maximum_anchor_pairs=100,
        )
        rows = cast(
            list[tuple[object, ...]],
            store._connection.execute(
                "SELECT anchor_rule_id, left_match_count, right_match_count, "
                "uniqueness_pass, evidence_action, eligible_as_training_truth "
                f'FROM "{result.table.table_name}" '
                "ORDER BY anchor_rule_id, left_record_key, right_record_key"
            ).fetchall(),
        )

    exact_rows = [row for row in rows if row[0] == "exact_text_date"]
    prefix_rows = [row for row in rows if row[0] == "text_prefix"]
    window_rows = [row for row in rows if row[0] == "date_window"]
    assert exact_rows == [("exact_text_date", 1, 1, True, "evidence_only", False)]
    assert prefix_rows
    assert any(row[3] is False for row in prefix_rows)
    assert window_rows
    assert all(row[3] is True for row in window_rows)
    assert all(row[4] == "evidence_only" for row in rows)
    assert all(row[5] is False for row in rows)
    assert result.evidence_row_count == len(rows)
    assert result.distinct_pair_count <= result.evidence_row_count
    assert "left-1" not in repr(result)
    assert "right-1" not in repr(result)


def test_anchor_budget_is_checked_before_materialisation() -> None:
    with DuckDBStore() as store:
        left, right = _prepared(store)
        with pytest.raises(AnchorBudgetExceeded) as exc_info:
            DuckDBAnchorEvidenceEvaluator(store).evaluate(
                left=left,
                right=right,
                anchors=(_anchors()[1],),
                maximum_anchor_pairs=1,
            )

    assert exc_info.value.code == "ML-ANCHOR-006"
    assert "left-" not in str(exc_info.value)
    assert "right-" not in str(exc_info.value)


def test_empty_anchor_configuration_returns_schema_stable_empty_evidence() -> None:
    with DuckDBStore() as store:
        left, right = _prepared(store)
        result = DuckDBAnchorEvidenceEvaluator(store).evaluate(
            left=left,
            right=right,
            anchors=(),
            maximum_anchor_pairs=10,
        )

    assert result.configured_anchor_count == 0
    assert result.evidence_row_count == 0
    assert result.distinct_pair_count == 0
    assert result.uniqueness_pass_count == 0


def test_unavailable_anchor_variable_is_value_safe() -> None:
    sentinel = "SYNTHETIC-SENSITIVE-COLUMN"
    anchor = DeterministicAnchorConfig.model_validate(
        {
            "id": "private_anchor",
            "predicate": {"kind": "exact", "variable": "unavailable_variable"},
        }
    )
    with DuckDBStore() as store:
        left, right = _prepared(store)
        left = PreparedDataset(left.dataset_id, left.table, {"other": sentinel}, {})
        with pytest.raises(AnchorEvidenceError) as exc_info:
            DuckDBAnchorEvidenceEvaluator(store).evaluate(
                left=left,
                right=right,
                anchors=(anchor,),
                maximum_anchor_pairs=10,
            )

    assert exc_info.value.code == "ML-ANCHOR-004"
    assert sentinel not in str(exc_info.value)
