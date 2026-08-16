from __future__ import annotations

from datetime import date
from typing import cast

import pytest

from mapel_linkage.comparisons import DuckDBComparisonFeatureBuilder
from mapel_linkage.configuration.models import ComparisonConfig
from mapel_linkage.domain import ComparisonFeatureError, TableRef
from mapel_linkage.io import ColumnSpec, DuckDBStore
from mapel_linkage.preprocessing import PreparedDataset

_PREPARED_COLUMNS = (
    ColumnSpec("__ml_record_key", "VARCHAR"),
    ColumnSpec("v_text", "VARCHAR"),
    ColumnSpec("m_text", "BOOLEAN"),
    ColumnSpec("v_date", "DATE"),
    ColumnSpec("m_date", "BOOLEAN"),
    ColumnSpec("v_number", "DOUBLE"),
    ColumnSpec("m_number", "BOOLEAN"),
    ColumnSpec("v_category", "VARCHAR"),
    ColumnSpec("m_category", "BOOLEAN"),
)
_CANDIDATE_COLUMNS = (
    ColumnSpec("left_record_key", "VARCHAR"),
    ColumnSpec("right_record_key", "VARCHAR"),
    ColumnSpec("retrieval_rule_ids", "VARCHAR"),
    ColumnSpec("retrieval_rule_count", "INTEGER"),
)


def _prepared_datasets(store: DuckDBStore) -> tuple[PreparedDataset, PreparedDataset]:
    left_table = store.create_table_from_rows(
        "comparison_left",
        _PREPARED_COLUMNS,
        (
            ("left-1", "alpha", False, date(2020, 1, 1), False, 10.0, False, "x", False),
            ("left-2", "alfa", False, date(2020, 1, 2), False, 10.5, False, "y", False),
            ("left-3", None, True, None, True, None, True, None, True),
            ("left-4", "abcd", False, date(2020, 1, 10), False, 12.0, False, "z", False),
        ),
    )
    right_table = store.create_table_from_rows(
        "comparison_right",
        _PREPARED_COLUMNS,
        (
            ("right-1", "alpha", False, date(2020, 1, 1), False, 10.0, False, "x", False),
            ("right-2", "alpha", False, date(2020, 1, 3), False, 11.0, False, "y", False),
            ("right-3", "beta", False, date(2020, 1, 4), False, 8.0, False, "q", False),
            ("right-4", "abce", False, date(2020, 1, 12), False, 14.0, False, "z", False),
        ),
    )
    variable_columns = {
        "text_value": "v_text",
        "date_value": "v_date",
        "number_value": "v_number",
        "category_value": "v_category",
    }
    missingness_columns = {
        "text_value": "m_text",
        "date_value": "m_date",
        "number_value": "m_number",
        "category_value": "m_category",
    }
    return (
        PreparedDataset("left", left_table, variable_columns, missingness_columns),
        PreparedDataset("right", right_table, variable_columns, missingness_columns),
    )


def _candidate_table(store: DuckDBStore) -> TableRef:
    return store.create_table_from_rows(
        "comparison_candidates",
        _CANDIDATE_COLUMNS,
        (
            ("left-1", "right-1", "rule_a,rule_b", 2),
            ("left-2", "right-2", "rule_a", 1),
            ("left-3", "right-3", "rule_b", 1),
            ("left-4", "right-4", "rule_c", 1),
        ),
    )


def _comparison_configs() -> tuple[ComparisonConfig, ...]:
    return tuple(
        ComparisonConfig.model_validate(payload)
        for payload in (
            {
                "id": "text_jaro",
                "variable": "text_value",
                "function": {"kind": "jaro_winkler"},
                "levels": [
                    {"kind": "missing"},
                    {"kind": "exact"},
                    {"kind": "threshold", "minimum": 0.8},
                    {"kind": "else"},
                ],
            },
            {
                "id": "text_edit",
                "variable": "text_value",
                "function": {"kind": "levenshtein"},
                "levels": [
                    {"kind": "missing"},
                    {"kind": "exact"},
                    {"kind": "threshold", "minimum": 0.75},
                    {"kind": "else"},
                ],
            },
            {
                "id": "text_qgram",
                "variable": "text_value",
                "function": {"kind": "qgram", "q": 2},
                "levels": [
                    {"kind": "missing"},
                    {"kind": "exact"},
                    {"kind": "threshold", "minimum": 0.6},
                    {"kind": "else"},
                ],
            },
            {
                "id": "date_distance",
                "variable": "date_value",
                "function": {"kind": "date_difference", "unit": "day"},
                "levels": [
                    {"kind": "missing"},
                    {"kind": "exact"},
                    {"kind": "maximum_difference", "value": 2.0},
                    {"kind": "else"},
                ],
            },
            {
                "id": "number_distance",
                "variable": "number_value",
                "function": {"kind": "numeric_difference", "scale": 1.0},
                "levels": [
                    {"kind": "missing"},
                    {"kind": "exact"},
                    {"kind": "maximum_difference", "value": 1.0},
                    {"kind": "else"},
                ],
            },
            {
                "id": "category_exact",
                "variable": "category_value",
                "function": {"kind": "categorical"},
                "levels": [
                    {"kind": "missing"},
                    {"kind": "exact"},
                    {"kind": "else"},
                ],
            },
        )
    )


def test_comparison_features_materialise_metrics_levels_and_missingness() -> None:
    comparisons = _comparison_configs()
    with DuckDBStore() as store:
        left, right = _prepared_datasets(store)
        candidates = _candidate_table(store)
        result = DuckDBComparisonFeatureBuilder(store).build(
            candidates=candidates,
            left=left,
            right=right,
            comparisons=comparisons,
        )
        text_jaro = result.columns["text_jaro"]
        text_edit = result.columns["text_edit"]
        text_qgram = result.columns["text_qgram"]
        date_distance = result.columns["date_distance"]
        number_distance = result.columns["number_distance"]
        category_exact = result.columns["category_exact"]
        rows = cast(
            list[tuple[object, ...]],
            store._connection.execute(
                "SELECT left_record_key, "
                f'"{text_jaro.value}", "{text_jaro.level}", '
                f'"{text_edit.value}", "{text_edit.level}", '
                f'"{text_qgram.value}", "{text_qgram.level}", '
                f'"{date_distance.value}", "{date_distance.level}", '
                f'"{number_distance.value}", "{number_distance.level}", '
                f'"{category_exact.level}", "{text_jaro.missing_any}" '
                f'FROM "{result.table.table_name}" ORDER BY left_record_key'
            ).fetchall(),
        )

    by_key = {cast(str, row[0]): row for row in rows}
    exact = by_key["left-1"]
    assert exact[2] == 1
    assert exact[4] == 1
    assert exact[6] == 1
    assert exact[8] == 1
    assert exact[10] == 1
    assert exact[11] == 1
    assert exact[12] is False

    fuzzy = by_key["left-2"]
    assert cast(float, fuzzy[1]) >= 0.8
    assert fuzzy[2] == 2
    assert cast(float, fuzzy[3]) == pytest.approx(0.6)
    assert fuzzy[4] == 3
    assert fuzzy[8] == 2
    assert fuzzy[10] == 2
    assert fuzzy[11] == 1

    missing = by_key["left-3"]
    assert missing[1] is None
    assert missing[2] == 0
    assert missing[3] is None
    assert missing[4] == 0
    assert missing[12] is True

    qgram = by_key["left-4"]
    assert cast(float, qgram[5]) == pytest.approx(2.0 / 3.0)
    assert qgram[6] == 2
    assert qgram[8] == 2
    assert qgram[10] == 3

    assert result.safe_summary()["candidate_pair_count"] == 4
    assert "left-1" not in repr(result)
    assert "right-1" not in repr(result)


def test_missing_comparison_without_missing_level_is_ignored() -> None:
    comparison = ComparisonConfig.model_validate(
        {
            "id": "ignored_missing",
            "variable": "text_value",
            "function": {"kind": "exact"},
            "levels": [{"kind": "exact"}, {"kind": "else"}],
        }
    )
    with DuckDBStore() as store:
        left, right = _prepared_datasets(store)
        candidates = store.create_table_from_rows(
            "missing_candidate",
            _CANDIDATE_COLUMNS,
            (("left-3", "right-3", "rule", 1),),
        )
        result = DuckDBComparisonFeatureBuilder(store).build(
            candidates=candidates,
            left=left,
            right=right,
            comparisons=(comparison,),
        )
        columns = result.columns[comparison.id]
        row = store._connection.execute(
            f'SELECT "{columns.value}", "{columns.level}" FROM "{result.table.table_name}"'
        ).fetchone()

    assert row == (None, None)


def test_unavailable_comparison_variable_is_value_safe() -> None:
    sentinel = "SYNTHETIC-SENSITIVE-COLUMN"
    comparison = ComparisonConfig.model_validate(
        {
            "id": "private_comparison",
            "variable": "unavailable_variable",
            "function": {"kind": "exact"},
            "levels": [{"kind": "missing"}, {"kind": "exact"}, {"kind": "else"}],
        }
    )
    with DuckDBStore() as store:
        left, right = _prepared_datasets(store)
        left = PreparedDataset(
            left.dataset_id,
            left.table,
            {"other": sentinel},
            left.missingness_columns,
        )
        with pytest.raises(ComparisonFeatureError) as exc_info:
            DuckDBComparisonFeatureBuilder(store).build(
                candidates=_candidate_table(store),
                left=left,
                right=right,
                comparisons=(comparison,),
            )

    assert exc_info.value.code == "ML-COMP-003"
    assert sentinel not in str(exc_info.value)
