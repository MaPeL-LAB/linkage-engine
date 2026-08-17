from __future__ import annotations

from datetime import date

from mapel_linkage.configuration.models import (
    AllPredicate,
    ComparisonConfig,
    FellegiSunterModelConfig,
)
from mapel_linkage.io import ColumnSpec, DuckDBStore
from mapel_linkage.models import SplinkSettingsPlanCompiler
from mapel_linkage.preprocessing import PreparedDataset


def _prepared(store: DuckDBStore) -> tuple[PreparedDataset, PreparedDataset]:
    columns = (
        ColumnSpec("__ml_record_key", "VARCHAR"),
        ColumnSpec("__ml_dataset_id", "VARCHAR"),
        ColumnSpec("canonical_text", "VARCHAR"),
        ColumnSpec("canonical_date", "DATE"),
    )
    left_table = store.create_table_from_rows(
        "splink_left",
        columns,
        (("l1", "left", "alpha", date(2020, 1, 1)),),
    )
    right_table = store.create_table_from_rows(
        "splink_right",
        columns,
        (("r1", "right", "alpha", date(2020, 1, 1)),),
    )
    values = {"text": "canonical_text", "date": "canonical_date"}
    missing = {"text": "missing_text", "date": "missing_date"}
    return (
        PreparedDataset("left", left_table, values, missing),
        PreparedDataset("right", right_table, values, missing),
    )


def _comparisons() -> tuple[ComparisonConfig, ...]:
    return tuple(
        ComparisonConfig.model_validate(payload)
        for payload in (
            {
                "id": "text_similarity",
                "variable": "text",
                "function": {"kind": "jaro_winkler"},
                "levels": [
                    {"kind": "missing"},
                    {"kind": "exact"},
                    {"kind": "threshold", "minimum": 0.8},
                    {"kind": "else"},
                ],
            },
            {
                "id": "date_distance",
                "variable": "date",
                "function": {"kind": "date_difference", "unit": "day"},
                "levels": [
                    {"kind": "missing"},
                    {"kind": "exact"},
                    {"kind": "maximum_difference", "value": 2.0},
                    {"kind": "else"},
                ],
            },
        )
    )


def test_package_owned_splink_plan_builds_settings_creator() -> None:
    with DuckDBStore() as store:
        left, right = _prepared(store)
        predicate = AllPredicate.model_validate(
            {
                "kind": "all",
                "terms": [
                    {"kind": "prefix_equal", "variable": "text", "length": 1},
                    {"kind": "date_window", "variable": "date", "maximum_days": 3},
                ],
            }
        )
        plan = SplinkSettingsPlanCompiler().compile(
            left=left,
            right=right,
            comparisons=_comparisons(),
            blocking_rules=(predicate,),
            model=FellegiSunterModelConfig.model_validate(
                {
                    "implementation": "splink_duckdb",
                    "model_id": "fs_baseline",
                    "probability_two_random_records_match": 0.01,
                    "u_max_pairs": 100,
                }
            ),
        )
        creator = plan.build_settings_creator()

    assert creator.__class__.__name__ == "SettingsCreator"
    assert plan.comparison_count == 2
    assert plan.blocking_rule_count == 1
    assert "record_key_a" not in repr(plan)
    assert "canonical_text" not in repr(plan)
