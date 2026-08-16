from __future__ import annotations

from datetime import date

from mapel_linkage.anchors import DuckDBAnchorEvidenceEvaluator
from mapel_linkage.comparisons import DuckDBComparisonFeatureBuilder
from mapel_linkage.configuration.models import ComparisonConfig, DeterministicAnchorConfig
from mapel_linkage.io import ColumnSpec, DuckDBStore
from mapel_linkage.preprocessing import PreparedDataset


def test_m2c_public_objects_do_not_render_record_values_or_pairs() -> None:
    private_left_key = "SYNTHETIC-PRIVATE-LEFT-KEY"
    private_right_key = "SYNTHETIC-PRIVATE-RIGHT-KEY"
    private_value = "SYNTHETIC-PRIVATE-CANONICAL-VALUE"
    prepared_columns = (
        ColumnSpec("__ml_record_key", "VARCHAR"),
        ColumnSpec("v_value", "VARCHAR"),
        ColumnSpec("m_value", "BOOLEAN"),
        ColumnSpec("v_date", "DATE"),
        ColumnSpec("m_date", "BOOLEAN"),
    )
    candidate_columns = (
        ColumnSpec("left_record_key", "VARCHAR"),
        ColumnSpec("right_record_key", "VARCHAR"),
        ColumnSpec("retrieval_rule_ids", "VARCHAR"),
        ColumnSpec("retrieval_rule_count", "INTEGER"),
    )
    comparison = ComparisonConfig.model_validate(
        {
            "id": "private_comparison",
            "variable": "value",
            "function": {"kind": "exact"},
            "levels": [{"kind": "missing"}, {"kind": "exact"}, {"kind": "else"}],
        }
    )
    anchor = DeterministicAnchorConfig.model_validate(
        {
            "id": "private_anchor",
            "predicate": {"kind": "exact", "variable": "value"},
        }
    )

    with DuckDBStore() as store:
        left_table = store.create_table_from_rows(
            "privacy_left",
            prepared_columns,
            ((private_left_key, private_value, False, date(2020, 1, 1), False),),
        )
        right_table = store.create_table_from_rows(
            "privacy_right",
            prepared_columns,
            ((private_right_key, private_value, False, date(2020, 1, 1), False),),
        )
        values = {"value": "v_value", "date": "v_date"}
        missing = {"value": "m_value", "date": "m_date"}
        left = PreparedDataset("left", left_table, values, missing)
        right = PreparedDataset("right", right_table, values, missing)
        candidates = store.create_table_from_rows(
            "privacy_candidates",
            candidate_columns,
            ((private_left_key, private_right_key, "private_rule", 1),),
        )
        features = DuckDBComparisonFeatureBuilder(store).build(
            candidates=candidates,
            left=left,
            right=right,
            comparisons=(comparison,),
        )
        evidence = DuckDBAnchorEvidenceEvaluator(store).evaluate(
            left=left,
            right=right,
            anchors=(anchor,),
            maximum_anchor_pairs=10,
        )
        rendered = " ".join(
            (
                repr(features),
                repr(features.table),
                repr(evidence),
                repr(evidence.table),
                repr(store),
            )
        )

    assert private_left_key not in rendered
    assert private_right_key not in rendered
    assert private_value not in rendered
    assert "private_rule" not in rendered
