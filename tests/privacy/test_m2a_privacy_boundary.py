from __future__ import annotations

from mapel_linkage.candidate_generation import BlockingRule, DuckDBCandidateGenerator, Exact
from mapel_linkage.io import ColumnSpec, DuckDBStore


def test_row_values_do_not_appear_in_public_objects() -> None:
    sentinel = "SYNTHETIC-ROW-SENTINEL-DO-NOT-PRINT"
    columns = (
        ColumnSpec("__ml_record_key", "VARCHAR"),
        ColumnSpec("v_code", "VARCHAR"),
    )
    with DuckDBStore() as store:
        left = store.create_table_from_rows(
            "left_privacy",
            columns,
            (("left-key", sentinel),),
        )
        right = store.create_table_from_rows(
            "right_privacy",
            columns,
            (("right-key", sentinel),),
        )
        result = DuckDBCandidateGenerator(store).generate(
            left=left,
            right=right,
            variable_columns={"code": "v_code"},
            rules=(BlockingRule("exact_code", Exact("code")),),
            maximum_candidate_pairs=10,
        )
        rendered = " ".join((repr(store), repr(left), repr(right), repr(result)))

    assert sentinel not in rendered
    assert "left-key" not in rendered
    assert "right-key" not in rendered
