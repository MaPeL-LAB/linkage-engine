from __future__ import annotations

from pathlib import Path

import pytest

from mapel_linkage.domain import DataPlaneError
from mapel_linkage.io import ColumnSpec, DuckDBStore


def test_store_creates_opaque_table_reference() -> None:
    with DuckDBStore() as store:
        ref = store.create_table_from_rows(
            "synthetic_left",
            (
                ColumnSpec("__ml_record_key", "VARCHAR"),
                ColumnSpec("v_name", "VARCHAR"),
            ),
            (("synthetic-left-1", "alpha"), ("synthetic-left-2", "beta")),
        )

    assert ref.row_count == 2
    assert len(ref.schema_digest) == 64
    assert "alpha" not in repr(ref)
    assert "synthetic-left-1" not in repr(ref)
    assert "/" not in repr(ref)


def test_store_representation_hides_database_path(tmp_path: Path) -> None:
    path = tmp_path / "restricted.duckdb"
    with DuckDBStore(path) as store:
        rendered = repr(store)

    assert str(path) not in rendered
    assert "<restricted>" in rendered


def test_store_rejects_unsafe_identifiers_without_echoing_value() -> None:
    unsafe = "table; DROP TABLE protected"
    with DuckDBStore() as store, pytest.raises(DataPlaneError) as exc_info:
        store.create_table_from_rows(
            unsafe,
            (ColumnSpec("value", "VARCHAR"),),
            (("synthetic",),),
        )

    assert unsafe not in str(exc_info.value)
    assert exc_info.value.code == "ML-DATA-002"


def test_store_rejects_row_shape_without_echoing_row() -> None:
    sentinel = "SYNTHETIC-SENTINEL-ROW-VALUE"
    with DuckDBStore() as store, pytest.raises(DataPlaneError) as exc_info:
        store.create_table_from_rows(
            "shape_test",
            (ColumnSpec("left_value", "VARCHAR"), ColumnSpec("right_value", "VARCHAR")),
            ((sentinel,),),
        )

    assert sentinel not in str(exc_info.value)
    assert exc_info.value.code == "ML-DATA-006"
