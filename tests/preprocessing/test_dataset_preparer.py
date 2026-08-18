from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

import pytest

from mapel_linkage.configuration import LinkageConfig, compile_config
from mapel_linkage.domain import PreprocessingError
from mapel_linkage.io import DuckDBStore
from mapel_linkage.preprocessing import ConfiguredDatasetPreparer, PreparedDataset
from tests.helpers import valid_payload


def _source_headers(renamed: bool) -> dict[str, dict[str, str]]:
    if renamed:
        return {
            "source_a": {
                "record": "Record Key A Renamed",
                "label": "Display Label A Renamed",
                "date": "Observed Date A Renamed",
                "group": "Group A Renamed",
            },
            "source_b": {
                "record": "record-key-b-renamed",
                "label": "Display Label B Renamed",
                "date": "Observed Date B Renamed",
                "group": "Group B Renamed",
            },
        }
    return {
        "source_a": {
            "record": "Record Key A",
            "label": "Display Label A",
            "date": "Observed Date A",
            "group": "Group A",
        },
        "source_b": {
            "record": "record-key-b",
            "label": "Display Label B",
            "date": "Observed Date B",
            "group": "Group B",
        },
    }


def _build_project(root: Path, *, renamed: bool = False) -> LinkageConfig:
    headers = _source_headers(renamed)
    data_root = root / "data"
    data_root.mkdir(parents=True)

    source_a_path = data_root / "source_a.csv"
    with source_a_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(headers["source_a"].values()))
        writer.writeheader()
        writer.writerow(
            {
                headers["source_a"]["record"]: "source-a-1",
                headers["source_a"]["label"]: "  \uff21LPHA   Beta ",
                headers["source_a"]["date"]: "2025-12-31",
                headers["source_a"]["group"]: "group-1",
            }
        )
        writer.writerow(
            {
                headers["source_a"]["record"]: "source-a-2",
                headers["source_a"]["label"]: "   ",
                headers["source_a"]["date"]: "2026-01-01",
                headers["source_a"]["group"]: "group-2",
            }
        )

    source_b_path = data_root / "source_b.jsonl"
    source_b_rows = (
        {
            headers["source_b"]["record"]: "source-b-1",
            headers["source_b"]["label"]: "alpha beta",
            headers["source_b"]["date"]: "2025-12-31",
            headers["source_b"]["group"]: "group-1",
        },
        {
            headers["source_b"]["record"]: "source-b-2",
            headers["source_b"]["label"]: "gamma",
            headers["source_b"]["date"]: "2026-01-02",
            headers["source_b"]["group"]: "group-2",
        },
    )
    source_b_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in source_b_rows),
        encoding="utf-8",
    )

    payload = valid_payload()
    datasets = cast(list[dict[str, Any]], payload["datasets"])
    datasets[0].update(
        {
            "path": "data/source_a.csv",
            "format": "csv",
            "record_id_column": headers["source_a"]["record"],
        }
    )
    datasets[1].update(
        {
            "path": "data/source_b.jsonl",
            "format": "jsonl",
            "record_id_column": headers["source_b"]["record"],
        }
    )
    variables = cast(list[dict[str, Any]], payload["variables"])
    variables[0]["source_columns"] = {
        "source_a": headers["source_a"]["label"],
        "source_b": headers["source_b"]["label"],
    }
    variables[1]["source_columns"] = {
        "source_a": headers["source_a"]["date"],
        "source_b": headers["source_b"]["date"],
    }
    variables[2]["source_columns"] = {
        "source_a": headers["source_a"]["group"],
        "source_b": headers["source_b"]["group"],
    }
    return LinkageConfig.model_validate(payload)


def _snapshot(store: DuckDBStore, prepared: PreparedDataset) -> list[tuple[object, ...]]:
    label_column = prepared.variable_columns["label_text"]
    missing_column = prepared.missingness_columns["label_text"]
    return cast(
        list[tuple[object, ...]],
        store._connection.execute(
            f'SELECT "__ml_record_key", "{label_column}", "{missing_column}" '
            f'FROM "{prepared.table.table_name}" ORDER BY "__ml_record_key"'
        ).fetchall(),
    )


def test_configured_csv_and_jsonl_sources_prepare_canonical_tables(tmp_path: Path) -> None:
    config = _build_project(tmp_path)
    plan = compile_config(config, project_root=tmp_path)

    with DuckDBStore() as store:
        catalog = ConfiguredDatasetPreparer(store).prepare_all(plan)
        left = catalog.require("source_a")
        right = catalog.require("source_b")
        left_rows = _snapshot(store, left)

    assert catalog.safe_summary() == {"dataset_count": 2, "row_count": 4}
    assert left.table.row_count == 2
    assert right.table.row_count == 2
    assert left.variable_columns == right.variable_columns
    assert left.missingness_columns == right.missingness_columns
    assert all(column.startswith("__ml_v_") for column in left.variable_columns.values())
    assert all(column.startswith("__ml_m_") for column in left.missingness_columns.values())
    assert {row[1] for row in left_rows} == {"alpha beta", None}
    assert {row[2] for row in left_rows} == {False, True}
    assert all(len(cast(str, row[0])) == 64 for row in left_rows)
    assert "source-a-1" not in {row[0] for row in left_rows}


def test_source_column_renaming_requires_configuration_changes_only(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    renamed_root = tmp_path / "renamed"
    original_plan = compile_config(_build_project(original_root), project_root=original_root)
    renamed_plan = compile_config(
        _build_project(renamed_root, renamed=True),
        project_root=renamed_root,
    )

    with DuckDBStore() as original_store:
        original = ConfiguredDatasetPreparer(original_store).prepare_all(original_plan)
        original_rows = _snapshot(original_store, original.require("source_a"))
    with DuckDBStore() as renamed_store:
        renamed = ConfiguredDatasetPreparer(renamed_store).prepare_all(renamed_plan)
        renamed_rows = _snapshot(renamed_store, renamed.require("source_a"))

    assert original_rows == renamed_rows


def test_duplicate_record_identifiers_are_rejected(tmp_path: Path) -> None:
    config = _build_project(tmp_path)
    source_path = tmp_path / "data/source_a.csv"
    text = source_path.read_text(encoding="utf-8")
    source_path.write_text(text + text.splitlines()[-1] + "\n", encoding="utf-8")
    plan = compile_config(config, project_root=tmp_path)

    with DuckDBStore() as store, pytest.raises(PreprocessingError) as exc_info:
        ConfiguredDatasetPreparer(store).prepare_dataset(plan, config.datasets[0])

    assert exc_info.value.code == "ML-PREP-005"
    assert "source-a-2" not in str(exc_info.value)


def test_direct_duckdb_source_attachment_is_deferred(tmp_path: Path) -> None:
    config = _build_project(tmp_path)
    payload = config.model_dump(mode="json")
    datasets = cast(list[dict[str, Any]], payload["datasets"])
    datasets[0]["format"] = "duckdb"
    deferred_config = LinkageConfig.model_validate(payload)
    plan = compile_config(deferred_config, project_root=tmp_path)

    with DuckDBStore() as store, pytest.raises(PreprocessingError) as exc_info:
        ConfiguredDatasetPreparer(store).prepare_dataset(plan, deferred_config.datasets[0])

    assert exc_info.value.code == "ML-PREP-003"
