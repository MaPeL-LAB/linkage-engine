from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, cast

import pytest

from mapel_linkage.configuration import LinkageConfig, compile_config
from mapel_linkage.domain import PreprocessingError
from mapel_linkage.io import DuckDBStore
from mapel_linkage.preprocessing import ConfiguredDatasetPreparer
from tests.helpers import valid_payload


def test_ingestion_errors_do_not_echo_paths_columns_or_values(tmp_path: Path) -> None:
    sentinel_column = "SYNTHETIC-SENSITIVE-COLUMN"
    sentinel_value = "SYNTHETIC-SENSITIVE-ROW-VALUE"
    data_root = tmp_path / "data"
    data_root.mkdir()
    source_path = data_root / "privacy.csv"
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("record", "actual"))
        writer.writeheader()
        writer.writerow({"record": "identifier-1", "actual": sentinel_value})

    payload = valid_payload()
    datasets = cast(list[dict[str, Any]], payload["datasets"])
    datasets[0].update(
        {
            "path": "data/privacy.csv",
            "format": "csv",
            "record_id_column": "record",
        }
    )
    variables = cast(list[dict[str, Any]], payload["variables"])
    variables[0]["source_columns"]["source_a"] = sentinel_column
    variables[1]["source_columns"]["source_a"] = "actual"
    config = LinkageConfig.model_validate(payload)
    plan = compile_config(config, project_root=tmp_path)

    with DuckDBStore() as store, pytest.raises(PreprocessingError) as exc_info:
        ConfiguredDatasetPreparer(store).prepare_dataset(plan, config.datasets[0])

    rendered = str(exc_info.value)
    assert sentinel_column not in rendered
    assert sentinel_value not in rendered
    assert str(source_path) not in rendered


def test_prepared_dataset_representation_is_structural_only(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    source_path = data_root / "privacy.csv"
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("record", "label", "date"))
        writer.writeheader()
        writer.writerow(
            {
                "record": "SYNTHETIC-ORIGINAL-ID",
                "label": "SYNTHETIC-PRIVATE-LABEL",
                "date": "2025-01-01",
            }
        )

    payload = valid_payload()
    datasets = cast(list[dict[str, Any]], payload["datasets"])
    datasets[0].update(
        {
            "path": "data/privacy.csv",
            "format": "csv",
            "record_id_column": "record",
        }
    )
    variables = cast(list[dict[str, Any]], payload["variables"])
    variables[0]["source_columns"]["source_a"] = "label"
    variables[1]["source_columns"]["source_a"] = "date"
    config = LinkageConfig.model_validate(payload)
    plan = compile_config(config, project_root=tmp_path)

    with DuckDBStore() as store:
        prepared = ConfiguredDatasetPreparer(store).prepare_dataset(plan, config.datasets[0])
        rendered = " ".join((repr(store), repr(prepared), repr(prepared.table)))

    assert "SYNTHETIC-ORIGINAL-ID" not in rendered
    assert "SYNTHETIC-PRIVATE-LABEL" not in rendered
    assert str(source_path) not in rendered
