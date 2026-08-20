from __future__ import annotations

import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from mapel_linkage.configuration import compile_config, load_config
from mapel_linkage.governance.errors import SafeError
from mapel_linkage.governance.manifests import create_run_manifest, write_manifest
from mapel_linkage.governance.paths import PathPolicy
from mapel_linkage.governance.safe_logging import (
    SafeLogEvent,
    SafeLogger,
    build_safe_log_event,
)
from tests.helpers import EXAMPLE_CONFIG, ROOT


def test_safe_logger_accepts_only_aggregate_fields() -> None:
    stream = io.StringIO()
    logger = logging.getLogger("mapel_linkage_test_safe")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(stream))
    SafeLogger(logger).emit(SafeLogEvent(event="config_validated", count=2, digest="a" * 12))
    payload = json.loads(stream.getvalue())
    assert payload == {"count": 2, "digest": "a" * 12, "event": "config_validated"}


def test_safe_log_validation_hides_rejected_value() -> None:
    sentinel = "SYNTHETIC-SENTINEL-ROW-VALUE"
    with pytest.raises(ValidationError) as caught:
        SafeLogEvent.model_validate({"event": "unsafe", "record": sentinel})
    assert sentinel not in str(caught.value)


def test_manifest_contains_no_paths_or_configuration_values(tmp_path: Path) -> None:
    loaded = load_config(EXAMPLE_CONFIG)
    plan = compile_config(loaded.config, project_root=ROOT)
    manifest = create_run_manifest(
        configuration_digest=plan.configuration_digest,
        registry_digest=plan.registry_digest,
        random_seed=plan.random_seed,
        dataset_count=plan.dataset_count,
        variable_count=plan.variable_count,
        run_id="a" * 32,
        created_at=datetime(2026, 8, 16, tzinfo=UTC),
    )
    text = manifest.model_dump_json()
    assert {
        "duckdb",
        "mapel-linkage-engine",
        "numpy",
        "pydantic",
        "PyYAML",
        "scikit-learn",
        "splink",
        "xgboost",
    }.issubset(manifest.package_versions)
    assert "source_a.parquet" not in text
    assert "record_key_a" not in text
    assert str(ROOT) not in text

    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data",),
        configured_output_roots=("artifacts",),
    )
    destination = write_manifest("artifacts/run/manifest.json", manifest, policy)
    assert destination.exists()
    assert "source_a.parquet" not in destination.read_text(encoding="utf-8")


def test_manifest_write_rejects_out_of_root(tmp_path: Path) -> None:
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data",),
        configured_output_roots=("artifacts",),
    )
    loaded = load_config(EXAMPLE_CONFIG)
    plan = compile_config(loaded.config, project_root=ROOT)
    manifest = create_run_manifest(
        configuration_digest=plan.configuration_digest,
        registry_digest=plan.registry_digest,
        random_seed=plan.random_seed,
        dataset_count=plan.dataset_count,
        variable_count=plan.variable_count,
    )
    with pytest.raises(SafeError) as caught:
        write_manifest("private/manifest.json", manifest, policy)
    assert str(tmp_path) not in str(caught.value)


@pytest.mark.parametrize("symlink_shape", ["destination", "ancestor"])
def test_output_resolution_rejects_terminal_and_ancestor_symlinks_without_target_write(
    tmp_path: Path,
    symlink_shape: str,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    external_target = external_root / "protected.json"
    external_target.write_text("unchanged\n", encoding="utf-8")
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data",),
        configured_output_roots=("artifacts",),
        host_input_roots=(tmp_path / "data",),
        host_output_roots=(artifact_root,),
    )

    if symlink_shape == "destination":
        run_directory = artifact_root / "run"
        run_directory.mkdir()
        (run_directory / "recipe-v1.json").symlink_to(external_target)
    else:
        (artifact_root / "run").symlink_to(external_root, target_is_directory=True)

    with pytest.raises(SafeError) as caught:
        policy.resolve_output("artifacts/run/recipe-v1.json")

    assert external_target.read_text(encoding="utf-8") == "unchanged\n"
    assert str(tmp_path) not in caught.value.render()


def test_safe_log_builder_hides_rejected_key_and_value() -> None:
    sensitive_key = "SYNTHETIC-SECRET-FIELD"
    sentinel = "SYNTHETIC-SECRET-VALUE"
    with pytest.raises(SafeError) as caught:
        build_safe_log_event({"event": "unsafe", sensitive_key: sentinel})
    assert sensitive_key not in caught.value.render()
    assert sentinel not in caught.value.render()


@pytest.mark.parametrize("field", ["record", "record_id", "candidate_pairs", "secret"])
def test_safe_log_builder_rejects_row_bearing_fields(field: str) -> None:
    sentinel = "SYNTHETIC-RESTRICTED-LOG-VALUE"
    with pytest.raises(SafeError) as caught:
        build_safe_log_event({"event": "unsafe", field: sentinel})
    assert field not in caught.value.render()
    assert sentinel not in caught.value.render()
