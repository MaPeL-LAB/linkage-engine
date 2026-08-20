from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mapel_linkage.configuration import ExecutionPlan, compile_config, load_config, load_config_text
from mapel_linkage.configuration import compiler as compiler_module
from mapel_linkage.governance.errors import SafeError, SafeErrorCode
from tests.helpers import ROOT, valid_payload, yaml_text


def _plan(payload: dict[str, Any] | None = None) -> ExecutionPlan:
    source_payload = valid_payload() if payload is None else payload
    loaded = load_config_text(yaml_text(source_payload), source_format="yaml")
    return compile_config(loaded.config, project_root=ROOT)


def test_example_compiles_to_stable_digest() -> None:
    first = _plan()
    second = _plan()
    assert first.configuration_digest == second.configuration_digest
    assert first.registry_digest == second.registry_digest
    assert first.dataset_count == 2
    assert "data/synthetic" not in repr(first)


def test_execution_plan_dataset_paths_are_immutable() -> None:
    plan = _plan()
    with pytest.raises(TypeError):
        plan.dataset_paths["source_a"] = Path("other")  # type: ignore[index]


def test_execution_plan_configuration_mappings_are_immutable() -> None:
    plan = _plan()
    with pytest.raises(TypeError):
        plan.config.variables[0].source_columns["source_a"] = "other"  # type: ignore[index]


def test_remote_uri_is_rejected_without_echo() -> None:
    payload = valid_payload()
    sentinel = "https://example.invalid/SYNTHETIC-SENTINEL"
    payload["datasets"][0]["path"] = sentinel
    loaded = load_config_text(yaml_text(payload), source_format="yaml")
    with pytest.raises(SafeError) as caught:
        compile_config(loaded.config, project_root=ROOT)
    assert caught.value.code == SafeErrorCode.PATH_POLICY
    assert sentinel not in caught.value.render()


def test_input_path_outside_configured_roots_is_rejected() -> None:
    payload = valid_payload()
    payload["datasets"][0]["path"] = "outside/source.parquet"
    loaded = load_config_text(yaml_text(payload), source_format="yaml")
    with pytest.raises(SafeError) as caught:
        compile_config(loaded.config, project_root=ROOT)
    assert caught.value.code == SafeErrorCode.PATH_POLICY


def test_configured_root_cannot_escape_host_boundary(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["privacy"]["allowed_input_roots"] = [str(tmp_path.parent)]
    loaded = load_config_text(yaml_text(payload), source_format="yaml")
    with pytest.raises(SafeError):
        compile_config(loaded.config, project_root=ROOT)


def test_project_root_cannot_be_self_approved() -> None:
    payload = valid_payload()
    payload["privacy"]["allowed_input_roots"] = ["."]
    loaded = load_config_text(yaml_text(payload), source_format="yaml")
    with pytest.raises(SafeError):
        compile_config(loaded.config, project_root=ROOT)


def test_label_source_path_uses_input_policy() -> None:
    payload = valid_payload()
    payload["labels"] = {
        "source": {"kind": "unverified_reference", "path": "outside/crosswalk.jsonl"},
        "permit_weak_labels_for_training": False,
        "permit_unverified_crosswalk": False,
    }
    payload["models"]["boosted_tree"]["enabled"] = False
    payload["models"]["ranking"]["enabled"] = False
    payload["calibration"]["source_model"] = "fs_baseline"
    loaded = load_config_text(yaml_text(payload), source_format="yaml")
    with pytest.raises(SafeError):
        compile_config(loaded.config, project_root=ROOT)


def test_path_with_ambiguous_whitespace_is_rejected() -> None:
    payload = valid_payload()
    payload["datasets"][0]["path"] = " data/synthetic/source_a.parquet"
    loaded = load_config_text(yaml_text(payload), source_format="yaml")
    with pytest.raises(SafeError):
        compile_config(loaded.config, project_root=ROOT)


def test_digest_changes_when_structural_configuration_changes() -> None:
    first = _plan()
    payload = valid_payload()
    payload["runtime"]["maximum_candidate_pairs"] += 1
    second = _plan(payload)
    assert first.configuration_digest != second.configuration_digest


def test_compiler_resolves_every_plural_model_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        compiler_module,
        "resolve_operation",
        lambda registry, key: calls.append((registry, key)),
    )
    config = load_config(ROOT / "configs/examples/synthetic_all_models.yaml").config

    compile_config(config, project_root=ROOT)

    assert {
        ("pair_model", "splink_duckdb"),
        ("pair_model", "xgboost_classifier"),
        ("pair_model", "lightgbm_classifier"),
        ("pair_model", "pytorch_pair_mlp"),
        ("pair_model", "stacking_logistic"),
        ("ranker", "xgboost_ranker"),
        ("ranker", "lightgbm_ranker"),
    }.issubset(calls)


@pytest.mark.parametrize(
    "unsafe_path",
    ["~/source.parquet", "//server/share/source.parquet", "file:data/source.parquet"],
)
def test_ambiguous_or_nonlocal_paths_are_rejected(unsafe_path: str) -> None:
    payload = valid_payload()
    payload["datasets"][0]["path"] = unsafe_path
    loaded = load_config_text(yaml_text(payload), source_format="yaml")
    with pytest.raises(SafeError):
        compile_config(loaded.config, project_root=ROOT)
