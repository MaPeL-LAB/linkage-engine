from __future__ import annotations

from typing import Any

import pytest

from mapel_linkage.configuration import compile_config, load_config_text
from mapel_linkage.governance.errors import SafeError
from tests.helpers import ROOT, valid_payload, yaml_text


def _mode_payload(mode: str, constraint: str) -> dict[str, Any]:
    payload = valid_payload()
    payload["project"]["linkage_mode"] = mode
    payload["project"]["assignment_constraint"] = constraint
    payload["assignment"]["constraint"] = constraint
    payload["assignment"]["solver"] = (
        "unconstrained" if constraint == "unconstrained" else "ortools_min_cost_flow"
    )
    payload["calibration"]["source_model"] = "xgb_pair_classifier"
    payload["mode_orchestration"] = {
        "artifact_schema_version": "1",
        "implementation": "synthetic_mode_v1",
        "pair_model_id": "xgb_pair_classifier",
    }
    if mode == "dedupe_only":
        payload["datasets"] = [payload["datasets"][0]]
        for variable in payload["variables"]:
            variable["source_columns"] = {"source_a": variable["source_columns"]["source_a"]}
        payload["labels"]["source"]["entity_group_columns"] = {"source_a": "synthetic_entity_id_a"}
        payload["labels"]["source"]["household_group_columns"] = {
            "source_a": "synthetic_household_id_a"
        }
    if mode in {"dedupe_only", "link_and_dedupe"}:
        payload["mode_orchestration"]["deduplication"] = {
            "algorithm": "clique",
            "minimum_probability": 0.75,
            "no_match_utility": 0.0,
            "maximum_cluster_size": 100,
            "maximum_candidate_edges": 100000,
            "deterministic_tie_breaking": True,
        }
    return payload


@pytest.mark.parametrize(
    ("mode", "constraint"),
    [
        ("link_only", "many_to_one"),
        ("link_only", "one_to_many"),
        ("link_only", "unconstrained"),
        ("dedupe_only", "unconstrained"),
        ("link_and_dedupe", "one_to_one"),
    ],
)
def test_supported_mode_combinations_compile_to_allow_list_dispatch(
    mode: str, constraint: str
) -> None:
    config = load_config_text(
        yaml_text(_mode_payload(mode, constraint)), source_format="yaml"
    ).config
    plan = compile_config(config, project_root=ROOT)
    assert plan.mode_dispatch_key == f"synthetic_mode_v1:{mode}:{constraint}"


def test_extended_mode_requires_explicit_orchestration_without_value_echo() -> None:
    payload = _mode_payload("link_only", "many_to_one")
    payload.pop("mode_orchestration")
    sentinel = "SYNTHETIC-SENTINEL-MODE"
    payload["project"]["project_id"] = sentinel
    with pytest.raises(SafeError) as caught:
        load_config_text(yaml_text(payload), source_format="yaml")
    assert sentinel not in caught.value.render()


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        (("project", "random_seed"), 1),
        (("mode_orchestration", "pair_model_id"), "fs_baseline"),
        (("mode_orchestration", "implementation"), "restricted.module:call"),
        (("mode_orchestration", "artifact_schema_version"), "2"),
    ],
)
def test_mode_orchestration_rejects_provenance_or_executable_drift(
    mutation: tuple[str, str], value: object
) -> None:
    payload = _mode_payload("dedupe_only", "unconstrained")
    payload[mutation[0]][mutation[1]] = value
    with pytest.raises(SafeError) as caught:
        load_config_text(yaml_text(payload), source_format="yaml")
    assert str(value) not in caught.value.render()


def test_deduplication_pair_budget_cannot_exceed_runtime_budget() -> None:
    payload = _mode_payload("dedupe_only", "unconstrained")
    payload["mode_orchestration"]["deduplication"]["maximum_candidate_edges"] = 100001
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


def test_mode_orchestration_rejects_assignment_utility_drift() -> None:
    payload = _mode_payload("link_only", "many_to_one")
    payload["assignment"]["no_match"]["utility"] = 0.25
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")
