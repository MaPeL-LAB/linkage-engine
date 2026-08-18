from __future__ import annotations

from typing import Any

import pytest
import yaml

from mapel_linkage.configuration import load_config, load_config_text
from mapel_linkage.governance.errors import SafeError, SafeErrorCode
from tests.helpers import ROOT, valid_payload, yaml_text


def test_unknown_key_is_rejected_without_exposing_value() -> None:
    payload = valid_payload()
    sentinel = "SYNTHETIC-SENTINEL-UNKNOWN-FIELD"
    payload["project"]["sql"] = sentinel
    with pytest.raises(SafeError) as caught:
        load_config_text(yaml_text(payload), source_format="yaml")
    rendered = caught.value.render()
    assert caught.value.code == SafeErrorCode.CONFIG_VALIDATION
    assert "project.*" in rendered
    assert sentinel not in rendered


def test_raw_sql_and_callable_paths_are_not_configuration_fields() -> None:
    for key in ("sql", "python", "callable", "function_path", "module", "import"):
        payload = valid_payload()
        payload["blocking"]["rules"][0][key] = "restricted.example:operation"
        with pytest.raises(SafeError) as caught:
            load_config_text(yaml_text(payload), source_format="yaml")
        assert key not in str(caught.value) or "Locations" in str(caught.value)
        assert "restricted.example:operation" not in str(caught.value)


def test_unsafe_yaml_constructor_is_rejected() -> None:
    text = "!!python/object/apply:os.system ['echo unsafe']"
    with pytest.raises(SafeError) as caught:
        load_config_text(text, source_format="yaml")
    assert caught.value.code == SafeErrorCode.CONFIG_PARSE
    assert "echo unsafe" not in caught.value.render()


def test_supervised_model_requires_eligible_labels() -> None:
    payload = valid_payload()
    payload.pop("labels")
    with pytest.raises(SafeError) as caught:
        load_config_text(yaml_text(payload), source_format="yaml")
    assert caught.value.code == SafeErrorCode.CONFIG_VALIDATION
    assert "eligible" not in caught.value.render().lower()


def test_unverified_reference_is_not_training_truth() -> None:
    payload = valid_payload()
    payload["labels"] = {
        "source": {"kind": "unverified_reference", "path": "private/crosswalk.jsonl"},
        "permit_weak_labels_for_training": False,
        "permit_unverified_crosswalk": False,
    }
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


def test_invalid_numeric_value_is_hidden() -> None:
    payload = valid_payload()
    sentinel = "SYNTHETIC-SENTINEL-BUDGET"
    payload["runtime"]["maximum_candidate_pairs"] = sentinel
    with pytest.raises(SafeError) as caught:
        load_config_text(yaml_text(payload), source_format="yaml")
    assert sentinel not in caught.value.render()
    assert "runtime.maximum_candidate_pairs" in caught.value.render()


def test_json_loads_the_same_schema() -> None:
    import json

    payload: dict[str, Any] = valid_payload()
    loaded = load_config_text(json.dumps(payload), source_format="json")
    assert loaded.config.project.linkage_mode == "link_only"
    assert loaded.source_format == "json"


def test_output_fields_are_deny_by_default() -> None:
    payload = valid_payload()
    payload["outputs"]["permitted_fields"] = []
    loaded = load_config_text(yaml_text(payload), source_format="yaml")
    assert loaded.config.outputs.permitted_fields == ()


def test_output_variable_requires_explicit_restricted_permission() -> None:
    payload = valid_payload()
    payload["outputs"]["permitted_variable_values"] = ["label_text"]
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


def test_mapping_key_is_sanitized_from_validation_location() -> None:
    payload = valid_payload()
    sensitive_key = "SYNTHETIC-SECRET-DATASET-ID"
    payload["variables"][0]["source_columns"] = {sensitive_key: 123}
    with pytest.raises(SafeError) as caught:
        load_config_text(yaml_text(payload), source_format="yaml")
    assert sensitive_key not in caught.value.render()
    assert "variables.*.source_columns.*" in caught.value.render()


def test_assignment_solver_must_match_constraint() -> None:
    payload = valid_payload()
    payload["assignment"]["solver"] = "unconstrained"
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


def test_yaml_alias_limit_is_enforced() -> None:
    aliases = ", ".join("*item" for _ in range(65))
    text = f"anchor: &item safe\naliases: [{aliases}]\n"
    with pytest.raises(SafeError) as caught:
        load_config_text(text, source_format="yaml")
    assert caught.value.code == SafeErrorCode.CONFIG_PARSE


def test_configuration_size_limit_is_enforced() -> None:
    oversized = "x" * (2 * 1024 * 1024 + 1)
    with pytest.raises(SafeError) as caught:
        load_config_text(oversized, source_format="yaml")
    assert caught.value.code == SafeErrorCode.CONFIG_READ


def test_duplicate_yaml_keys_are_rejected() -> None:
    text = 'schema_version: "0.1"\nschema_version: "0.1"\n'
    with pytest.raises(SafeError) as caught:
        load_config_text(text, source_format="yaml")
    assert caught.value.code == SafeErrorCode.CONFIG_PARSE


def test_duplicate_json_keys_are_rejected() -> None:
    text = '{"schema_version":"0.1","schema_version":"0.1"}'
    with pytest.raises(SafeError) as caught:
        load_config_text(text, source_format="json")
    assert caught.value.code == SafeErrorCode.CONFIG_PARSE


def test_yaml_merge_keys_are_rejected() -> None:
    text = "base: &base {kind: exact}\nitem: {<<: *base, variable: label_text}\n"
    with pytest.raises(SafeError) as caught:
        load_config_text(text, source_format="yaml")
    assert caught.value.code == SafeErrorCode.CONFIG_PARSE


def test_deeply_nested_configuration_is_rejected() -> None:
    nested: object = "leaf"
    for _ in range(66):
        nested = {"node": nested}
    with pytest.raises(SafeError) as caught:
        load_config_text(yaml.safe_dump(nested), source_format="yaml")
    assert caught.value.code == SafeErrorCode.CONFIG_PARSE


def test_non_json_yaml_scalar_is_rejected() -> None:
    with pytest.raises(SafeError) as caught:
        load_config_text("value: 2026-08-16\n", source_format="yaml")
    assert caught.value.code == SafeErrorCode.CONFIG_PARSE


def test_non_finite_json_number_is_rejected() -> None:
    with pytest.raises(SafeError) as caught:
        load_config_text('{"value": NaN}', source_format="json")
    assert caught.value.code == SafeErrorCode.CONFIG_PARSE


def test_supervised_label_requirement_cannot_be_disabled() -> None:
    payload = valid_payload()
    payload["models"]["boosted_tree"]["require_verified_labels"] = False
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


def test_calibration_cannot_reference_a_disabled_pair_model() -> None:
    payload = valid_payload()
    payload["calibration"]["source_model"] = "xgb_pair_classifier"
    payload["models"]["boosted_tree"]["enabled"] = False
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


def test_calibration_may_reference_validation_selected_champion() -> None:
    payload = valid_payload()
    payload["calibration"]["source_model"] = "selected_champion"
    loaded = load_config_text(yaml_text(payload), source_format="yaml")
    assert loaded.config.calibration.source_model == "selected_champion"


def test_synthetic_entity_truth_must_cover_every_dataset() -> None:
    payload = valid_payload()
    payload["labels"]["source"]["entity_group_columns"].pop("source_b")
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


def test_similarity_comparison_rejects_difference_levels() -> None:
    payload = valid_payload()
    payload["comparisons"][0]["levels"] = [
        {"kind": "missing"},
        {"kind": "exact"},
        {"kind": "maximum_difference", "value": 1},
        {"kind": "else"},
    ]
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


def test_explicit_missingness_requires_missing_level() -> None:
    payload = valid_payload()
    payload["comparisons"][0]["levels"] = [
        {"kind": "exact"},
        {"kind": "threshold", "minimum": 0.90},
        {"kind": "else"},
    ]
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("runtime", "deterministic_mode"),
        ("assignment", "deterministic_tie_breaking"),
    ],
)
def test_deterministic_safeguards_cannot_be_disabled(section: str, field: str) -> None:
    payload = valid_payload()
    payload[section][field] = False
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


def test_all_validation_partitions_must_be_nonempty() -> None:
    payload = valid_payload()
    payload["validation"]["split"]["decision_fraction"] = 0.0
    payload["validation"]["split"]["training_fraction"] = 0.65
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


def test_boosted_tree_plan_has_bounded_deterministic_defaults() -> None:
    loaded = load_config_text(yaml_text(valid_payload()), source_format="yaml")
    model = loaded.config.models.boosted_tree
    assert model is not None
    assert model.implementation == "xgboost_classifier"
    assert model.n_jobs == 1
    assert model.deterministic_mode is True
    assert model.maximum_training_pairs == 100000
    assert model.hard_negative_fraction == 0.75


def test_boosted_tree_training_budget_cannot_exceed_runtime_pair_budget() -> None:
    payload = valid_payload()
    payload["models"]["boosted_tree"]["maximum_training_pairs"] = 100001
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


def test_ranking_training_budget_cannot_exceed_runtime_pair_budget() -> None:
    payload = valid_payload()
    payload["models"]["ranking"]["maximum_training_pairs"] = 100001
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


def test_selected_champion_requires_two_enabled_pair_models() -> None:
    payload = valid_payload()
    payload["models"]["boosted_tree"]["enabled"] = False
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


def test_boosted_tree_single_thread_safeguard_cannot_be_disabled() -> None:
    payload = valid_payload()
    payload["models"]["boosted_tree"]["n_jobs"] = 2
    with pytest.raises(SafeError):
        load_config_text(yaml_text(payload), source_format="yaml")


def test_generic_local_template_matches_live_schema() -> None:
    loaded = load_config(ROOT / "configs" / "templates" / "local_project.template.yaml")
    assert loaded.config.project.linkage_mode == "link_only"
    assert loaded.config.calibration.source_model == "selected_champion"
    assert loaded.config.model_selection.test_partition_may_select_model is False
