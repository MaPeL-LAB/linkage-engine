from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mapel_linkage.cli.main import main
from mapel_linkage.governance.atomic import atomic_write_text as real_atomic_write_text
from tests.helpers import EXAMPLE_CONFIG, ROOT, valid_payload, yaml_text


def _cli_mode_payload(mode: str, constraint: str) -> dict[str, Any]:
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


def test_status_is_current_and_distinguishes_integration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["status"]) == 0
    output = capsys.readouterr().out
    assert "workflow_integrated=" in output
    assert "component_only=" in output
    assert "generated-synthetic two-source link_only" in output
    assert "configuration-driven all-model portfolio" in output
    assert "I1C synthetic mode dispatch is allow-listed only" in output
    assert "link_and_dedupe with one-to-one assignment" in output
    assert "development candidate" not in output
    assert "record-level" not in output


def test_status_json_is_machine_readable_and_safe(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["engine_version"]
    assert payload["summary"]["operational_validation"] == "not_established"
    assert payload["summary"]["merge_authority"] == "none"
    assert any(
        item["capability_id"] == "beta_calibration"
        and item["workflow_status"] == "workflow_integrated"
        for item in payload["capabilities"]
    )
    assert all(
        item["operational_validation"] == "not_established" for item in payload["capabilities"]
    )


def test_status_details_lists_component_and_workflow_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["status", "--details"]) == 0
    output = capsys.readouterr().out

    assert "lightgbm_pair_classifier\tcomponent=implemented\tworkflow=workflow_integrated" in output
    assert (
        "approved_recipe_inference\tcomponent=implemented\tworkflow=workflow_integrated" in output
    )
    assert (
        "single_source_deduplication\tcomponent=implemented\tworkflow=workflow_integrated" in output
    )
    assert "operational_validation=not_established" in output


def test_validate_config_succeeds_without_printing_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "validate-config",
                "--config",
                str(EXAMPLE_CONFIG),
                "--project-root",
                str(ROOT),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Configuration valid" in output
    assert str(EXAMPLE_CONFIG) not in output
    assert "record_key_a" not in output


def test_validate_config_error_does_not_echo_value_or_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "SYNTHETIC-SENTINEL-CLI-CONFIG"
    path = tmp_path / "private-config.yaml"
    path.write_text(f"schema_version: {sentinel}\n", encoding="utf-8")
    assert (
        main(
            [
                "validate-config",
                "--config",
                str(path),
                "--project-root",
                str(ROOT),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "ML-CONFIG-003" in captured.err
    assert sentinel not in captured.err
    assert str(path) not in captured.err


@pytest.mark.parametrize("command", ["run", "run-model-portfolio", "run-linkage-mode"])
def test_target_command_fails_without_echoing_config(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([command, "--config", "private/project.yaml"]) == 2
    captured = capsys.readouterr()
    assert "ML-CLI-002" in captured.err
    assert captured.out == ""
    assert "private/project.yaml" not in captured.err


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
def test_cli_reaches_each_allow_listed_synthetic_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    constraint: str,
) -> None:
    config_path = tmp_path / "mode.yaml"
    config_path.write_text(
        yaml_text(_cli_mode_payload(mode, constraint)),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "run-linkage-mode",
                "--config",
                str(config_path),
                "--project-root",
                str(tmp_path),
                "--synthetic-demo",
                "--entity-count",
                "100",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    summary = json.loads(output)
    assert summary["linkage_mode"] == mode
    assert summary["assignment_constraint"] == constraint
    assert summary["operational_validation"] == "not_established"
    assert summary["merge_authority"] == "none"
    for forbidden in ("A000000", "B000000", str(tmp_path), str(config_path)):
        assert forbidden not in output


def test_cli_rejects_non_orchestrated_mode_without_private_path_echo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "run-linkage-mode",
                "--config",
                str(EXAMPLE_CONFIG),
                "--project-root",
                str(ROOT),
                "--synthetic-demo",
                "--entity-count",
                "100",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "ML-MODE-010" in captured.err
    assert str(EXAMPLE_CONFIG) not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    ("failure_point", "mode", "constraint", "expected_code"),
    [
        ("model", "link_only", "many_to_one", "ML-MODE-014"),
        ("calibrator", "link_only", "many_to_one", "ML-MODE-015"),
        ("recipe", "link_only", "many_to_one", "ML-MODE-016"),
        ("mode_artifact", "dedupe_only", "unconstrained", "ML-MODE-016"),
    ],
)
def test_linkage_mode_cli_translates_artifact_filesystem_errors_without_path_echo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    mode: str,
    constraint: str,
    expected_code: str,
) -> None:
    config_path = tmp_path / "mode.yaml"
    config_path.write_text(yaml_text(_cli_mode_payload(mode, constraint)), encoding="utf-8")

    def fail_with_private_path(*_args: object, **_kwargs: object) -> None:
        raise OSError(f"restricted path: {tmp_path}")

    if failure_point == "model":
        monkeypatch.setattr(
            "mapel_linkage.pipeline.synthetic_mode_workflow.write_xgboost_artifact",
            fail_with_private_path,
        )
    elif failure_point == "calibrator":
        monkeypatch.setattr(
            "mapel_linkage.pipeline.synthetic_mode_workflow.write_calibrator_artifact",
            fail_with_private_path,
        )
    else:
        target_name = "recipe-v1.json" if failure_point == "recipe" else "mode-run-v1.json"

        def fail_selected_atomic_write(destination: Path, text: str) -> None:
            if destination.name == target_name:
                raise OSError(f"restricted path: {tmp_path}")
            real_atomic_write_text(destination, text)

        monkeypatch.setattr(
            "mapel_linkage.pipeline.synthetic_mode_workflow.atomic_write_text",
            fail_selected_atomic_write,
        )

    assert (
        main(
            [
                "run-linkage-mode",
                "--config",
                str(config_path),
                "--project-root",
                str(tmp_path),
                "--synthetic-demo",
                "--entity-count",
                "100",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert expected_code in captured.err
    assert "Traceback" not in captured.err
    assert str(tmp_path) not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("managed_root", ["artifacts", "data", "private"])
def test_linkage_mode_cli_rejects_symlinked_managed_roots_without_external_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    managed_root: str,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    (project_root / managed_root).symlink_to(external_root, target_is_directory=True)
    config_path = project_root / "mode.yaml"
    config_path.write_text(
        yaml_text(_cli_mode_payload("link_only", "many_to_one")),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "run-linkage-mode",
                "--config",
                str(config_path),
                "--project-root",
                str(project_root),
                "--synthetic-demo",
                "--entity-count",
                "100",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "ML-PATH-001" in captured.err
    assert "Traceback" not in captured.err
    assert str(project_root) not in captured.err
    assert str(external_root) not in captured.err
    assert not tuple(external_root.iterdir())
    assert captured.out == ""


def test_synthetic_entity_count_upper_bound_is_safe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "run",
                "--config",
                str(EXAMPLE_CONFIG),
                "--project-root",
                str(ROOT),
                "--synthetic-demo",
                "--entity-count",
                "100001",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "ML-CLI-003" in captured.err
    assert captured.out == ""


def test_model_portfolio_fold_bound_fails_before_config_access(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "run-model-portfolio",
                "--config",
                "private/project.yaml",
                "--synthetic-demo",
                "--k-folds",
                "11",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "ML-CLI-004" in captured.err
    assert "private/project.yaml" not in captured.err
    assert captured.out == ""


def test_emit_config_schema(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    destination = tmp_path / "schema.json"
    assert main(["emit-config-schema", "--output", str(destination)]) == 0
    output = capsys.readouterr().out
    assert "Schema written" in output
    assert str(destination) not in output
    assert destination.is_file()


def test_emit_config_schema_error_hides_destination(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "existing-directory"
    destination.mkdir()
    assert main(["emit-config-schema", "--output", str(destination)]) == 2
    captured = capsys.readouterr()
    assert "ML-CONFIG-005" in captured.err
    assert str(destination) not in captured.err


def test_cli_recommend_pipeline_similarity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "recommend-pipeline",
                "--config",
                str(EXAMPLE_CONFIG),
                "--project-root",
                str(ROOT),
                "--method",
                "similarity",
                "--registry-dir",
                str(tmp_path / "empty_registry"),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["out_of_distribution"] is True
    assert payload["synthetic_evidence_retrieved"] is False
    assert payload["recommendation"]["coverage_status"] == "structural_only"
    assert payload["recommendation"]["shortlist"][0]["pair_model_family"] == "fellegi_sunter"


def test_cli_adjudication_lifecycle_workflow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reviews_file = tmp_path / "reviews.jsonl"
    reviews_file.write_text(
        json.dumps(
            {
                "event_id": "rev_1",
                "pair_digest": "a" * 64,
                "left_record_key": "s1",
                "right_record_key": "t1",
                "decision": "match",
                "confidence": 0.95,
                "reviewer_id": "rev_user_1",
                "timestamp": "2026-08-19T10:00:00Z",
                "protocol_version": "v1.0",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    ledger_dest = tmp_path / "ledger.json"

    # 1. Test import-reviews
    assert (
        main(
            [
                "import-reviews",
                "--reviews",
                str(reviews_file),
                "--ledger-path",
                str(ledger_dest),
            ]
        )
        == 0
    )
    import_out = json.loads(capsys.readouterr().out)
    assert import_out["total_imported"] == 1
    assert ledger_dest.is_file()

    # 2. Test resolve-consensus
    assert (
        main(
            [
                "resolve-consensus",
                "--reviews",
                str(reviews_file),
                "--policy",
                "majority_vote",
            ]
        )
        == 0
    )
    cons_out = json.loads(capsys.readouterr().out)
    assert cons_out["total_pairs"] == 1
    assert cons_out["resolved_pairs"] == 1

    # 3. Test promote-labels
    labels_dest = tmp_path / "promoted_labels.json"
    assert (
        main(
            [
                "promote-labels",
                "--reviews",
                str(reviews_file),
                "--output",
                str(labels_dest),
                "--partition",
                "training",
            ]
        )
        == 0
    )
    prom_out = json.loads(capsys.readouterr().out)
    assert prom_out["retraining_triggered"] is False
    assert labels_dest.is_file()


def test_cli_recommend_pipeline_meta_ranker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "recommend-pipeline",
                "--config",
                str(EXAMPLE_CONFIG),
                "--project-root",
                str(ROOT),
                "--method",
                "meta-ranker",
                "--registry-dir",
                str(tmp_path / "empty_registry"),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["fallback_to_similarity"] is True
    assert payload["recommendation_authority"] == "advisory_only"


def test_cli_sample_review_queue(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    in_queue = tmp_path / "in_queue.jsonl"
    in_queue.write_text(
        json.dumps(
            {
                "relationship_id": "rel_1",
                "source_record_ref": "s1",
                "target_record_ref": "t1",
                "relationship_status": "review_required",
                "calibrated_probability": 0.52,
                "candidate_rank": 1,
                "probability_margin": 0.04,
                "review_reason_codes": ["review_probability_region"],
                "model_version": "v1.0",
                "decision_rule_id": "rule_1",
                "assignment_method": "ortools",
                "run_id": "run_01",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_queue = tmp_path / "out_queue.jsonl"
    assert (
        main(
            [
                "sample-review-queue",
                "--input-queue",
                str(in_queue),
                "--output",
                str(out_queue),
                "--strategy",
                "uncertainty",
                "--budget",
                "10",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["relationship_count"] == 1
    assert out_queue.is_file()
    sampled_lines = out_queue.read_text(encoding="utf-8").strip().splitlines()
    assert len(sampled_lines) == 1
    assert "active_learning_uncertainty" in json.loads(sampled_lines[0])["review_reason_codes"]
