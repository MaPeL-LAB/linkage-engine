from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mapel_linkage.cli.main import main
from tests.helpers import ROOT


def test_plan_advisor_corpus_emits_aggregate_design(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["plan-advisor-corpus", "--shards", "32"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert captured.err == ""
    assert payload["design"]["family_count"] == 64
    assert payload["design"]["instance_count"] == 280
    assert payload["design"]["meta_training_family_count"] == 40
    assert payload["design"]["conformal_family_count"] == 8
    assert payload["design"]["locked_evaluation_family_count"] == 8
    assert payload["design"]["ood_holdout_family_count"] == 8
    assert payload["readiness"]["execution_ready"] is True
    assert payload["readiness"]["readiness_schema_version"] == "2"
    assert payload["readiness"]["planned_replicates_per_instance"] == 5
    assert payload["readiness"]["expected_run_count"] == 9_800
    assert payload["readiness"]["missing_required_adapter_run_count"] == 4_200
    assert payload["readiness"]["advisor_evidence_ready"] is False
    assert payload["shard_plan"]["shard_count"] == 32
    assert payload["design"]["operational_validity"] == "not_established"


def test_run_advisor_corpus_requires_approval_before_path_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")

    assert (
        main(
            [
                "run-advisor-corpus",
                "--project-root",
                str(project),
                "--registry-dir",
                "private/advisor",
                "--shard-index",
                "0",
                "--approval-reference",
                "synthetic-owner",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ML-BENCH-CORPUS-002" in captured.err
    assert not (project / "private").exists()


def test_audit_advisor_corpus_fails_closed_without_creating_a_registry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")

    assert (
        main(
            [
                "audit-advisor-corpus",
                "--project-root",
                str(project),
                "--registry-dir",
                "private/missing",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ML-BENCH-CORPUS-004" in captured.err
    assert str(project) not in captured.err
    assert not (project / "private").exists()


def test_run_advisor_corpus_rejects_absolute_and_symlink_paths_without_leakage(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "registry_link").symlink_to(outside, target_is_directory=True)
    approval_reference = "private-approval-sentinel"

    for registry_path in (str(outside), "registry_link/nested"):
        assert (
            main(
                [
                    "run-advisor-corpus",
                    "--project-root",
                    str(project),
                    "--registry-dir",
                    registry_path,
                    "--shard-index",
                    "0",
                    "--approve-execution",
                    "--approval-reference",
                    approval_reference,
                ]
            )
            == 2
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "ML-BENCH-CORPUS-003" in captured.err
        assert str(project) not in captured.err
        assert str(outside) not in captured.err
        assert approval_reference not in captured.err


def test_external_driver_is_syntax_valid_and_dry_run_does_not_write(tmp_path: Path) -> None:
    driver = ROOT / "scripts" / "run_advisor_corpus.sh"
    assert "private/benchmark_registry/advisor_v2_execution_v2" in driver.read_text(
        encoding="utf-8"
    )
    syntax = subprocess.run(
        ["bash", "-n", str(driver)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    registry_relative = "private/advisor_corpus_dry_run_test"
    registry = ROOT / registry_relative
    assert not registry.exists()
    completed = subprocess.run(
        [
            str(driver),
            "--dry-run",
            "--python",
            sys.executable,
            "--registry-dir",
            registry_relative,
            "--shards",
            "32",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"family_count": 64' in completed.stdout
    assert '"instance_count": 280' in completed.stdout
    assert "Changed: none (dry-run planning only)." in completed.stdout
    assert str(ROOT) not in completed.stdout
    assert sys.executable not in completed.stdout
    assert not registry.exists()
