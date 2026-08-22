from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.run_m8_rollback_drill import (
    BASELINE_COMMIT,
    CANDIDATE_COMMIT,
    RollbackDrillError,
    _validate_retained_artifacts,
    _wheel_package_state,
)
from tests.helpers import ROOT

DRIVER = ROOT / "scripts" / "run_m8_rollback_drill.py"


def _dry_run(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DRIVER), "--dry-run", *extra],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_rollback_plan_is_deterministic_aggregate_only_and_no_write(tmp_path: Path) -> None:
    output = ROOT / "artifacts" / f"rollback-plan-{tmp_path.name}"
    first = _dry_run("--output-dir", str(output.relative_to(ROOT)))
    second = _dry_run("--output-dir", str(output.relative_to(ROOT)))

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert not output.exists()
    plan = json.loads(first.stdout)
    assert plan["candidate_commit"] == CANDIDATE_COMMIT
    assert plan["baseline_commit"] == BASELINE_COMMIT
    assert plan["constraints_match"] is True
    assert plan["authority_contracts_unchanged"] is True
    assert plan["candidate_config_digest"] == plan["baseline_config_digest"]
    assert plan["candidate_constraints_digest"] == plan["baseline_constraints_digest"]
    assert plan["data_policy"] == "synthetic_only"
    assert plan["report_classification"] == "aggregate_only"
    assert plan["contains_record_data"] is False
    assert plan["contains_identifiers"] is False
    assert plan["contains_candidate_pairs"] is False
    assert plan["contains_local_paths"] is False
    assert plan["release_authority"] == "none"
    assert plan["operational_validity"] == "not_established"
    assert str(ROOT) not in first.stdout


def test_rollback_plan_rejects_mutable_or_unsafe_inputs_without_writing(tmp_path: Path) -> None:
    mutable = _dry_run("--candidate-commit", "main")
    assert mutable.returncode == 2
    assert "full immutable commit digest" in mutable.stderr
    assert str(ROOT) not in mutable.stderr

    outside = _dry_run("--output-dir", f"../rollback-{tmp_path.name}")
    assert outside.returncode == 2
    assert "repository-relative" in outside.stderr
    assert str(ROOT) not in outside.stderr


def test_rollback_execution_requires_explicit_approval() -> None:
    completed = subprocess.run(
        [sys.executable, str(DRIVER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "requires explicit approval" in completed.stderr
    assert str(ROOT) not in completed.stderr


def test_wheel_package_state_excludes_metadata_and_bytecode(tmp_path: Path) -> None:
    wheel = tmp_path / "synthetic.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("mapel_linkage/__init__.py", "__version__ = '1'\n")
        archive.writestr("mapel_linkage/module.py", "VALUE = 1\n")
        archive.writestr("mapel_linkage/__pycache__/module.pyc", b"ignored")
        archive.writestr("package.dist-info/METADATA", "ignored\n")

    state = _wheel_package_state(wheel)

    assert state["file_count"] == 2
    assert state["names"] == (
        "mapel_linkage/__init__.py",
        "mapel_linkage/module.py",
    )
    assert isinstance(state["digest"], str)
    assert len(state["digest"]) == 64


def test_retained_artifact_validation_fails_closed_on_tampering(tmp_path: Path) -> None:
    candidate = tmp_path / "retained" / "candidate" / "candidate.whl"
    baseline = tmp_path / "retained" / "baseline" / "baseline.whl"
    candidate.parent.mkdir(parents=True)
    baseline.parent.mkdir(parents=True)
    candidate.write_bytes(b"candidate")
    baseline.write_bytes(b"baseline")
    payload = {
        "candidate_wheel_digest": hashlib.sha256(b"candidate").hexdigest(),
        "baseline_wheel_digest": hashlib.sha256(b"baseline").hexdigest(),
    }
    _validate_retained_artifacts(tmp_path, payload)

    candidate.write_bytes(b"tampered")
    with pytest.raises(RollbackDrillError, match="Retained rollback artifacts") as captured:
        _validate_retained_artifacts(tmp_path, payload)
    assert captured.value.code == "ML-ROLLBACK-009"
