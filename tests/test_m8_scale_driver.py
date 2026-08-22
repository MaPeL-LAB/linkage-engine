from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from tests.helpers import ROOT

PYTHON_DRIVER = ROOT / "scripts" / "run_m8_scale_benchmarks.py"
SHELL_DRIVER = ROOT / "scripts" / "run_m8_scale_benchmarks.sh"


def _dry_run(output: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PYTHON_DRIVER),
            "--python",
            sys.executable,
            "--output-dir",
            output,
            "--entity-counts",
            "100,250",
            "--repetitions",
            "1",
            "--workers",
            "10",
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_scale_plan_is_deterministic_bounded_and_no_write(tmp_path: Path) -> None:
    relative_output = f"artifacts/m8-dry-run-{tmp_path.name}"
    output = ROOT / relative_output
    assert not output.exists()

    first = _dry_run(relative_output)
    second = _dry_run(relative_output)

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    plan = json.loads(first.stdout)
    assert plan["benchmark_id"] == "m8_complete_synthetic_scale_v2"
    assert plan["workers"] == 10
    assert plan["maximum_workers"] == 10
    assert [item["entity_count"] for item in plan["cases"]] == [100, 250]
    assert plan["dry_run"] is True
    assert plan["contains_record_data"] is False
    assert plan["contains_identifiers"] is False
    assert plan["contains_candidate_pairs"] is False
    assert plan["contains_local_paths"] is False
    assert plan["operational_validity"] == "not_established"
    assert not output.exists()


def test_scale_plan_rejects_unsafe_scope_without_writing(tmp_path: Path) -> None:
    relative_output = f"artifacts/m8-invalid-{tmp_path.name}"
    output = ROOT / relative_output
    completed = subprocess.run(
        [
            sys.executable,
            str(PYTHON_DRIVER),
            "--python",
            sys.executable,
            "--output-dir",
            relative_output,
            "--entity-counts",
            "100",
            "--repetitions",
            "1",
            "--workers",
            "11",
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "failed closed" in completed.stderr
    assert str(ROOT) not in completed.stderr
    assert not output.exists()


def test_scale_plan_rejects_entity_counts_above_verified_budget_envelope(
    tmp_path: Path,
) -> None:
    relative_output = f"artifacts/m8-count-{tmp_path.name}"
    output = ROOT / relative_output
    completed = subprocess.run(
        [
            sys.executable,
            str(PYTHON_DRIVER),
            "--python",
            sys.executable,
            "--output-dir",
            relative_output,
            "--entity-counts",
            "501",
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "failed closed" in completed.stderr
    assert "501" not in completed.stderr
    assert str(ROOT) not in completed.stderr
    assert not output.exists()


def test_scale_plan_accepts_only_package_owned_synthetic_config(tmp_path: Path) -> None:
    relative_output = f"artifacts/m8-config-{tmp_path.name}"
    completed = subprocess.run(
        [
            sys.executable,
            str(PYTHON_DRIVER),
            "--python",
            sys.executable,
            "--config",
            "configs/examples/synthetic_all_models.yaml",
            "--output-dir",
            relative_output,
            "--entity-counts",
            "100",
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "failed closed" in completed.stderr
    assert "synthetic_all_models" not in completed.stderr
    assert not (ROOT / relative_output).exists()


def test_shell_scale_driver_has_valid_syntax_and_no_write_dry_run(tmp_path: Path) -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(SHELL_DRIVER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0
    relative_output = f"artifacts/m8-shell-{tmp_path.name}"
    output = ROOT / relative_output
    completed = subprocess.run(
        [
            str(SHELL_DRIVER),
            "--python",
            sys.executable,
            "--entity-counts",
            "100",
            "--repetitions",
            "1",
            "--workers",
            "10",
            "--output-dir",
            relative_output,
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Changed: none (dry-run planning only)." in completed.stdout
    assert "Next command: scripts/run_m8_scale_benchmarks.sh" in completed.stdout
    assert not output.exists()


def test_shell_scale_driver_accepts_no_forwarded_arguments(tmp_path: Path) -> None:
    python_stub = tmp_path / "python-stub"
    python_stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    python_stub.chmod(0o700)

    completed = subprocess.run(
        [str(SHELL_DRIVER), "--python", str(python_stub)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "unbound variable" not in completed.stderr
    assert "Changed: wrote or resumed aggregate synthetic scale evidence" in completed.stdout
    assert "Next command: python scripts/verify_release_readiness.py --expect-blocked" in (
        completed.stdout
    )


def test_scale_driver_executes_and_resumes_minimum_synthetic_case(tmp_path: Path) -> None:
    run_suffix = hashlib.sha256(str(tmp_path).encode("utf-8")).hexdigest()[:12]
    relative_output = f"artifacts/test_m8_scale_driver_runtime-{run_suffix}"
    output = ROOT / relative_output
    assert not output.exists()
    command = [
        sys.executable,
        str(PYTHON_DRIVER),
        "--python",
        sys.executable,
        "--output-dir",
        relative_output,
        "--entity-counts",
        "100",
        "--repetitions",
        "1",
        "--workers",
        "1",
    ]
    try:
        first = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert first.returncode == 0
        first_summary = json.loads(first.stdout)
        assert first_summary["newly_completed_case_count"] == 1
        assert first_summary["resumed_case_count"] == 0

        second = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert second.returncode == 0
        second_summary = json.loads(second.stdout)
        assert second_summary["newly_completed_case_count"] == 0
        assert second_summary["resumed_case_count"] == 1
        assert second_summary["summary_digest"] == first_summary["summary_digest"]

        plan_text = (output / "plan.json").read_text(encoding="utf-8")
        case_path = next((output / "cases").glob("*.json"))
        case_text = case_path.read_text(encoding="utf-8")
        summary_text = (output / "summary.json").read_text(encoding="utf-8")
        for text in (plan_text, case_text, summary_text):
            assert str(ROOT) not in text
            payload = json.loads(text)
            assert payload["contains_record_data"] is False
            assert payload["contains_identifiers"] is False
            assert payload["contains_candidate_pairs"] is False
            assert payload["contains_local_paths"] is False
            assert payload["operational_validity"] == "not_established"

        tampered_case = json.loads(case_text)
        tampered_case["contains_local_paths"] = True
        body = {key: value for key, value in tampered_case.items() if key != "report_digest"}
        tampered_case["report_digest"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        case_path.write_text(
            json.dumps(tampered_case, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rejected = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode == 2
        assert "failed closed" in rejected.stderr
        assert str(ROOT) not in rejected.stderr
    finally:
        if output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
