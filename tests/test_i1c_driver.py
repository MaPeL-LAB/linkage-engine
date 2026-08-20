from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from tests.helpers import ROOT

DRIVER = ROOT / "scripts" / "run_i1c_linkage_modes.sh"


def test_i1c_driver_rejects_executable_without_python_marker() -> None:
    completed = subprocess.run(
        [str(DRIVER), "--dry-run", "--python", "/usr/bin/true"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "did not identify as Python 3.12" in completed.stderr
    assert completed.stdout == ""


def test_i1c_driver_dry_run_uses_private_path_placeholder() -> None:
    completed = subprocess.run(
        [str(DRIVER), "--dry-run", "--python", sys.executable],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--python '<PYTHON_3_12_PATH>' --full" in completed.stdout
    assert sys.executable not in completed.stdout
    assert "Changed: none (dry-run verification plan only)." in completed.stdout


def test_i1c_driver_trap_reports_safely_quoted_actual_command(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/bin/sh\n"
        'if [ "${1:-}" = "-c" ]; then\n'
        '  case "${2:-}" in\n'
        "    *mapel-i1c-python*) echo 'mapel-i1c-python:3.12'; exit 0 ;;\n"
        f'    *) exec {shlex.quote(sys.executable)} "$@" ;;\n'
        "  esac\n"
        "fi\n"
        "exit 9\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o700)

    completed = subprocess.run(
        [str(DRIVER), "--python", str(fake_python)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 9
    assert "command: <python> -m ruff check" in completed.stderr
    assert str(fake_python) not in completed.stderr
    assert "command: I1C lint" not in completed.stderr


def test_i1c_driver_can_dry_run_the_full_gate_without_execution() -> None:
    completed = subprocess.run(
        [str(DRIVER), "--dry-run", "--full", "--python", sys.executable],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "[verify] complete test suite" in completed.stdout
    assert "[verify] distribution verification" in completed.stdout
    assert "[verify] external distribution source copy" in completed.stdout
    assert r"\<verification-cache\>/distribution-source" in completed.stdout
    assert "data/private" not in completed.stdout
    assert "--python '<PYTHON_3_12_PATH>' --full" in completed.stdout
    assert sys.executable not in completed.stdout


def test_i1c_driver_snapshots_only_repository_candidate_content() -> None:
    source = DRIVER.read_text(encoding="utf-8")

    assert '"ls-files", "-co", "--exclude-standard", "-z"' in source
    assert "--ignored=matching" not in source
    assert 'root.rglob("*")' not in source
    assert '"private",' in source
    assert "external distribution source copy" in source
