from __future__ import annotations

from pathlib import Path

import pytest

from mapel_linkage.cli.main import main
from tests.helpers import EXAMPLE_CONFIG, ROOT


def test_status(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status"]) == 0
    output = capsys.readouterr().out
    assert "M2D" in output
    assert "record-level" not in output


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


def test_target_command_fails_without_echoing_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["run", "--config", "private/project.yaml"]) == 2
    captured = capsys.readouterr()
    assert "ML-PREALPHA-001" in captured.err
    assert captured.out == ""
    assert "private/project.yaml" not in captured.err


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
