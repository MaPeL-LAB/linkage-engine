import pytest

from mapel_linkage.cli.main import main


def test_status(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status"]) == 0
    assert "pre-alpha" in capsys.readouterr().out


def test_target_command_fails_without_echoing_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["run", "--config", "private/project.yaml"]) == 2
    output = capsys.readouterr().out
    assert "ML-PREALPHA-001" in output
    assert "private/project.yaml" not in output
