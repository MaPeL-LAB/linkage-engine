from __future__ import annotations

import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

import pytest

from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.pipeline.local_workspace import initialise_local_project, run_doctor


def test_local_project_initializer_creates_only_generic_ignored_workspace(tmp_path: Path) -> None:
    created = initialise_local_project(tmp_path)
    expected = {
        "private/config",
        "private/labels",
        "private/adjudication",
        "private/outputs",
        "data/raw",
        "data/derived",
        "artifacts/models",
        "artifacts/runs",
        "artifacts/reports",
    }
    assert expected.issubset({str(path.relative_to(tmp_path)) for path in created if path.is_dir()})
    guidance = (tmp_path / "private" / "LOCAL_ONLY_README.md").read_text(encoding="utf-8")
    assert "unverified crosswalk" in guidance.lower()
    assert not list(tmp_path.rglob("*.jsonl"))
    assert not list(tmp_path.rglob("*.parquet"))


def test_doctor_uses_aggregate_checks_without_exposing_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(importlib_metadata, "version", lambda _: "1.0")
    report = run_doctor(tmp_path)
    assert report.ready_for_synthetic_run is True
    summary = report.safe_summary()
    assert summary["failed_check_count"] == 0
    assert str(tmp_path) not in repr(report)
    assert str(tmp_path) not in str(summary)


def test_local_project_initializer_preserves_existing_guidance(tmp_path: Path) -> None:
    guidance = tmp_path / "private" / "LOCAL_ONLY_README.md"
    guidance.parent.mkdir(parents=True)
    guidance.write_text("operator-owned guidance\n", encoding="utf-8")

    initialise_local_project(tmp_path)

    assert guidance.read_text(encoding="utf-8") == "operator-owned guidance\n"


def test_local_project_initializer_rejects_broad_root() -> None:
    with pytest.raises(PipelineError, match="ML-PIPE-023"):
        initialise_local_project(Path.home())
