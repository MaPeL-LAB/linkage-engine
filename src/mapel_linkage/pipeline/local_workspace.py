"""Safe local-workspace bootstrap and dependency diagnostics."""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

from mapel_linkage import __version__
from mapel_linkage.domain.errors import PipelineError

_REQUIRED_DISTRIBUTIONS = (
    "duckdb",
    "networkx",
    "numpy",
    "ortools",
    "pydantic",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "splink",
    "xgboost",
)
_LOCAL_DIRECTORIES = (
    "private/config",
    "private/labels",
    "private/adjudication",
    "private/outputs",
    "data/raw",
    "data/derived",
    "artifacts/models",
    "artifacts/runs",
    "artifacts/reports",
)


def _validated_workspace_root(directory: Path) -> Path:
    root = directory.resolve(strict=False)
    filesystem_root = Path(root.anchor).resolve(strict=False)
    if root in {filesystem_root, Path.home().resolve(strict=False)}:
        raise PipelineError("ML-PIPE-023", "A broad workspace root is not permitted.")
    return root


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True, slots=True)
class DoctorReport:
    engine_version: str
    python_version: str
    platform_name: str
    checks: tuple[DoctorCheck, ...]
    ready_for_synthetic_run: bool
    real_data_validation_status: str = "not_established"

    def safe_summary(self) -> dict[str, object]:
        return {
            "engine_version": self.engine_version,
            "python_version": self.python_version,
            "platform": self.platform_name,
            "ready_for_synthetic_run": self.ready_for_synthetic_run,
            "check_count": len(self.checks),
            "failed_check_count": sum(check.status != "pass" for check in self.checks),
            "checks": [
                {"name": check.name, "status": check.status, "detail": check.detail}
                for check in self.checks
            ],
            "real_data_validation_status": self.real_data_validation_status,
        }


def run_doctor(project_root: Path) -> DoctorReport:
    root = _validated_workspace_root(project_root)
    checks: list[DoctorCheck] = []
    python_ok = sys.version_info[:2] == (3, 12)
    checks.append(
        DoctorCheck(
            "python_3_12",
            "pass" if python_ok else "fail",
            "supported" if python_ok else "Python 3.12 is required",
        )
    )
    dependencies_ok = True
    for distribution in _REQUIRED_DISTRIBUTIONS:
        try:
            version = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            version = "unavailable"
            dependencies_ok = False
        checks.append(
            DoctorCheck(
                f"dependency_{distribution.lower().replace('-', '_')}",
                "pass" if version != "unavailable" else "fail",
                version,
            )
        )
    boundary_ok = True
    for relative in ("private", "data", "artifacts"):
        target = root / relative
        probe: Path | None = None
        try:
            if target.is_symlink() or not target.resolve(strict=False).is_relative_to(root):
                raise OSError
            target.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=".mapel-write-probe-",
                dir=target,
                delete=False,
            ) as handle:
                handle.write("synthetic-only-probe\n")
                probe = Path(handle.name)
        except OSError:
            boundary_ok = False
        finally:
            if probe is not None:
                with suppress(OSError):
                    probe.unlink()
    checks.append(
        DoctorCheck(
            "local_restricted_roots",
            "pass" if boundary_ok else "fail",
            "writable" if boundary_ok else "not writable",
        )
    )
    return DoctorReport(
        engine_version=__version__,
        python_version=platform.python_version(),
        platform_name=platform.system(),
        checks=tuple(checks),
        ready_for_synthetic_run=python_ok and dependencies_ok and boundary_ok,
    )


def initialise_local_project(directory: Path) -> tuple[Path, ...]:
    """Create only ignored operational directories and generic local guidance."""

    root = _validated_workspace_root(directory)
    created: list[Path] = []
    try:
        root.mkdir(parents=True, exist_ok=True)
        for relative in _LOCAL_DIRECTORIES:
            path = root / relative
            if path.is_symlink() or not path.resolve(strict=False).is_relative_to(root):
                raise OSError
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
        guidance = root / "private" / "LOCAL_ONLY_README.md"
        if guidance.is_symlink():
            raise OSError
        if not guidance.exists():
            guidance.write_text(
                "# Local-only Linkage Engine workspace\n\n"
                "This directory may contain authorised local configuration, verified labels, "
                "adjudication records, models, and outputs. Do not commit its contents.\n\n"
                "An unverified crosswalk is reference evidence only and must not be used as "
                "training, calibration, threshold-selection, or test truth.\n",
                encoding="utf-8",
            )
        created.append(guidance)
        manifest = root / "artifacts" / "local_workspace_manifest.json"
        if manifest.is_symlink():
            raise OSError
        if not manifest.exists():
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "0.1",
                        "engine_version": __version__,
                        "directory_count": len(_LOCAL_DIRECTORIES),
                        "contains_record_data": False,
                        "git_commit_authority": "none",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        created.append(manifest)
    except OSError:
        raise PipelineError(
            "ML-PIPE-014", "The local project workspace could not be created."
        ) from None
    return tuple(created)
