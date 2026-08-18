#!/usr/bin/env python3
"""Build a privacy-safe local handoff bundle from the verified source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, root: Path) -> None:
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_handoff(root: Path, *, replace_build_output: bool = False) -> Path:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("ERROR: Python 3.12 is required for the tested handoff envelope.")
    root = root.resolve(strict=True)
    if root in {Path(root.anchor).resolve(), Path.home().resolve()}:
        raise SystemExit("ERROR: A broad project root is not permitted.")
    reports = root / "artifacts" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    dist = root / "dist"
    if dist.resolve(strict=False).parent != root:
        raise SystemExit("ERROR: The build output path is outside the project root.")
    if dist.exists() and any(dist.iterdir()) and not replace_build_output:
        raise SystemExit(
            "ERROR: dist is not empty; rerun with --replace-build-output to replace it."
        )
    if dist.exists() and replace_build_output:
        shutil.rmtree(dist)

    _run([sys.executable, "scripts/generate_config_schema.py"], root=root)
    _run([sys.executable, "scripts/generate_repository_manifest.py"], root=root)
    _run([sys.executable, "scripts/verify_repository.py"], root=root)
    _run([sys.executable, "-m", "pytest"], root=root)
    _run([sys.executable, "-m", "build"], root=root)
    _run([sys.executable, "scripts/verify_repository.py", "--distribution", "dist"], root=root)

    audit_path = reports / "dependency-audit.json"
    sbom_path = reports / "software-bill-of-materials.cdx.json"
    _run(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--local",
            "--strict",
            "--progress-spinner",
            "off",
            "--format",
            "json",
            "--output",
            str(audit_path),
        ],
        root=root,
    )
    _run(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--local",
            "--strict",
            "--progress-spinner",
            "off",
            "--format",
            "cyclonedx-json",
            "--output",
            str(sbom_path),
        ],
        root=root,
    )

    distributions = sorted(path for path in dist.iterdir() if path.is_file())
    checksum_path = dist / "SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in distributions),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "0.1",
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "distribution_files": [
            {"name": path.name, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in distributions
        ],
        "checksum_file": checksum_path.name,
        "dependency_audit": str(audit_path.relative_to(root)),
        "software_bill_of_materials": str(sbom_path.relative_to(root)),
        "contains_record_data": False,
        "contains_operational_configuration": False,
        "package_publication_authority": "none",
        "real_data_validation_status": "not_established",
        "warning": (
            "Synthetic testing establishes software behaviour only; it does not validate "
            "linkage accuracy on real populations or systems."
        ),
    }
    destination = reports / "local_handoff_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--replace-build-output",
        action="store_true",
        help="Explicitly replace a non-empty project-local dist directory.",
    )
    args = parser.parse_args()
    destination = build_handoff(
        args.project_root.resolve(strict=True),
        replace_build_output=bool(args.replace_build_output),
    )
    print(f"Local handoff built: manifest={destination.relative_to(args.project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
