"""Verify packaging, wheel integrity, metadata, entrypoints, and distribution hygiene."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SUBPACKAGES = {
    "adjudication",
    "anchors",
    "assignment",
    "benchmarking",
    "calibration",
    "candidate_generation",
    "cli",
    "clustering",
    "comparisons",
    "configuration",
    "decisions",
    "domain",
    "governance",
    "io",
    "models",
    "pipeline",
    "preprocessing",
    "profiling",
    "recommendation",
    "synthetic",
    "validation",
}

FORBIDDEN_PATTERNS = (
    "tests/",
    "tests\\",
    ".csv",
    ".parquet",
    ".db",
    ".duckdb",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
)


def verify_distribution(dist_dir: Path | None = None) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) if dist_dir is None else dist_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build sdist and wheel
        cmd = [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--outdir",
            str(output_dir),
            str(ROOT),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"Distribution build failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )

        wheels = list(output_dir.glob("*.whl"))
        sdists = list(output_dir.glob("*.tar.gz"))

        if not wheels:
            raise RuntimeError("No .whl package found in build output.")
        if not sdists:
            raise RuntimeError("No .tar.gz package found in build output.")

        target_wheel = wheels[0]
        target_sdist = sdists[0]

        # 2. Inspect Wheel
        wheel_subpackages: set[str] = set()
        wheel_entry_points_valid = False
        wheel_forbidden_files: list[str] = []

        with zipfile.ZipFile(target_wheel, "r") as zf:
            namelist = zf.namelist()
            for name in namelist:
                for forbidden in FORBIDDEN_PATTERNS:
                    if forbidden in name:
                        wheel_forbidden_files.append(name)

                if name.startswith("mapel_linkage/"):
                    parts = name.split("/")
                    if len(parts) >= 2 and parts[1]:
                        wheel_subpackages.add(parts[1])

                if name.endswith("entry_points.txt"):
                    ep_text = zf.read(name).decode("utf-8")
                    if "mapel-linkage = mapel_linkage.cli.main:main" in ep_text:
                        wheel_entry_points_valid = True

        if wheel_forbidden_files:
            raise RuntimeError(f"Forbidden files in wheel: {wheel_forbidden_files[:5]}")

        missing_subpackages = EXPECTED_SUBPACKAGES - wheel_subpackages
        if missing_subpackages:
            raise RuntimeError(f"Missing expected subpackages in wheel: {missing_subpackages}")

        if not wheel_entry_points_valid:
            raise RuntimeError(
                "Entrypoint 'mapel-linkage = mapel_linkage.cli.main:main' not found in wheel."
            )

        # 3. Inspect sdist
        with tarfile.open(target_sdist, "r:gz") as tf:
            sdist_names = tf.getnames()
            sdist_forbidden = [
                n
                for n in sdist_names
                if any(f in n for f in (".parquet", ".db", ".duckdb", ".pkl", ".pt"))
            ]
            if sdist_forbidden:
                raise RuntimeError(f"Forbidden files in sdist: {sdist_forbidden[:5]}")

        return {
            "status": "verified",
            "wheel_name": target_wheel.name,
            "wheel_size_bytes": target_wheel.stat().st_size,
            "sdist_name": target_sdist.name,
            "sdist_size_bytes": target_sdist.stat().st_size,
            "subpackages_verified": sorted(wheel_subpackages),
            "entry_points_valid": True,
            "forbidden_data_files_detected": 0,
        }


def main() -> int:
    try:
        report = verify_distribution()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ERROR: Distribution verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
