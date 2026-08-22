#!/usr/bin/env python3
"""Verify M8 release controls while failing closed on release authorization."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "docs" / "release" / "RELEASE_READINESS_POLICY.json"
EXPECTED_BLOCKERS = (
    "api_stability_not_frozen",
    "artifact_migration_tool_not_implemented",
    "compatibility_matrix_incomplete",
    "external_security_review_not_approved",
    "licence_not_selected",
    "operational_validation_not_established",
    "rollback_drill_not_completed",
    "scale_evidence_not_completed",
)
EXPECTED_DOCUMENTS = (
    "docs/release/API_STABILITY_POLICY.md",
    "docs/release/ARTIFACT_MIGRATION_POLICY.md",
    "docs/release/COMPATIBILITY_MATRIX.md",
    "docs/release/ERROR_CODE_CATALOGUE.md",
    "docs/release/MODEL_CARDS.md",
    "docs/release/PRIVATE_RELEASE_AND_ROLLBACK.md",
    "docs/release/SCALE_BENCHMARK_POLICY.md",
    "docs/release/SECURITY_AND_DEPENDENCY_REVIEW.md",
)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _load_policy() -> dict[str, Any]:
    if POLICY_PATH.is_symlink() or not POLICY_PATH.is_file():
        raise ValueError("Release-readiness policy is unavailable or path-unsafe.")
    text = POLICY_PATH.read_text(encoding="utf-8")
    if len(text.encode("utf-8")) > 64 * 1024:
        raise ValueError("Release-readiness policy exceeds its aggregate bound.")
    payload = json.loads(text)
    if not isinstance(payload, dict) or text != _canonical_json(payload):
        raise ValueError("Release-readiness policy is not canonical JSON.")
    return payload


def _verify_release_controls() -> dict[str, object]:
    errors: list[str] = []
    try:
        policy = _load_policy()
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "invalid",
            "release_ready": False,
            "release_authorized": False,
            "operational_validity": "not_established",
            "blockers": EXPECTED_BLOCKERS,
            "error_count": 1,
            "errors": (str(error),),
        }

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    classifiers = project.get("classifiers", [])
    checks: tuple[tuple[bool, str], ...] = (
        (policy.get("schema_version") == "1", "unsupported release policy schema"),
        (
            policy.get("distribution_name") == project.get("name") == "mapel-linkage-engine",
            "distribution identity is inconsistent",
        ),
        (
            policy.get("package_version") == project.get("version"),
            "package version is inconsistent",
        ),
        (
            project.get("requires-python") == ">=3.12,<3.13",
            "Python compatibility contract is inconsistent",
        ),
        (
            "Private :: Do Not Upload" in classifiers,
            "private publication guard is absent",
        ),
        ("license" not in project and not (ROOT / "LICENSE").exists(), "licence gate drifted"),
        (policy.get("release_channel") == "private_candidate_only", "release channel drifted"),
        (policy.get("current_status") == "blocked", "release status is not fail-closed"),
        (policy.get("release_authorized") is False, "release authorization was asserted"),
        (policy.get("publication_authority") == "none", "publication authority was asserted"),
        (policy.get("deployment_authority") == "none", "deployment authority was asserted"),
        (
            policy.get("artifact_migration_authority") == "none",
            "artifact migration authority was asserted",
        ),
        (
            policy.get("operational_validity") == "not_established",
            "operational validity was overstated",
        ),
        (
            policy.get("automatic_publication") == "prohibited",
            "automatic publication is not prohibited",
        ),
        (
            policy.get("synthetic_testing_establishes_operational_validity") is False,
            "synthetic evidence was treated as operational validation",
        ),
        (
            tuple(policy.get("current_blockers", ())) == EXPECTED_BLOCKERS,
            "release blockers are incomplete or reordered",
        ),
        (
            tuple(policy.get("required_documents", ())) == EXPECTED_DOCUMENTS,
            "required release documents are incomplete or reordered",
        ),
        (
            tuple(policy.get("required_ci_checks", ())) == ("all-models", "quality"),
            "required CI checks are inconsistent",
        ),
    )
    errors.extend(message for passed, message in checks if not passed)

    for relative in EXPECTED_DOCUMENTS:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            errors.append("required release document is unavailable or path-unsafe")
            continue
        text = path.read_text(encoding="utf-8")
        if "| NA |" in text or "| N/A |" in text:
            errors.append("release tables must use '-' for unavailable results")

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    if "pip-audit --local --strict" not in workflow or "cyclonedx-json" not in workflow:
        errors.append("CI dependency-audit or SBOM control is absent")
    action_uses = re.findall(r"uses:\s*([^\s]+)", workflow)
    if not action_uses or any(
        re.fullmatch(r"[^@]+@[0-9a-f]{40}", action) is None for action in action_uses
    ):
        errors.append("GitHub Actions are not pinned to immutable commit SHAs")
    if "python scripts/verify_release_readiness.py --expect-blocked" not in workflow:
        errors.append("CI does not verify the fail-closed release policy")

    scale = policy.get("scale_benchmark")
    if not isinstance(scale, dict) or scale != {
        "default_workers": 10,
        "maximum_workers": 10,
        "output_classification": "aggregate_synthetic_only",
        "resumable": True,
    }:
        errors.append("scale-benchmark release policy is inconsistent")
    for relative in (
        "scripts/run_m8_scale_benchmarks.py",
        "scripts/run_m8_scale_benchmarks.sh",
    ):
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            errors.append("scale-benchmark entry point is unavailable or path-unsafe")

    return {
        "status": "invalid" if errors else "verified_blocked",
        "release_ready": False,
        "release_authorized": False,
        "operational_validity": "not_established",
        "blockers": EXPECTED_BLOCKERS,
        "error_count": len(errors),
        "errors": tuple(errors),
    }


def verify_release_controls() -> dict[str, object]:
    try:
        return _verify_release_controls()
    except Exception:
        return {
            "status": "invalid",
            "release_ready": False,
            "release_authorized": False,
            "operational_validity": "not_established",
            "blockers": EXPECTED_BLOCKERS,
            "error_count": 1,
            "errors": ("release control verification encountered an internal failure",),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expect-blocked",
        action="store_true",
        help="Pass only when Phase 1 controls are valid and release remains explicitly blocked.",
    )
    args = parser.parse_args()
    report = verify_release_controls()
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "invalid":
        return 1
    if args.expect_blocked:
        return 0
    print(
        "ERROR: Release is not authorized; use --expect-blocked only for governance verification.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
