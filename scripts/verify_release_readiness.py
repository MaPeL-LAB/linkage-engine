#!/usr/bin/env python3
"""Verify M8 release controls while failing closed on release authorization."""

from __future__ import annotations

import argparse
import hashlib
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
    "compatibility_matrix_incomplete",
    "external_security_review_not_approved",
    "licence_not_selected",
    "operational_validation_not_established",
)
EXPECTED_DOCUMENTS = (
    "docs/release/API_STABILITY_POLICY.md",
    "docs/release/ARTIFACT_MIGRATION_EVIDENCE.md",
    "docs/release/ARTIFACT_MIGRATION_POLICY.md",
    "docs/release/COMPATIBILITY_MATRIX.md",
    "docs/release/ERROR_CODE_CATALOGUE.md",
    "docs/release/MODEL_CARDS.md",
    "docs/release/PRIVATE_RELEASE_AND_ROLLBACK.md",
    "docs/release/ROLLBACK_DRILL_EVIDENCE.md",
    "docs/release/SCALE_BENCHMARK_EVIDENCE_V2.md",
    "docs/release/SCALE_BENCHMARK_POLICY.md",
    "docs/release/SECURITY_AND_DEPENDENCY_REVIEW.md",
)
EXPECTED_SCALE_BENCHMARK: dict[str, object] = {
    "benchmark_id": "m8_complete_synthetic_scale_v2",
    "default_workers": 10,
    "evidence_review": "approved_development_envelope",
    "maximum_entity_count": 500,
    "maximum_workers": 10,
    "output_classification": "aggregate_synthetic_only",
    "plan_digest": "442d59215bf6572979bb96ce1b3881c88b7974e627bd731a083d20b2eb05a48d",
    "resumable": True,
    "summary_digest": "4d4c015b1f1e289c57516a76b6b3730d277a761e2310b53be1f76fab651f7465",
}
EXPECTED_ARTIFACT_MIGRATION: dict[str, object] = {
    "artifact_kind": "run_manifest",
    "conflicting_replay": "rejected",
    "dry_run_required": True,
    "evidence_review": "implemented_and_verified",
    "exact_replay": "idempotent",
    "plan_schema_version": "1",
    "report_classification": "aggregate_only",
    "source_overwrite": False,
    "source_schema_version": "0.1",
    "target_schema_version": "1",
    "transformation": "run_manifest_0_1_to_1",
}
EXPECTED_ROLLBACK_DRILL: dict[str, object] = {
    "approval_reference": "codex_task_owner_approval_2026-08-22",
    "authority_contract_digest": (
        "4de937aca72fc9e4275ee8cb1ab03979948bc1c3c159eefdc1f1e70abe5c6f4c"
    ),
    "baseline_commit": "5050626583236fe1a7778eabc363a31764385285",
    "baseline_package_tree_digest": (
        "b106ede07796592fa8c799400d25e42ab31ef801a8995b32ef5b10db566cbf5c"
    ),
    "baseline_wheel_digest": ("492c9c30c76059e316e9d477792ddef796669024483bc8b368e2d0a2dbd9b475"),
    "candidate_commit": "81762675996eae77ccb16210936d630f092a3e7b",
    "candidate_package_tree_digest": (
        "ecbc7fa8cddea0de30f0caf33f19a3a82be598b6fd11ba3d2dc360c0a953e221"
    ),
    "candidate_wheel_digest": ("974e76df6f6f4783eeec0f9f4f499f565417cdd8aaea8d2d653ca483f024a926"),
    "check_count": 13,
    "config_digest": "9c4b3b630316cb6802aaddcd61e9bb712184274aec06988981c9d5bb71f3eb06",
    "constraints_digest": "a527f3013c3e076e804f757e17b6d3c64eaf4a514c8706ac17ea38e83f017423",
    "data_policy": "synthetic_only",
    "dependency_environment_digest": (
        "711ee5cc4b885cab1d997074c0ca17b9ef1f8f69b041f0e484a8c2bcb5661508"
    ),
    "dependency_layer_independently_isolated": False,
    "drill_id": "m8_synthetic_rollback_v1",
    "evidence_review": "completed_and_verified",
    "failed_candidate_evidence_overwritten": False,
    "implementation_digest": ("248c80f27ac61b0c205d1b25574db391f955415c64a865b3494ca6c26686e15d"),
    "installation_surface": "isolated_venv_with_verified_dependency_layer",
    "operational_validity": "not_established",
    "passed_check_count": 13,
    "removed_candidate_file_count": 1,
    "report_classification": "aggregate_only",
    "report_digest": "fd664bebd3d8e8d328812ea2dffaee80894838d112d4d7dcd0b2d9f389771f89",
    "summary_digest": "0515cd83ec9b9e9abd91ebe7cb660c6e6d788ca1e7642bc65a7cf46927ab0763",
    "synthetic_testing_establishes_operational_validity": False,
}


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
    if not isinstance(scale, dict) or scale != EXPECTED_SCALE_BENCHMARK:
        errors.append("scale-benchmark release policy is inconsistent")

    migration = policy.get("artifact_migration")
    if not isinstance(migration, dict) or migration != EXPECTED_ARTIFACT_MIGRATION:
        errors.append("artifact-migration release policy is inconsistent")

    rollback = policy.get("rollback_drill")
    if not isinstance(rollback, dict) or rollback != EXPECTED_ROLLBACK_DRILL:
        errors.append("rollback-drill release policy is inconsistent")

    rollback_source = ROOT / "scripts" / "run_m8_rollback_drill.py"
    if rollback_source.is_symlink() or not rollback_source.is_file():
        errors.append("rollback-drill implementation is unavailable or path-unsafe")
    else:
        rollback_source_digest = hashlib.sha256(rollback_source.read_bytes()).hexdigest()
        if rollback_source_digest != EXPECTED_ROLLBACK_DRILL["implementation_digest"]:
            errors.append("rollback-drill implementation digest drifted from reviewed evidence")
        rollback_source_text = rollback_source.read_text(encoding="utf-8")
        for required_literal in (
            'DRILL_ID = "m8_synthetic_rollback_v1"',
            "SEED = 20260816",
            "ENTITY_COUNT = 100",
            '"data_policy": "synthetic_only"',
            '"release_authority": "none"',
            '"operational_validity": "not_established"',
            '"failed_candidate_evidence_overwritten": False',
        ):
            if required_literal not in rollback_source_text:
                errors.append("rollback-drill implementation boundary is incomplete")

    rollback_evidence_path = ROOT / "docs" / "release" / "ROLLBACK_DRILL_EVIDENCE.md"
    if rollback_evidence_path.is_file() and not rollback_evidence_path.is_symlink():
        rollback_evidence = rollback_evidence_path.read_text(encoding="utf-8")
        for key in (
            "candidate_commit",
            "baseline_commit",
            "candidate_wheel_digest",
            "baseline_wheel_digest",
            "implementation_digest",
            "report_digest",
            "summary_digest",
        ):
            if f"`{EXPECTED_ROLLBACK_DRILL[key]}`" not in rollback_evidence:
                errors.append("rollback-drill evidence binding is incomplete")
        for required_claim in (
            "All 13 checks passed",
            "does not authorize release",
            "not independently reinstalled",
            "`operational_validity=not_established`",
        ):
            if required_claim not in rollback_evidence:
                errors.append("rollback-drill evidence boundary is incomplete")

    migration_evidence_path = ROOT / "docs" / "release" / "ARTIFACT_MIGRATION_EVIDENCE.md"
    if migration_evidence_path.is_file() and not migration_evidence_path.is_symlink():
        migration_evidence = migration_evidence_path.read_text(encoding="utf-8")
        for required_claim in (
            "`run_manifest_0_1_to_1`",
            "`source_schema_version=0.1`",
            "`target_schema_version=1`",
            "dry-run plan is required",
            "does not authorize release",
            "`operational_validity=not_established`",
        ):
            if required_claim not in migration_evidence:
                errors.append("artifact-migration evidence boundary is incomplete")

    migration_source = ROOT / "src" / "mapel_linkage" / "governance" / "artifact_migration.py"
    if migration_source.is_symlink() or not migration_source.is_file():
        errors.append("artifact-migration implementation is unavailable or path-unsafe")
    else:
        migration_text = migration_source.read_text(encoding="utf-8")
        for required_literal in (
            '_RUN_MANIFEST_KIND: Literal["run_manifest"] = "run_manifest"',
            '_RUN_MANIFEST_SOURCE_VERSION: Literal["0.1"] = "0.1"',
            '_RUN_MANIFEST_TARGET_VERSION: Literal["1"] = "1"',
            '"ML-MIGRATE-007"',
            '"ML-MIGRATE-008"',
            'migration_authority: Literal["none"] = "none"',
            'release_authority: Literal["none"] = "none"',
            'operational_validity: Literal["not_established"] = "not_established"',
        ):
            if required_literal not in migration_text:
                errors.append("artifact-migration implementation drifted from reviewed evidence")

    cli_source = (ROOT / "src" / "mapel_linkage" / "cli" / "main.py").read_text(encoding="utf-8")
    if '"migrate-artifact"' not in cli_source or '"--dry-run"' not in cli_source:
        errors.append("artifact-migration CLI planning boundary is absent")

    evidence_path = ROOT / "docs" / "release" / "SCALE_BENCHMARK_EVIDENCE_V2.md"
    if evidence_path.is_file() and not evidence_path.is_symlink():
        evidence = evidence_path.read_text(encoding="utf-8")
        for key in ("benchmark_id", "plan_digest", "summary_digest"):
            if f"`{EXPECTED_SCALE_BENCHMARK[key]}`" not in evidence:
                errors.append("scale-benchmark evidence binding is incomplete")
        for required_claim in (
            "explicitly approved by the repository owner on 2026-08-22",
            "does not authorize release",
            "`operational_validity=not_established`",
        ):
            if required_claim not in evidence:
                errors.append("scale-benchmark evidence boundary is incomplete")

    scale_driver = (ROOT / "scripts" / "run_m8_scale_benchmarks.py").read_text(encoding="utf-8")
    for required_literal in (
        'DEFAULT_COUNTS = "100,200,300,400,500"',
        "DEFAULT_WORKERS = 10",
        "MAXIMUM_WORKERS = 10",
        "MAXIMUM_ENTITY_COUNT = 500",
        'BENCHMARK_ID = "m8_complete_synthetic_scale_v2"',
    ):
        if required_literal not in scale_driver:
            errors.append("scale-benchmark implementation drifted from reviewed evidence")
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
