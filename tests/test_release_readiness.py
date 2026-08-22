from __future__ import annotations

import json
import subprocess
import sys

from tests.helpers import ROOT

POLICY = ROOT / "docs" / "release" / "RELEASE_READINESS_POLICY.json"
SCALE_EVIDENCE = ROOT / "docs" / "release" / "SCALE_BENCHMARK_EVIDENCE_V2.md"
MIGRATION_EVIDENCE = ROOT / "docs" / "release" / "ARTIFACT_MIGRATION_EVIDENCE.md"
VERIFY = ROOT / "scripts" / "verify_release_readiness.py"
GENERATE_ERRORS = ROOT / "scripts" / "generate_error_code_catalogue.py"


def test_release_policy_is_canonical_truthful_and_explicitly_blocked() -> None:
    text = POLICY.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert text == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert payload["current_status"] == "blocked"
    assert payload["release_authorized"] is False
    assert payload["publication_authority"] == "none"
    assert payload["deployment_authority"] == "none"
    assert payload["operational_validity"] == "not_established"
    assert payload["synthetic_testing_establishes_operational_validity"] is False
    assert payload["scale_benchmark"]["default_workers"] == 10
    assert payload["scale_benchmark"]["maximum_workers"] == 10
    assert payload["scale_benchmark"]["maximum_entity_count"] == 500
    assert payload["scale_benchmark"]["evidence_review"] == "approved_development_envelope"
    assert payload["scale_benchmark"]["benchmark_id"] == "m8_complete_synthetic_scale_v2"
    assert "licence_not_selected" in payload["current_blockers"]
    assert "scale_evidence_not_completed" not in payload["current_blockers"]
    assert "artifact_migration_tool_not_implemented" not in payload["current_blockers"]
    assert "operational_validation_not_established" in payload["current_blockers"]
    assert len(payload["current_blockers"]) == 6

    migration = payload["artifact_migration"]
    assert migration == {
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

    evidence = SCALE_EVIDENCE.read_text(encoding="utf-8")
    assert f"`{payload['scale_benchmark']['plan_digest']}`" in evidence
    assert f"`{payload['scale_benchmark']['summary_digest']}`" in evidence
    assert "does not authorize release" in evidence
    assert "`operational_validity=not_established`" in evidence

    migration_evidence = MIGRATION_EVIDENCE.read_text(encoding="utf-8")
    assert "`run_manifest_0_1_to_1`" in migration_evidence
    assert "dry-run plan is required" in migration_evidence
    assert "does not authorize release" in migration_evidence
    assert "`operational_validity=not_established`" in migration_evidence


def test_release_control_verifier_passes_only_in_expected_blocked_mode() -> None:
    expected = subprocess.run(
        [sys.executable, str(VERIFY), "--expect-blocked"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert expected.returncode == 0
    report = json.loads(expected.stdout)
    assert report["status"] == "verified_blocked"
    assert report["release_ready"] is False
    assert report["error_count"] == 0

    release_attempt = subprocess.run(
        [sys.executable, str(VERIFY)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert release_attempt.returncode == 1
    assert "Release is not authorized" in release_attempt.stderr
    assert json.loads(release_attempt.stdout)["release_authorized"] is False


def test_error_catalogue_and_release_documentation_are_current() -> None:
    completed = subprocess.run(
        [sys.executable, str(GENERATE_ERRORS), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    catalogue = (ROOT / "docs" / "release" / "ERROR_CODE_CATALOGUE.md").read_text(encoding="utf-8")
    assert "Catalogue count:" in catalogue
    assert "`ML-CONFIG-001`" in catalogue

    for path in (ROOT / "docs" / "release").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert "| NA |" not in text
        assert "| N/A |" not in text


def test_ci_and_precommit_verify_release_controls() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    hooks = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    command = "python scripts/verify_release_readiness.py --expect-blocked"
    assert command in workflow
    assert command in hooks
    assert "python scripts/generate_error_code_catalogue.py --check" in workflow
    assert "python scripts/generate_error_code_catalogue.py --check" in hooks


def test_milestone_status_uses_canonical_integrated_capability_truth() -> None:
    milestones = (ROOT / "docs" / "implementation" / "MILESTONES.md").read_text(encoding="utf-8")

    for milestone in ("M3", "M5", "M6"):
        line = next(line for line in milestones.splitlines() if line.startswith(f"| {milestone} "))
        assert "Workflow integrated" in line
        assert "workflow integration pending" not in line
    m7_line = next(line for line in milestones.splitlines() if line.startswith("| M7 "))
    assert "Multi-source workflow integrated" in m7_line
    m8_line = next(line for line in milestones.splitlines() if line.startswith("| M8 "))
    assert "Phase 1 controls integrated; release blocked" in m8_line
