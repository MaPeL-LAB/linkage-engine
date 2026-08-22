"""Fail-closed registry, governance, artifact, and CLI tests for advisor-v3.1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from mapel_linkage.benchmarking.advisor_catalogue import AdvisorFamilyRole
from mapel_linkage.benchmarking.advisor_v3_catalogue import (
    advisor_v3_family_roles,
    build_advisor_v3_geometry_coherence,
)
from mapel_linkage.benchmarking.advisor_v31_remediation import (
    AdvisorV31RemediationApproval,
    AdvisorV31RemediationReadinessManifest,
    advisor_v31_analysis_provenance_digest,
    build_advisor_v31_protocol_amendment,
)
from mapel_linkage.benchmarking.registry import BenchmarkRegistry
from mapel_linkage.cli.main import main
from mapel_linkage.recommendation.distance_v3 import (
    MechanismAwareTaskMetaFeatureVector,
)
from mapel_linkage.recommendation.qualification_v3 import (
    AdvisorV3QualificationPolicy,
    AdvisorV31FamilyUtilityEvidence,
    AdvisorV31QualificationApproval,
    AdvisorV31QualificationArtifact,
    AdvisorV31QualificationReadinessArtifact,
    build_advisor_v31_qualification_readiness,
    deserialize_advisor_v31_qualification_artifact,
    evaluate_advisor_v31_aggregate_evidence,
    load_advisor_v31_qualification_artifact,
    serialize_advisor_v31_qualification_artifact,
    write_advisor_v31_qualification_artifact,
)
from mapel_linkage.recommendation.qualification_v31 import (
    build_advisor_v31_qualification_readiness_from_registries,
    load_advisor_v31_qualification_governance,
)
from mapel_linkage.recommendation.utility import REQUIRED_ADVISOR_RECIPE_TOKENS
from tests.helpers import ROOT


def _canonical(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def _aggregate_inputs() -> tuple[
    AdvisorV31QualificationReadinessArtifact,
    AdvisorV31QualificationApproval,
    tuple[AdvisorV31FamilyUtilityEvidence, ...],
    dict[str, AdvisorFamilyRole],
    dict[str, MechanismAwareTaskMetaFeatureVector],
]:
    digest = "a" * 64
    readiness = build_advisor_v31_qualification_readiness(
        amendment_digest=digest,
        source_execution_approval_digest=digest,
        source_execution_provenance_digest=digest,
        source_v3_readiness_digest=digest,
        source_registry_snapshot_digest=digest,
        analysis_provenance_digest=digest,
        remediation_approval_digest=digest,
        remediation_readiness_digest=digest,
        advisor_evidence_ready=True,
    )
    approval = AdvisorV31QualificationApproval(
        approval_reference="owner_approved_v31_synthetic",
        human_approved=True,
        locked_evaluation_access_authorized=True,
        ood_evaluation_access_authorized=True,
        amendment_digest=digest,
        source_execution_approval_digest=digest,
        source_execution_provenance_digest=digest,
        source_v3_readiness_digest=digest,
        source_registry_snapshot_digest=digest,
        analysis_provenance_digest=digest,
        remediation_approval_digest=digest,
        remediation_readiness_digest=digest,
        policy_digest=AdvisorV3QualificationPolicy().policy_digest,
        evaluation_algorithm_digest=readiness.evaluation_algorithm_digest,
    )
    roles = dict(advisor_v3_family_roles())
    vectors = {
        family_id: MechanismAwareTaskMetaFeatureVector(
            linkage_mode="link_only",
            assignment_constraint="one_to_one",
            label_volume_class="sparse",
            script_variation_rate=(index % 4) / 3,
            punctuation_variation_rate=(index % 5) / 4,
            tokenization_variation_rate=(index % 3) / 2,
            missingness_mean=(index % 7) / 6,
            missingness_asymmetry=(index % 6) / 5,
            frequency_concentration=(index % 8) / 7,
            candidate_ambiguity_scale=(index % 9) / 8,
            duplicate_signature_rate=(index % 10) / 9,
            planned_training_label_budget_scale=(index % 11) / 10,
        )
        for index, family_id in enumerate(sorted(roles))
    }
    evidence = tuple(
        AdvisorV31FamilyUtilityEvidence(
            family_id=family_id,
            family_role=role,
            recipe_token=token,
            mean_utility=0.25 + 0.1 * token_index + (family_index % 5) / 100,
            run_count=20,
            evidence_digest=hashlib.sha256(f"{family_id}:{token}".encode()).hexdigest(),
        )
        for family_index, (family_id, role) in enumerate(sorted(roles.items()))
        if role != "ood_holdout"
        for token_index, token in enumerate(REQUIRED_ADVISOR_RECIPE_TOKENS)
    )
    return readiness, approval, evidence, roles, vectors


def _qualification_artifact() -> AdvisorV31QualificationArtifact:
    readiness, approval, evidence, roles, vectors = _aggregate_inputs()
    report = evaluate_advisor_v31_aggregate_evidence(
        readiness=readiness,
        approval=approval,
        evidence=evidence,
        role_by_family=roles,
        family_vectors=vectors,
    )
    return AdvisorV31QualificationArtifact(report=report, report_digest=report.report_digest)


def _write_remediation_governance(registry: BenchmarkRegistry) -> tuple[str, str]:
    amendment = build_advisor_v31_protocol_amendment()
    analysis_digest = advisor_v31_analysis_provenance_digest()
    source_digest = "b" * 64
    geometry_digest = build_advisor_v3_geometry_coherence().geometry_coherence_digest
    approval = AdvisorV31RemediationApproval(
        approval_reference="owner_approved_v31_remediation",
        human_approved=True,
        amendment_digest=amendment.amendment_digest,
        source_execution_approval_digest=source_digest,
        source_execution_provenance_digest=source_digest,
        source_v3_readiness_digest=source_digest,
        source_registry_snapshot_digest=source_digest,
        analysis_provenance_digest=analysis_digest,
        recomputed_geometry_coherence_digest=geometry_digest,
    )
    readiness = AdvisorV31RemediationReadinessManifest(
        amendment_digest=amendment.amendment_digest,
        source_execution_approval_digest=source_digest,
        source_execution_provenance_digest=source_digest,
        source_v3_preregistration_digest=source_digest,
        source_v3_readiness_digest=source_digest,
        source_registry_snapshot_digest=source_digest,
        analysis_provenance_digest=analysis_digest,
        remediation_approval_digest=approval.approval_digest,
        recomputed_geometry_coherence_digest=geometry_digest,
        source_completed_run_count=11_760,
        source_family_manifest_count=84,
        source_instance_manifest_count=336,
        source_catalogue_integrity_checked=True,
        source_failure_sidecar_count=6_730,
        successful_qualification_required_family_count=72,
        successful_qualification_required_evidence_cell_count=1_440,
        successful_qualification_required_adapter_run_count=4_320,
        failed_qualification_required_adapter_run_count=0,
        missing_qualification_required_adapter_run_count=0,
        complete_ood_geometry_family_count=12,
        completed_ood_diagnostic_adapter_run_count=720,
        non_success_ood_diagnostic_adapter_run_count=10,
        ineligible_nonrequired_recipe_run_count=6_720,
        advisor_evidence_ready=True,
    )
    governance = registry.root_directory / "governance"
    governance.mkdir(parents=True, exist_ok=True)
    (governance / f"amendment.v3.1.{amendment.amendment_digest}.json").write_text(
        _canonical(amendment), encoding="utf-8"
    )
    (governance / f"approval.v3.1.{approval.approval_digest}.json").write_text(
        _canonical(approval), encoding="utf-8"
    )
    (governance / f"readiness.v3.1.{readiness.readiness_digest}.json").write_text(
        _canonical(readiness), encoding="utf-8"
    )
    return approval.approval_digest, readiness.readiness_digest


def test_v31_artifact_is_canonical_semantic_and_path_safe(tmp_path: Path) -> None:
    artifact = _qualification_artifact()
    text = serialize_advisor_v31_qualification_artifact(artifact)

    assert deserialize_advisor_v31_qualification_artifact(text) == artifact
    assert "family.advisor_v3" not in json.dumps(artifact.safe_summary(), sort_keys=True)
    payload = json.loads(text)
    payload["report"]["stage2_similarity"]["mean_regret"] = 0.5
    payload["report_digest"] = hashlib.sha256(
        json.dumps(payload["report"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValidationError, match="derived regret metrics"):
        deserialize_advisor_v31_qualification_artifact(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        write_advisor_v31_qualification_artifact(linked / "qualification.json", artifact)
    assert not (outside / "qualification.json").exists()

    path = tmp_path / "qualification.json"
    write_advisor_v31_qualification_artifact(path, artifact)
    assert load_advisor_v31_qualification_artifact(path) == artifact
    write_advisor_v31_qualification_artifact(path, artifact)


def test_committed_v31_qualification_artifact_is_canonical_and_fail_closed() -> None:
    artifact = load_advisor_v31_qualification_artifact(
        ROOT / "docs" / "evidence" / "advisor_v31_qualification_20260822.json"
    )

    assert artifact.artifact_digest == (
        "b633eac62b463f871ec2c34dec4a8481a81346e7acf3a112f83151ad33342fac"
    )
    assert artifact.report.qualification_status == "not_qualified"
    assert artifact.report.stage2_qualified is True
    assert artifact.report.stage3_qualified is False
    assert artifact.report.failed_gate_codes == ("stage3_regret_improvement",)
    assert artifact.report.fallback_to_similarity_required is True
    assert artifact.report.operational_validity == "not_established"


def test_v31_governance_selection_is_exact_and_current(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path / "remediation")
    approval_digest, readiness_digest = _write_remediation_governance(registry)

    selected = load_advisor_v31_qualification_governance(
        remediation_registry=registry,
        committed_amendment_path=(
            ROOT / "docs" / "evidence" / "advisor_v31_protocol_amendment_20260821.json"
        ),
        remediation_approval_digest=approval_digest,
        remediation_readiness_digest=readiness_digest,
    )

    assert selected.remediation_approval.approval_digest == approval_digest
    assert selected.remediation_readiness.readiness_digest == readiness_digest
    with pytest.raises(FileNotFoundError, match="unavailable"):
        load_advisor_v31_qualification_governance(
            remediation_registry=registry,
            committed_amendment_path=(
                ROOT / "docs" / "evidence" / "advisor_v31_protocol_amendment_20260821.json"
            ),
            remediation_approval_digest="c" * 64,
            remediation_readiness_digest=readiness_digest,
        )


def test_v31_governance_only_reaudit_binds_recomputed_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mapel_linkage.recommendation.qualification_v31 as module

    source = BenchmarkRegistry(tmp_path / "source")
    remediation = BenchmarkRegistry(tmp_path / "remediation")
    approval_digest, readiness_digest = _write_remediation_governance(remediation)
    source_digest = "b" * 64
    inspection = SimpleNamespace(
        approval=SimpleNamespace(
            approval_digest=source_digest,
            execution_provenance_digest=source_digest,
        ),
        readiness=SimpleNamespace(readiness_digest=source_digest),
    )
    monkeypatch.setattr(module, "inspect_frozen_advisor_v3_corpus", lambda _registry: inspection)
    monkeypatch.setattr(
        module,
        "frozen_advisor_v3_snapshot_digest",
        lambda _registry, _inspection: source_digest,
    )

    readiness = build_advisor_v31_qualification_readiness_from_registries(
        source_registry=source,
        remediation_registry=remediation,
        committed_amendment_path=(
            ROOT / "docs" / "evidence" / "advisor_v31_protocol_amendment_20260821.json"
        ),
        remediation_approval_digest=approval_digest,
        remediation_readiness_digest=readiness_digest,
    )

    assert readiness.qualification_evaluation_accessed is False
    assert readiness.locked_evaluation_access_authorized is False
    assert readiness.ood_evaluation_access_authorized is False
    assert readiness.remediation_approval_digest == approval_digest


@pytest.mark.parametrize("approval_flag", [(), ("--approve-locked-evaluation",)])
def test_v31_cli_requires_both_separate_approvals_before_path_access(
    approval_flag: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        [
            "qualify-advisor-v31",
            "--source-registry-dir",
            "private/benchmark_registry/source",
            "--remediation-registry-dir",
            "private/benchmark_registry/remediation",
            "--remediation-approval-digest",
            "a" * 64,
            "--remediation-readiness-digest",
            "b" * 64,
            "--approval-reference",
            "owner_approved_v31_synthetic",
            *approval_flag,
        ]
    )

    assert result == 2
    assert "Separate explicit approval" in capsys.readouterr().err
