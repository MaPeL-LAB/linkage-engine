"""Fail-closed registry-to-artifact workflow for advisor-v3.1 qualification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from mapel_linkage.benchmarking.advisor_catalogue import AdvisorFamilyRole
from mapel_linkage.benchmarking.advisor_v3_catalogue import (
    advisor_v3_family_roles,
    build_advisor_v3_generator,
)
from mapel_linkage.benchmarking.advisor_v31_remediation import (
    AdvisorV31ProtocolAmendmentManifest,
    AdvisorV31RemediationApproval,
    AdvisorV31RemediationReadinessManifest,
    FrozenAdvisorV3CorpusInspection,
    advisor_v31_analysis_provenance_digest,
    frozen_advisor_v3_snapshot_digest,
    inspect_frozen_advisor_v3_corpus,
    load_committed_advisor_v31_protocol_amendment,
)
from mapel_linkage.benchmarking.registry import BenchmarkRegistry
from mapel_linkage.recommendation.distance_v3 import (
    extract_advisor_v3_family_meta_features,
)
from mapel_linkage.recommendation.meta_learning import (
    aggregate_family_recipe_evidence,
)
from mapel_linkage.recommendation.qualification_v3 import (
    AdvisorV3QualificationPolicy,
    AdvisorV31FamilyUtilityEvidence,
    AdvisorV31QualificationApproval,
    AdvisorV31QualificationArtifact,
    AdvisorV31QualificationReadinessArtifact,
    build_advisor_v31_qualification_readiness,
    evaluate_advisor_v31_aggregate_evidence,
)
from mapel_linkage.recommendation.utility import AdvisorRecipeToken

_MAX_GOVERNANCE_BYTES = 4 * 1024 * 1024
_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_ALLOWED_GOVERNANCE_PREFIXES = (
    "amendment.v3.1.",
    "approval.v3.1.",
    "readiness.v3.1.",
)
_TOKEN_BY_RECIPE_ID: dict[str, AdvisorRecipeToken] = {
    "recipe.fellegi_sunter_reference": "fellegi_sunter",
    "recipe.xgboost_classifier": "xgboost_classifier",
    "recipe.xgboost_ranker": "xgboost_ranker",
}


def _canonical_text(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def _require_digest(value: str, *, label: str) -> str:
    if _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Advisor-v3.1 {label} is not a canonical digest.")
    return value


def _load_exact_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_GOVERNANCE_BYTES:
            raise FileNotFoundError
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            "The selected advisor-v3.1 governance artifact is unavailable."
        ) from None
    except (OSError, UnicodeError):
        raise ValueError(
            "The selected advisor-v3.1 governance artifact could not be read safely."
        ) from None
    model = model_type.model_validate_json(text)
    if text != _canonical_text(model):
        raise ValueError("The selected advisor-v3.1 governance artifact is not canonical.")
    return model


def _validate_governance_only_registry(registry: BenchmarkRegistry) -> None:
    root = registry.root_directory
    governance = root / "governance"
    if root.is_symlink() or governance.is_symlink() or not governance.is_dir():
        raise ValueError("The advisor-v3.1 remediation registry layout is path-unsafe.")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Advisor-v3.1 remediation registries cannot contain symbolic links.")
        if not path.is_file():
            continue
        if not path.is_relative_to(governance):
            raise ValueError("Advisor-v3.1 remediation registries must be governance-only.")
        if (
            path.parent != governance
            or path.suffix != ".json"
            or not path.name.startswith(_ALLOWED_GOVERNANCE_PREFIXES)
        ):
            raise ValueError("Advisor-v3.1 remediation governance contains an unknown artifact.")


@dataclass(frozen=True, slots=True)
class AdvisorV31QualificationGovernance:
    """Exact selected governance artifacts for one later qualification request."""

    amendment: AdvisorV31ProtocolAmendmentManifest
    remediation_approval: AdvisorV31RemediationApproval
    remediation_readiness: AdvisorV31RemediationReadinessManifest


def load_advisor_v31_qualification_governance(
    *,
    remediation_registry: BenchmarkRegistry,
    committed_amendment_path: Path,
    remediation_approval_digest: str,
    remediation_readiness_digest: str,
) -> AdvisorV31QualificationGovernance:
    """Load exact digest-selected remediation governance without outcome access."""

    _validate_governance_only_registry(remediation_registry)
    approval_digest = _require_digest(
        remediation_approval_digest, label="remediation approval digest"
    )
    readiness_digest = _require_digest(
        remediation_readiness_digest, label="remediation readiness digest"
    )
    committed = load_committed_advisor_v31_protocol_amendment(committed_amendment_path)
    governance = remediation_registry.root_directory / "governance"
    amendment = _load_exact_model(
        governance / f"amendment.v3.1.{committed.amendment_digest}.json",
        AdvisorV31ProtocolAmendmentManifest,
    )
    approval = _load_exact_model(
        governance / f"approval.v3.1.{approval_digest}.json",
        AdvisorV31RemediationApproval,
    )
    readiness = _load_exact_model(
        governance / f"readiness.v3.1.{readiness_digest}.json",
        AdvisorV31RemediationReadinessManifest,
    )
    if amendment != committed or approval.approval_digest != approval_digest:
        raise ValueError("Advisor-v3.1 remediation governance digest selection is inconsistent.")
    if (
        readiness.readiness_digest != readiness_digest
        or readiness.remediation_approval_digest != approval_digest
        or not readiness.advisor_evidence_ready
    ):
        raise ValueError("Advisor-v3.1 remediation readiness failed closed.")
    shared = (
        "amendment_digest",
        "source_execution_approval_digest",
        "source_execution_provenance_digest",
        "source_v3_readiness_digest",
        "source_registry_snapshot_digest",
        "analysis_provenance_digest",
        "recomputed_geometry_coherence_digest",
    )
    if any(getattr(approval, name) != getattr(readiness, name) for name in shared):
        raise ValueError("Advisor-v3.1 remediation approval and readiness are misbound.")
    if approval.amendment_digest != amendment.amendment_digest:
        raise ValueError("Advisor-v3.1 remediation approval does not bind the amendment.")
    if readiness.analysis_provenance_digest != advisor_v31_analysis_provenance_digest():
        raise ValueError(
            "Advisor-v3.1 remediation readiness is stale for the current evaluator source."
        )
    return AdvisorV31QualificationGovernance(
        amendment=amendment,
        remediation_approval=approval,
        remediation_readiness=readiness,
    )


def _build_readiness_and_inspection(
    *,
    source_registry: BenchmarkRegistry,
    remediation_registry: BenchmarkRegistry,
    committed_amendment_path: Path,
    remediation_approval_digest: str,
    remediation_readiness_digest: str,
) -> tuple[AdvisorV31QualificationReadinessArtifact, FrozenAdvisorV3CorpusInspection]:
    governance = load_advisor_v31_qualification_governance(
        remediation_registry=remediation_registry,
        committed_amendment_path=committed_amendment_path,
        remediation_approval_digest=remediation_approval_digest,
        remediation_readiness_digest=remediation_readiness_digest,
    )
    inspection = inspect_frozen_advisor_v3_corpus(source_registry)
    remediation = governance.remediation_readiness
    if (
        inspection.approval.approval_digest != remediation.source_execution_approval_digest
        or inspection.approval.execution_provenance_digest
        != remediation.source_execution_provenance_digest
        or inspection.readiness.readiness_digest != remediation.source_v3_readiness_digest
        or frozen_advisor_v3_snapshot_digest(source_registry, inspection)
        != remediation.source_registry_snapshot_digest
    ):
        raise ValueError("Advisor-v3.1 remediation does not bind the exact frozen source corpus.")
    readiness = build_advisor_v31_qualification_readiness(
        amendment_digest=governance.amendment.amendment_digest,
        source_execution_approval_digest=remediation.source_execution_approval_digest,
        source_execution_provenance_digest=remediation.source_execution_provenance_digest,
        source_v3_readiness_digest=remediation.source_v3_readiness_digest,
        source_registry_snapshot_digest=remediation.source_registry_snapshot_digest,
        analysis_provenance_digest=remediation.analysis_provenance_digest,
        remediation_approval_digest=governance.remediation_approval.approval_digest,
        remediation_readiness_digest=remediation.readiness_digest,
        advisor_evidence_ready=remediation.advisor_evidence_ready,
    )
    return readiness, inspection


def build_advisor_v31_qualification_readiness_from_registries(
    *,
    source_registry: BenchmarkRegistry,
    remediation_registry: BenchmarkRegistry,
    committed_amendment_path: Path,
    remediation_approval_digest: str,
    remediation_readiness_digest: str,
) -> AdvisorV31QualificationReadinessArtifact:
    """Re-audit exact bindings without authorizing or executing locked/OOD evaluation."""

    readiness, _ = _build_readiness_and_inspection(
        source_registry=source_registry,
        remediation_registry=remediation_registry,
        committed_amendment_path=committed_amendment_path,
        remediation_approval_digest=remediation_approval_digest,
        remediation_readiness_digest=remediation_readiness_digest,
    )
    return readiness


def qualify_advisor_v31_registry(
    *,
    source_registry: BenchmarkRegistry,
    remediation_registry: BenchmarkRegistry,
    committed_amendment_path: Path,
    remediation_approval_digest: str,
    remediation_readiness_digest: str,
    approval_reference: str,
    human_approved: Literal[True],
    locked_evaluation_access_authorized: Literal[True],
    ood_evaluation_access_authorized: Literal[True],
) -> AdvisorV31QualificationArtifact:
    """Execute one explicitly approved aggregate v3.1 qualification."""

    if not (
        human_approved is True
        and locked_evaluation_access_authorized is True
        and ood_evaluation_access_authorized is True
    ):
        raise ValueError("Advisor-v3.1 qualification requires explicit protected-role approval.")
    readiness, inspection = _build_readiness_and_inspection(
        source_registry=source_registry,
        remediation_registry=remediation_registry,
        committed_amendment_path=committed_amendment_path,
        remediation_approval_digest=remediation_approval_digest,
        remediation_readiness_digest=remediation_readiness_digest,
    )
    policy = AdvisorV3QualificationPolicy()
    approval = AdvisorV31QualificationApproval(
        approval_reference=approval_reference,
        human_approved=human_approved,
        locked_evaluation_access_authorized=locked_evaluation_access_authorized,
        ood_evaluation_access_authorized=ood_evaluation_access_authorized,
        amendment_digest=readiness.amendment_digest,
        source_execution_approval_digest=readiness.source_execution_approval_digest,
        source_execution_provenance_digest=readiness.source_execution_provenance_digest,
        source_v3_readiness_digest=readiness.source_v3_readiness_digest,
        source_registry_snapshot_digest=readiness.source_registry_snapshot_digest,
        analysis_provenance_digest=readiness.analysis_provenance_digest,
        remediation_approval_digest=readiness.remediation_approval_digest,
        remediation_readiness_digest=readiness.remediation_readiness_digest,
        policy_digest=policy.policy_digest,
        evaluation_algorithm_digest=readiness.evaluation_algorithm_digest,
    )
    roles: dict[str, AdvisorFamilyRole] = dict(advisor_v3_family_roles())
    recipe_token_by_digest = {
        recipe_digest: _TOKEN_BY_RECIPE_ID[recipe_id]
        for recipe_digest, recipe_id in inspection.recipe_ids_by_digest
        if recipe_id in _TOKEN_BY_RECIPE_ID
    }
    family_evidence = aggregate_family_recipe_evidence(
        registry=source_registry,
        records=inspection.records,
        role_by_family=roles,
        recipe_token_by_digest=recipe_token_by_digest,
    )
    evidence = tuple(
        AdvisorV31FamilyUtilityEvidence(
            family_id=item.family_id,
            family_role=item.family_role,
            recipe_token=item.recipe_token,
            mean_utility=item.mean_utility,
            run_count=item.run_count,
            evidence_digest=item.evidence_digest,
        )
        for item in family_evidence
    )
    generator = build_advisor_v3_generator()
    vectors = extract_advisor_v3_family_meta_features(
        generator,
        family_ids=frozenset(roles),
    )
    report = evaluate_advisor_v31_aggregate_evidence(
        readiness=readiness,
        approval=approval,
        evidence=evidence,
        role_by_family=roles,
        family_vectors=vectors,
    )
    return AdvisorV31QualificationArtifact(report=report, report_digest=report.report_digest)


__all__ = [
    "AdvisorV31QualificationGovernance",
    "build_advisor_v31_qualification_readiness_from_registries",
    "load_advisor_v31_qualification_governance",
    "qualify_advisor_v31_registry",
]
