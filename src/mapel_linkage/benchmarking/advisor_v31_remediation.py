"""Performance-metric-blind advisor-v3.1 qualification-input remediation governance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from mapel_linkage.benchmarking.advisor_v3_catalogue import (
    AdvisorV3CorpusReadinessManifest,
    advisor_v3_family_roles,
    build_advisor_v3_corpus_design,
    build_advisor_v3_generator,
    build_advisor_v3_geometry_coherence,
    build_advisor_v3_preregistration,
    build_advisor_v3_shard_plan,
    validate_advisor_v3_geometry_coherence,
)
from mapel_linkage.benchmarking.advisor_v3_execution import (
    AdvisorV3CorpusExecutionApproval,
    AdvisorV3ShardExecutionReport,
    advisor_v3_execution_provenance_digest,
)
from mapel_linkage.benchmarking.contracts import (
    BenchmarkFailureRecord,
    BenchmarkRunRecord,
    BenchmarkRunStatus,
)
from mapel_linkage.benchmarking.registry import BenchmarkRegistry
from mapel_linkage.benchmarking.runner import (
    BenchmarkPortfolioRunner,
    benchmark_replicate_seed,
    benchmark_run_id,
)
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.recommendation.qualification_v3 import (
    advisor_v31_evaluation_algorithm_digest,
)

_MAX_AMENDMENT_BYTES = 4 * 1024 * 1024
_REQUIRED_RECIPE_IDS = frozenset(
    {
        "recipe.fellegi_sunter_reference",
        "recipe.xgboost_classifier",
        "recipe.xgboost_ranker",
    }
)
_PORTFOLIO_RECIPE_IDS = (
    "recipe.fellegi_sunter_reference",
    "recipe.xgboost_classifier",
    "recipe.xgboost_ranker",
    "recipe.lightgbm_classifier",
    "recipe.pytorch_matcher",
    "recipe.single_source_dedupe",
    "recipe.multi_source_resolver",
)


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_text(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


Digest = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]


class AdvisorV31ProtocolAmendmentManifest(BaseModel):
    """Post-corpus, pre-qualification amendment limited to role evidence needs."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    amendment_schema_version: Literal["3.1"] = "3.1"
    amendment_id: Literal["advisor_v31_role_evidence_20260821"] = (
        "advisor_v31_role_evidence_20260821"
    )
    source_preregistration_digest: Digest
    source_design_digest: Digest
    source_catalogue_manifest_digest: Digest
    source_role_manifest_digest: Digest
    source_qualification_policy_digest: Digest
    source_evaluation_algorithm_digest: Digest
    amended_evaluation_algorithm_digest: Digest
    source_geometry_coherence_digest: Digest
    qualification_required_roles: tuple[
        Literal["meta_training", "conformal", "locked_evaluation"], ...
    ] = (
        "meta_training",
        "conformal",
        "locked_evaluation",
    )
    qualification_required_family_count: Literal[72] = 72
    qualification_required_evidence_cell_count: Literal[1440] = 1_440
    qualification_required_adapter_run_count: Literal[4320] = 4_320
    ood_geometry_family_count: Literal[12] = 12
    ood_required_evidence: Literal["observable_mechanism_profile_and_distance_geometry"] = (
        "observable_mechanism_profile_and_distance_geometry"
    )
    ood_recipe_utility_requirement: Literal["none"] = "none"
    ood_recipe_metric_use_for_fit_threshold_or_qualification: Literal["prohibited"] = "prohibited"
    ood_adapter_status_use: Literal["aggregate_diagnostic_integrity_only"] = (
        "aggregate_diagnostic_integrity_only"
    )
    source_registry_mutation: Literal["prohibited"] = "prohibited"
    remediation_registry_content: Literal["governance_only_digest_bound_reference"] = (
        "governance_only_digest_bound_reference"
    )
    amendment_trigger_scope: Literal["adapter_status_and_failure_code_metadata"] = (
        "adapter_status_and_failure_code_metadata"
    )
    adapter_status_metadata_accessed_to_select_amendment: Literal[True] = True
    failure_code_metadata_accessed_to_select_amendment: Literal[True] = True
    performance_metric_values_accessed_to_select_amendment: Literal[False] = False
    performance_thresholds_changed: Literal[False] = False
    catalogue_or_family_roles_changed: Literal[False] = False
    seeds_or_replicates_changed: Literal[False] = False
    utility_policy_changed: Literal[False] = False
    qualification_evaluation_accessed: Literal[False] = False
    locked_and_ood_evaluation_requires_later_human_approval: Literal[True] = True
    synthetic_only: Literal[True] = True
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    release_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        source = build_advisor_v3_preregistration()
        design = build_advisor_v3_corpus_design()
        geometry = build_advisor_v3_geometry_coherence()
        expected = {
            "source_preregistration_digest": source.preregistration_digest,
            "source_design_digest": design.design_digest,
            "source_catalogue_manifest_digest": design.catalogue_manifest_digest,
            "source_role_manifest_digest": design.role_manifest_digest,
            "source_qualification_policy_digest": source.qualification_policy_digest,
            "source_evaluation_algorithm_digest": source.evaluation_algorithm_digest,
            "amended_evaluation_algorithm_digest": advisor_v31_evaluation_algorithm_digest(),
            "source_geometry_coherence_digest": geometry.geometry_coherence_digest,
        }
        if {name: getattr(self, name) for name in expected} != expected:
            raise ValueError("Advisor-v3.1 amendment bindings are stale or conflicting.")
        if tuple(self.qualification_required_roles) != (
            "meta_training",
            "conformal",
            "locked_evaluation",
        ):
            raise ValueError("Advisor-v3.1 role evidence requirements cannot be reordered.")
        return self

    @property
    def amendment_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {**self.model_dump(mode="json"), "amendment_digest": self.amendment_digest}


class AdvisorV31RemediationApproval(BaseModel):
    """Human approval bound to one immutable source snapshot and amendment."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    approval_schema_version: Literal["3.1"] = "3.1"
    approval_reference: Annotated[
        StrictStr, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$", repr=False)
    ]
    human_approved: Literal[True]
    amendment_digest: Digest
    source_execution_approval_digest: Digest
    source_execution_provenance_digest: Digest
    source_v3_readiness_digest: Digest
    source_registry_snapshot_digest: Digest
    analysis_provenance_digest: Digest
    recomputed_geometry_coherence_digest: Digest
    source_registry_mutation_authorized: Literal[False] = False
    performance_metric_access_authorized: Literal[False] = False
    locked_evaluation_access_authorized: Literal[False] = False
    ood_evaluation_access_authorized: Literal[False] = False
    automatic_promotion_authorized: Literal[False] = False
    release_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    @property
    def approval_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        payload = self.model_dump(mode="json", exclude={"approval_reference"})
        return {**payload, "approval_digest": self.approval_digest}


class AdvisorV31RemediationReadinessManifest(BaseModel):
    """Fail-closed role-specific evidence readiness without qualification access."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    readiness_schema_version: Literal["3.1"] = "3.1"
    execution_protocol_id: Literal["advisor_v31_governance_only_remediation_v1"] = (
        "advisor_v31_governance_only_remediation_v1"
    )
    amendment_digest: Digest
    source_execution_approval_digest: Digest
    source_execution_provenance_digest: Digest
    source_v3_preregistration_digest: Digest
    source_v3_readiness_digest: Digest
    source_registry_snapshot_digest: Digest
    analysis_provenance_digest: Digest
    recomputed_geometry_coherence_digest: Digest
    observable_geometry_recomputed: Literal[True] = True
    source_v3_advisor_evidence_ready: Literal[False] = False
    source_expected_run_count: Literal[11760] = 11_760
    source_completed_run_count: Annotated[StrictInt, Field(ge=0, le=11760)]
    source_expected_family_manifest_count: Literal[84] = 84
    source_family_manifest_count: Annotated[StrictInt, Field(ge=0, le=84)]
    source_expected_instance_manifest_count: Literal[336] = 336
    source_instance_manifest_count: Annotated[StrictInt, Field(ge=0, le=336)]
    source_catalogue_integrity_checked: StrictBool
    source_sidecar_digest_integrity_checked: Literal[True] = True
    source_failure_sidecar_count: Annotated[StrictInt, Field(ge=0, le=11760)]
    metric_payloads_parsed_for_digest_integrity_only: Literal[True] = True
    performance_metric_values_inspected_or_used: Literal[False] = False
    diagnostic_status_metadata_accessed: Literal[True] = True
    qualification_required_family_count: Literal[72] = 72
    successful_qualification_required_family_count: Annotated[StrictInt, Field(ge=0, le=72)]
    qualification_required_evidence_cell_count: Literal[1440] = 1_440
    successful_qualification_required_evidence_cell_count: Annotated[
        StrictInt, Field(ge=0, le=1440)
    ]
    expected_qualification_required_adapter_run_count: Literal[4320] = 4_320
    successful_qualification_required_adapter_run_count: Annotated[StrictInt, Field(ge=0, le=4320)]
    failed_qualification_required_adapter_run_count: Annotated[StrictInt, Field(ge=0, le=4320)]
    missing_qualification_required_adapter_run_count: Annotated[StrictInt, Field(ge=0, le=4320)]
    expected_ood_geometry_family_count: Literal[12] = 12
    complete_ood_geometry_family_count: Annotated[StrictInt, Field(ge=0, le=12)]
    expected_ood_diagnostic_adapter_run_count: Literal[720] = 720
    completed_ood_diagnostic_adapter_run_count: Annotated[StrictInt, Field(ge=0, le=720)]
    non_success_ood_diagnostic_adapter_run_count: Annotated[StrictInt, Field(ge=0, le=720)]
    expected_ineligible_nonrequired_recipe_run_count: Literal[6720] = 6_720
    ineligible_nonrequired_recipe_run_count: Annotated[StrictInt, Field(ge=0, le=6720)]
    advisor_evidence_ready: StrictBool
    qualification_evaluation_accessed: Literal[False] = False
    locked_evaluation_access_authorized: Literal[False] = False
    ood_evaluation_access_authorized: Literal[False] = False
    ood_recipe_metric_use_for_fit_threshold_or_qualification: Literal["prohibited"] = "prohibited"
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_readiness(self) -> Self:
        if (
            self.recomputed_geometry_coherence_digest
            != build_advisor_v3_geometry_coherence().geometry_coherence_digest
        ):
            raise ValueError("Advisor-v3.1 recomputed geometry binding is stale.")
        if (
            self.successful_qualification_required_adapter_run_count
            + self.failed_qualification_required_adapter_run_count
            + self.missing_qualification_required_adapter_run_count
            != self.expected_qualification_required_adapter_run_count
        ):
            raise ValueError("Advisor-v3.1 required-role adapter counts are inconsistent.")
        if (
            self.non_success_ood_diagnostic_adapter_run_count
            > self.completed_ood_diagnostic_adapter_run_count
        ):
            raise ValueError("Advisor-v3.1 OOD diagnostic counts are inconsistent.")
        if self.source_failure_sidecar_count != (
            self.ineligible_nonrequired_recipe_run_count
            + self.failed_qualification_required_adapter_run_count
            + self.non_success_ood_diagnostic_adapter_run_count
        ):
            raise ValueError("Advisor-v3.1 failure-sidecar counts are inconsistent.")
        expected_ready = (
            self.source_completed_run_count == self.source_expected_run_count
            and self.source_family_manifest_count == self.source_expected_family_manifest_count
            and self.source_instance_manifest_count == self.source_expected_instance_manifest_count
            and self.source_catalogue_integrity_checked
            and self.successful_qualification_required_family_count
            == self.qualification_required_family_count
            and self.successful_qualification_required_evidence_cell_count
            == self.qualification_required_evidence_cell_count
            and self.successful_qualification_required_adapter_run_count
            == self.expected_qualification_required_adapter_run_count
            and self.failed_qualification_required_adapter_run_count == 0
            and self.missing_qualification_required_adapter_run_count == 0
            and self.complete_ood_geometry_family_count == self.expected_ood_geometry_family_count
            and self.completed_ood_diagnostic_adapter_run_count
            == self.expected_ood_diagnostic_adapter_run_count
            and self.ineligible_nonrequired_recipe_run_count
            == self.expected_ineligible_nonrequired_recipe_run_count
        )
        if self.advisor_evidence_ready != expected_ready:
            raise ValueError(
                "Advisor-v3.1 evidence readiness must fail closed on any required gap."
            )
        return self

    @property
    def readiness_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {**self.model_dump(mode="json"), "readiness_digest": self.readiness_digest}


def build_advisor_v31_protocol_amendment() -> AdvisorV31ProtocolAmendmentManifest:
    """Build the canonical amendment without reading any benchmark registry."""

    source = build_advisor_v3_preregistration()
    design = build_advisor_v3_corpus_design()
    geometry = build_advisor_v3_geometry_coherence()
    return AdvisorV31ProtocolAmendmentManifest(
        source_preregistration_digest=source.preregistration_digest,
        source_design_digest=design.design_digest,
        source_catalogue_manifest_digest=design.catalogue_manifest_digest,
        source_role_manifest_digest=design.role_manifest_digest,
        source_qualification_policy_digest=source.qualification_policy_digest,
        source_evaluation_algorithm_digest=source.evaluation_algorithm_digest,
        amended_evaluation_algorithm_digest=advisor_v31_evaluation_algorithm_digest(),
        source_geometry_coherence_digest=geometry.geometry_coherence_digest,
    )


def load_committed_advisor_v31_protocol_amendment(
    path: Path,
) -> AdvisorV31ProtocolAmendmentManifest:
    """Load the exact source-controlled amendment and reject path substitution."""

    try:
        if any(component.is_symlink() for component in (path, *path.parents)) or not path.is_file():
            raise FileNotFoundError
        if path.stat().st_size > _MAX_AMENDMENT_BYTES:
            raise ValueError("Advisor-v3.1 amendment exceeds its aggregate size bound.")
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError("The committed advisor-v3.1 amendment is unavailable.") from None
    except OSError:
        raise ValueError("The committed advisor-v3.1 amendment could not be read safely.") from None
    canonical = build_advisor_v31_protocol_amendment()
    if text != _canonical_text(canonical):
        raise ValueError("The committed advisor-v3.1 amendment is not canonical.")
    manifest = AdvisorV31ProtocolAmendmentManifest.model_validate_json(text)
    if manifest != canonical:
        raise ValueError("The committed advisor-v3.1 amendment is stale or conflicting.")
    return manifest


def _recompute_and_validate_observable_geometry(
    amendment: AdvisorV31ProtocolAmendmentManifest,
) -> str:
    observed = validate_advisor_v3_geometry_coherence()
    if observed.geometry_coherence_digest != amendment.source_geometry_coherence_digest:
        raise ValueError("Advisor-v3.1 observable geometry conflicts with its amendment binding.")
    return observed.geometry_coherence_digest


@dataclass(frozen=True, slots=True)
class FrozenAdvisorV3CorpusInspection:
    """Integrity-checked v3 evidence bound to its persisted execution provenance."""

    approval: AdvisorV3CorpusExecutionApproval
    readiness: AdvisorV3CorpusReadinessManifest
    records: tuple[BenchmarkRunRecord, ...]
    failure_digests: tuple[str, ...]
    governance_artifact_digests: tuple[str, ...]
    recipe_ids_by_digest: tuple[tuple[str, str], ...]


def advisor_v31_analysis_provenance_digest() -> str:
    """Bind current source, dependencies, and environment without replacing v3 provenance."""

    return advisor_v3_execution_provenance_digest(BenchmarkPortfolioRunner())


def frozen_advisor_v3_provenance_digest(
    records: tuple[BenchmarkRunRecord, ...],
) -> str:
    """Reconstruct the original aggregate provenance digest from immutable run records."""

    provenance = {
        (record.engine_commit, record.dependency_lock_digest, record.environment_digest)
        for record in records
    }
    if len(provenance) != 1:
        raise ValueError("Advisor-v3 source evidence has mixed execution provenance.")
    engine_source, dependency_lock, environment = next(iter(provenance))
    return _digest(
        {
            "provenance_schema_id": "advisor_v3_execution_provenance_v1",
            "engine_source_digest": engine_source,
            "dependency_lock_digest": dependency_lock,
            "environment_digest": environment,
        }
    )


def _read_governance_text(path: Path) -> str:
    try:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError
        if path.stat().st_size > _MAX_AMENDMENT_BYTES:
            raise ValueError("Advisor-v3 source governance exceeds its aggregate size bound.")
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError("Advisor-v3 source governance is incomplete.") from None
    except OSError:
        raise ValueError("Advisor-v3 source governance could not be read safely.") from None


def _load_frozen_v3_approval(registry: BenchmarkRegistry) -> AdvisorV3CorpusExecutionApproval:
    paths = tuple(sorted((registry.root_directory / "governance").glob("approval.v3.*.json")))
    if len(paths) != 1:
        raise ValueError("Advisor-v3 source must contain exactly one execution approval.")
    text = _read_governance_text(paths[0])
    approval = AdvisorV3CorpusExecutionApproval.model_validate_json(text)
    if paths[0].name != f"approval.v3.{approval.approval_digest}.json" or text != _canonical_text(
        approval
    ):
        raise ValueError("Advisor-v3 source approval filename failed its digest binding.")
    return approval


def _validate_frozen_v3_static_governance(
    registry: BenchmarkRegistry,
    approval: AdvisorV3CorpusExecutionApproval,
) -> AdvisorV3CorpusReadinessManifest:
    governance = registry.root_directory / "governance"
    design = build_advisor_v3_corpus_design()
    preregistration = build_advisor_v3_preregistration()
    plan = build_advisor_v3_shard_plan()
    expected_static = (
        (
            governance / f"design.v3.{design.design_digest}.json",
            _canonical_text(design),
        ),
        (
            governance / f"preregistration.v3.{preregistration.preregistration_digest}.json",
            _canonical_text(preregistration),
        ),
        (
            governance / f"shards.v3.{plan.plan_digest}.json",
            _canonical_text(plan),
        ),
    )
    for path, expected_text in expected_static:
        if _read_governance_text(path) != expected_text:
            raise ValueError("Advisor-v3 source static governance is stale or conflicting.")
    readiness_path = governance / f"readiness.v3.{approval.readiness_digest}.json"
    readiness_text = _read_governance_text(readiness_path)
    readiness = AdvisorV3CorpusReadinessManifest.model_validate_json(readiness_text)
    if readiness.readiness_digest != approval.readiness_digest or readiness_text != _canonical_text(
        readiness
    ):
        raise ValueError("Advisor-v3 source execution readiness is not approval-bound.")
    return readiness


def _validate_frozen_v3_completions(
    registry: BenchmarkRegistry,
    approval: AdvisorV3CorpusExecutionApproval,
) -> None:
    plan_digest = build_advisor_v3_preregistration().shard_plan_digest
    governance = registry.root_directory / "governance"
    paths = tuple(sorted(governance.glob(f"completion.v3.{plan_digest}.*.json")))
    if len(paths) != 42:
        raise ValueError("Advisor-v3 source does not contain all whole-family completions.")
    observed_indices: list[int] = []
    for path in paths:
        text = _read_governance_text(path)
        report = AdvisorV3ShardExecutionReport.model_validate_json(text)
        expected_name = f"completion.v3.{plan_digest}.{report.shard_index:03d}.json"
        if (
            path.name != expected_name
            or report.approval_digest != approval.approval_digest
            or report.readiness_digest != approval.readiness_digest
            or report.execution_provenance_digest != approval.execution_provenance_digest
            or not report.shard_complete
            or report.completed_run_count != 280
            or report.newly_persisted_run_count != 0
            or report.resumed_run_count != 280
            or text != _canonical_text(report)
        ):
            raise ValueError("Advisor-v3 source shard completion governance is inconsistent.")
        observed_indices.append(report.shard_index)
    if observed_indices != list(range(42)):
        raise ValueError("Advisor-v3 source shard completions are not canonical.")


def _validate_failure_sidecar_binding(
    record: BenchmarkRunRecord,
    failure: BenchmarkFailureRecord,
    *,
    recipe_id: str,
) -> None:
    if (
        failure.run_id != record.run_id
        or failure.family_id != record.family_id
        or failure.instance_id != record.instance_id
        or failure.replicate_id != record.replicate_id
        or failure.recipe_id != recipe_id
        or failure.status is not record.status
        or failure.failure_code != record.failure_code
    ):
        raise ValueError("Advisor-v3 source failure sidecar conflicts with its run record.")


def _validate_exact_artifact_coverage(
    directory: Path,
    *,
    expected_run_ids: frozenset[str],
    artifact_name: str,
) -> None:
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Advisor-v3 source {artifact_name} directory is path-unsafe.")
        entries = tuple(directory.iterdir())
        if any(
            entry.is_symlink() or not entry.is_file() or entry.suffix != ".json"
            for entry in entries
        ):
            raise ValueError(
                f"Advisor-v3 source {artifact_name} contains a path-unsafe or non-JSON artifact."
            )
        observed_run_ids = frozenset(entry.stem for entry in entries)
    except ValueError:
        raise
    except OSError:
        raise ValueError(
            f"Advisor-v3 source {artifact_name} coverage could not be inspected safely."
        ) from None
    if observed_run_ids != expected_run_ids:
        raise ValueError(f"Advisor-v3 source {artifact_name} coverage is not exact.")


def _inspect_frozen_v3_records(
    registry: BenchmarkRegistry,
    approval: AdvisorV3CorpusExecutionApproval,
    initial_readiness: AdvisorV3CorpusReadinessManifest,
) -> FrozenAdvisorV3CorpusInspection:
    generator = build_advisor_v3_generator()
    expected: dict[str, tuple[str, str, str, int, str]] = {}
    for _, instance_ids in (
        (family_id, tuple(item.instance_id for item in generator.list_instances(family_id)))
        for family_id, _ in advisor_v3_family_roles()
    ):
        for instance_id in instance_ids:
            family_id = generator.get_instance(instance_id).family_id
            for replicate_number in range(5):
                replicate_id = f"replicate.{replicate_number:07d}"
                seed = benchmark_replicate_seed(
                    instance_id=instance_id,
                    replicate_number=replicate_number,
                    base_seed=20260816,
                )
                for recipe_id in _PORTFOLIO_RECIPE_IDS:
                    expected[
                        benchmark_run_id(
                            instance_id=instance_id,
                            recipe_id=recipe_id,
                            replicate_id=replicate_id,
                        )
                    ] = (family_id, instance_id, replicate_id, seed, recipe_id)
    records = registry.list_run_records()
    if {record.run_id for record in records} != set(expected):
        raise ValueError("Advisor-v3 source run coverage differs from its fixed grid.")
    record_ids = frozenset(record.run_id for record in records)
    success_ids = frozenset(
        record.run_id for record in records if record.status is BenchmarkRunStatus.SUCCESS
    )
    non_success_ids = record_ids - success_ids
    _validate_exact_artifact_coverage(
        registry.runs_dir,
        expected_run_ids=record_ids,
        artifact_name="run-record",
    )
    _validate_exact_artifact_coverage(
        registry.metrics_dir,
        expected_run_ids=success_ids,
        artifact_name="metric-sidecar",
    )
    _validate_exact_artifact_coverage(
        registry.failures_dir,
        expected_run_ids=non_success_ids,
        artifact_name="failure-sidecar",
    )
    recipe_digests: dict[str, str] = {}
    required_success_by_cell: dict[tuple[str, str], set[str]] = {}
    required_success_by_family: dict[str, set[str]] = {}
    required_success = 0
    required_failure = 0
    failure_digests: list[str] = []
    for record in records:
        family_id, instance_id, replicate_id, seed, recipe_id = expected[record.run_id]
        if (
            record.family_id != family_id
            or record.instance_id != instance_id
            or record.replicate_id != replicate_id
            or record.random_seed != seed
        ):
            raise ValueError("Advisor-v3 source run identity or seed binding is inconsistent.")
        previous_digest = recipe_digests.setdefault(recipe_id, record.pipeline_recipe_digest)
        if previous_digest != record.pipeline_recipe_digest:
            raise ValueError("Advisor-v3 source recipe provenance is internally inconsistent.")
        if record.status is not BenchmarkRunStatus.SUCCESS:
            failure = registry.load_failure_record(record.run_id)
            if failure is None:
                raise ValueError("Advisor-v3 source failure sidecar is missing.")
            _validate_failure_sidecar_binding(record, failure, recipe_id=recipe_id)
            failure_digests.append(failure.failure_digest)
        if recipe_id not in _REQUIRED_RECIPE_IDS:
            continue
        if record.status is BenchmarkRunStatus.SUCCESS:
            required_success += 1
            required_success_by_cell.setdefault((instance_id, replicate_id), set()).add(recipe_id)
            required_success_by_family.setdefault(family_id, set()).add(recipe_id)
        else:
            required_failure += 1
    if frozen_advisor_v3_provenance_digest(records) != approval.execution_provenance_digest:
        raise ValueError("Advisor-v3 source execution provenance failed its frozen binding.")
    if len(set(recipe_digests.values())) != len(_PORTFOLIO_RECIPE_IDS):
        raise ValueError("Advisor-v3 source recipe provenance contains a digest collision.")
    successful_cells = sum(
        _REQUIRED_RECIPE_IDS.issubset(tokens) for tokens in required_success_by_cell.values()
    )
    successful_families = sum(
        _REQUIRED_RECIPE_IDS.issubset(tokens) for tokens in required_success_by_family.values()
    )
    terminal = AdvisorV3CorpusReadinessManifest.model_validate(
        {
            **initial_readiness.model_dump(mode="json"),
            "execution_status": "complete",
            "completed_run_count": len(records),
            "successful_evidence_cell_count": successful_cells,
            "successful_required_adapter_run_count": required_success,
            "failed_required_adapter_run_count": required_failure,
            "missing_required_adapter_run_count": 5_040 - required_success - required_failure,
            "successful_overlap_family_count": successful_families,
            "held_out_digest_integrity_checked": True,
            "advisor_evidence_ready": (
                successful_cells == 1_680
                and required_success == 5_040
                and required_failure == 0
                and successful_families == 84
            ),
        }
    )
    audit_paths = tuple(
        sorted((registry.root_directory / "governance").glob("readiness.audit.v3.*.json"))
    )
    if len(audit_paths) != 1:
        raise ValueError("Advisor-v3 source must contain exactly one terminal readiness audit.")
    retained_text = _read_governance_text(audit_paths[0])
    retained = AdvisorV3CorpusReadinessManifest.model_validate_json(retained_text)
    if (
        audit_paths[0].name != f"readiness.audit.v3.{retained.readiness_digest}.json"
        or retained != terminal
        or retained_text != _canonical_text(retained)
    ):
        raise ValueError("Advisor-v3 source terminal readiness failed its digest binding.")
    governance = registry.root_directory / "governance"
    design = build_advisor_v3_corpus_design()
    preregistration = build_advisor_v3_preregistration()
    plan = build_advisor_v3_shard_plan()
    expected_governance_paths = {
        governance / f"approval.v3.{approval.approval_digest}.json",
        governance / f"design.v3.{design.design_digest}.json",
        governance / f"preregistration.v3.{preregistration.preregistration_digest}.json",
        governance / f"readiness.v3.{approval.readiness_digest}.json",
        governance / f"readiness.audit.v3.{retained.readiness_digest}.json",
        governance / f"shards.v3.{plan.plan_digest}.json",
        *(
            governance / f"completion.v3.{plan.plan_digest}.{shard_index:03d}.json"
            for shard_index in range(42)
        ),
    }
    lock_directory = governance / "locks.v3"
    observed_governance_entries = set(governance.iterdir())
    if observed_governance_entries != expected_governance_paths | {lock_directory} or any(
        path.is_symlink() or path.suffix != ".json" for path in expected_governance_paths
    ):
        raise ValueError("Advisor-v3 source governance artifact coverage is not exact.")
    expected_lock_paths = {
        lock_directory / f"shard.{shard_index:03d}.lock" for shard_index in range(42)
    }
    if (
        lock_directory.is_symlink()
        or not lock_directory.is_dir()
        or set(lock_directory.iterdir()) != expected_lock_paths
        or any(path.is_symlink() or not path.is_file() for path in expected_lock_paths)
    ):
        raise ValueError("Advisor-v3 source shard-lock coverage is not exact.")
    return FrozenAdvisorV3CorpusInspection(
        approval=approval,
        readiness=retained,
        records=records,
        failure_digests=tuple(sorted(failure_digests)),
        governance_artifact_digests=tuple(
            sorted(
                hashlib.sha256(_read_governance_text(path).encode("utf-8")).hexdigest()
                for path in expected_governance_paths
            )
        ),
        recipe_ids_by_digest=tuple(
            sorted(
                (recipe_digest, recipe_id) for recipe_id, recipe_digest in recipe_digests.items()
            )
        ),
    )


def inspect_frozen_advisor_v3_corpus(
    registry: BenchmarkRegistry,
) -> FrozenAdvisorV3CorpusInspection:
    """Validate completed v3 evidence against persisted, not current, provenance."""

    approval = _load_frozen_v3_approval(registry)
    initial_readiness = _validate_frozen_v3_static_governance(registry, approval)
    _validate_frozen_v3_completions(registry, approval)
    return _inspect_frozen_v3_records(registry, approval, initial_readiness)


def _validate_source_catalogue(registry: BenchmarkRegistry) -> tuple[int, int, bool, int]:
    generator = build_advisor_v3_generator()
    roles = dict(advisor_v3_family_roles())
    expected_families = {family_id: generator.get_family(family_id) for family_id in roles}
    expected_instances = {
        instance.instance_id: instance
        for family_id in roles
        for instance in generator.list_instances(family_id)
    }
    families = {item.family_id: item for item in registry.list_families()}
    instances = {item.instance_id: item for item in registry.list_instances()}
    if set(families) - set(expected_families) or set(instances) - set(expected_instances):
        raise ValueError("The advisor-v3 source catalogue contains unregistered evidence.")
    if any(item != expected_families[identifier] for identifier, item in families.items()):
        raise ValueError("The advisor-v3 source family catalogue conflicts with its digest.")
    if any(item != expected_instances[identifier] for identifier, item in instances.items()):
        raise ValueError("The advisor-v3 source instance catalogue conflicts with its digest.")
    complete = set(families) == set(expected_families) and set(instances) == set(expected_instances)
    complete_ood = sum(
        role == "ood_holdout"
        and all(
            instance.instance_id in instances for instance in generator.list_instances(family_id)
        )
        for family_id, role in roles.items()
    )
    return len(families), len(instances), complete, complete_ood


def _frozen_source_snapshot_digest(
    registry: BenchmarkRegistry,
    inspection: FrozenAdvisorV3CorpusInspection,
) -> str:
    """Bind all aggregate manifests and sidecars without exposing their identifiers."""

    return _digest(
        {
            "snapshot_schema_id": "advisor_v31_frozen_source_snapshot_v1",
            "source_execution_approval_digest": inspection.approval.approval_digest,
            "source_execution_provenance_digest": (inspection.approval.execution_provenance_digest),
            "source_terminal_readiness_digest": inspection.readiness.readiness_digest,
            "run_digests": tuple(sorted(record.run_digest for record in inspection.records)),
            "failure_sidecar_digests": inspection.failure_digests,
            "source_governance_artifact_digests": inspection.governance_artifact_digests,
            "family_manifest_digests": tuple(
                sorted(manifest.family_digest for manifest in registry.list_families())
            ),
            "instance_manifest_digests": tuple(
                sorted(manifest.instance_digest for manifest in registry.list_instances())
            ),
            "contains_record_values": False,
            "contains_identifiers": False,
            "contains_candidate_pairs": False,
        }
    )


def _role_evidence_counts(
    records: tuple[BenchmarkRunRecord, ...],
    *,
    recipe_ids_by_digest: Mapping[str, str],
) -> dict[str, int]:
    roles = dict(advisor_v3_family_roles())
    required_success_by_cell: dict[tuple[str, str], set[str]] = {}
    required_success_cells_by_family: dict[str, int] = {}
    family_by_instance: dict[str, str] = {}
    required_success = 0
    required_failure = 0
    required_observed = 0
    ood_diagnostic = 0
    ood_non_success = 0
    nonrequired_ineligible = 0
    for record in records:
        try:
            role = roles[record.family_id]
            recipe_id = recipe_ids_by_digest[record.pipeline_recipe_digest]
        except KeyError:
            raise ValueError(
                "Advisor-v3.1 evidence has an unknown role or recipe binding."
            ) from None
        family_by_instance[record.instance_id] = record.family_id
        if recipe_id not in _REQUIRED_RECIPE_IDS:
            if record.status is BenchmarkRunStatus.INELIGIBLE:
                nonrequired_ineligible += 1
            continue
        if role == "ood_holdout":
            ood_diagnostic += 1
            if record.status is not BenchmarkRunStatus.SUCCESS:
                ood_non_success += 1
            continue
        required_observed += 1
        if record.status is BenchmarkRunStatus.SUCCESS:
            required_success += 1
            required_success_by_cell.setdefault(
                (record.instance_id, record.replicate_id), set()
            ).add(recipe_id)
        else:
            required_failure += 1
    for (instance_id, _), successful_recipes in required_success_by_cell.items():
        if _REQUIRED_RECIPE_IDS.issubset(successful_recipes):
            family_id = family_by_instance[instance_id]
            required_success_cells_by_family[family_id] = (
                required_success_cells_by_family.get(family_id, 0) + 1
            )
    return {
        "required_success": required_success,
        "required_failure": required_failure,
        "required_missing": 4_320 - required_observed,
        "successful_cells": sum(required_success_cells_by_family.values()),
        "successful_families": sum(
            count == 20 for count in required_success_cells_by_family.values()
        ),
        "ood_diagnostic": ood_diagnostic,
        "ood_non_success": ood_non_success,
        "nonrequired_ineligible": nonrequired_ineligible,
    }


def _reject_non_governance_destination_content(registry: BenchmarkRegistry) -> None:
    root = registry.root_directory
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Advisor-v3.1 remediation paths cannot contain symbolic links.")
        if path.is_file() and not path.is_relative_to(root / "governance"):
            raise ValueError("Advisor-v3.1 remediation registries must be governance-only.")


def _persist_exact(path: Path, model: BaseModel) -> None:
    text = _canonical_text(model)
    try:
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("Advisor-v3.1 governance paths cannot be symbolic links.")
        if path.exists():
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                raise FileExistsError("A conflicting advisor-v3.1 governance artifact exists.")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.is_symlink():
            raise ValueError("Advisor-v3.1 governance paths cannot be symbolic links.")
        atomic_write_text(path, text)
    except (FileExistsError, ValueError):
        raise
    except OSError:
        raise ValueError("Advisor-v3.1 governance could not be persisted safely.") from None


def audit_advisor_v31_remediation(
    *,
    source_registry: BenchmarkRegistry,
    remediation_registry: BenchmarkRegistry,
    committed_amendment_path: Path,
    remediation_approval_reference: str,
) -> AdvisorV31RemediationReadinessManifest:
    """Write a separate governance-only readiness reference without qualification access."""

    source_root = source_registry.root_directory.resolve(strict=True)
    remediation_root = remediation_registry.root_directory.resolve(strict=True)
    if source_root == remediation_root:
        raise ValueError("Advisor-v3 source and v3.1 remediation registries must be distinct.")
    _reject_non_governance_destination_content(remediation_registry)
    amendment = load_committed_advisor_v31_protocol_amendment(committed_amendment_path)
    geometry_digest = _recompute_and_validate_observable_geometry(amendment)
    inspection = inspect_frozen_advisor_v3_corpus(source_registry)
    if inspection.readiness.advisor_evidence_ready:
        raise ValueError("Advisor-v3.1 remediation is invalid for an already-ready v3 source.")
    family_count, instance_count, catalogue_complete, complete_ood = _validate_source_catalogue(
        source_registry
    )
    counts = _role_evidence_counts(
        inspection.records,
        recipe_ids_by_digest=dict(inspection.recipe_ids_by_digest),
    )
    snapshot_digest = _frozen_source_snapshot_digest(source_registry, inspection)
    ready = (
        len(inspection.records) == 11_760
        and family_count == 84
        and instance_count == 336
        and catalogue_complete
        and counts["required_success"] == 4_320
        and counts["required_failure"] == 0
        and counts["required_missing"] == 0
        and counts["successful_cells"] == 1_440
        and counts["successful_families"] == 72
        and complete_ood == 12
        and counts["ood_diagnostic"] == 720
        and counts["nonrequired_ineligible"] == 6_720
    )
    readiness = AdvisorV31RemediationReadinessManifest(
        amendment_digest=amendment.amendment_digest,
        source_execution_approval_digest=inspection.approval.approval_digest,
        source_execution_provenance_digest=inspection.approval.execution_provenance_digest,
        source_v3_preregistration_digest=build_advisor_v3_preregistration().preregistration_digest,
        source_v3_readiness_digest=inspection.readiness.readiness_digest,
        source_registry_snapshot_digest=snapshot_digest,
        analysis_provenance_digest=advisor_v31_analysis_provenance_digest(),
        recomputed_geometry_coherence_digest=geometry_digest,
        source_completed_run_count=len(inspection.records),
        source_family_manifest_count=family_count,
        source_instance_manifest_count=instance_count,
        source_catalogue_integrity_checked=catalogue_complete,
        source_failure_sidecar_count=len(inspection.failure_digests),
        successful_qualification_required_family_count=counts["successful_families"],
        successful_qualification_required_evidence_cell_count=counts["successful_cells"],
        successful_qualification_required_adapter_run_count=counts["required_success"],
        failed_qualification_required_adapter_run_count=counts["required_failure"],
        missing_qualification_required_adapter_run_count=counts["required_missing"],
        complete_ood_geometry_family_count=complete_ood,
        completed_ood_diagnostic_adapter_run_count=counts["ood_diagnostic"],
        non_success_ood_diagnostic_adapter_run_count=counts["ood_non_success"],
        ineligible_nonrequired_recipe_run_count=counts["nonrequired_ineligible"],
        advisor_evidence_ready=ready,
    )
    approval = AdvisorV31RemediationApproval(
        approval_reference=remediation_approval_reference,
        human_approved=True,
        amendment_digest=amendment.amendment_digest,
        source_execution_approval_digest=inspection.approval.approval_digest,
        source_execution_provenance_digest=inspection.approval.execution_provenance_digest,
        source_v3_readiness_digest=inspection.readiness.readiness_digest,
        source_registry_snapshot_digest=snapshot_digest,
        analysis_provenance_digest=readiness.analysis_provenance_digest,
        recomputed_geometry_coherence_digest=geometry_digest,
    )
    governance = remediation_registry.root_directory / "governance"
    for path, artifact in (
        (governance / f"amendment.v3.1.{amendment.amendment_digest}.json", amendment),
        (governance / f"approval.v3.1.{approval.approval_digest}.json", approval),
        (governance / f"readiness.v3.1.{readiness.readiness_digest}.json", readiness),
    ):
        _persist_exact(path, artifact)
    return readiness


__all__ = [
    "AdvisorV31ProtocolAmendmentManifest",
    "AdvisorV31RemediationApproval",
    "AdvisorV31RemediationReadinessManifest",
    "FrozenAdvisorV3CorpusInspection",
    "advisor_v31_analysis_provenance_digest",
    "audit_advisor_v31_remediation",
    "build_advisor_v31_protocol_amendment",
    "frozen_advisor_v3_provenance_digest",
    "inspect_frozen_advisor_v3_corpus",
    "load_committed_advisor_v31_protocol_amendment",
]
