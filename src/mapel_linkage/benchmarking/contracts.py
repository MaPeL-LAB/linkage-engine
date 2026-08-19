"""Versioned aggregate contracts for the future synthetic benchmark library."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

Identifier = Annotated[
    StrictStr,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$"),
]
Digest = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
CommitHash = Annotated[StrictStr, Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")]


class BenchmarkEvidenceScope(StrEnum):
    """Permitted evidence classes for benchmark records."""

    GLOBAL_SYNTHETIC = "global_synthetic"
    LOCAL_SCHEMA_MATCHED_SYNTHETIC = "local_schema_matched_synthetic"


class BenchmarkRunStatus(StrEnum):
    """Complete status vocabulary; unsuccessful runs remain evidence."""

    SUCCESS = "success"
    FAILED_FIT = "failed_fit"
    TIMEOUT = "timeout"
    MEMORY_FAILURE = "memory_failure"
    INELIGIBLE = "ineligible"
    ABSTAINED = "abstained"
    NUMERICAL_FAILURE = "numerical_failure"
    CANDIDATE_BUDGET_FAILURE = "candidate_budget_failure"


class BenchmarkNode(BaseModel):
    """Strict immutable benchmark contract."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        hide_input_in_errors=True,
    )


class ScenarioFamilyManifest(BenchmarkNode):
    """One scientifically coherent generating mechanism or corruption regime."""

    scenario_schema_version: Literal["1"] = "1"
    family_id: Identifier
    mechanism_tags: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=64)]
    observable_profile_schema_version: Literal["1"] = "1"
    latent_scenario_manifest_digest: Digest
    prospectively_held_out: StrictBool
    evidence_scope: BenchmarkEvidenceScope = BenchmarkEvidenceScope.GLOBAL_SYNTHETIC
    contains_real_data: Literal[False] = False
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_tags(self) -> Self:
        if len(self.mechanism_tags) != len(set(self.mechanism_tags)):
            raise ValueError("Scenario-family mechanism tags must be unique.")
        if self.evidence_scope is not BenchmarkEvidenceScope.GLOBAL_SYNTHETIC:
            raise ValueError("Global scenario-family manifests must use synthetic evidence only.")
        return self

    @property
    def family_digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class ScenarioInstanceManifest(BenchmarkNode):
    """One parameterised instance within a scenario family."""

    scenario_schema_version: Literal["1"] = "1"
    family_id: Identifier
    instance_id: Identifier
    family_digest: Digest
    latent_parameter_manifest_digest: Digest
    observable_profile_digest: Digest
    planned_replicates: Annotated[StrictInt, Field(ge=1, le=10_000)]
    evidence_scope: BenchmarkEvidenceScope = BenchmarkEvidenceScope.GLOBAL_SYNTHETIC
    contains_record_values: Literal[False] = False
    operational_validity: Literal["not_established"] = "not_established"

    @property
    def instance_digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class BenchmarkRunRecord(BenchmarkNode):
    """Aggregate result for one scenario instance, recipe, seed, and environment."""

    registry_schema_version: Literal["1"] = "1"
    run_id: Identifier
    family_id: Identifier
    instance_id: Identifier
    replicate_id: Identifier
    task_profile_digest: Digest
    pipeline_recipe_digest: Digest
    engine_commit: CommitHash
    dependency_lock_digest: Digest
    environment_digest: Digest
    random_seed: Annotated[StrictInt, Field(ge=0, le=4_294_967_295)]
    status: BenchmarkRunStatus
    failure_code: Identifier | None = None
    aggregate_metrics_digest: Digest | None = None
    stage_artifact_manifest_digest: Digest | None = None
    runtime_ms: Annotated[StrictInt, Field(ge=0)] | None = None
    peak_memory_mb: Annotated[StrictInt, Field(ge=0)] | None = None
    evidence_scope: BenchmarkEvidenceScope = BenchmarkEvidenceScope.GLOBAL_SYNTHETIC
    contains_record_values: Literal[False] = False
    contains_identifiers: Literal[False] = False
    contains_candidate_pairs: Literal[False] = False
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_status_payload(self) -> Self:
        if self.status is BenchmarkRunStatus.SUCCESS:
            if self.failure_code is not None:
                raise ValueError("A successful benchmark run cannot retain a failure code.")
            if self.aggregate_metrics_digest is None or self.runtime_ms is None:
                raise ValueError(
                    "A successful benchmark run requires aggregate metrics and runtime."
                )
        elif self.failure_code is None:
            raise ValueError("An unsuccessful benchmark run requires a stable failure code.")
        return self

    @property
    def run_digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "run_digest": self.run_digest,
            "family_id": self.family_id,
            "instance_id": self.instance_id,
            "replicate_id": self.replicate_id,
            "status": self.status.value,
            "failure_code": self.failure_code,
            "runtime_ms": self.runtime_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "evidence_scope": self.evidence_scope.value,
            "contains_record_values": self.contains_record_values,
            "contains_identifiers": self.contains_identifiers,
            "contains_candidate_pairs": self.contains_candidate_pairs,
            "operational_validity": self.operational_validity,
        }


class BenchmarkRegistrySnapshot(BenchmarkNode):
    """Immutable aggregate registry snapshot used by future advisor stages."""

    registry_schema_version: Literal["1"] = "1"
    snapshot_id: Identifier
    records: Annotated[tuple[BenchmarkRunRecord, ...], Field(max_length=100_000)] = ()
    evidence_scope: BenchmarkEvidenceScope = BenchmarkEvidenceScope.GLOBAL_SYNTHETIC
    contains_record_values: Literal[False] = False
    contains_identifiers: Literal[False] = False
    contains_candidate_pairs: Literal[False] = False
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_records(self) -> Self:
        run_ids = [record.run_id for record in self.records]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("Benchmark registry run IDs must be unique.")
        if any(record.evidence_scope is not self.evidence_scope for record in self.records):
            raise ValueError("Registry records must share the registry evidence scope.")
        return self

    @property
    def registry_digest(self) -> str:
        payload = {
            "registry_schema_version": self.registry_schema_version,
            "snapshot_id": self.snapshot_id,
            "run_digests": [record.run_digest for record in self.records],
            "evidence_scope": self.evidence_scope.value,
            "contains_record_values": self.contains_record_values,
            "contains_identifiers": self.contains_identifiers,
            "contains_candidate_pairs": self.contains_candidate_pairs,
            "operational_validity": self.operational_validity,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, object]:
        status_counts = {status.value: 0 for status in BenchmarkRunStatus}
        for record in self.records:
            status_counts[record.status.value] += 1
        return {
            "snapshot_id": self.snapshot_id,
            "registry_digest": self.registry_digest,
            "run_count": len(self.records),
            "family_count": len({record.family_id for record in self.records}),
            "instance_count": len({record.instance_id for record in self.records}),
            "status_counts": status_counts,
            "evidence_scope": self.evidence_scope.value,
            "contains_record_values": self.contains_record_values,
            "contains_identifiers": self.contains_identifiers,
            "contains_candidate_pairs": self.contains_candidate_pairs,
            "operational_validity": self.operational_validity,
        }


class BenchmarkAggregateMetrics(BenchmarkNode):
    """Aggregate evaluation metrics for a single benchmark run."""

    candidate_recall: float = Field(ge=0.0, le=1.0)
    candidate_recall_at_k: dict[str, float] = Field(default_factory=dict)
    sensitivity: float = Field(ge=0.0, le=1.0)
    positive_predictive_value: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    calibration_intercept: float
    calibration_slope: float
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    runtime_ms: Annotated[StrictInt, Field(ge=0)]
    peak_memory_mb: Annotated[StrictInt, Field(ge=0)]
    evaluation_scope: str = "synthetic_benchmark"
    operational_validity: Literal["not_established"] = "not_established"

    @property
    def metrics_digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, object]:
        return {
            "candidate_recall": self.candidate_recall,
            "candidate_recall_at_k": self.candidate_recall_at_k,
            "sensitivity": self.sensitivity,
            "positive_predictive_value": self.positive_predictive_value,
            "brier_score": self.brier_score,
            "calibration_intercept": self.calibration_intercept,
            "calibration_slope": self.calibration_slope,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "runtime_ms": self.runtime_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "metrics_digest": self.metrics_digest,
            "evaluation_scope": self.evaluation_scope,
            "operational_validity": self.operational_validity,
        }


class BenchmarkFailureRecord(BenchmarkNode):
    """Retained failure evidence for an unsuccessful benchmark run."""

    run_id: Identifier
    family_id: Identifier
    instance_id: Identifier
    replicate_id: Identifier
    recipe_id: Identifier
    status: BenchmarkRunStatus
    failure_code: Identifier
    error_message: StrictStr
    evidence_scope: BenchmarkEvidenceScope = BenchmarkEvidenceScope.GLOBAL_SYNTHETIC
    operational_validity: Literal["not_established"] = "not_established"

    @property
    def failure_digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "family_id": self.family_id,
            "instance_id": self.instance_id,
            "replicate_id": self.replicate_id,
            "recipe_id": self.recipe_id,
            "status": self.status.value,
            "failure_code": self.failure_code,
            "error_message": self.error_message,
            "failure_digest": self.failure_digest,
            "evidence_scope": self.evidence_scope.value,
            "operational_validity": self.operational_validity,
        }


class CoverageSummaryReport(BenchmarkNode):
    """Aggregate benchmark coverage and evidence quality summary."""

    report_schema_version: Literal["1"] = "1"
    report_id: Identifier
    snapshot_digest: Digest
    family_count: Annotated[StrictInt, Field(ge=0)]
    instance_count: Annotated[StrictInt, Field(ge=0)]
    replicate_count: Annotated[StrictInt, Field(ge=0)]
    run_count: Annotated[StrictInt, Field(ge=0)]
    successful_run_count: Annotated[StrictInt, Field(ge=0)]
    failed_run_count: Annotated[StrictInt, Field(ge=0)]
    status_counts: dict[str, int]
    recipe_by_family_coverage: dict[str, tuple[str, ...]]
    pairwise_comparison_counts: dict[str, int]
    held_out_mechanism_count: Annotated[StrictInt, Field(ge=0)]
    failure_rates_by_recipe: dict[str, float]
    evidence_scope: BenchmarkEvidenceScope = BenchmarkEvidenceScope.GLOBAL_SYNTHETIC
    contains_record_values: Literal[False] = False
    operational_validity: Literal["not_established"] = "not_established"

    @property
    def report_digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, object]:
        return {
            "report_schema_version": self.report_schema_version,
            "report_id": self.report_id,
            "report_digest": self.report_digest,
            "snapshot_digest": self.snapshot_digest,
            "family_count": self.family_count,
            "instance_count": self.instance_count,
            "replicate_count": self.replicate_count,
            "run_count": self.run_count,
            "successful_run_count": self.successful_run_count,
            "failed_run_count": self.failed_run_count,
            "status_counts": self.status_counts,
            "recipe_by_family_coverage": self.recipe_by_family_coverage,
            "pairwise_comparison_counts": self.pairwise_comparison_counts,
            "held_out_mechanism_count": self.held_out_mechanism_count,
            "failure_rates_by_recipe": self.failure_rates_by_recipe,
            "evidence_scope": self.evidence_scope.value,
            "contains_record_values": self.contains_record_values,
            "operational_validity": self.operational_validity,
        }


__all__ = [
    "BenchmarkAggregateMetrics",
    "BenchmarkEvidenceScope",
    "BenchmarkFailureRecord",
    "BenchmarkRegistrySnapshot",
    "BenchmarkRunRecord",
    "BenchmarkRunStatus",
    "CoverageSummaryReport",
    "ScenarioFamilyManifest",
    "ScenarioInstanceManifest",
]
