"""Strict aggregate-only artifacts for bounded synthetic deduplication modes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal

from mapel_linkage.domain.errors import PipelineError

_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MAX_BYTES = 262_144


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_digest(value: object) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise PipelineError("ML-MODE-001", "A synthetic mode artifact digest is invalid.")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PipelineError("ML-MODE-002", "A synthetic mode artifact has duplicate keys.")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class SyntheticModeOrchestrationArtifact:
    """Immutable executable-evidence binding for dedupe-only mode orchestration."""

    linkage_mode: Literal["dedupe_only", "link_and_dedupe"]
    assignment_constraint: Literal["one_to_one", "unconstrained"]
    configuration_digest: str
    registry_digest: str
    synthetic_bundle_digest: str
    generator_version: str
    random_seed: Literal[20260816]
    candidate_plan_digests: tuple[str, ...]
    calibrated_evidence_digests: tuple[str, ...]
    feature_schema_digest: str
    champion_model_id: str
    champion_model_version: str
    champion_artifact_digest: str
    calibrator_digest: str
    partition_manifest_digest: str
    deduplication_plan_digest: str
    assignment_plan_digest: str
    schema_version: Literal["1"] = "1"
    data_origin: Literal["package_generated_synthetic"] = "package_generated_synthetic"
    evaluation_scope: Literal["synthetic_mechanical_evaluation"] = "synthetic_mechanical_evaluation"
    operational_validation: Literal["not_established"] = "not_established"
    decision_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"

    def __post_init__(self) -> None:
        expected_count = 1 if self.linkage_mode == "dedupe_only" else 3
        expected_constraint = (
            "unconstrained" if self.linkage_mode == "dedupe_only" else "one_to_one"
        )
        if (
            self.schema_version != "1"
            or self.linkage_mode not in {"dedupe_only", "link_and_dedupe"}
            or self.assignment_constraint != expected_constraint
            or self.random_seed != 20260816
            or len(self.candidate_plan_digests) != expected_count
            or len(self.calibrated_evidence_digests) != expected_count
            or len(set(self.candidate_plan_digests)) != expected_count
            or len(set(self.calibrated_evidence_digests)) != expected_count
            or _VERSION.fullmatch(self.generator_version) is None
            or _IDENTIFIER.fullmatch(self.champion_model_id) is None
            or _IDENTIFIER.fullmatch(self.champion_model_version) is None
            or self.data_origin != "package_generated_synthetic"
            or self.evaluation_scope != "synthetic_mechanical_evaluation"
            or self.operational_validation != "not_established"
            or self.decision_authority != "none"
            or self.merge_authority != "none"
        ):
            raise PipelineError("ML-MODE-003", "A synthetic mode artifact contract is invalid.")
        for digest in (
            self.configuration_digest,
            self.registry_digest,
            self.synthetic_bundle_digest,
            *self.candidate_plan_digests,
            *self.calibrated_evidence_digests,
            self.feature_schema_digest,
            self.champion_artifact_digest,
            self.calibrator_digest,
            self.partition_manifest_digest,
            self.deduplication_plan_digest,
            self.assignment_plan_digest,
        ):
            _require_digest(digest)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "linkage_mode": self.linkage_mode,
            "assignment_constraint": self.assignment_constraint,
            "configuration_digest": self.configuration_digest,
            "registry_digest": self.registry_digest,
            "synthetic_bundle_digest": self.synthetic_bundle_digest,
            "generator_version": self.generator_version,
            "random_seed": self.random_seed,
            "candidate_plan_digests": list(self.candidate_plan_digests),
            "calibrated_evidence_digests": list(self.calibrated_evidence_digests),
            "feature_schema_digest": self.feature_schema_digest,
            "champion_model_id": self.champion_model_id,
            "champion_model_version": self.champion_model_version,
            "champion_artifact_digest": self.champion_artifact_digest,
            "calibrator_digest": self.calibrator_digest,
            "partition_manifest_digest": self.partition_manifest_digest,
            "deduplication_plan_digest": self.deduplication_plan_digest,
            "assignment_plan_digest": self.assignment_plan_digest,
            "data_origin": self.data_origin,
            "evaluation_scope": self.evaluation_scope,
            "operational_validation": self.operational_validation,
            "decision_authority": self.decision_authority,
            "merge_authority": self.merge_authority,
        }

    @property
    def artifact_digest(self) -> str:
        return _canonical_digest(self._payload())

    def safe_summary(self) -> dict[str, object]:
        return {
            "artifact_digest": self.artifact_digest,
            "linkage_mode": self.linkage_mode,
            "assignment_constraint": self.assignment_constraint,
            "candidate_evidence_count": len(self.calibrated_evidence_digests),
            "champion_model_id": self.champion_model_id,
            "champion_model_version": self.champion_model_version,
            "operational_validation": self.operational_validation,
            "decision_authority": self.decision_authority,
            "merge_authority": self.merge_authority,
        }


def serialize_mode_orchestration_artifact(
    artifact: SyntheticModeOrchestrationArtifact,
) -> str:
    payload = artifact._payload()
    payload["artifact_digest"] = artifact.artifact_digest
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def deserialize_mode_orchestration_artifact(
    text: str,
) -> SyntheticModeOrchestrationArtifact:
    if len(text.encode("utf-8")) > _MAX_BYTES:
        raise PipelineError("ML-MODE-004", "A synthetic mode artifact exceeds its size limit.")
    try:
        raw = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except PipelineError:
        raise
    except (TypeError, ValueError):
        raise PipelineError("ML-MODE-005", "A synthetic mode artifact is not valid JSON.") from None
    expected_keys = {
        "schema_version",
        "linkage_mode",
        "assignment_constraint",
        "configuration_digest",
        "registry_digest",
        "synthetic_bundle_digest",
        "generator_version",
        "random_seed",
        "candidate_plan_digests",
        "calibrated_evidence_digests",
        "feature_schema_digest",
        "champion_model_id",
        "champion_model_version",
        "champion_artifact_digest",
        "calibrator_digest",
        "partition_manifest_digest",
        "deduplication_plan_digest",
        "assignment_plan_digest",
        "data_origin",
        "evaluation_scope",
        "operational_validation",
        "decision_authority",
        "merge_authority",
        "artifact_digest",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise PipelineError("ML-MODE-006", "A synthetic mode artifact schema is invalid.")
    candidate_digests = raw["candidate_plan_digests"]
    evidence_digests = raw["calibrated_evidence_digests"]
    if (
        not isinstance(candidate_digests, list)
        or not all(isinstance(value, str) for value in candidate_digests)
        or not isinstance(evidence_digests, list)
        or not all(isinstance(value, str) for value in evidence_digests)
        or type(raw["random_seed"]) is not int
        or raw["random_seed"] != 20260816
    ):
        raise PipelineError("ML-MODE-006", "A synthetic mode artifact schema is invalid.")
    try:
        artifact = SyntheticModeOrchestrationArtifact(
            schema_version=raw["schema_version"],
            linkage_mode=raw["linkage_mode"],
            assignment_constraint=raw["assignment_constraint"],
            configuration_digest=raw["configuration_digest"],
            registry_digest=raw["registry_digest"],
            synthetic_bundle_digest=raw["synthetic_bundle_digest"],
            generator_version=raw["generator_version"],
            random_seed=20260816,
            candidate_plan_digests=tuple(candidate_digests),
            calibrated_evidence_digests=tuple(evidence_digests),
            feature_schema_digest=raw["feature_schema_digest"],
            champion_model_id=raw["champion_model_id"],
            champion_model_version=raw["champion_model_version"],
            champion_artifact_digest=raw["champion_artifact_digest"],
            calibrator_digest=raw["calibrator_digest"],
            partition_manifest_digest=raw["partition_manifest_digest"],
            deduplication_plan_digest=raw["deduplication_plan_digest"],
            assignment_plan_digest=raw["assignment_plan_digest"],
            data_origin=raw["data_origin"],
            evaluation_scope=raw["evaluation_scope"],
            operational_validation=raw["operational_validation"],
            decision_authority=raw["decision_authority"],
            merge_authority=raw["merge_authority"],
        )
    except (PipelineError, TypeError, ValueError):
        raise PipelineError("ML-MODE-006", "A synthetic mode artifact schema is invalid.") from None
    if (
        not isinstance(raw["artifact_digest"], str)
        or artifact.artifact_digest != raw["artifact_digest"]
    ):
        raise PipelineError("ML-MODE-007", "A synthetic mode artifact integrity check failed.")
    if text != serialize_mode_orchestration_artifact(artifact):
        raise PipelineError("ML-MODE-007", "A synthetic mode artifact integrity check failed.")
    return artifact


@dataclass(frozen=True, slots=True)
class SyntheticModeRunArtifact:
    """Aggregate-only result bound to one verified mode orchestration artifact."""

    linkage_mode: Literal["dedupe_only", "link_and_dedupe"]
    orchestration_artifact_digest: str
    configuration_digest: str
    result_digest: str
    input_record_count: int
    candidate_pair_count: int
    cluster_count: int
    selected_edge_count: int
    schema_version: Literal["1"] = "1"
    operational_validation: Literal["not_established"] = "not_established"
    decision_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"

    def __post_init__(self) -> None:
        for digest in (
            self.orchestration_artifact_digest,
            self.configuration_digest,
            self.result_digest,
        ):
            _require_digest(digest)
        if any(
            type(value) is not int or value < 0
            for value in (
                self.input_record_count,
                self.candidate_pair_count,
                self.cluster_count,
                self.selected_edge_count,
            )
        ):
            raise PipelineError("ML-MODE-008", "Synthetic mode aggregate counts are invalid.")
        if (
            self.schema_version != "1"
            or self.linkage_mode not in {"dedupe_only", "link_and_dedupe"}
            or self.operational_validation != "not_established"
            or self.decision_authority != "none"
            or self.merge_authority != "none"
        ):
            raise PipelineError("ML-MODE-009", "A synthetic mode run contract is invalid.")

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "linkage_mode": self.linkage_mode,
            "orchestration_artifact_digest": self.orchestration_artifact_digest,
            "configuration_digest": self.configuration_digest,
            "result_digest": self.result_digest,
            "input_record_count": self.input_record_count,
            "candidate_pair_count": self.candidate_pair_count,
            "cluster_count": self.cluster_count,
            "selected_edge_count": self.selected_edge_count,
            "operational_validation": self.operational_validation,
            "decision_authority": self.decision_authority,
            "merge_authority": self.merge_authority,
        }

    @property
    def run_digest(self) -> str:
        return _canonical_digest(self._payload())

    def safe_summary(self) -> dict[str, object]:
        return {**self._payload(), "run_digest": self.run_digest}


def serialize_mode_run_artifact(artifact: SyntheticModeRunArtifact) -> str:
    return json.dumps(artifact.safe_summary(), sort_keys=True, separators=(",", ":")) + "\n"


def deserialize_mode_run_artifact(text: str) -> SyntheticModeRunArtifact:
    if len(text.encode("utf-8")) > _MAX_BYTES:
        raise PipelineError("ML-MODE-004", "A synthetic mode artifact exceeds its size limit.")
    try:
        raw = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except PipelineError:
        raise
    except (TypeError, ValueError):
        raise PipelineError("ML-MODE-005", "A synthetic mode artifact is not valid JSON.") from None
    expected = {
        "schema_version",
        "linkage_mode",
        "orchestration_artifact_digest",
        "configuration_digest",
        "result_digest",
        "input_record_count",
        "candidate_pair_count",
        "cluster_count",
        "selected_edge_count",
        "operational_validation",
        "decision_authority",
        "merge_authority",
        "run_digest",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != expected
        or any(
            type(raw[key]) is not int
            for key in (
                "input_record_count",
                "candidate_pair_count",
                "cluster_count",
                "selected_edge_count",
            )
        )
    ):
        raise PipelineError("ML-MODE-006", "A synthetic mode artifact schema is invalid.")
    try:
        artifact = SyntheticModeRunArtifact(
            schema_version=raw["schema_version"],
            linkage_mode=raw["linkage_mode"],
            orchestration_artifact_digest=raw["orchestration_artifact_digest"],
            configuration_digest=raw["configuration_digest"],
            result_digest=raw["result_digest"],
            input_record_count=raw["input_record_count"],
            candidate_pair_count=raw["candidate_pair_count"],
            cluster_count=raw["cluster_count"],
            selected_edge_count=raw["selected_edge_count"],
            operational_validation=raw["operational_validation"],
            decision_authority=raw["decision_authority"],
            merge_authority=raw["merge_authority"],
        )
    except (PipelineError, TypeError, ValueError):
        raise PipelineError("ML-MODE-006", "A synthetic mode artifact schema is invalid.") from None
    if not isinstance(raw["run_digest"], str) or artifact.run_digest != raw["run_digest"]:
        raise PipelineError("ML-MODE-007", "A synthetic mode artifact integrity check failed.")
    if text != serialize_mode_run_artifact(artifact):
        raise PipelineError("ML-MODE-007", "A synthetic mode artifact integrity check failed.")
    return artifact


__all__ = [
    "SyntheticModeOrchestrationArtifact",
    "SyntheticModeRunArtifact",
    "deserialize_mode_orchestration_artifact",
    "deserialize_mode_run_artifact",
    "serialize_mode_orchestration_artifact",
    "serialize_mode_run_artifact",
]
