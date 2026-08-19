"""Immutable stage-artifact and out-of-fold provenance contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

Digest = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
Identifier = Annotated[
    StrictStr,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$"),
]
RunId = Annotated[
    StrictStr,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"),
]
StageName = Literal[
    "configuration",
    "preprocessing",
    "candidate_generation",
    "comparison_features",
    "model_training",
    "model_scoring",
    "champion_selection",
    "calibration",
    "ranking",
    "assignment",
    "decisions",
    "adjudication",
    "evaluation",
    "multi_source_resolution",
]
ArtifactKind = Literal[
    "execution_plan",
    "canonical_table",
    "candidate_table",
    "feature_table",
    "model_manifest",
    "score_table",
    "selection_manifest",
    "calibrator_manifest",
    "ranking_manifest",
    "assignment_table",
    "relationship_table",
    "review_queue",
    "evaluation_report",
    "cluster_crosswalk",
]


class ArtifactNode(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        hide_input_in_errors=True,
    )


class StageArtifactRef(ArtifactNode):
    """Aggregate-only reference to a restricted or unrestricted stage artifact."""

    artifact_id: Identifier
    stage: StageName
    kind: ArtifactKind
    run_id: RunId
    engine_version: StrictStr
    artifact_digest: Digest
    configuration_digest: Digest
    schema_digest: Digest | None = None
    upstream_artifact_digests: Annotated[tuple[Digest, ...], Field(max_length=64)] = ()
    contains_row_level_data: StrictBool
    restricted: StrictBool
    decision_authority: Literal["none", "evidence_only"] = "none"
    merge_authority: Literal["none"] = "none"

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        if self.contains_row_level_data and not self.restricted:
            raise ValueError("Row-level stage artifacts must remain restricted.")
        if len(self.upstream_artifact_digests) != len(set(self.upstream_artifact_digests)):
            raise ValueError("Stage-artifact upstream digests must be unique.")
        if self.artifact_digest in self.upstream_artifact_digests:
            raise ValueError("A stage artifact cannot depend on itself.")
        return self

    @property
    def lineage_digest(self) -> str:
        payload = {
            "artifact_id": self.artifact_id,
            "stage": self.stage,
            "kind": self.kind,
            "run_id": self.run_id,
            "engine_version": self.engine_version,
            "artifact_digest": self.artifact_digest,
            "configuration_digest": self.configuration_digest,
            "schema_digest": self.schema_digest,
            "upstream_artifact_digests": self.upstream_artifact_digests,
            "contains_row_level_data": self.contains_row_level_data,
            "restricted": self.restricted,
            "decision_authority": self.decision_authority,
            "merge_authority": self.merge_authority,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, int | str | bool | None]:
        return {
            "artifact_id": self.artifact_id,
            "stage": self.stage,
            "kind": self.kind,
            "run_id": self.run_id,
            "engine_version": self.engine_version,
            "artifact_digest": self.artifact_digest,
            "configuration_digest": self.configuration_digest,
            "schema_digest": self.schema_digest,
            "upstream_artifact_count": len(self.upstream_artifact_digests),
            "contains_row_level_data": self.contains_row_level_data,
            "restricted": self.restricted,
            "decision_authority": self.decision_authority,
            "merge_authority": self.merge_authority,
            "lineage_digest": self.lineage_digest,
        }


class OutOfFoldPredictionManifest(ArtifactNode):
    """Aggregate provenance for stacking inputs created within training data only."""

    model_id: Identifier
    model_version: StrictStr
    model_artifact_digest: Digest
    feature_schema_digest: Digest
    label_authority_digest: Digest
    split_manifest_digest: Digest
    fold_count: Annotated[StrictInt, Field(ge=2, le=100)]
    pair_count: Annotated[StrictInt, Field(gt=0)]
    prediction_digest: Digest
    partition: Literal["training_oof"] = "training_oof"
    test_partition_used: Literal[False] = False
    calibration_partition_used: Literal[False] = False
    decision_partition_used: Literal[False] = False
    decision_authority: Literal["evidence_only"] = "evidence_only"
    merge_authority: Literal["none"] = "none"

    @property
    def manifest_digest(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, int | str | bool]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_artifact_digest": self.model_artifact_digest,
            "feature_schema_digest": self.feature_schema_digest,
            "label_authority_digest": self.label_authority_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "fold_count": self.fold_count,
            "pair_count": self.pair_count,
            "prediction_digest": self.prediction_digest,
            "partition": self.partition,
            "test_partition_used": self.test_partition_used,
            "calibration_partition_used": self.calibration_partition_used,
            "decision_partition_used": self.decision_partition_used,
            "decision_authority": self.decision_authority,
            "merge_authority": self.merge_authority,
            "manifest_digest": self.manifest_digest,
        }


class StageArtifactLedger(ArtifactNode):
    """Ordered acyclic lineage ledger over aggregate artifact references."""

    external_input_digests: Annotated[tuple[Digest, ...], Field(max_length=64)] = ()
    artifacts: Annotated[tuple[StageArtifactRef, ...], Field(min_length=1, max_length=512)]

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        if len(self.external_input_digests) != len(set(self.external_input_digests)):
            raise ValueError("External input digests must be unique.")
        ids: set[str] = set()
        digests: set[str] = set()
        known = set(self.external_input_digests)
        for artifact in self.artifacts:
            if artifact.artifact_id in ids or artifact.artifact_digest in digests:
                raise ValueError("Stage-artifact ledger entries must be unique.")
            if not set(artifact.upstream_artifact_digests).issubset(known):
                raise ValueError("A stage artifact references unavailable upstream evidence.")
            ids.add(artifact.artifact_id)
            digests.add(artifact.artifact_digest)
            known.add(artifact.artifact_digest)
        return self

    @property
    def ledger_digest(self) -> str:
        payload = {
            "external_input_digests": self.external_input_digests,
            "lineage_digests": tuple(artifact.lineage_digest for artifact in self.artifacts),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "external_input_count": len(self.external_input_digests),
            "artifact_count": len(self.artifacts),
            "ledger_digest": self.ledger_digest,
        }


__all__ = [
    "OutOfFoldPredictionManifest",
    "StageArtifactLedger",
    "StageArtifactRef",
]
