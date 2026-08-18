"""Native JSON persistence and integrity checks for candidate rankers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mapel_linkage.domain.errors import RankingError
from mapel_linkage.governance.atomic import atomic_write_bytes, atomic_write_text
from mapel_linkage.governance.paths import PathPolicy
from mapel_linkage.models.ranking.contracts import XGBoostRankingArtifact

_MAX_MODEL_BYTES = 128 * 1024 * 1024
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class WrittenRankingArtifact:
    model_path: Path = field(repr=False)
    manifest_path: Path = field(repr=False)
    model_digest: str

    def safe_summary(self) -> dict[str, str]:
        return {"model_digest": self.model_digest, "artifact_format": "xgboost_json"}


def _manifest(artifact: XGBoostRankingArtifact) -> dict[str, object]:
    return {
        "schema_version": "0.1",
        **artifact.safe_summary(),
        "engine_version": artifact.engine_version,
        "configuration_digest": artifact.configuration_digest,
        "random_seed": artifact.random_seed,
        "training_partition": artifact.training_partition,
        "feature_names": list(artifact.feature_names),
    }


def write_ranking_artifact(
    *,
    artifact: XGBoostRankingArtifact,
    model_path: str,
    manifest_path: str,
    policy: PathPolicy,
) -> WrittenRankingArtifact:
    model_destination = policy.resolve_output(model_path)
    manifest_destination = policy.resolve_output(manifest_path)
    if model_destination == manifest_destination:
        raise RankingError("ML-RANK-024", "Ranker model and manifest paths must differ.")
    model_destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_bytes(model_destination, artifact.model_json)
        atomic_write_text(
            manifest_destination,
            json.dumps(_manifest(artifact), indent=2, sort_keys=True) + "\n",
        )
    except (OSError, TypeError, ValueError):
        raise RankingError(
            "ML-RANK-025", "A ranking artifact could not be written safely."
        ) from None
    return WrittenRankingArtifact(model_destination, manifest_destination, artifact.model_digest)


def read_ranking_artifact(
    *,
    model_path: str,
    manifest_path: str,
    policy: PathPolicy,
) -> XGBoostRankingArtifact:
    model_source = policy.resolve_output(model_path)
    manifest_source = policy.resolve_output(manifest_path)
    try:
        if (
            model_source.suffix != ".json"
            or manifest_source.suffix != ".json"
            or not model_source.is_file()
            or not manifest_source.is_file()
            or model_source.stat().st_size > _MAX_MODEL_BYTES
            or manifest_source.stat().st_size > _MAX_MANIFEST_BYTES
        ):
            raise OSError
        model_json = model_source.read_bytes()
        raw = json.loads(manifest_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RankingError("ML-RANK-026", "A ranking artifact could not be read safely.") from None
    if not isinstance(raw, dict):
        raise RankingError("ML-RANK-027", "A ranking artifact manifest is invalid.")
    expected_keys = {
        "schema_version",
        "model_id",
        "model_version",
        "engine_version",
        "configuration_digest",
        "random_seed",
        "training_pair_count",
        "training_query_count",
        "feature_schema_digest",
        "label_authority_digest",
        "selection_digest",
        "parameter_digest",
        "model_digest",
        "artifact_digest",
        "xgboost_version",
        "query_side",
        "top_k",
        "decision_authority",
        "relationship_authority",
        "real_data_validation_status",
        "training_partition",
        "feature_names",
    }
    if (
        set(raw) != expected_keys
        or raw.get("schema_version") != "0.1"
        or raw.get("model_version") != "m2g-xgboost-ranker-v1"
        or raw.get("training_partition") != "training"
        or raw.get("decision_authority") != "ranking_only"
        or raw.get("relationship_authority") != "none"
        or raw.get("real_data_validation_status") != "not_established"
    ):
        raise RankingError("ML-RANK-027", "A ranking artifact manifest is invalid.")
    try:
        artifact = XGBoostRankingArtifact(
            model_id=str(raw["model_id"]),
            model_version=raw["model_version"],
            engine_version=str(raw["engine_version"]),
            configuration_digest=str(raw["configuration_digest"]),
            random_seed=int(raw["random_seed"]),
            training_pair_count=int(raw["training_pair_count"]),
            training_query_count=int(raw["training_query_count"]),
            feature_schema_digest=str(raw["feature_schema_digest"]),
            label_authority_digest=str(raw["label_authority_digest"]),
            selection_digest=str(raw["selection_digest"]),
            parameter_digest=str(raw["parameter_digest"]),
            model_digest=str(raw["model_digest"]),
            artifact_digest=str(raw["artifact_digest"]),
            xgboost_version=str(raw["xgboost_version"]),
            query_side=raw["query_side"],
            top_k=int(raw["top_k"]),
            feature_names=tuple(str(value) for value in raw["feature_names"]),
            model_json=model_json,
        )
    except (KeyError, TypeError, ValueError):
        raise RankingError("ML-RANK-027", "A ranking artifact manifest is invalid.") from None
    return artifact
