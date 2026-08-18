"""Native LightGBM learning-to-rank adapter with no identity-decision authority."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from mapel_linkage import __version__
from mapel_linkage.configuration.models import RankingModelConfig
from mapel_linkage.domain.errors import RankingError
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.paths import PathPolicy
from mapel_linkage.models.ranking.artifacts import WrittenRankingArtifact
from mapel_linkage.models.ranking.contracts import (
    LightGBMRankingArtifact,
    RankingFeatureMatrix,
    RankingMatrix,
    RankingScoreBatch,
    canonical_digest,
    ranking_artifact_digest,
)

try:
    import lightgbm as _lightgbm
except ModuleNotFoundError:  # pragma: no cover - optional dependency boundary
    _lightgbm = None  # type: ignore[assignment]


def _require_lightgbm() -> Any:
    if _lightgbm is None:
        raise RankingError("ML-RANK-017", "The LightGBM ranking dependency is unavailable.")
    return _lightgbm


class LightGBMRanker:
    """Train and score a query-grouped candidate ranker using LightGBM."""

    @staticmethod
    def fit(
        *,
        matrix: RankingMatrix,
        model: RankingModelConfig,
        random_seed: int,
        configuration_digest: str,
    ) -> LightGBMRankingArtifact:
        if model.implementation != "lightgbm_ranker" or matrix.partition != "training":
            raise RankingError("ML-RANK-018", "The ranking training contract is invalid.")
        if matrix.pair_count > model.maximum_training_pairs:
            raise RankingError(
                "ML-RANK-019", "The ranking matrix exceeds its configured pair budget."
            )
        if (
            random_seed < 0
            or len(configuration_digest) != 64
            or any(character not in "0123456789abcdef" for character in configuration_digest)
        ):
            raise RankingError("ML-RANK-039", "Ranking training provenance is invalid.")
        lgb = _require_lightgbm()
        dataset = lgb.Dataset(
            matrix.features,
            label=matrix.relevance,
            group=list(matrix.group_sizes),
            feature_name=list(matrix.feature_names),
            free_raw_data=False,
        )
        parameters: dict[str, object] = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [model.top_k],
            "learning_rate": model.learning_rate,
            "max_depth": model.max_depth,
            "min_child_samples": 1,
            "min_data_in_bin": 1,
            "random_state": random_seed,
            "num_threads": 1,
            "verbosity": -1,
            "deterministic": True,
        }
        booster = lgb.train(parameters, dataset, num_boost_round=model.n_estimators)
        model_str = booster.model_to_string()
        model_digest = hashlib.sha256(model_str.encode("utf-8")).hexdigest()
        parameter_digest = canonical_digest(parameters)
        artifact_digest = ranking_artifact_digest(
            model_id=model.model_id,
            model_version="m5-lightgbm-ranker-v1",
            engine_version=__version__,
            configuration_digest=configuration_digest,
            random_seed=random_seed,
            training_pair_count=matrix.pair_count,
            training_query_count=matrix.query_count,
            feature_schema_digest=matrix.feature_schema_digest,
            label_authority_digest=matrix.label_authority_digest,
            selection_digest=matrix.selection_digest,
            parameter_digest=parameter_digest,
            model_digest=model_digest,
            lightgbm_version=str(lgb.__version__),
            query_side=matrix.query_side,
            top_k=model.top_k,
            feature_names=matrix.feature_names,
        )
        return LightGBMRankingArtifact(
            model_id=model.model_id,
            model_version="m5-lightgbm-ranker-v1",
            engine_version=__version__,
            configuration_digest=configuration_digest,
            random_seed=random_seed,
            training_pair_count=matrix.pair_count,
            training_query_count=matrix.query_count,
            feature_schema_digest=matrix.feature_schema_digest,
            label_authority_digest=matrix.label_authority_digest,
            selection_digest=matrix.selection_digest,
            parameter_digest=parameter_digest,
            model_digest=model_digest,
            artifact_digest=artifact_digest,
            lightgbm_version=str(lgb.__version__),
            query_side=matrix.query_side,
            top_k=model.top_k,
            feature_names=matrix.feature_names,
            model_str=model_str,
        )

    @staticmethod
    def score(
        *,
        matrix: RankingFeatureMatrix,
        model: LightGBMRankingArtifact,
    ) -> RankingScoreBatch:
        if (
            matrix.feature_schema_digest != model.feature_schema_digest
            or matrix.feature_names != model.feature_names
        ):
            raise RankingError("ML-RANK-020", "Ranking features do not match the model artifact.")
        if matrix.query_side != model.query_side:
            raise RankingError("ML-RANK-021", "Ranking query orientation is inconsistent.")
        lgb = _require_lightgbm()
        try:
            booster = lgb.Booster(model_str=model.model_str)
            raw_scores = np.asarray(booster.predict(matrix.features), dtype=np.float64)
        except Exception:
            raise RankingError(
                "ML-RANK-022", "The ranking model returned invalid scores."
            ) from None
        if (
            raw_scores.ndim != 1
            or len(raw_scores) != matrix.pair_count
            or not np.all(np.isfinite(raw_scores))
        ):
            raise RankingError("ML-RANK-022", "The ranking model returned invalid scores.")
        ranks = np.empty(matrix.pair_count, dtype=np.int64)
        membership = np.zeros(matrix.pair_count, dtype=np.int64)
        offset = 0
        for size in matrix.group_sizes:
            indices = list(range(offset, offset + size))
            ordered = sorted(
                indices, key=lambda index: (-float(raw_scores[index]), matrix.pair_digests[index])
            )
            for rank, index in enumerate(ordered, start=1):
                ranks[index] = rank
                membership[index] = int(rank <= model.top_k)
            offset += size
        return RankingScoreBatch(
            pair_references=matrix.pair_references,
            pair_digests=matrix.pair_digests,
            query_keys=matrix.query_keys,
            scores=raw_scores,
            ranks=ranks,
            top_k_membership=membership,
            model_id=model.model_id,
            model_version=model.model_version,
            model_digest=model.model_digest,
            query_side=model.query_side,
            top_k=model.top_k,
        )


def write_lightgbm_ranker_artifact(
    *,
    artifact: LightGBMRankingArtifact,
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
    manifest_payload = {
        "schema_version": "0.1",
        **artifact.safe_summary(),
        "engine_version": artifact.engine_version,
        "configuration_digest": artifact.configuration_digest,
        "random_seed": artifact.random_seed,
        "training_partition": artifact.training_partition,
        "feature_names": list(artifact.feature_names),
    }
    try:
        atomic_write_text(model_destination, artifact.model_str)
        atomic_write_text(
            manifest_destination,
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        )
    except (OSError, TypeError, ValueError):
        raise RankingError(
            "ML-RANK-025", "A ranking artifact could not be written safely."
        ) from None
    return WrittenRankingArtifact(model_destination, manifest_destination, artifact.model_digest)


def read_lightgbm_ranker_artifact(
    *,
    model_path: str,
    manifest_path: str,
    policy: PathPolicy,
) -> LightGBMRankingArtifact:
    model_source = policy.resolve_output(model_path)
    manifest_source = policy.resolve_output(manifest_path)
    try:
        if (
            model_source.suffix not in (".txt", ".json")
            or manifest_source.suffix != ".json"
            or not model_source.is_file()
            or not manifest_source.is_file()
            or model_source.stat().st_size > 128 * 1024 * 1024
            or manifest_source.stat().st_size > 2 * 1024 * 1024
        ):
            raise OSError
        model_str = model_source.read_text(encoding="utf-8")
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
        "lightgbm_version",
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
        or raw.get("model_version") != "m5-lightgbm-ranker-v1"
        or raw.get("training_partition") != "training"
        or raw.get("decision_authority") != "ranking_only"
        or raw.get("relationship_authority") != "none"
        or raw.get("real_data_validation_status") != "not_established"
    ):
        raise RankingError("ML-RANK-027", "A ranking artifact manifest is invalid.")
    try:
        artifact = LightGBMRankingArtifact(
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
            lightgbm_version=str(raw["lightgbm_version"]),
            query_side=raw["query_side"],
            top_k=int(raw["top_k"]),
            feature_names=tuple(str(value) for value in raw["feature_names"]),
            model_str=model_str,
        )
    except (KeyError, TypeError, ValueError):
        raise RankingError("ML-RANK-027", "A ranking artifact manifest is invalid.") from None
    return artifact
