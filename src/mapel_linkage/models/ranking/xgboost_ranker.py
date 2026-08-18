"""Native XGBoost learning-to-rank adapter with no identity-decision authority."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from mapel_linkage import __version__
from mapel_linkage.configuration.models import RankingModelConfig
from mapel_linkage.domain.errors import RankingError
from mapel_linkage.models.ranking.contracts import (
    RankingFeatureMatrix,
    RankingMatrix,
    RankingScoreBatch,
    XGBoostRankingArtifact,
    canonical_digest,
    ranking_artifact_digest,
)

_xgboost: Any
try:
    import xgboost as _xgboost
except ImportError:  # pragma: no cover - optional dependency boundary
    _xgboost = None


def _require_xgboost() -> Any:
    if _xgboost is None:
        raise RankingError("ML-RANK-017", "The XGBoost ranking dependency is unavailable.")
    return _xgboost


def _model_bytes(booster: Any) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ranker.json"
        booster.save_model(path)
        return path.read_bytes()


def _load_model(model_json: bytes) -> Any:
    xgb = _require_xgboost()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ranker.json"
        path.write_bytes(model_json)
        booster = xgb.Booster()
        booster.load_model(path)
        return booster


class XGBoostCandidateRanker:
    """Train and score a query-grouped candidate ranker."""

    @staticmethod
    def fit(
        *,
        matrix: RankingMatrix,
        model: RankingModelConfig,
        random_seed: int,
        configuration_digest: str,
    ) -> XGBoostRankingArtifact:
        if model.implementation != "xgboost_ranker" or matrix.partition != "training":
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
        xgb = _require_xgboost()
        dmatrix = xgb.DMatrix(
            matrix.features,
            label=matrix.relevance,
            feature_names=list(matrix.feature_names),
            missing=np.nan,
        )
        dmatrix.set_group(list(matrix.group_sizes))
        parameters = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg",
            "tree_method": "hist",
            "seed": random_seed,
            "nthread": 1,
            "max_depth": model.max_depth,
            "eta": model.learning_rate,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        }
        booster = xgb.train(parameters, dmatrix, num_boost_round=model.n_estimators)
        model_json = _model_bytes(booster)
        model_digest = hashlib.sha256(model_json).hexdigest()
        parameter_digest = canonical_digest(parameters)
        artifact_digest = ranking_artifact_digest(
            model_id=model.model_id,
            model_version="m2g-xgboost-ranker-v1",
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
            xgboost_version=str(xgb.__version__),
            query_side=matrix.query_side,
            top_k=model.top_k,
            feature_names=matrix.feature_names,
        )
        return XGBoostRankingArtifact(
            model_id=model.model_id,
            model_version="m2g-xgboost-ranker-v1",
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
            xgboost_version=str(xgb.__version__),
            query_side=matrix.query_side,
            top_k=model.top_k,
            feature_names=matrix.feature_names,
            model_json=model_json,
        )

    @staticmethod
    def score(
        *,
        matrix: RankingFeatureMatrix,
        model: XGBoostRankingArtifact,
    ) -> RankingScoreBatch:
        if (
            matrix.feature_schema_digest != model.feature_schema_digest
            or matrix.feature_names != model.feature_names
        ):
            raise RankingError("ML-RANK-020", "Ranking features do not match the model artifact.")
        if matrix.query_side != model.query_side:
            raise RankingError("ML-RANK-021", "Ranking query orientation is inconsistent.")
        xgb = _require_xgboost()
        booster = _load_model(model.model_json)
        dmatrix = xgb.DMatrix(
            matrix.features,
            feature_names=list(matrix.feature_names),
            missing=np.nan,
        )
        raw_scores = np.asarray(booster.predict(dmatrix), dtype=np.float64)
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
