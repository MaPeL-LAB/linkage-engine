"""Candidate-ranking matrices, artifacts, and evidence-only score contracts."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from mapel_linkage.domain.errors import RankingError
from mapel_linkage.governance.labels import LabelSourceKind

_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_FEATURE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ranking_artifact_digest(
    *,
    model_id: str,
    model_version: str,
    engine_version: str,
    configuration_digest: str,
    random_seed: int,
    training_pair_count: int,
    training_query_count: int,
    feature_schema_digest: str,
    label_authority_digest: str,
    selection_digest: str,
    parameter_digest: str,
    model_digest: str,
    xgboost_version: str | None = None,
    lightgbm_version: str | None = None,
    query_side: str,
    top_k: int,
    feature_names: tuple[str, ...],
) -> str:
    payload: dict[str, object] = {
        "model_id": model_id,
        "model_version": model_version,
        "engine_version": engine_version,
        "configuration_digest": configuration_digest,
        "random_seed": random_seed,
        "training_pair_count": training_pair_count,
        "training_query_count": training_query_count,
        "feature_schema_digest": feature_schema_digest,
        "label_authority_digest": label_authority_digest,
        "selection_digest": selection_digest,
        "parameter_digest": parameter_digest,
        "model_digest": model_digest,
        "query_side": query_side,
        "top_k": top_k,
        "feature_names": feature_names,
        "decision_authority": "ranking_only",
        "relationship_authority": "none",
        "real_data_validation_status": "not_established",
    }
    if xgboost_version is not None:
        payload["xgboost_version"] = xgboost_version
    if lightgbm_version is not None:
        payload["lightgbm_version"] = lightgbm_version
    return canonical_digest(payload)


def immutable_float_matrix(values: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64).copy()
    matrix.setflags(write=False)
    return matrix


def immutable_float_vector(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64).copy()
    vector.setflags(write=False)
    return vector


def immutable_int_vector(values: NDArray[np.int64]) -> NDArray[np.int64]:
    vector = np.asarray(values, dtype=np.int64).copy()
    vector.setflags(write=False)
    return vector


def _pair_digest(left: str, right: str) -> str:
    return hashlib.sha256(f"{left}\x00{right}".encode()).hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class RankingFeatureMatrix:
    """Private query groups and immutable canonical comparison features."""

    features: NDArray[np.float64] = field(repr=False)
    pair_references: tuple[tuple[str, str], ...] = field(repr=False)
    pair_digests: tuple[str, ...] = field(repr=False)
    query_keys: tuple[str, ...] = field(repr=False)
    group_sizes: tuple[int, ...]
    feature_names: tuple[str, ...] = field(repr=False)
    feature_schema_digest: str
    query_side: Literal["source", "target"]

    def __post_init__(self) -> None:
        features = immutable_float_matrix(self.features)
        count = len(self.pair_references)
        if features.ndim != 2:
            raise RankingError("ML-RANK-001", "A ranking matrix has invalid dimensions.")
        if (
            features.shape[0] != count
            or len(self.pair_digests) != count
            or len(self.query_keys) != count
        ):
            raise RankingError("ML-RANK-002", "A ranking matrix has invalid row coverage.")
        if features.shape[1] != len(self.feature_names) or not self.feature_names:
            raise RankingError("ML-RANK-003", "A ranking feature schema is invalid.")
        if np.any(np.isinf(features)):
            raise RankingError("ML-RANK-030", "A ranking matrix contains infinite features.")
        if len(set(self.feature_names)) != len(self.feature_names) or any(
            _FEATURE_IDENTIFIER_PATTERN.fullmatch(name) is None for name in self.feature_names
        ):
            raise RankingError("ML-RANK-003", "A ranking feature schema is invalid.")
        if (
            sum(self.group_sizes) != count
            or not self.group_sizes
            or any(size < 1 for size in self.group_sizes)
        ):
            raise RankingError("ML-RANK-004", "Ranking query groups are invalid.")
        if len(set(self.pair_digests)) != count or len(set(self.pair_references)) != count:
            raise RankingError("ML-RANK-006", "Duplicate ranking pairs were rejected.")
        side_index = 0 if self.query_side == "source" else 1
        if any(
            query != pair[side_index]
            for query, pair in zip(self.query_keys, self.pair_references, strict=True)
        ):
            raise RankingError("ML-RANK-031", "Ranking queries do not match pair orientation.")
        observed_group_sizes = tuple(
            sum(1 for _ in group) for _, group in itertools.groupby(self.query_keys)
        )
        if observed_group_sizes != self.group_sizes:
            raise RankingError("ML-RANK-032", "Ranking group sizes do not match query rows.")
        if any(
            digest != _pair_digest(left, right)
            for (left, right), digest in zip(
                self.pair_references,
                self.pair_digests,
                strict=True,
            )
        ):
            raise RankingError("ML-RANK-033", "A ranking pair digest is inconsistent.")
        if tuple(sorted(zip(self.query_keys, self.pair_digests, strict=True))) != tuple(
            zip(self.query_keys, self.pair_digests, strict=True)
        ):
            raise RankingError("ML-RANK-007", "Ranking rows are not deterministically grouped.")
        for digest in (self.feature_schema_digest, *self.pair_digests):
            if _DIGEST_PATTERN.fullmatch(digest) is None:
                raise RankingError("ML-RANK-008", "A ranking digest is invalid.")
        object.__setattr__(self, "features", features)

    @property
    def pair_count(self) -> int:
        return len(self.pair_references)

    @property
    def query_count(self) -> int:
        return len(self.group_sizes)

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "pair_count": self.pair_count,
            "query_count": self.query_count,
            "feature_count": len(self.feature_names),
            "query_side": self.query_side,
            "feature_schema_digest": self.feature_schema_digest,
        }


@dataclass(frozen=True, slots=True, repr=False)
class RankingMatrix(RankingFeatureMatrix):
    """Verified labelled ranking matrix for training or independent evaluation."""

    relevance: NDArray[np.float64] = field(repr=False)
    partition: Literal["training", "validation", "test"]
    label_source_kind: LabelSourceKind
    label_authority_digest: str
    selection_digest: str
    excluded_singleton_query_count: int = 0
    excluded_uninformative_query_count: int = 0

    def __post_init__(self) -> None:
        RankingFeatureMatrix.__post_init__(self)
        relevance = immutable_float_vector(self.relevance)
        if relevance.ndim != 1 or len(relevance) != self.pair_count:
            raise RankingError("ML-RANK-002", "A ranking matrix has invalid row coverage.")
        if not np.all(np.isfinite(relevance)) or np.any(relevance < 0.0):
            raise RankingError("ML-RANK-005", "Ranking relevance labels are invalid.")
        for digest in (self.label_authority_digest, self.selection_digest):
            if _DIGEST_PATTERN.fullmatch(digest) is None:
                raise RankingError("ML-RANK-008", "A ranking digest is invalid.")
        object.__setattr__(self, "relevance", relevance)

    def safe_summary(self) -> dict[str, int | str]:
        return {
            **super().safe_summary(),
            "partition": self.partition,
            "label_authority_digest": self.label_authority_digest,
            "selection_digest": self.selection_digest,
            "excluded_singleton_query_count": self.excluded_singleton_query_count,
            "excluded_uninformative_query_count": self.excluded_uninformative_query_count,
        }


@dataclass(frozen=True, slots=True, repr=False)
class XGBoostRankingArtifact:
    model_id: str
    model_version: Literal["m2g-xgboost-ranker-v1"]
    engine_version: str
    configuration_digest: str
    random_seed: int
    training_pair_count: int
    training_query_count: int
    feature_schema_digest: str
    label_authority_digest: str
    selection_digest: str
    parameter_digest: str
    model_digest: str
    artifact_digest: str
    xgboost_version: str
    query_side: Literal["source", "target"]
    top_k: int
    feature_names: tuple[str, ...] = field(repr=False)
    model_json: bytes = field(repr=False)
    training_partition: Literal["training"] = "training"
    decision_authority: Literal["ranking_only"] = "ranking_only"
    relationship_authority: Literal["none"] = "none"
    real_data_validation_status: Literal["not_established"] = "not_established"

    def __post_init__(self) -> None:
        for value in (self.model_id, self.model_version):
            if _IDENTIFIER_PATTERN.fullmatch(value) is None:
                raise RankingError("ML-RANK-009", "A ranking model identifier is invalid.")
        if (
            self.random_seed < 0
            or self.training_pair_count <= 0
            or self.training_query_count <= 0
            or self.training_query_count > self.training_pair_count
            or not 1 <= self.top_k <= 1000
        ):
            raise RankingError("ML-RANK-010", "A ranking artifact has invalid aggregate counts.")
        if (
            len(set(self.feature_names)) != len(self.feature_names)
            or not self.feature_names
            or any(
                _FEATURE_IDENTIFIER_PATTERN.fullmatch(name) is None for name in self.feature_names
            )
        ):
            raise RankingError("ML-RANK-034", "A ranking artifact feature schema is invalid.")
        for digest in (
            self.configuration_digest,
            self.feature_schema_digest,
            self.label_authority_digest,
            self.selection_digest,
            self.parameter_digest,
            self.model_digest,
            self.artifact_digest,
        ):
            if _DIGEST_PATTERN.fullmatch(digest) is None:
                raise RankingError("ML-RANK-011", "A ranking artifact digest is invalid.")
        if hashlib.sha256(self.model_json).hexdigest() != self.model_digest:
            raise RankingError("ML-RANK-012", "Ranking model integrity failed.")
        try:
            json.loads(self.model_json.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RankingError("ML-RANK-013", "The ranking model must use native JSON.") from None
        expected_artifact_digest = ranking_artifact_digest(
            model_id=self.model_id,
            model_version=self.model_version,
            engine_version=self.engine_version,
            configuration_digest=self.configuration_digest,
            random_seed=self.random_seed,
            training_pair_count=self.training_pair_count,
            training_query_count=self.training_query_count,
            feature_schema_digest=self.feature_schema_digest,
            label_authority_digest=self.label_authority_digest,
            selection_digest=self.selection_digest,
            parameter_digest=self.parameter_digest,
            model_digest=self.model_digest,
            xgboost_version=self.xgboost_version,
            query_side=self.query_side,
            top_k=self.top_k,
            feature_names=self.feature_names,
        )
        if expected_artifact_digest != self.artifact_digest:
            raise RankingError("ML-RANK-028", "Ranking manifest integrity failed.")

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "training_pair_count": self.training_pair_count,
            "training_query_count": self.training_query_count,
            "feature_schema_digest": self.feature_schema_digest,
            "label_authority_digest": self.label_authority_digest,
            "selection_digest": self.selection_digest,
            "parameter_digest": self.parameter_digest,
            "model_digest": self.model_digest,
            "artifact_digest": self.artifact_digest,
            "xgboost_version": self.xgboost_version,
            "query_side": self.query_side,
            "top_k": self.top_k,
            "decision_authority": self.decision_authority,
            "relationship_authority": self.relationship_authority,
            "real_data_validation_status": self.real_data_validation_status,
        }


@dataclass(frozen=True, slots=True, repr=False)
class RankingScoreBatch:
    pair_references: tuple[tuple[str, str], ...] = field(repr=False)
    pair_digests: tuple[str, ...] = field(repr=False)
    query_keys: tuple[str, ...] = field(repr=False)
    scores: NDArray[np.float64] = field(repr=False)
    ranks: NDArray[np.int64] = field(repr=False)
    top_k_membership: NDArray[np.int64] = field(repr=False)
    model_id: str
    model_version: str
    model_digest: str
    query_side: Literal["source", "target"]
    top_k: int
    decision_authority: Literal["ranking_only"] = "ranking_only"
    relationship_authority: Literal["none"] = "none"

    def __post_init__(self) -> None:
        scores = immutable_float_vector(self.scores)
        ranks = immutable_int_vector(self.ranks)
        membership = immutable_int_vector(self.top_k_membership)
        count = len(self.pair_references)
        if any(
            len(values) != count
            for values in (self.pair_digests, self.query_keys, scores, ranks, membership)
        ):
            raise RankingError("ML-RANK-014", "Ranking output coverage is invalid.")
        if (
            count == 0
            or len(set(self.pair_references)) != count
            or len(set(self.pair_digests)) != count
        ):
            raise RankingError("ML-RANK-035", "Ranking output pair coverage is invalid.")
        side_index = 0 if self.query_side == "source" else 1
        if any(
            query != pair[side_index]
            for query, pair in zip(self.query_keys, self.pair_references, strict=True)
        ) or any(
            digest != _pair_digest(left, right)
            for (left, right), digest in zip(
                self.pair_references,
                self.pair_digests,
                strict=True,
            )
        ):
            raise RankingError("ML-RANK-036", "Ranking output provenance is inconsistent.")
        if (
            not np.all(np.isfinite(scores))
            or np.any(ranks < 1)
            or np.any(~np.isin(membership, (0, 1)))
        ):
            raise RankingError("ML-RANK-015", "Ranking output values are invalid.")
        if (
            not 1 <= self.top_k <= 1000
            or _IDENTIFIER_PATTERN.fullmatch(self.model_id) is None
            or _IDENTIFIER_PATTERN.fullmatch(self.model_version) is None
            or _DIGEST_PATTERN.fullmatch(self.model_digest) is None
        ):
            raise RankingError("ML-RANK-037", "Ranking output model provenance is invalid.")
        for _, group in itertools.groupby(range(count), key=self.query_keys.__getitem__):
            indices = tuple(group)
            expected_ranks = list(range(1, len(indices) + 1))
            if sorted(int(ranks[index]) for index in indices) != expected_ranks or any(
                int(membership[index]) != int(int(ranks[index]) <= self.top_k) for index in indices
            ):
                raise RankingError("ML-RANK-038", "Ranking output ranks are inconsistent.")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "ranks", ranks)
        object.__setattr__(self, "top_k_membership", membership)

    @property
    def pair_count(self) -> int:
        return len(self.pair_references)

    @property
    def query_count(self) -> int:
        return len(set(self.query_keys))

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "pair_count": self.pair_count,
            "query_count": self.query_count,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_digest": self.model_digest,
            "query_side": self.query_side,
            "top_k": self.top_k,
            "decision_authority": self.decision_authority,
            "relationship_authority": self.relationship_authority,
        }


@dataclass(frozen=True, slots=True, repr=False)
class LightGBMRankingArtifact:
    model_id: str
    model_version: Literal["m5-lightgbm-ranker-v1"]
    engine_version: str
    configuration_digest: str
    random_seed: int
    training_pair_count: int
    training_query_count: int
    feature_schema_digest: str
    label_authority_digest: str
    selection_digest: str
    parameter_digest: str
    model_digest: str
    artifact_digest: str
    lightgbm_version: str
    query_side: Literal["source", "target"]
    top_k: int
    feature_names: tuple[str, ...] = field(repr=False)
    model_str: str = field(repr=False)
    training_partition: Literal["training"] = "training"
    decision_authority: Literal["ranking_only"] = "ranking_only"
    relationship_authority: Literal["none"] = "none"
    real_data_validation_status: Literal["not_established"] = "not_established"

    def __post_init__(self) -> None:
        for value in (self.model_id, self.model_version):
            if _IDENTIFIER_PATTERN.fullmatch(value) is None:
                raise RankingError("ML-RANK-009", "A ranking model identifier is invalid.")
        if (
            self.random_seed < 0
            or self.training_pair_count <= 0
            or self.training_query_count <= 0
            or self.training_query_count > self.training_pair_count
            or not 1 <= self.top_k <= 1000
        ):
            raise RankingError("ML-RANK-010", "A ranking artifact has invalid aggregate counts.")
        if (
            len(set(self.feature_names)) != len(self.feature_names)
            or not self.feature_names
            or any(
                _FEATURE_IDENTIFIER_PATTERN.fullmatch(name) is None for name in self.feature_names
            )
        ):
            raise RankingError("ML-RANK-034", "A ranking artifact feature schema is invalid.")
        for digest in (
            self.configuration_digest,
            self.feature_schema_digest,
            self.label_authority_digest,
            self.selection_digest,
            self.parameter_digest,
            self.model_digest,
            self.artifact_digest,
        ):
            if _DIGEST_PATTERN.fullmatch(digest) is None:
                raise RankingError("ML-RANK-011", "A ranking artifact digest is invalid.")
        if hashlib.sha256(self.model_str.encode("utf-8")).hexdigest() != self.model_digest:
            raise RankingError("ML-RANK-012", "Ranking model integrity failed.")
        if not self.model_str.strip():
            raise RankingError("ML-RANK-013", "The ranking model string cannot be empty.")
        expected_artifact_digest = ranking_artifact_digest(
            model_id=self.model_id,
            model_version=self.model_version,
            engine_version=self.engine_version,
            configuration_digest=self.configuration_digest,
            random_seed=self.random_seed,
            training_pair_count=self.training_pair_count,
            training_query_count=self.training_query_count,
            feature_schema_digest=self.feature_schema_digest,
            label_authority_digest=self.label_authority_digest,
            selection_digest=self.selection_digest,
            parameter_digest=self.parameter_digest,
            model_digest=self.model_digest,
            lightgbm_version=self.lightgbm_version,
            query_side=self.query_side,
            top_k=self.top_k,
            feature_names=self.feature_names,
        )
        if expected_artifact_digest != self.artifact_digest:
            raise RankingError("ML-RANK-028", "Ranking manifest integrity failed.")

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "training_pair_count": self.training_pair_count,
            "training_query_count": self.training_query_count,
            "feature_schema_digest": self.feature_schema_digest,
            "label_authority_digest": self.label_authority_digest,
            "selection_digest": self.selection_digest,
            "parameter_digest": self.parameter_digest,
            "model_digest": self.model_digest,
            "artifact_digest": self.artifact_digest,
            "lightgbm_version": self.lightgbm_version,
            "query_side": self.query_side,
            "top_k": self.top_k,
            "decision_authority": self.decision_authority,
            "relationship_authority": self.relationship_authority,
            "real_data_validation_status": self.real_data_validation_status,
        }


class CandidateRanker(Protocol):
    """Protocol for candidate ranking models."""

    @staticmethod
    def fit(
        *,
        matrix: RankingMatrix,
        model: Any,
        random_seed: int,
        configuration_digest: str,
    ) -> Any: ...

    @staticmethod
    def score(
        *,
        matrix: RankingFeatureMatrix,
        model: Any,
    ) -> RankingScoreBatch: ...
