"""Immutable, value-hidden integrity metadata for pair-model score batches.

This contract can detect drift after a package scorer has produced evidence.  It is
not an authorization capability and never replaces replay through a typed fitted
model artifact.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from mapel_linkage.assignment.contracts import pair_digest
from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.domain.sql_identifiers import quote_identifier
from mapel_linkage.io.duckdb_store import DuckDBStore
from mapel_linkage.models.fellegi_sunter import (
    SplinkNativeModelArtifact,
    SplinkNativeScoreResult,
)

_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_PAIR_SCORE_EVIDENCE_ISSUER = object()

type PairScoreProbabilityStatus = Literal[
    "model_score_uncalibrated",
    "model_posterior_uncalibrated",
]


class _RecipeBinding(Protocol):
    @property
    def champion_model_id(self) -> str: ...

    @property
    def champion_model_version(self) -> str: ...

    @property
    def champion_artifact_digest(self) -> str: ...

    @property
    def configuration_digest(self) -> str: ...

    @property
    def feature_schema_digest(self) -> str: ...


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _score_digest(scores: NDArray[np.float64]) -> str:
    digest = hashlib.sha256()
    digest.update(b"pair_score_evidence_v1\x00")
    digest.update(str(scores.shape).encode("ascii"))
    digest.update(scores.tobytes(order="C"))
    return digest.hexdigest()


def _require_digest(value: str) -> None:
    if _DIGEST_PATTERN.fullmatch(value) is None:
        raise PipelineError("ML-PIPE-067", "Pair-score evidence provenance is invalid.")


def issue_native_splink_score_evidence(
    *,
    store: DuckDBStore,
    score_result: SplinkNativeScoreResult,
    model_artifact: SplinkNativeModelArtifact,
    pair_references: tuple[tuple[str, str], ...],
    pair_digests: tuple[str, ...],
) -> PairScoreEvidenceBatch:
    """Issue a partition batch only after rechecking the native scorer table.

    This is an integrity bridge for tournament evaluation, not generic inference
    authorization. Native Splink inference still requires native prepared-data replay.
    """
    if (
        len(pair_references) != len(pair_digests)
        or len(set(pair_references)) != len(pair_references)
        or any(
            digest != pair_digest(left, right)
            for (left, right), digest in zip(
                pair_references,
                pair_digests,
                strict=True,
            )
        )
        or score_result.model_id != model_artifact.model_id
        or score_result.model_version != model_artifact.model_version
        or not hmac.compare_digest(score_result.artifact_digest, model_artifact.artifact_digest)
        or score_result.pair_count < len(pair_references)
    ):
        raise PipelineError("ML-PIPE-080", "Native score evidence binding is invalid.")
    rows = store._fetch_model_rows(
        "SELECT left_record_key, right_record_key, __ml_fs_match_weight, "
        "__ml_fs_model_probability, __ml_fs_model_id, __ml_fs_model_version, "
        "__ml_fs_parameter_digest, __ml_fs_probability_status, "
        "__ml_fs_decision_authority FROM "
        f"{quote_identifier(score_result.table.table_name)} "
        "ORDER BY left_record_key, right_record_key"
    )
    if len(rows) != score_result.pair_count:
        raise PipelineError("ML-PIPE-080", "Native score evidence binding is invalid.")
    scores_by_pair: dict[tuple[str, str], float] = {}
    digest_rows: list[dict[str, str | float]] = []
    for row in rows:
        left, right = str(row[0]), str(row[1])
        match_weight, probability = row[2], row[3]
        if (
            not isinstance(match_weight, (int, float))
            or not isinstance(probability, (int, float))
            or not np.isfinite(float(match_weight))
            or not np.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
            or str(row[4]) != model_artifact.model_id
            or str(row[5]) != model_artifact.model_version
            or str(row[6]) != model_artifact.parameter_digest
            or str(row[7]) != model_artifact.probability_status
            or str(row[8]) != "evidence_only"
        ):
            raise PipelineError("ML-PIPE-080", "Native score evidence binding is invalid.")
        pair = (left, right)
        if pair in scores_by_pair:
            raise PipelineError("ML-PIPE-080", "Native score evidence binding is invalid.")
        scores_by_pair[pair] = float(probability)
        native_pair_digest = hashlib.sha256(f"{left}\x1f{right}".encode()).hexdigest()
        digest_rows.append(
            {
                "pair_digest": native_pair_digest,
                "match_weight": float(match_weight),
                "probability": float(probability),
            }
        )
    if not hmac.compare_digest(
        _canonical_digest(sorted(digest_rows, key=lambda item: str(item["pair_digest"]))),
        score_result.score_digest,
    ):
        raise PipelineError("ML-PIPE-080", "Native score evidence binding is invalid.")
    try:
        ordered_scores = np.asarray(
            [scores_by_pair[pair] for pair in pair_references],
            dtype=np.float64,
        )
    except KeyError:
        raise PipelineError("ML-PIPE-080", "Native score evidence binding is invalid.") from None
    return PairScoreEvidenceBatch._issue(
        pair_digests=pair_digests,
        scores=ordered_scores,
        champion_model_id=model_artifact.model_id,
        champion_model_version=model_artifact.model_version,
        champion_artifact_digest=model_artifact.artifact_digest,
        configuration_digest=model_artifact.configuration_digest,
        feature_schema_digest=model_artifact.feature_schema_digest,
        probability_status="model_posterior_uncalibrated",
    )


@dataclass(frozen=True, slots=True, repr=False)
class PairScoreEvidenceArtifact:
    """Aggregate integrity metadata for one ordered score vector and fitted model."""

    champion_model_id: str
    champion_model_version: str
    champion_artifact_digest: str
    configuration_digest: str
    feature_schema_digest: str
    pair_count: int
    ordered_pair_digest: str
    score_digest: str
    probability_status: PairScoreProbabilityStatus
    evidence_digest: str
    schema_version: Literal["1"] = "1"
    calibration_status: Literal["not_calibrated"] = "not_calibrated"
    evidence_authority: Literal["evidence_only"] = "evidence_only"
    relationship_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    def __post_init__(self) -> None:
        if (
            _IDENTIFIER_PATTERN.fullmatch(self.champion_model_id) is None
            or _IDENTIFIER_PATTERN.fullmatch(self.champion_model_version) is None
            or isinstance(self.pair_count, bool)
            or not isinstance(self.pair_count, int)
            or self.pair_count < 1
            or self.probability_status
            not in {"model_score_uncalibrated", "model_posterior_uncalibrated"}
            or self.schema_version != "1"
            or self.calibration_status != "not_calibrated"
            or self.evidence_authority != "evidence_only"
            or self.relationship_authority != "none"
            or self.assignment_authority != "none"
            or self.merge_authority != "none"
            or self.operational_validity != "not_established"
        ):
            raise PipelineError("ML-PIPE-067", "Pair-score evidence provenance is invalid.")
        for value in (
            self.champion_artifact_digest,
            self.configuration_digest,
            self.feature_schema_digest,
            self.ordered_pair_digest,
            self.score_digest,
            self.evidence_digest,
        ):
            _require_digest(value)

    def safe_summary(self) -> dict[str, str | int]:
        """Return aggregate metadata without scores, pairs, or record identifiers."""
        return {
            "schema_version": self.schema_version,
            "champion_model_id": self.champion_model_id,
            "champion_model_version": self.champion_model_version,
            "champion_artifact_digest": self.champion_artifact_digest,
            "configuration_digest": self.configuration_digest,
            "feature_schema_digest": self.feature_schema_digest,
            "pair_count": self.pair_count,
            "ordered_pair_digest": self.ordered_pair_digest,
            "score_digest": self.score_digest,
            "probability_status": self.probability_status,
            "calibration_status": self.calibration_status,
            "evidence_authority": self.evidence_authority,
            "relationship_authority": self.relationship_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "operational_validity": self.operational_validity,
            "evidence_digest": self.evidence_digest,
        }

    def __repr__(self) -> str:
        return "<PairScoreEvidenceArtifact aggregate-only>"


def _artifact_digest_payload(artifact: PairScoreEvidenceArtifact) -> dict[str, str | int]:
    return {
        key: value for key, value in artifact.safe_summary().items() if key != "evidence_digest"
    }


@dataclass(frozen=True, slots=True, init=False, repr=False)
class PairScoreEvidenceBatch:
    """Opaque score vector with exact ordered-pair and model provenance linkage.

    The private issuer prevents accidental direct construction, but Python process
    boundaries are not security boundaries. Approved inference must recompute scores
    through the recipe-bound typed model artifact and may use this batch only to
    verify that separately retained evidence has not drifted.
    """

    artifact: PairScoreEvidenceArtifact
    pair_digests: tuple[str, ...] = field(repr=False)
    scores: NDArray[np.float64] = field(repr=False)
    _issuer: object = field(repr=False)

    def __new__(cls) -> PairScoreEvidenceBatch:
        raise TypeError("PairScoreEvidenceBatch instances are issued by package model scorers.")

    @classmethod
    def _issue(
        cls,
        *,
        pair_digests: tuple[str, ...],
        scores: NDArray[np.float64] | tuple[float, ...] | list[float],
        champion_model_id: str,
        champion_model_version: str,
        champion_artifact_digest: str,
        configuration_digest: str,
        feature_schema_digest: str,
        probability_status: PairScoreProbabilityStatus,
    ) -> PairScoreEvidenceBatch:
        """Create integrity metadata after a package-owned scorer returns evidence."""
        score_array = np.asarray(scores, dtype="<f8").copy()
        score_array.setflags(write=False)
        ordered_pair_digest = _canonical_digest({"pair_digests": pair_digests})
        provisional = PairScoreEvidenceArtifact(
            champion_model_id=champion_model_id,
            champion_model_version=champion_model_version,
            champion_artifact_digest=champion_artifact_digest,
            configuration_digest=configuration_digest,
            feature_schema_digest=feature_schema_digest,
            pair_count=len(pair_digests),
            ordered_pair_digest=ordered_pair_digest,
            score_digest=_score_digest(score_array),
            probability_status=probability_status,
            evidence_digest="0" * 64,
        )
        artifact = PairScoreEvidenceArtifact(
            champion_model_id=provisional.champion_model_id,
            champion_model_version=provisional.champion_model_version,
            champion_artifact_digest=provisional.champion_artifact_digest,
            configuration_digest=provisional.configuration_digest,
            feature_schema_digest=provisional.feature_schema_digest,
            pair_count=provisional.pair_count,
            ordered_pair_digest=provisional.ordered_pair_digest,
            score_digest=provisional.score_digest,
            probability_status=provisional.probability_status,
            evidence_digest=_canonical_digest(_artifact_digest_payload(provisional)),
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "artifact", artifact)
        object.__setattr__(instance, "pair_digests", tuple(pair_digests))
        object.__setattr__(instance, "scores", score_array)
        object.__setattr__(instance, "_issuer", _PAIR_SCORE_EVIDENCE_ISSUER)
        instance.assert_valid_contract()
        return instance

    def assert_valid_contract(self) -> None:
        """Recompute all value-hidden digests and reject forged or mutated evidence."""
        try:
            scores = np.asarray(self.scores, dtype="<f8")
            artifact = self.artifact
        except (AttributeError, TypeError, ValueError):
            raise PipelineError(
                "ML-PIPE-067", "Pair-score evidence provenance is invalid."
            ) from None
        if (
            getattr(self, "_issuer", None) is not _PAIR_SCORE_EVIDENCE_ISSUER
            or scores.ndim != 1
            or len(self.pair_digests) != artifact.pair_count
            or scores.shape[0] != artifact.pair_count
            or len(set(self.pair_digests)) != artifact.pair_count
            or not np.all(np.isfinite(scores))
            or np.any(scores < 0.0)
            or np.any(scores > 1.0)
        ):
            raise PipelineError("ML-PIPE-067", "Pair-score evidence provenance is invalid.")
        for value in self.pair_digests:
            _require_digest(value)
        if (
            artifact.ordered_pair_digest != _canonical_digest({"pair_digests": self.pair_digests})
            or artifact.score_digest != _score_digest(scores)
            or artifact.evidence_digest != _canonical_digest(_artifact_digest_payload(artifact))
        ):
            raise PipelineError("ML-PIPE-067", "Pair-score evidence provenance is invalid.")

    def assert_matches(
        self,
        *,
        recipe: _RecipeBinding,
        pair_digests: tuple[str, ...],
    ) -> None:
        """Require exact recipe, fitted-artifact, schema, and ordered-pair parity."""
        self.assert_valid_contract()
        artifact = self.artifact
        if (
            self.pair_digests != pair_digests
            or artifact.champion_model_id != recipe.champion_model_id
            or artifact.champion_model_version != recipe.champion_model_version
            or not hmac.compare_digest(
                artifact.champion_artifact_digest,
                recipe.champion_artifact_digest,
            )
            or not hmac.compare_digest(
                artifact.configuration_digest,
                recipe.configuration_digest,
            )
            or not hmac.compare_digest(
                artifact.feature_schema_digest,
                recipe.feature_schema_digest,
            )
        ):
            raise PipelineError(
                "ML-PIPE-068",
                "Pair-score evidence does not match the approved pipeline recipe.",
            )

    def assert_model_binding(
        self,
        *,
        model_id: str,
        model_version: str,
        model_artifact_digest: str,
        configuration_digest: str,
        feature_schema_digest: str,
        pair_digests: tuple[str, ...],
    ) -> None:
        """Require exact fitted-model, schema, and ordered-pair integrity linkage."""
        self.assert_valid_contract()
        artifact = self.artifact
        if (
            self.pair_digests != pair_digests
            or artifact.champion_model_id != model_id
            or artifact.champion_model_version != model_version
            or not hmac.compare_digest(
                artifact.champion_artifact_digest,
                model_artifact_digest,
            )
            or not hmac.compare_digest(
                artifact.configuration_digest,
                configuration_digest,
            )
            or not hmac.compare_digest(
                artifact.feature_schema_digest,
                feature_schema_digest,
            )
        ):
            raise PipelineError(
                "ML-PIPE-068",
                "Pair-score evidence does not match the fitted model artifact.",
            )

    def assert_scores(self, scores: NDArray[np.float64]) -> None:
        """Require exact parity with scores recomputed by a typed fitted artifact."""
        self.assert_valid_contract()
        try:
            recomputed = np.asarray(scores, dtype="<f8")
        except (TypeError, ValueError):
            raise PipelineError(
                "ML-PIPE-068",
                "Pair-score evidence does not match the approved pipeline recipe.",
            ) from None
        if (
            recomputed.ndim != 1
            or recomputed.shape[0] != self.pair_count
            or not np.all(np.isfinite(recomputed))
            or not hmac.compare_digest(_score_digest(recomputed), self.artifact.score_digest)
        ):
            raise PipelineError(
                "ML-PIPE-068",
                "Pair-score evidence does not match the approved pipeline recipe.",
            )

    @property
    def evidence_digest(self) -> str:
        return self.artifact.evidence_digest

    @property
    def pair_count(self) -> int:
        return self.artifact.pair_count

    def safe_summary(self) -> dict[str, str | int]:
        return self.artifact.safe_summary()

    def __repr__(self) -> str:
        return "<PairScoreEvidenceBatch value-hidden>"


__all__ = [
    "PairScoreEvidenceArtifact",
    "PairScoreEvidenceBatch",
    "issue_native_splink_score_evidence",
]
