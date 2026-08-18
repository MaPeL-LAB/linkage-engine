"""Privacy-safe calibration and champion-selection contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mapel_linkage.domain.errors import CalibrationError, ModelSelectionError
from mapel_linkage.governance.labels import LabelPartition

CalibrationMethod = Literal["sigmoid", "isotonic"]
ProbabilityStatus = Literal["calibrated_probability"]
DecisionAuthority = Literal["evidence_only"]
ThresholdAuthority = Literal["none"]
RealDataValidationStatus = Literal["not_established"]

_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


def canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_digest(value: str, *, code: str, message: str) -> None:
    if _DIGEST_PATTERN.fullmatch(value) is None:
        raise CalibrationError(code, message)


def immutable_float_vector(
    values: NDArray[np.float64] | list[float] | tuple[float, ...],
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64).copy()
    array.setflags(write=False)
    return array


def immutable_label_vector(
    values: NDArray[np.int8] | list[int] | tuple[int, ...],
) -> NDArray[np.int8]:
    array = np.asarray(values, dtype=np.int8).copy()
    array.setflags(write=False)
    return array


def _pair_digest(left: str, right: str) -> str:
    return hashlib.sha256(f"{left}\x00{right}".encode()).hexdigest()


def _freeze_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    frozen: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            frozen[key] = tuple(value)
        elif isinstance(value, dict):
            raise CalibrationError("ML-CAL-042", "Nested calibrator payloads are not permitted.")
        else:
            frozen[key] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True, repr=False)
class PairScoreBatch:
    """Protected pair scores aligned with an eligible verified label snapshot."""

    pair_references: tuple[tuple[str, str], ...] = field(repr=False)
    pair_digests: tuple[str, ...] = field(repr=False)
    scores: NDArray[np.float64] = field(repr=False)
    labels: NDArray[np.int8] = field(repr=False)
    partition: LabelPartition
    source_model_family: str
    source_model_id: str
    source_model_version: str
    source_evidence_digest: str
    feature_schema_digest: str
    label_authority_digest: str
    partition_manifest_digest: str
    champion_selection_digest: str | None = None

    def __post_init__(self) -> None:
        if self.partition not in {"validation", "calibration", "decision", "test"}:
            raise CalibrationError(
                "ML-CAL-001", "Scores for this calibration boundary use an invalid partition."
            )
        count = len(self.pair_references)
        if count == 0 or len(self.pair_digests) != count:
            raise CalibrationError("ML-CAL-002", "A protected score batch has invalid coverage.")
        scores = immutable_float_vector(self.scores)
        labels = immutable_label_vector(self.labels)
        if scores.ndim != 1 or labels.ndim != 1 or len(scores) != count or len(labels) != count:
            raise CalibrationError("ML-CAL-003", "A protected score batch has invalid dimensions.")
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise CalibrationError("ML-CAL-004", "A protected score batch contains invalid scores.")
        if not np.all(np.isin(labels, np.asarray([0, 1], dtype=np.int8))):
            raise CalibrationError("ML-CAL-005", "A protected score batch contains invalid labels.")
        if len(set(self.pair_digests)) != count or len(set(self.pair_references)) != count:
            raise CalibrationError("ML-CAL-006", "Duplicate protected pair scores were rejected.")
        if any(
            digest != _pair_digest(left, right)
            for (left, right), digest in zip(
                self.pair_references,
                self.pair_digests,
                strict=True,
            )
        ):
            raise CalibrationError("ML-CAL-043", "A protected pair digest is inconsistent.")
        for digest in (
            *self.pair_digests,
            self.source_evidence_digest,
            self.feature_schema_digest,
            self.label_authority_digest,
            self.partition_manifest_digest,
        ):
            require_digest(digest, code="ML-CAL-007", message="A calibration digest is invalid.")
        if self.champion_selection_digest is not None:
            require_digest(
                self.champion_selection_digest,
                code="ML-CAL-008",
                message="A champion-selection digest is invalid.",
            )
        for value in (self.source_model_family, self.source_model_id, self.source_model_version):
            if _IDENTIFIER_PATTERN.fullmatch(value) is None:
                raise CalibrationError("ML-CAL-009", "A source-model identifier is invalid.")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "labels", labels)

    @property
    def pair_count(self) -> int:
        return len(self.pair_references)

    @property
    def positive_count(self) -> int:
        return int(self.labels.sum())

    @property
    def negative_count(self) -> int:
        return self.pair_count - self.positive_count

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "partition": self.partition,
            "pair_count": self.pair_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "source_model_family": self.source_model_family,
            "source_model_id": self.source_model_id,
            "source_model_version": self.source_model_version,
            "label_authority_digest": self.label_authority_digest,
        }


@dataclass(frozen=True, slots=True)
class ModelEvaluationCandidate:
    """Aggregate validation evidence eligible for champion selection."""

    model_family: str
    model_id: str
    model_version: str
    evidence_digest: str
    feature_schema_digest: str
    validation_label_authority_digest: str
    partition_manifest_digest: str
    average_precision: float
    brier_score: float
    pair_count: int
    training_label_authority_digest: str | None = None
    evaluation_partition: Literal["validation"] = "validation"
    decision_authority: Literal["evidence_only"] = "evidence_only"
    real_data_validation_status: Literal["not_established"] = "not_established"

    def __post_init__(self) -> None:
        for value in (self.model_family, self.model_id, self.model_version):
            if _IDENTIFIER_PATTERN.fullmatch(value) is None:
                raise ModelSelectionError(
                    "ML-SELECT-001", "A champion-candidate identifier is invalid."
                )
        for digest in (
            self.evidence_digest,
            self.feature_schema_digest,
            self.validation_label_authority_digest,
            self.partition_manifest_digest,
        ):
            if _DIGEST_PATTERN.fullmatch(digest) is None:
                raise ModelSelectionError(
                    "ML-SELECT-002", "A champion-candidate digest is invalid."
                )
        if (
            self.training_label_authority_digest is not None
            and _DIGEST_PATTERN.fullmatch(self.training_label_authority_digest) is None
        ):
            raise ModelSelectionError("ML-SELECT-003", "A training-authority digest is invalid.")
        if self.pair_count <= 0:
            raise ModelSelectionError(
                "ML-SELECT-004", "A champion candidate has no validation pairs."
            )
        if not math.isfinite(self.average_precision) or not 0.0 <= self.average_precision <= 1.0:
            raise ModelSelectionError("ML-SELECT-005", "Average precision is invalid.")
        if not math.isfinite(self.brier_score) or not 0.0 <= self.brier_score <= 1.0:
            raise ModelSelectionError("ML-SELECT-006", "Brier score is invalid.")

    def safe_summary(self) -> dict[str, float | int | str]:
        return {
            "model_family": self.model_family,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "average_precision": self.average_precision,
            "brier_score": self.brier_score,
            "pair_count": self.pair_count,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class ChampionSelection:
    """Immutable record of validation-only model-family selection."""

    selected_model_family: str
    selected_model_id: str
    selected_model_version: str
    selected_evidence_digest: str
    selected_feature_schema_digest: str
    selected_training_label_authority_digest: str | None
    validation_label_authority_digest: str
    partition_manifest_digest: str
    primary_metric: Literal["average_precision", "brier_score"]
    secondary_metric: Literal["average_precision", "brier_score"]
    selection_digest: str
    candidate_summaries: tuple[Mapping[str, float | int | str], ...] = field(repr=False)
    test_partition_used: Literal[False] = False
    calibration_partition_used: Literal[False] = False
    decision_authority: Literal["evidence_only"] = "evidence_only"

    def __post_init__(self) -> None:
        for digest in (
            self.selected_evidence_digest,
            self.selected_feature_schema_digest,
            self.validation_label_authority_digest,
            self.partition_manifest_digest,
            self.selection_digest,
        ):
            if _DIGEST_PATTERN.fullmatch(digest) is None:
                raise ModelSelectionError(
                    "ML-SELECT-007", "A champion-selection digest is invalid."
                )

    def safe_summary(self) -> dict[str, str | bool]:
        return {
            "selected_model_family": self.selected_model_family,
            "selected_model_id": self.selected_model_id,
            "selected_model_version": self.selected_model_version,
            "primary_metric": self.primary_metric,
            "secondary_metric": self.secondary_metric,
            "selection_digest": self.selection_digest,
            "test_partition_used": self.test_partition_used,
            "calibration_partition_used": self.calibration_partition_used,
        }


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower_bound: float
    upper_bound: float
    pair_count: int
    mean_probability: float
    observed_fraction: float
    absolute_gap: float


@dataclass(frozen=True, slots=True)
class CalibrationDiagnostics:
    pair_count: int
    positive_count: int
    negative_count: int
    brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    calibration_intercept: float
    calibration_slope: float
    mean_probability: float
    observed_fraction: float
    reliability_bins: tuple[ReliabilityBin, ...]
    evaluation_scope: Literal["protected_calibration_fit"] = "protected_calibration_fit"
    real_data_validation_status: Literal["not_established"] = "not_established"

    def __post_init__(self) -> None:
        if (
            self.pair_count <= 0
            or self.positive_count <= 0
            or self.negative_count <= 0
            or self.positive_count + self.negative_count != self.pair_count
        ):
            raise CalibrationError("ML-CAL-044", "Calibration diagnostic counts are invalid.")
        bounded = (
            self.brier_score,
            self.expected_calibration_error,
            self.maximum_calibration_error,
            self.mean_probability,
            self.observed_fraction,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in bounded):
            raise CalibrationError("ML-CAL-045", "Calibration diagnostics are invalid.")
        if not math.isfinite(self.calibration_intercept) or not math.isfinite(
            self.calibration_slope
        ):
            raise CalibrationError("ML-CAL-045", "Calibration diagnostics are invalid.")

    def safe_summary(self) -> dict[str, float | int | str]:
        return {
            "pair_count": self.pair_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "calibration_intercept": self.calibration_intercept,
            "calibration_slope": self.calibration_slope,
            "evaluation_scope": self.evaluation_scope,
            "real_data_validation_status": self.real_data_validation_status,
        }


@dataclass(frozen=True, slots=True, repr=False)
class CalibratorArtifact:
    method: CalibrationMethod
    calibrator_version: str
    engine_version: str
    numpy_version: str
    source_model_family: str
    source_model_id: str
    source_model_version: str
    source_evidence_digest: str
    feature_schema_digest: str
    champion_selection_digest: str
    validation_label_authority_digest: str
    calibration_label_authority_digest: str
    partition_manifest_digest: str
    calibration_pair_count: int
    positive_count: int
    negative_count: int
    payload: Mapping[str, object] = field(repr=False)
    payload_digest: str
    calibrator_digest: str
    diagnostics: CalibrationDiagnostics
    probability_status: ProbabilityStatus = "calibrated_probability"
    calibration_status: Literal["calibrated_on_protected_partition"] = (
        "calibrated_on_protected_partition"
    )
    decision_authority: DecisionAuthority = "evidence_only"
    threshold_authority: ThresholdAuthority = "none"
    assignment_authority: Literal["none"] = "none"
    real_data_validation_status: RealDataValidationStatus = "not_established"

    def __post_init__(self) -> None:
        payload = _freeze_payload(self.payload)
        object.__setattr__(self, "payload", payload)
        for digest in (
            self.source_evidence_digest,
            self.feature_schema_digest,
            self.champion_selection_digest,
            self.validation_label_authority_digest,
            self.calibration_label_authority_digest,
            self.partition_manifest_digest,
            self.payload_digest,
            self.calibrator_digest,
        ):
            require_digest(digest, code="ML-CAL-010", message="A calibrator digest is invalid.")
        if self.calibration_pair_count <= 0 or self.positive_count <= 0 or self.negative_count <= 0:
            raise CalibrationError("ML-CAL-011", "A calibrator requires both outcome classes.")
        if self.positive_count + self.negative_count != self.calibration_pair_count:
            raise CalibrationError("ML-CAL-046", "Calibrator class counts are inconsistent.")
        if canonical_digest(dict(payload)) != self.payload_digest:
            raise CalibrationError("ML-CAL-047", "Calibrator payload integrity failed.")
        if self.diagnostics.pair_count != self.calibration_pair_count:
            raise CalibrationError("ML-CAL-048", "Calibrator diagnostics are inconsistent.")
        if not self.calibrator_version or len(self.calibrator_version) > 128:
            raise CalibrationError("ML-CAL-049", "Calibrator provenance is invalid.")
        for value in (
            self.source_model_family,
            self.source_model_id,
            self.source_model_version,
        ):
            if _IDENTIFIER_PATTERN.fullmatch(value) is None:
                raise CalibrationError("ML-CAL-049", "Calibrator provenance is invalid.")

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "method": self.method,
            "calibrator_version": self.calibrator_version,
            "source_model_family": self.source_model_family,
            "source_model_id": self.source_model_id,
            "source_model_version": self.source_model_version,
            "calibration_pair_count": self.calibration_pair_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "calibrator_digest": self.calibrator_digest,
            "probability_status": self.probability_status,
            "decision_authority": self.decision_authority,
            "threshold_authority": self.threshold_authority,
            "real_data_validation_status": self.real_data_validation_status,
        }


@dataclass(frozen=True, slots=True, repr=False)
class CalibratedScoreBatch:
    pair_references: tuple[tuple[str, str], ...] = field(repr=False)
    pair_digests: tuple[str, ...] = field(repr=False)
    probabilities: NDArray[np.float64] = field(repr=False)
    source_model_family: str
    source_model_id: str
    source_model_version: str
    source_evidence_digest: str
    feature_schema_digest: str
    calibrator_method: CalibrationMethod
    calibrator_version: str
    calibrator_digest: str
    champion_selection_digest: str
    probability_status: ProbabilityStatus = "calibrated_probability"
    decision_authority: DecisionAuthority = "evidence_only"
    threshold_authority: ThresholdAuthority = "none"
    real_data_validation_status: RealDataValidationStatus = "not_established"

    def __post_init__(self) -> None:
        probabilities = immutable_float_vector(self.probabilities)
        if probabilities.ndim != 1 or len(probabilities) != len(self.pair_references):
            raise CalibrationError("ML-CAL-012", "Calibrated score coverage is invalid.")
        if len(self.pair_digests) != len(self.pair_references):
            raise CalibrationError("ML-CAL-013", "Calibrated pair digests are incomplete.")
        count = len(self.pair_references)
        if (
            count == 0
            or len(set(self.pair_references)) != count
            or len(set(self.pair_digests)) != count
        ):
            raise CalibrationError("ML-CAL-050", "Calibrated pair coverage is invalid.")
        if any(
            digest != _pair_digest(left, right)
            for (left, right), digest in zip(
                self.pair_references,
                self.pair_digests,
                strict=True,
            )
        ):
            raise CalibrationError("ML-CAL-051", "A calibrated pair digest is inconsistent.")
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
        ):
            raise CalibrationError("ML-CAL-014", "Calibrated probabilities are invalid.")
        for digest in (
            self.source_evidence_digest,
            self.feature_schema_digest,
            self.calibrator_digest,
            self.champion_selection_digest,
        ):
            require_digest(digest, code="ML-CAL-052", message="A calibrated digest is invalid.")
        for value in (
            self.source_model_family,
            self.source_model_id,
            self.source_model_version,
        ):
            if _IDENTIFIER_PATTERN.fullmatch(value) is None:
                raise CalibrationError("ML-CAL-053", "Calibrated provenance is invalid.")
        if not self.calibrator_version or len(self.calibrator_version) > 128:
            raise CalibrationError("ML-CAL-053", "Calibrated provenance is invalid.")
        object.__setattr__(self, "probabilities", probabilities)

    @property
    def pair_count(self) -> int:
        return len(self.pair_references)

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "pair_count": self.pair_count,
            "source_model_family": self.source_model_family,
            "source_model_id": self.source_model_id,
            "source_model_version": self.source_model_version,
            "calibrator_method": self.calibrator_method,
            "calibrator_version": self.calibrator_version,
            "calibrator_digest": self.calibrator_digest,
            "probability_status": self.probability_status,
            "decision_authority": self.decision_authority,
        }
