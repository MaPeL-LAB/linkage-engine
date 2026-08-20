"""Aggregate pair-model discrimination diagnostics with no decision authority."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from mapel_linkage.domain.errors import BoostedTreeError


def _bounded_metric(value: float) -> float:
    metric = float(value)
    if not math.isfinite(metric) or metric < -1e-12 or metric > 1.0 + 1e-12:
        raise BoostedTreeError("ML-BOOST-043", "A validation metric is outside its bounds.")
    return min(max(metric, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class PairValidationReport:
    """Aggregate validation diagnostics without pair-level predictions."""

    pair_count: int
    positive_count: int
    negative_count: int
    average_precision: float
    roc_auc: float
    brier_score: float
    sensitivity: float
    positive_predictive_value: float
    false_link_rate: float
    missed_link_rate: float
    diagnostic_threshold: float
    precision_recall_points: tuple[tuple[float, float, float | None], ...] = field(repr=False)
    evaluation_scope: str
    partition_manifest_digest: str
    threshold_authority: str = "diagnostic_only"
    calibration_status: str = "not_calibrated"
    decision_authority: str = "evidence_only"
    real_data_validation_status: str = "not_established"

    def safe_summary(self) -> dict[str, object]:
        return {
            "pair_count": self.pair_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "average_precision": self.average_precision,
            "roc_auc": self.roc_auc,
            "brier_score": self.brier_score,
            "sensitivity": self.sensitivity,
            "positive_predictive_value": self.positive_predictive_value,
            "false_link_rate": self.false_link_rate,
            "missed_link_rate": self.missed_link_rate,
            "diagnostic_threshold": self.diagnostic_threshold,
            "precision_recall_points": [
                {"precision": precision, "recall": recall, "threshold": threshold}
                for precision, recall, threshold in self.precision_recall_points
            ],
            "evaluation_scope": self.evaluation_scope,
            "partition_manifest_digest": self.partition_manifest_digest,
            "threshold_authority": self.threshold_authority,
            "calibration_status": self.calibration_status,
            "decision_authority": self.decision_authority,
            "real_data_validation_status": self.real_data_validation_status,
        }


def evaluate_binary_scores(
    *,
    labels: NDArray[np.int8],
    scores: NDArray[np.float64],
    diagnostic_threshold: float,
    evaluation_scope: str,
    partition_manifest_digest: str,
) -> PairValidationReport:
    """Evaluate uncalibrated pair evidence at a fixed diagnostic threshold."""

    target = np.asarray(labels, dtype=np.int8)
    probability_like = np.asarray(scores, dtype=np.float64)
    if target.ndim != 1 or probability_like.ndim != 1 or len(target) != len(probability_like):
        raise BoostedTreeError("ML-BOOST-030", "Validation labels and model scores do not align.")
    if len(target) == 0 or not np.all(np.isin(target, (0, 1))):
        raise BoostedTreeError(
            "ML-BOOST-031", "Validation requires non-empty binary verified labels."
        )
    if not np.any(target == 1) or not np.any(target == 0):
        raise BoostedTreeError(
            "ML-BOOST-032", "Validation requires verified matches and nonmatches."
        )
    if (
        not np.all(np.isfinite(probability_like))
        or np.any(probability_like < 0.0)
        or np.any(probability_like > 1.0)
    ):
        raise BoostedTreeError(
            "ML-BOOST-033", "Model scores must be finite and within the unit interval."
        )
    if not math.isfinite(diagnostic_threshold) or not 0.0 < diagnostic_threshold < 1.0:
        raise BoostedTreeError(
            "ML-BOOST-034", "The diagnostic threshold must be strictly between zero and one."
        )

    if len(partition_manifest_digest) != 64 or any(
        character not in "0123456789abcdef" for character in partition_manifest_digest
    ):
        raise BoostedTreeError(
            "ML-BOOST-042", "Validation requires a protected partition-manifest digest."
        )

    predicted = probability_like >= diagnostic_threshold
    positives = target == 1
    negatives = target == 0
    true_positive = int(np.sum(predicted & positives))
    false_positive = int(np.sum(predicted & negatives))
    false_negative = int(np.sum(~predicted & positives))
    positive_count = int(np.sum(positives))
    negative_count = int(np.sum(negatives))

    sensitivity = true_positive / positive_count
    predicted_positive = true_positive + false_positive
    ppv = true_positive / predicted_positive if predicted_positive else 0.0
    false_link_rate = false_positive / predicted_positive if predicted_positive else 0.0
    missed_link_rate = false_negative / positive_count
    precision_values, recall_values, thresholds = precision_recall_curve(target, probability_like)
    precision_recall_points = tuple(
        (
            float(precision_values[index]),
            float(recall_values[index]),
            (float(thresholds[index]) if index < len(thresholds) else None),
        )
        for index in range(len(precision_values))
    )

    return PairValidationReport(
        pair_count=len(target),
        positive_count=positive_count,
        negative_count=negative_count,
        average_precision=_bounded_metric(average_precision_score(target, probability_like)),
        roc_auc=_bounded_metric(roc_auc_score(target, probability_like)),
        brier_score=_bounded_metric(brier_score_loss(target, probability_like)),
        sensitivity=float(sensitivity),
        positive_predictive_value=float(ppv),
        false_link_rate=float(false_link_rate),
        missed_link_rate=float(missed_link_rate),
        diagnostic_threshold=float(diagnostic_threshold),
        precision_recall_points=precision_recall_points,
        evaluation_scope=evaluation_scope,
        partition_manifest_digest=partition_manifest_digest,
    )
