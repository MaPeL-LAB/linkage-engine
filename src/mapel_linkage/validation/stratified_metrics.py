"""Aggregate pair-performance diagnostics by missingness and candidate-set size."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
)

from mapel_linkage.domain.errors import ValidationReportError


@dataclass(frozen=True, slots=True)
class PairPerformanceStratum:
    stratum: str
    pair_count: int
    positive_count: int
    negative_count: int
    average_precision: float | None
    brier_score: float
    sensitivity: float
    positive_predictive_value: float
    false_link_rate: float
    missed_link_rate: float

    def safe_summary(self) -> dict[str, float | int | str]:
        return {
            "stratum": self.stratum,
            "pair_count": self.pair_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "average_precision": (
                self.average_precision if self.average_precision is not None else "-"
            ),
            "brier_score": self.brier_score,
            "sensitivity": self.sensitivity,
            "positive_predictive_value": self.positive_predictive_value,
            "false_link_rate": self.false_link_rate,
            "missed_link_rate": self.missed_link_rate,
        }


@dataclass(frozen=True, slots=True)
class StratifiedPairValidationReport:
    pair_count: int
    missingness_pattern_strata: tuple[PairPerformanceStratum, ...] = field(repr=False)
    candidate_set_size_strata: tuple[PairPerformanceStratum, ...] = field(repr=False)
    partition_manifest_digest: str
    evaluation_scope: str = "synthetic_mechanical_evaluation"
    real_data_validation_status: str = "not_established"

    def safe_summary(self) -> dict[str, object]:
        return {
            "pair_count": self.pair_count,
            "missingness_pattern_strata": [
                stratum.safe_summary() for stratum in self.missingness_pattern_strata
            ],
            "candidate_set_size_strata": [
                stratum.safe_summary() for stratum in self.candidate_set_size_strata
            ],
            "partition_manifest_digest": self.partition_manifest_digest,
            "evaluation_scope": self.evaluation_scope,
            "real_data_validation_status": self.real_data_validation_status,
        }


def candidate_set_size_band(size: int) -> str:
    if size < 1:
        return "size_0"
    if size == 1:
        return "size_1"
    if size <= 5:
        return "size_2_to_5"
    if size <= 10:
        return "size_6_to_10"
    return "size_11_plus"


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _stratum(
    *,
    name: str,
    labels: NDArray[np.int8],
    probabilities: NDArray[np.float64],
    threshold: float,
) -> PairPerformanceStratum:
    positive = labels == 1
    negative = labels == 0
    predicted = probabilities >= threshold
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & negative))
    false_negative = int(np.sum(~predicted & positive))
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    average_precision: float | None = None
    if positive_count and negative_count:
        average_precision = float(average_precision_score(labels, probabilities))
    return PairPerformanceStratum(
        stratum=name,
        pair_count=len(labels),
        positive_count=positive_count,
        negative_count=negative_count,
        average_precision=average_precision,
        brier_score=float(brier_score_loss(labels, probabilities)),
        sensitivity=_safe_divide(true_positive, positive_count),
        positive_predictive_value=_safe_divide(
            true_positive,
            true_positive + false_positive,
        ),
        false_link_rate=_safe_divide(false_positive, true_positive + false_positive),
        missed_link_rate=_safe_divide(false_negative, positive_count),
    )


def _grouped_strata(
    *,
    group_names: tuple[str, ...],
    labels: NDArray[np.int8],
    probabilities: NDArray[np.float64],
    threshold: float,
) -> tuple[PairPerformanceStratum, ...]:
    output: list[PairPerformanceStratum] = []
    for group in sorted(set(group_names)):
        indices = np.asarray(
            [index for index, value in enumerate(group_names) if value == group],
            dtype=np.int64,
        )
        output.append(
            _stratum(
                name=group,
                labels=labels[indices],
                probabilities=probabilities[indices],
                threshold=threshold,
            )
        )
    return tuple(output)


def evaluate_stratified_pair_performance(
    *,
    labels: NDArray[np.int8],
    probabilities: NDArray[np.float64],
    missingness_patterns: tuple[str, ...],
    candidate_set_sizes: tuple[int, ...],
    diagnostic_threshold: float,
    partition_manifest_digest: str,
) -> StratifiedPairValidationReport:
    target = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    count = len(target)
    if (
        target.ndim != 1
        or scores.ndim != 1
        or len(scores) != count
        or len(missingness_patterns) != count
        or len(candidate_set_sizes) != count
        or count == 0
    ):
        raise ValidationReportError(
            "ML-VALID-STRAT-001",
            "Stratified pair-performance inputs have invalid coverage.",
        )
    if not np.all(np.isin(target, np.asarray([0, 1], dtype=np.int8))):
        raise ValidationReportError(
            "ML-VALID-STRAT-002",
            "Stratified pair-performance labels are invalid.",
        )
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValidationReportError(
            "ML-VALID-STRAT-003",
            "Stratified pair-performance probabilities are invalid.",
        )
    if not math.isfinite(diagnostic_threshold) or not 0.0 < diagnostic_threshold < 1.0:
        raise ValidationReportError(
            "ML-VALID-STRAT-004",
            "The stratified diagnostic threshold is invalid.",
        )
    if len(partition_manifest_digest) != 64 or any(
        character not in "0123456789abcdef" for character in partition_manifest_digest
    ):
        raise ValidationReportError(
            "ML-VALID-STRAT-005",
            "The stratified partition digest is invalid.",
        )
    if any(not value or len(value) > 128 for value in missingness_patterns):
        raise ValidationReportError(
            "ML-VALID-STRAT-006",
            "A missingness stratum identifier is invalid.",
        )
    if any(size < 1 for size in candidate_set_sizes):
        raise ValidationReportError(
            "ML-VALID-STRAT-007",
            "A candidate-set-size stratum is invalid.",
        )

    size_bands = tuple(candidate_set_size_band(size) for size in candidate_set_sizes)
    return StratifiedPairValidationReport(
        pair_count=count,
        missingness_pattern_strata=_grouped_strata(
            group_names=missingness_patterns,
            labels=target,
            probabilities=scores,
            threshold=diagnostic_threshold,
        ),
        candidate_set_size_strata=_grouped_strata(
            group_names=size_bands,
            labels=target,
            probabilities=scores,
            threshold=diagnostic_threshold,
        ),
        partition_manifest_digest=partition_manifest_digest,
    )
