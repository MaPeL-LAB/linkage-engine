"""Protected decision-partition evidence for configured relationship thresholds."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mapel_linkage.domain.errors import ValidationReportError


@dataclass(frozen=True, slots=True)
class DecisionThresholdReport:
    """Aggregate evidence for configured thresholds on the protected decision split."""

    pair_count: int
    positive_count: int
    negative_count: int
    confirmed_threshold: float
    review_threshold: float
    no_match_threshold: float
    confirmed_sensitivity: float
    confirmed_positive_predictive_value: float
    confirmed_false_link_rate: float
    confirmed_missed_link_rate: float
    no_match_specificity: float
    no_match_false_no_match_rate: float
    review_region_fraction: float
    decision_policy_digest: str
    partition_manifest_digest: str
    decision_partition_used: Literal[True] = True
    test_partition_used: Literal[False] = False
    threshold_authority: Literal["synthetic_benchmark_only"] = "synthetic_benchmark_only"
    operational_authority: Literal["none"] = "none"
    evaluation_scope: Literal["synthetic_mechanical_evaluation"] = "synthetic_mechanical_evaluation"
    real_data_validation_status: Literal["not_established"] = "not_established"

    def safe_summary(self) -> dict[str, bool | float | int | str]:
        return {
            "pair_count": self.pair_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "confirmed_threshold": self.confirmed_threshold,
            "review_threshold": self.review_threshold,
            "no_match_threshold": self.no_match_threshold,
            "confirmed_sensitivity": self.confirmed_sensitivity,
            "confirmed_positive_predictive_value": self.confirmed_positive_predictive_value,
            "confirmed_false_link_rate": self.confirmed_false_link_rate,
            "confirmed_missed_link_rate": self.confirmed_missed_link_rate,
            "no_match_specificity": self.no_match_specificity,
            "no_match_false_no_match_rate": self.no_match_false_no_match_rate,
            "review_region_fraction": self.review_region_fraction,
            "decision_policy_digest": self.decision_policy_digest,
            "partition_manifest_digest": self.partition_manifest_digest,
            "decision_partition_used": self.decision_partition_used,
            "test_partition_used": self.test_partition_used,
            "threshold_authority": self.threshold_authority,
            "operational_authority": self.operational_authority,
            "evaluation_scope": self.evaluation_scope,
            "real_data_validation_status": self.real_data_validation_status,
        }


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _require_digest(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValidationReportError(
            "ML-VALID-THRESH-001",
            "A protected decision-threshold digest is invalid.",
        )


def evaluate_configured_decision_thresholds(
    *,
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int8],
    confirmed_threshold: float,
    review_threshold: float,
    no_match_threshold: float,
    partition_manifest_digest: str,
) -> DecisionThresholdReport:
    """Evaluate configured thresholds without selecting or operationally approving them."""

    _require_digest(partition_manifest_digest)
    scores = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int8)
    if scores.ndim != 1 or truth.ndim != 1 or len(scores) != len(truth) or len(scores) == 0:
        raise ValidationReportError(
            "ML-VALID-THRESH-002",
            "Protected decision-threshold inputs have invalid coverage.",
        )
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValidationReportError(
            "ML-VALID-THRESH-003",
            "Protected decision-threshold probabilities are invalid.",
        )
    if not np.all(np.isin(truth, np.asarray([0, 1], dtype=np.int8))):
        raise ValidationReportError(
            "ML-VALID-THRESH-004",
            "Protected decision-threshold labels are invalid.",
        )
    if not (
        math.isfinite(confirmed_threshold)
        and math.isfinite(review_threshold)
        and math.isfinite(no_match_threshold)
        and 0.0 <= no_match_threshold < review_threshold <= confirmed_threshold <= 1.0
    ):
        raise ValidationReportError(
            "ML-VALID-THRESH-005",
            "Configured decision thresholds are invalid.",
        )

    positives = truth == 1
    negatives = truth == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    if positive_count == 0 or negative_count == 0:
        raise ValidationReportError(
            "ML-VALID-THRESH-006",
            "Protected decision-threshold evidence requires both verified classes.",
        )

    confirmed = scores >= confirmed_threshold
    true_positive = int(np.sum(confirmed & positives))
    false_positive = int(np.sum(confirmed & negatives))
    false_negative = int(np.sum(~confirmed & positives))

    proposed_no_match = scores <= no_match_threshold
    true_no_match = int(np.sum(proposed_no_match & negatives))
    false_no_match = int(np.sum(proposed_no_match & positives))

    review_region = (scores >= review_threshold) & (scores < confirmed_threshold)
    policy_payload = {
        "confirmed_threshold": confirmed_threshold,
        "review_threshold": review_threshold,
        "no_match_threshold": no_match_threshold,
        "partition_manifest_digest": partition_manifest_digest,
        "threshold_authority": "synthetic_benchmark_only",
        "test_partition_used": False,
    }
    decision_policy_digest = hashlib.sha256(
        json.dumps(policy_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return DecisionThresholdReport(
        pair_count=len(scores),
        positive_count=positive_count,
        negative_count=negative_count,
        confirmed_threshold=confirmed_threshold,
        review_threshold=review_threshold,
        no_match_threshold=no_match_threshold,
        confirmed_sensitivity=_safe_divide(true_positive, positive_count),
        confirmed_positive_predictive_value=_safe_divide(
            true_positive,
            true_positive + false_positive,
        ),
        confirmed_false_link_rate=_safe_divide(
            false_positive,
            true_positive + false_positive,
        ),
        confirmed_missed_link_rate=_safe_divide(false_negative, positive_count),
        no_match_specificity=_safe_divide(true_no_match, negative_count),
        no_match_false_no_match_rate=_safe_divide(false_no_match, positive_count),
        review_region_fraction=float(review_region.mean()),
        decision_policy_digest=decision_policy_digest,
        partition_manifest_digest=partition_manifest_digest,
    )
