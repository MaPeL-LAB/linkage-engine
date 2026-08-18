from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mapel_linkage.domain.errors import ValidationReportError
from mapel_linkage.validation import evaluate_configured_decision_thresholds


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_decision_threshold_evidence_uses_only_the_protected_decision_partition() -> None:
    report = evaluate_configured_decision_thresholds(
        probabilities=np.asarray([0.02, 0.08, 0.18, 0.42, 0.61, 0.82, 0.95, 0.99]),
        labels=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8),
        confirmed_threshold=0.95,
        review_threshold=0.60,
        no_match_threshold=0.20,
        partition_manifest_digest=digest("decision-partition"),
    )
    assert report.decision_partition_used is True
    assert report.test_partition_used is False
    assert report.threshold_authority == "synthetic_benchmark_only"
    assert report.operational_authority == "none"
    assert report.confirmed_sensitivity == 0.5
    assert report.confirmed_positive_predictive_value == 1.0
    assert report.no_match_specificity == 0.75


def test_decision_threshold_evidence_rejects_invalid_regions_and_single_class_truth() -> None:
    with pytest.raises(ValidationReportError):
        evaluate_configured_decision_thresholds(
            probabilities=np.asarray([0.1, 0.9]),
            labels=np.asarray([0, 1], dtype=np.int8),
            confirmed_threshold=0.8,
            review_threshold=0.9,
            no_match_threshold=0.2,
            partition_manifest_digest=digest("decision-partition"),
        )
    with pytest.raises(ValidationReportError):
        evaluate_configured_decision_thresholds(
            probabilities=np.asarray([0.1, 0.2]),
            labels=np.asarray([0, 0], dtype=np.int8),
            confirmed_threshold=0.9,
            review_threshold=0.6,
            no_match_threshold=0.2,
            partition_manifest_digest=digest("decision-partition"),
        )
