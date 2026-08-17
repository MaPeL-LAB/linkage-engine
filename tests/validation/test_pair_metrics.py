from __future__ import annotations

import numpy as np
import pytest

from mapel_linkage.domain.errors import BoostedTreeError
from mapel_linkage.validation import evaluate_binary_scores


def test_pair_metrics_are_aggregate_and_diagnostic_only() -> None:
    report = evaluate_binary_scores(
        labels=np.asarray([1, 1, 0, 0], dtype=np.int8),
        scores=np.asarray([0.95, 0.80, 0.35, 0.10], dtype=np.float64),
        diagnostic_threshold=0.5,
        evaluation_scope="synthetic_mechanical_evaluation",
        partition_manifest_digest="a" * 64,
    )

    assert report.average_precision == pytest.approx(1.0)
    assert report.roc_auc == pytest.approx(1.0)
    assert report.sensitivity == pytest.approx(1.0)
    assert report.positive_predictive_value == pytest.approx(1.0)
    assert report.threshold_authority == "diagnostic_only"
    assert report.calibration_status == "not_calibrated"
    assert report.decision_authority == "evidence_only"


def test_pair_metrics_require_both_verified_classes() -> None:
    with pytest.raises(BoostedTreeError) as captured:
        evaluate_binary_scores(
            labels=np.asarray([1, 1], dtype=np.int8),
            scores=np.asarray([0.8, 0.9], dtype=np.float64),
            diagnostic_threshold=0.5,
            evaluation_scope="synthetic_mechanical_evaluation",
            partition_manifest_digest="a" * 64,
        )

    assert captured.value.code == "ML-BOOST-032"


def test_pair_metrics_require_partition_manifest_digest() -> None:
    with pytest.raises(BoostedTreeError) as captured:
        evaluate_binary_scores(
            labels=np.asarray([1, 0], dtype=np.int8),
            scores=np.asarray([0.8, 0.2], dtype=np.float64),
            diagnostic_threshold=0.5,
            evaluation_scope="synthetic_mechanical_evaluation",
            partition_manifest_digest="invalid",
        )

    assert captured.value.code == "ML-BOOST-042"
