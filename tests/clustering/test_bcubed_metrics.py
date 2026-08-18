"""Unit tests for BCubed evaluation metrics, purity, pairwise metrics, and constraints."""

from __future__ import annotations

import contextlib
import math
import re
from typing import Any

try:
    import pytest
except ImportError:

    class _PytestShim:
        @staticmethod
        def approx(expected: float, abs: float | None = None) -> Any:
            class _Approx:
                def __eq__(self, actual: Any) -> bool:
                    tolerance = abs if abs is not None else 1e-6
                    return math.isclose(actual, expected, abs_tol=tolerance)

            return _Approx()

        @staticmethod
        @contextlib.contextmanager
        def raises(expected_exception: type[BaseException], match: str | None = None) -> Any:
            class _ExcInfo:
                value: BaseException

            exc_info = _ExcInfo()
            try:
                yield exc_info
            except expected_exception as e:
                exc_info.value = e
                if match and not re.search(match, str(e)):
                    raise AssertionError(f"Pattern {match!r} did not match {str(e)!r}") from e
            else:
                raise AssertionError(
                    f"Expected exception {expected_exception.__name__} was not raised"
                )

    pytest = _PytestShim()  # type: ignore[assignment]

from mapel_linkage.clustering import (
    BCubedMetrics,
    ClusterPurityMetrics,
    ConstraintViolationMetrics,
    MultiSourceEvaluationReport,
    PairwiseClusterMetrics,
    calculate_bcubed_metrics,
    calculate_cluster_purity,
    calculate_constraint_violations,
    calculate_pairwise_metrics,
    evaluate_multisource_clustering,
)
from mapel_linkage.domain.errors import ClusteringError

_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")


def test_bcubed_metrics_perfect_clustering() -> None:
    # Ground truth: {1: {a, b}, 2: {c, d}}
    # Prediction:   {1: {a, b}, 2: {c, d}}
    true_clusters = {"a": "c1", "b": "c1", "c": "c2", "d": "c2"}
    pred_clusters = {"a": "p1", "b": "p1", "c": "p2", "d": "p2"}

    metrics = calculate_bcubed_metrics(true_clusters, pred_clusters)

    assert isinstance(metrics, BCubedMetrics)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1_score == pytest.approx(1.0)
    assert metrics.item_count == 4


def test_bcubed_metrics_all_singletons_prediction() -> None:
    # Ground truth: all 4 records in one cluster {a, b, c, d}
    # Prediction: 4 singletons {a}, {b}, {c}, {d}
    true_clusters = {"a": "c1", "b": "c1", "c": "c1", "d": "c1"}
    pred_clusters = {"a": "p1", "b": "p2", "c": "p3", "d": "p4"}

    metrics = calculate_bcubed_metrics(true_clusters, pred_clusters)

    # Precision for each: |{r} ∩ {a,b,c,d}| / |{r}| = 1 / 1 = 1.0 -> Avg P = 1.0
    # Recall for each: |{r} ∩ {a,b,c,d}| / |{a,b,c,d}| = 1 / 4 = 0.25 -> Avg R = 0.25
    # F1 = 2 * 1.0 * 0.25 / (1.25) = 0.40
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(0.25)
    assert metrics.f1_score == pytest.approx(0.40)
    assert metrics.item_count == 4


def test_bcubed_metrics_all_merged_prediction() -> None:
    # Ground truth: 4 singletons {a}, {b}, {c}, {d}
    # Prediction: all merged {a, b, c, d}
    true_clusters = {"a": "c1", "b": "c2", "c": "c3", "d": "c4"}
    pred_clusters = {"a": "p1", "b": "p1", "c": "p1", "d": "p1"}

    metrics = calculate_bcubed_metrics(true_clusters, pred_clusters)

    # Precision for each: |{a,b,c,d} ∩ {r}| / |{a,b,c,d}| = 1 / 4 = 0.25 -> Avg P = 0.25
    # Recall for each: |{a,b,c,d} ∩ {r}| / |{r}| = 1 / 1 = 1.0 -> Avg R = 1.0
    # F1 = 2 * 0.25 * 1.0 / (1.25) = 0.40
    assert metrics.precision == pytest.approx(0.25)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1_score == pytest.approx(0.40)


def test_bcubed_metrics_partial_overlap() -> None:
    # Ground truth: {a, b}, {c, d}
    # Prediction:   {a, b, c}, {d}
    true_clusters = {"a": "c1", "b": "c1", "c": "c2", "d": "c2"}
    pred_clusters = {"a": "p1", "b": "p1", "c": "p1", "d": "p2"}

    metrics = calculate_bcubed_metrics(true_clusters, pred_clusters)

    # P(a) = 2/3, P(b) = 2/3, P(c) = 1/3, P(d) = 1/1 = 1.0 -> Sum P = 8/3 -> Avg P = 2/3
    # R(a) = 2/2 = 1, R(b) = 1, R(c) = 1/2 = 0.5, R(d) = 1/2 = 0.5 -> Sum R = 3 -> Avg R = 0.75
    # F1 = 2 * (2/3) * (3/4) / (2/3 + 3/4) = 1 / (17/12) = 12/17 ≈ 0.705882
    assert metrics.precision == pytest.approx(2.0 / 3.0)
    assert metrics.recall == pytest.approx(0.75)
    assert metrics.f1_score == pytest.approx(12.0 / 17.0)


def test_bcubed_metrics_different_input_formats() -> None:
    # Format 1: dict[record_key, cluster_id]
    true_dict = {"a": "c1", "b": "c1", "c": "c2"}
    pred_dict = {"a": "p1", "b": "p1", "c": "p2"}
    res1 = calculate_bcubed_metrics(true_dict, pred_dict)

    # Format 2: dict[cluster_id, list[record_key]]
    true_mapping = {"c1": ["a", "b"], "c2": ["c"]}
    pred_mapping = {"p1": ["a", "b"], "p2": ["c"]}
    res2 = calculate_bcubed_metrics(true_mapping, pred_mapping)

    # Format 3: list[list[record_key]]
    true_list = [["a", "b"], ["c"]]
    pred_list = [["a", "b"], ["c"]]
    res3 = calculate_bcubed_metrics(true_list, pred_list)

    assert res1.f1_score == res2.f1_score == res3.f1_score == pytest.approx(1.0)


def test_cluster_purity_metrics() -> None:
    # Ground truth: {a, b}, {c, d}
    # Prediction:   {a, b, c}, {d}
    true_clusters = {"a": "c1", "b": "c1", "c": "c2", "d": "c2"}
    pred_clusters = {"a": "p1", "b": "p1", "c": "p1", "d": "p2"}

    purity = calculate_cluster_purity(true_clusters, pred_clusters)

    assert isinstance(purity, ClusterPurityMetrics)
    # p1: 2 'c1' and 1 'c2' -> max is 2
    # p2: 1 'c2' -> max is 1
    # Purity = (2 + 1) / 4 = 0.75
    assert purity.purity == pytest.approx(0.75)

    # Inverse purity:
    # c1: 2 in 'p1' -> max 2
    # c2: 1 in 'p1', 1 in 'p2' -> max 1
    # Inverse purity = (2 + 1) / 4 = 0.75
    assert purity.inverse_purity == pytest.approx(0.75)
    assert purity.total_records == 4
    assert purity.predicted_cluster_count == 2
    assert purity.true_cluster_count == 2


def test_pairwise_cluster_metrics() -> None:
    # Records: a, b, c, d (total pairs = 6)
    # Ground truth: {a, b}, {c, d} -> True positive pairs: (a,b), (c,d) [2 pairs]
    # Prediction:   {a, b, c}, {d} -> Predicted pairs: (a,b), (a,c), (b,c) [3 pairs]
    # TP: (a,b) -> 1 pair
    # FP: (a,c), (b,c) -> 2 pairs
    # FN: (c,d) -> 1 pair
    # TN: (a,d), (b,d) -> 2 pairs
    # Total = 1 + 2 + 1 + 2 = 6
    true_clusters = {"a": "c1", "b": "c1", "c": "c2", "d": "c2"}
    pred_clusters = {"a": "p1", "b": "p1", "c": "p1", "d": "p2"}

    pairwise = calculate_pairwise_metrics(true_clusters, pred_clusters)

    assert isinstance(pairwise, PairwiseClusterMetrics)
    assert pairwise.tp == 1
    assert pairwise.fp == 2
    assert pairwise.fn == 1
    assert pairwise.tn == 2
    assert pairwise.total_pairs == 6
    assert pairwise.precision == pytest.approx(1.0 / 3.0)
    assert pairwise.recall == pytest.approx(1.0 / 2.0)
    # F1 = 2 * (1/3) * (1/2) / (1/3 + 1/2) = (1/3) / (5/6) = 2/5 = 0.40
    assert pairwise.f1_score == pytest.approx(0.40)


def test_constraint_violation_metrics() -> None:
    pred_clusters = {"a": "p1", "b": "p1", "c": "p2", "d": "p2"}

    cannot_link = [("a", "b"), ("a", "c")]  # (a,b) violated since both in p1; (a,c) respected
    must_link = [("a", "d"), ("c", "d")]  # (a,d) violated since in p1 vs p2; (c,d) respected

    record_to_ds = {"a": "ds1", "b": "ds1", "c": "ds1", "d": "ds2"}
    # In p1: 'a' (ds1) and 'b' (ds1) -> dataset collision!
    # In p2: 'c' (ds1) and 'd' (ds2) -> no collision

    violations = calculate_constraint_violations(
        pred_clusters,
        cannot_link_pairs=cannot_link,
        must_link_pairs=must_link,
        record_to_dataset=record_to_ds,
    )

    assert isinstance(violations, ConstraintViolationMetrics)
    assert violations.cannot_link_checked == 2
    assert violations.cannot_link_violations == 1
    assert violations.cannot_link_violation_rate == pytest.approx(0.5)
    assert violations.must_link_checked == 2
    assert violations.must_link_violations == 1
    assert violations.must_link_violation_rate == pytest.approx(0.5)
    assert violations.dataset_collisions == 1
    assert violations.total_clusters == 2


def test_evaluate_multisource_clustering_report() -> None:
    true_clusters = {"a": "c1", "b": "c1", "c": "c2", "d": "c2"}
    pred_clusters = {"a": "p1", "b": "p1", "c": "p2", "d": "p2"}

    report = evaluate_multisource_clustering(
        true_clusters,
        pred_clusters,
        cannot_link_pairs=[("a", "c")],
        must_link_pairs=[("a", "b")],
        record_to_dataset={"a": "ds1", "b": "ds2", "c": "ds1", "d": "ds2"},
        evaluation_scope="test_evaluation_run",
    )

    assert isinstance(report, MultiSourceEvaluationReport)
    assert report.bcubed_precision == pytest.approx(1.0)
    assert report.bcubed_recall == pytest.approx(1.0)
    assert report.bcubed_f1 == pytest.approx(1.0)
    assert report.cluster_purity == pytest.approx(1.0)
    assert report.cannot_link_violations == 0
    assert report.must_link_violations == 0
    assert report.dataset_collisions == 0
    assert _DIGEST_PATTERN.fullmatch(report.evaluation_digest) is not None

    summary = report.safe_summary()
    assert summary["evaluation_scope"] == "test_evaluation_run"
    assert summary["total_records"] == 4
    assert summary["bcubed_f1"] == 1.0


def test_clustering_metrics_validation_errors() -> None:
    # Empty universe error
    with pytest.raises(ClusteringError, match="ML-CLUSTER-001"):
        calculate_bcubed_metrics({}, {})

    # Mismatched record universes
    with pytest.raises(ClusteringError, match="ML-CLUSTER-001"):
        calculate_bcubed_metrics({"a": "c1"}, {"b": "p1"})

    # Duplicate record in mapping
    with pytest.raises(ClusteringError, match="ML-CLUSTER-001"):
        calculate_bcubed_metrics({"c1": ["a", "b"], "c2": ["a"]}, {"p1": ["a", "b"]})

    # Out of bounds metrics
    with pytest.raises(ClusteringError, match="ML-CLUSTER-012"):
        BCubedMetrics(precision=1.5, recall=1.0, f1_score=1.0, item_count=1)

    with pytest.raises(ClusteringError, match="ML-CLUSTER-012"):
        ClusterPurityMetrics(
            purity=-0.1,
            inverse_purity=1.0,
            total_records=1,
            predicted_cluster_count=1,
            true_cluster_count=1,
        )
