"""Multi-source entity resolution evaluation metrics including BCubed, purity, and constraints."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from mapel_linkage.domain.errors import ClusteringError

_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")


def _canonical_digest(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _normalize_cluster_mapping(
    clusters: Mapping[str, str]
    | Mapping[str, Iterable[str]]
    | Sequence[Iterable[str]]
    | Iterable[Iterable[str]],
    universe_name: str = "cluster_partition",
) -> dict[str, str]:
    """Normalize various cluster partition formats into a dict[record_key, cluster_id]."""
    record_to_cluster: dict[str, str] = {}

    if isinstance(clusters, Mapping):
        if not clusters:
            return {}

        first_val = next(iter(clusters.values()))
        if isinstance(first_val, str):
            for k, v in clusters.items():
                rec_k = str(k).strip()
                clust_id = str(v).strip()
                if not rec_k:
                    raise ClusteringError(
                        "ML-CLUSTER-001", f"Record key in {universe_name} cannot be empty."
                    )
                if not clust_id:
                    raise ClusteringError(
                        "ML-CLUSTER-001", f"Cluster ID in {universe_name} cannot be empty."
                    )
                record_to_cluster[rec_k] = clust_id
        elif isinstance(first_val, (set, frozenset, list, tuple, Sequence, Iterable)):
            for clust_id_raw, member_keys in clusters.items():
                clust_id = str(clust_id_raw).strip()
                if not clust_id:
                    raise ClusteringError(
                        "ML-CLUSTER-001", f"Cluster ID in {universe_name} cannot be empty."
                    )
                for member in member_keys:
                    rec_k = str(member).strip()
                    if not rec_k:
                        raise ClusteringError(
                            "ML-CLUSTER-001", f"Record key in {universe_name} cannot be empty."
                        )
                    if rec_k in record_to_cluster:
                        raise ClusteringError(
                            "ML-CLUSTER-001",
                            f"Record appears in multiple clusters in {universe_name}.",
                        )
                    record_to_cluster[rec_k] = clust_id
        else:
            raise ClusteringError(
                "ML-CLUSTER-001", f"Unsupported mapping value type in {universe_name}."
            )
    elif isinstance(clusters, (list, tuple, Sequence, Iterable)):
        for idx, member_keys in enumerate(clusters):
            clust_id = f"cluster_{idx}"
            for member in member_keys:
                rec_k = str(member).strip()
                if not rec_k:
                    raise ClusteringError(
                        "ML-CLUSTER-001", f"Record key in {universe_name} cannot be empty."
                    )
                if rec_k in record_to_cluster:
                    raise ClusteringError(
                        "ML-CLUSTER-001",
                        f"Record appears in multiple clusters in {universe_name}.",
                    )
                record_to_cluster[rec_k] = clust_id
    else:
        raise ClusteringError(
            "ML-CLUSTER-001", f"Unsupported cluster representation type in {universe_name}."
        )

    return record_to_cluster


@dataclass(frozen=True, slots=True)
class BCubedMetrics:
    """BCubed precision, recall, and F1 evaluation metrics."""

    precision: float
    recall: float
    f1_score: float
    item_count: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.precision) or not (0.0 <= self.precision <= 1.0):
            raise ClusteringError("ML-CLUSTER-012", "BCubed precision is outside [0, 1].")
        if not math.isfinite(self.recall) or not (0.0 <= self.recall <= 1.0):
            raise ClusteringError("ML-CLUSTER-012", "BCubed recall is outside [0, 1].")
        if not math.isfinite(self.f1_score) or not (0.0 <= self.f1_score <= 1.0):
            raise ClusteringError("ML-CLUSTER-012", "BCubed F1 score is outside [0, 1].")
        if self.item_count < 0:
            raise ClusteringError("ML-CLUSTER-012", "Item count cannot be negative.")

    def safe_summary(self) -> dict[str, float | int]:
        return {
            "bcubed_precision": self.precision,
            "bcubed_recall": self.recall,
            "bcubed_f1": self.f1_score,
            "item_count": self.item_count,
        }


@dataclass(frozen=True, slots=True)
class ClusterPurityMetrics:
    """Cluster purity and inverse purity evaluation metrics."""

    purity: float
    inverse_purity: float
    total_records: int
    predicted_cluster_count: int
    true_cluster_count: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.purity) or not (0.0 <= self.purity <= 1.0):
            raise ClusteringError("ML-CLUSTER-012", "Cluster purity is outside [0, 1].")
        if not math.isfinite(self.inverse_purity) or not (0.0 <= self.inverse_purity <= 1.0):
            raise ClusteringError("ML-CLUSTER-012", "Inverse purity is outside [0, 1].")
        if self.total_records < 0:
            raise ClusteringError("ML-CLUSTER-012", "Total records count cannot be negative.")

    def safe_summary(self) -> dict[str, float | int]:
        return {
            "cluster_purity": self.purity,
            "inverse_purity": self.inverse_purity,
            "total_records": self.total_records,
            "predicted_cluster_count": self.predicted_cluster_count,
            "true_cluster_count": self.true_cluster_count,
        }


@dataclass(frozen=True, slots=True)
class PairwiseClusterMetrics:
    """Pairwise link evaluation metrics (precision, recall, F1, TP, FP, FN, TN)."""

    precision: float
    recall: float
    f1_score: float
    tp: int
    fp: int
    fn: int
    tn: int
    total_pairs: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.precision) or not (0.0 <= self.precision <= 1.0):
            raise ClusteringError("ML-CLUSTER-012", "Pairwise precision is outside [0, 1].")
        if not math.isfinite(self.recall) or not (0.0 <= self.recall <= 1.0):
            raise ClusteringError("ML-CLUSTER-012", "Pairwise recall is outside [0, 1].")
        if not math.isfinite(self.f1_score) or not (0.0 <= self.f1_score <= 1.0):
            raise ClusteringError("ML-CLUSTER-012", "Pairwise F1 score is outside [0, 1].")
        if any(count < 0 for count in (self.tp, self.fp, self.fn, self.tn, self.total_pairs)):
            raise ClusteringError("ML-CLUSTER-012", "Pair counts cannot be negative.")
        if self.tp + self.fp + self.fn + self.tn != self.total_pairs:
            raise ClusteringError("ML-CLUSTER-012", "Pairwise contingency sums do not match total.")

    def safe_summary(self) -> dict[str, float | int]:
        return {
            "pairwise_precision": self.precision,
            "pairwise_recall": self.recall,
            "pairwise_f1": self.f1_score,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "total_pairs": self.total_pairs,
        }


@dataclass(frozen=True, slots=True)
class ConstraintViolationMetrics:
    """Evaluation metrics for cannot-link, must-link, and dataset collision constraints."""

    cannot_link_violations: int
    cannot_link_checked: int
    cannot_link_violation_rate: float
    must_link_violations: int
    must_link_checked: int
    must_link_violation_rate: float
    dataset_collisions: int
    total_clusters: int

    def __post_init__(self) -> None:
        if self.cannot_link_violations < 0 or self.cannot_link_checked < 0:
            raise ClusteringError(
                "ML-CLUSTER-012", "Cannot-link violation counts cannot be negative."
            )
        if self.must_link_violations < 0 or self.must_link_checked < 0:
            raise ClusteringError(
                "ML-CLUSTER-012", "Must-link violation counts cannot be negative."
            )
        if not math.isfinite(self.cannot_link_violation_rate) or not (
            0.0 <= self.cannot_link_violation_rate <= 1.0
        ):
            raise ClusteringError("ML-CLUSTER-012", "Cannot-link violation rate is outside [0, 1].")
        if not math.isfinite(self.must_link_violation_rate) or not (
            0.0 <= self.must_link_violation_rate <= 1.0
        ):
            raise ClusteringError("ML-CLUSTER-012", "Must-link violation rate is outside [0, 1].")

    def safe_summary(self) -> dict[str, float | int]:
        return {
            "cannot_link_violations": self.cannot_link_violations,
            "cannot_link_checked": self.cannot_link_checked,
            "cannot_link_violation_rate": self.cannot_link_violation_rate,
            "must_link_violations": self.must_link_violations,
            "must_link_checked": self.must_link_checked,
            "must_link_violation_rate": self.must_link_violation_rate,
            "dataset_collisions": self.dataset_collisions,
            "total_clusters": self.total_clusters,
        }


@dataclass(frozen=True, slots=True, repr=False)
class MultiSourceEvaluationReport:
    """Comprehensive immutable evaluation report for multi-source entity resolution."""

    bcubed_precision: float
    bcubed_recall: float
    bcubed_f1: float
    cluster_purity: float
    inverse_purity: float
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float
    pairwise_tp: int
    pairwise_fp: int
    pairwise_fn: int
    pairwise_tn: int
    total_records: int
    total_true_clusters: int
    total_predicted_clusters: int
    cannot_link_violations: int
    cannot_link_constraints_count: int
    cannot_link_violation_rate: float
    must_link_violations: int
    must_link_constraints_count: int
    must_link_violation_rate: float
    dataset_collisions: int
    evaluation_digest: str
    evaluation_scope: str = "multisource_resolution_evaluation"

    def __post_init__(self) -> None:
        if _DIGEST_PATTERN.fullmatch(self.evaluation_digest) is None:
            raise ClusteringError("ML-CLUSTER-013", "Evaluation digest format is invalid.")
        for name, val in (
            ("bcubed_precision", self.bcubed_precision),
            ("bcubed_recall", self.bcubed_recall),
            ("bcubed_f1", self.bcubed_f1),
            ("cluster_purity", self.cluster_purity),
            ("inverse_purity", self.inverse_purity),
            ("pairwise_precision", self.pairwise_precision),
            ("pairwise_recall", self.pairwise_recall),
            ("pairwise_f1", self.pairwise_f1),
            ("cannot_link_violation_rate", self.cannot_link_violation_rate),
            ("must_link_violation_rate", self.must_link_violation_rate),
        ):
            if not math.isfinite(val) or not (0.0 <= val <= 1.0):
                raise ClusteringError(
                    "ML-CLUSTER-012", f"Metric {name} is outside valid bounds [0, 1]."
                )

    def safe_summary(self) -> dict[str, object]:
        """Value-safe summary of clustering evaluation without sensitive participant IDs."""
        return {
            "evaluation_scope": self.evaluation_scope,
            "total_records": self.total_records,
            "total_true_clusters": self.total_true_clusters,
            "total_predicted_clusters": self.total_predicted_clusters,
            "bcubed_precision": self.bcubed_precision,
            "bcubed_recall": self.bcubed_recall,
            "bcubed_f1": self.bcubed_f1,
            "cluster_purity": self.cluster_purity,
            "inverse_purity": self.inverse_purity,
            "pairwise_precision": self.pairwise_precision,
            "pairwise_recall": self.pairwise_recall,
            "pairwise_f1": self.pairwise_f1,
            "pairwise_tp": self.pairwise_tp,
            "pairwise_fp": self.pairwise_fp,
            "pairwise_fn": self.pairwise_fn,
            "pairwise_tn": self.pairwise_tn,
            "cannot_link_violations": self.cannot_link_violations,
            "cannot_link_constraints_count": self.cannot_link_constraints_count,
            "cannot_link_violation_rate": self.cannot_link_violation_rate,
            "must_link_violations": self.must_link_violations,
            "must_link_constraints_count": self.must_link_constraints_count,
            "must_link_violation_rate": self.must_link_violation_rate,
            "dataset_collisions": self.dataset_collisions,
            "evaluation_digest": self.evaluation_digest,
        }


def calculate_bcubed_metrics(
    true_clusters: Mapping[str, str]
    | Mapping[str, Iterable[str]]
    | Sequence[Iterable[str]]
    | Iterable[Iterable[str]],
    predicted_clusters: Mapping[str, str]
    | Mapping[str, Iterable[str]]
    | Sequence[Iterable[str]]
    | Iterable[Iterable[str]],
) -> BCubedMetrics:
    """Calculate BCubed Precision, BCubed Recall, and BCubed F1 score.

    BCubed evaluates precision and recall at the individual item level:
    - Precision(e) = |C(e) ∩ L(e)| / |C(e)|
    - Recall(e) = |C(e) ∩ L(e)| / |L(e)|
    where C(e) is e's predicted cluster and L(e) is e's ground truth cluster.
    """
    true_map = _normalize_cluster_mapping(true_clusters, "true_clusters")
    pred_map = _normalize_cluster_mapping(predicted_clusters, "predicted_clusters")

    if not true_map or not pred_map:
        raise ClusteringError(
            "ML-CLUSTER-001", "Cluster universes cannot be empty for BCubed metric evaluation."
        )

    true_records = set(true_map.keys())
    pred_records = set(pred_map.keys())

    if true_records != pred_records:
        raise ClusteringError(
            "ML-CLUSTER-001",
            "True and predicted record universes must contain identical records.",
        )

    total_records = len(true_records)
    true_sizes: Counter[str] = Counter(true_map.values())
    pred_sizes: Counter[str] = Counter(pred_map.values())

    contingency: Counter[tuple[str, str]] = Counter(
        (pred_map[r], true_map[r]) for r in true_records
    )

    sum_precision = 0.0
    sum_recall = 0.0

    for r in true_records:
        p_c = pred_map[r]
        t_c = true_map[r]
        intersection = contingency[(p_c, t_c)]

        sum_precision += intersection / pred_sizes[p_c]
        sum_recall += intersection / true_sizes[t_c]

    precision = sum_precision / total_records
    recall = sum_recall / total_records

    f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall > 0.0 else 0.0

    return BCubedMetrics(
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        item_count=total_records,
    )


def calculate_cluster_purity(
    true_clusters: Mapping[str, str]
    | Mapping[str, Iterable[str]]
    | Sequence[Iterable[str]]
    | Iterable[Iterable[str]],
    predicted_clusters: Mapping[str, str]
    | Mapping[str, Iterable[str]]
    | Sequence[Iterable[str]]
    | Iterable[Iterable[str]],
) -> ClusterPurityMetrics:
    """Calculate cluster purity and inverse purity."""
    true_map = _normalize_cluster_mapping(true_clusters, "true_clusters")
    pred_map = _normalize_cluster_mapping(predicted_clusters, "predicted_clusters")

    if not true_map or not pred_map:
        raise ClusteringError(
            "ML-CLUSTER-001", "Cluster universes cannot be empty for purity evaluation."
        )

    true_records = set(true_map.keys())
    pred_records = set(pred_map.keys())

    if true_records != pred_records:
        raise ClusteringError(
            "ML-CLUSTER-001",
            "True and predicted record universes must contain identical records.",
        )

    total_records = len(true_records)
    records_by_pred: dict[str, list[str]] = defaultdict(list)
    records_by_true: dict[str, list[str]] = defaultdict(list)

    for r in true_records:
        records_by_pred[pred_map[r]].append(true_map[r])
        records_by_true[true_map[r]].append(pred_map[r])

    purity_sum = sum(max(Counter(members).values()) for members in records_by_pred.values())
    inv_purity_sum = sum(max(Counter(members).values()) for members in records_by_true.values())

    purity = purity_sum / total_records
    inv_purity = inv_purity_sum / total_records

    return ClusterPurityMetrics(
        purity=float(purity),
        inverse_purity=float(inv_purity),
        total_records=total_records,
        predicted_cluster_count=len(records_by_pred),
        true_cluster_count=len(records_by_true),
    )


def calculate_pairwise_metrics(
    true_clusters: Mapping[str, str]
    | Mapping[str, Iterable[str]]
    | Sequence[Iterable[str]]
    | Iterable[Iterable[str]],
    predicted_clusters: Mapping[str, str]
    | Mapping[str, Iterable[str]]
    | Sequence[Iterable[str]]
    | Iterable[Iterable[str]],
) -> PairwiseClusterMetrics:
    """Calculate pairwise Precision, Recall, F1, TP, FP, FN, TN over all record pairs."""
    true_map = _normalize_cluster_mapping(true_clusters, "true_clusters")
    pred_map = _normalize_cluster_mapping(predicted_clusters, "predicted_clusters")

    if not true_map or not pred_map:
        raise ClusteringError(
            "ML-CLUSTER-001", "Cluster universes cannot be empty for pairwise metric evaluation."
        )

    true_records = set(true_map.keys())
    pred_records = set(pred_map.keys())

    if true_records != pred_records:
        raise ClusteringError(
            "ML-CLUSTER-001",
            "True and predicted record universes must contain identical records.",
        )

    n = len(true_records)
    total_pairs = n * (n - 1) // 2

    if total_pairs == 0:
        return PairwiseClusterMetrics(
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            tp=0,
            fp=0,
            fn=0,
            tn=0,
            total_pairs=0,
        )

    true_counts: Counter[str] = Counter(true_map.values())
    pred_counts: Counter[str] = Counter(pred_map.values())
    contingency: Counter[tuple[str, str]] = Counter(
        (pred_map[r], true_map[r]) for r in true_records
    )

    pairs_true = sum(c * (c - 1) // 2 for c in true_counts.values())
    pairs_pred = sum(c * (c - 1) // 2 for c in pred_counts.values())
    tp = sum(c * (c - 1) // 2 for c in contingency.values())

    fp = pairs_pred - tp
    fn = pairs_true - tp
    tn = total_pairs - tp - fp - fn

    precision = tp / pairs_pred if pairs_pred > 0 else (1.0 if pairs_true == 0 else 0.0)
    recall = tp / pairs_true if pairs_true > 0 else 1.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall > 0.0 else 0.0

    return PairwiseClusterMetrics(
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        total_pairs=total_pairs,
    )


def calculate_constraint_violations(
    predicted_clusters: Mapping[str, str]
    | Mapping[str, Iterable[str]]
    | Sequence[Iterable[str]]
    | Iterable[Iterable[str]],
    cannot_link_pairs: Iterable[tuple[str, str]] = (),
    must_link_pairs: Iterable[tuple[str, str]] = (),
    record_to_dataset: Mapping[str, str] | None = None,
) -> ConstraintViolationMetrics:
    """Evaluate cannot-link, must-link, and intra-dataset collisions in predicted clusters."""
    pred_map = _normalize_cluster_mapping(predicted_clusters, "predicted_clusters")

    cannot_link_list = list(cannot_link_pairs)
    must_link_list = list(must_link_pairs)

    cl_violations = 0
    cl_checked = 0
    for left, right in cannot_link_list:
        if left in pred_map and right in pred_map:
            cl_checked += 1
            if pred_map[left] == pred_map[right]:
                cl_violations += 1

    ml_violations = 0
    ml_checked = 0
    for left, right in must_link_list:
        if left in pred_map and right in pred_map:
            ml_checked += 1
            if pred_map[left] != pred_map[right]:
                ml_violations += 1

    cl_rate = (cl_violations / cl_checked) if cl_checked > 0 else 0.0
    ml_rate = (ml_violations / ml_checked) if ml_checked > 0 else 0.0

    dataset_collisions = 0
    if record_to_dataset:
        cluster_dataset_counts: dict[str, set[str]] = defaultdict(set)
        for rec_k, c_id in pred_map.items():
            ds_id = record_to_dataset.get(rec_k)
            if ds_id is not None:
                if ds_id in cluster_dataset_counts[c_id]:
                    dataset_collisions += 1
                else:
                    cluster_dataset_counts[c_id].add(ds_id)

    total_clusters = len(set(pred_map.values()))

    return ConstraintViolationMetrics(
        cannot_link_violations=cl_violations,
        cannot_link_checked=cl_checked,
        cannot_link_violation_rate=float(cl_rate),
        must_link_violations=ml_violations,
        must_link_checked=ml_checked,
        must_link_violation_rate=float(ml_rate),
        dataset_collisions=dataset_collisions,
        total_clusters=total_clusters,
    )


def evaluate_multisource_clustering(
    true_clusters: Mapping[str, str]
    | Mapping[str, Iterable[str]]
    | Sequence[Iterable[str]]
    | Iterable[Iterable[str]],
    predicted_clusters: Mapping[str, str]
    | Mapping[str, Iterable[str]]
    | Sequence[Iterable[str]]
    | Iterable[Iterable[str]],
    cannot_link_pairs: Iterable[tuple[str, str]] = (),
    must_link_pairs: Iterable[tuple[str, str]] = (),
    record_to_dataset: Mapping[str, str] | None = None,
    evaluation_scope: str = "multisource_resolution_evaluation",
) -> MultiSourceEvaluationReport:
    """Generate a comprehensive multi-source evaluation report with canonical digest."""
    bcubed = calculate_bcubed_metrics(true_clusters, predicted_clusters)
    purity = calculate_cluster_purity(true_clusters, predicted_clusters)
    pairwise = calculate_pairwise_metrics(true_clusters, predicted_clusters)
    constraints = calculate_constraint_violations(
        predicted_clusters,
        cannot_link_pairs=cannot_link_pairs,
        must_link_pairs=must_link_pairs,
        record_to_dataset=record_to_dataset,
    )

    digest_payload = {
        "evaluation_scope": evaluation_scope,
        "total_records": bcubed.item_count,
        "total_true_clusters": purity.true_cluster_count,
        "total_predicted_clusters": purity.predicted_cluster_count,
        "bcubed_precision": round(bcubed.precision, 8),
        "bcubed_recall": round(bcubed.recall, 8),
        "bcubed_f1": round(bcubed.f1_score, 8),
        "cluster_purity": round(purity.purity, 8),
        "inverse_purity": round(purity.inverse_purity, 8),
        "pairwise_precision": round(pairwise.precision, 8),
        "pairwise_recall": round(pairwise.recall, 8),
        "pairwise_f1": round(pairwise.f1_score, 8),
        "cannot_link_violations": constraints.cannot_link_violations,
        "must_link_violations": constraints.must_link_violations,
        "dataset_collisions": constraints.dataset_collisions,
    }
    digest = _canonical_digest(digest_payload)

    return MultiSourceEvaluationReport(
        bcubed_precision=bcubed.precision,
        bcubed_recall=bcubed.recall,
        bcubed_f1=bcubed.f1_score,
        cluster_purity=purity.purity,
        inverse_purity=purity.inverse_purity,
        pairwise_precision=pairwise.precision,
        pairwise_recall=pairwise.recall,
        pairwise_f1=pairwise.f1_score,
        pairwise_tp=pairwise.tp,
        pairwise_fp=pairwise.fp,
        pairwise_fn=pairwise.fn,
        pairwise_tn=pairwise.tn,
        total_records=bcubed.item_count,
        total_true_clusters=purity.true_cluster_count,
        total_predicted_clusters=purity.predicted_cluster_count,
        cannot_link_violations=constraints.cannot_link_violations,
        cannot_link_constraints_count=constraints.cannot_link_checked,
        cannot_link_violation_rate=constraints.cannot_link_violation_rate,
        must_link_violations=constraints.must_link_violations,
        must_link_constraints_count=constraints.must_link_checked,
        must_link_violation_rate=constraints.must_link_violation_rate,
        dataset_collisions=constraints.dataset_collisions,
        evaluation_digest=digest,
        evaluation_scope=evaluation_scope,
    )
