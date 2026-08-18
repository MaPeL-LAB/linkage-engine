"""Aggregate candidate-retrieval diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from mapel_linkage.domain.errors import ValidationReportError


@dataclass(frozen=True, slots=True)
class CandidateRetrievalReport:
    source_record_count: int
    target_record_count: int
    candidate_pair_count: int
    cartesian_pair_count: int
    cartesian_reduction_fraction: float
    true_relationship_count: int
    retrieved_true_relationship_count: int
    candidate_recall: float
    zero_candidate_source_count: int
    mean_candidate_set_size: float
    median_candidate_set_size: float
    percentile_95_candidate_set_size: float
    percentile_99_candidate_set_size: float
    maximum_candidate_set_size: int
    rule_pair_counts: tuple[tuple[str, int], ...] = field(repr=False)
    rule_true_relationship_counts: tuple[tuple[str, int], ...] = field(repr=False)
    evaluation_scope: str = "synthetic_mechanical_evaluation"
    real_data_validation_status: str = "not_established"

    def safe_summary(self) -> dict[str, object]:
        return {
            "source_record_count": self.source_record_count,
            "target_record_count": self.target_record_count,
            "candidate_pair_count": self.candidate_pair_count,
            "cartesian_pair_count": self.cartesian_pair_count,
            "cartesian_reduction_fraction": self.cartesian_reduction_fraction,
            "true_relationship_count": self.true_relationship_count,
            "retrieved_true_relationship_count": self.retrieved_true_relationship_count,
            "candidate_recall": self.candidate_recall,
            "zero_candidate_source_count": self.zero_candidate_source_count,
            "mean_candidate_set_size": self.mean_candidate_set_size,
            "median_candidate_set_size": self.median_candidate_set_size,
            "percentile_95_candidate_set_size": self.percentile_95_candidate_set_size,
            "percentile_99_candidate_set_size": self.percentile_99_candidate_set_size,
            "maximum_candidate_set_size": self.maximum_candidate_set_size,
            "rule_pair_counts": dict(self.rule_pair_counts),
            "rule_true_relationship_counts": dict(self.rule_true_relationship_counts),
            "evaluation_scope": self.evaluation_scope,
            "real_data_validation_status": self.real_data_validation_status,
        }


def evaluate_candidate_retrieval(
    *,
    source_record_keys: tuple[str, ...],
    target_record_keys: tuple[str, ...],
    candidate_pairs: tuple[tuple[str, str], ...],
    true_pairs: frozenset[tuple[str, str]],
    rule_ids_by_pair: dict[tuple[str, str], tuple[str, ...]] | None = None,
) -> CandidateRetrievalReport:
    if len(set(source_record_keys)) != len(source_record_keys) or len(
        set(target_record_keys)
    ) != len(target_record_keys):
        raise ValidationReportError(
            "ML-VALID-001", "Candidate evaluation record universes are invalid."
        )
    if len(set(candidate_pairs)) != len(candidate_pairs):
        raise ValidationReportError("ML-VALID-002", "Duplicate candidate pairs were rejected.")
    source_set = set(source_record_keys)
    target_set = set(target_record_keys)
    if any(left not in source_set or right not in target_set for left, right in candidate_pairs):
        raise ValidationReportError(
            "ML-VALID-003", "Candidate evaluation includes an unknown record."
        )
    if any(left not in source_set or right not in target_set for left, right in true_pairs):
        raise ValidationReportError("ML-VALID-004", "Truth evaluation includes an unknown record.")
    candidate_set = set(candidate_pairs)
    retrieved_truth = candidate_set & set(true_pairs)
    per_source = Counter(left for left, _ in candidate_pairs)
    sizes = np.asarray(
        [per_source.get(source, 0) for source in source_record_keys], dtype=np.float64
    )
    cartesian = len(source_record_keys) * len(target_record_keys)
    rule_pair_counts: Counter[str] = Counter()
    rule_true_counts: Counter[str] = Counter()
    if rule_ids_by_pair is not None:
        for pair in candidate_pairs:
            for rule in rule_ids_by_pair.get(pair, ()):
                rule_pair_counts[rule] += 1
                if pair in true_pairs:
                    rule_true_counts[rule] += 1
    return CandidateRetrievalReport(
        source_record_count=len(source_record_keys),
        target_record_count=len(target_record_keys),
        candidate_pair_count=len(candidate_pairs),
        cartesian_pair_count=cartesian,
        cartesian_reduction_fraction=(
            0.0 if cartesian == 0 else 1.0 - len(candidate_pairs) / cartesian
        ),
        true_relationship_count=len(true_pairs),
        retrieved_true_relationship_count=len(retrieved_truth),
        candidate_recall=(1.0 if not true_pairs else len(retrieved_truth) / len(true_pairs)),
        zero_candidate_source_count=int(np.sum(sizes == 0)),
        mean_candidate_set_size=float(np.mean(sizes)) if len(sizes) else 0.0,
        median_candidate_set_size=float(np.median(sizes)) if len(sizes) else 0.0,
        percentile_95_candidate_set_size=float(np.percentile(sizes, 95)) if len(sizes) else 0.0,
        percentile_99_candidate_set_size=float(np.percentile(sizes, 99)) if len(sizes) else 0.0,
        maximum_candidate_set_size=int(np.max(sizes)) if len(sizes) else 0,
        rule_pair_counts=tuple(sorted(rule_pair_counts.items())),
        rule_true_relationship_counts=tuple(sorted(rule_true_counts.items())),
    )
