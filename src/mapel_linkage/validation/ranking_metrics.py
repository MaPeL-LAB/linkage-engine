"""Aggregate candidate-ranking diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mapel_linkage.domain.errors import ValidationReportError
from mapel_linkage.models.ranking import RankingScoreBatch


@dataclass(frozen=True, slots=True)
class RankingValidationReport:
    eligible_query_count: int
    retrieved_true_query_count: int
    top1_true_query_count: int
    top1_fraction: float
    mean_reciprocal_rank: float
    mean_true_match_rank: float
    median_true_match_rank: float
    recall_at_k: tuple[tuple[int, float], ...] = field(repr=False)
    evaluation_scope: str = "synthetic_mechanical_evaluation"
    decision_authority: str = "none"
    real_data_validation_status: str = "not_established"

    def safe_summary(self) -> dict[str, object]:
        return {
            "eligible_query_count": self.eligible_query_count,
            "retrieved_true_query_count": self.retrieved_true_query_count,
            "top1_true_query_count": self.top1_true_query_count,
            "top1_fraction": self.top1_fraction,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "mean_true_match_rank": self.mean_true_match_rank,
            "median_true_match_rank": self.median_true_match_rank,
            "recall_at_k": {str(k): value for k, value in self.recall_at_k},
            "evaluation_scope": self.evaluation_scope,
            "decision_authority": self.decision_authority,
            "real_data_validation_status": self.real_data_validation_status,
        }


def evaluate_ranking(
    *,
    scores: RankingScoreBatch,
    true_pair_digests: frozenset[str],
    eligible_query_keys: tuple[str, ...],
    k_values: tuple[int, ...],
) -> RankingValidationReport:
    if (
        not k_values
        or tuple(sorted(set(k_values))) != k_values
        or any(value <= 0 for value in k_values)
    ):
        raise ValidationReportError("ML-VALID-005", "Ranking recall thresholds are invalid.")
    eligible = set(eligible_query_keys)
    query_true_ranks: dict[str, list[int]] = {query: [] for query in eligible}
    for index, query in enumerate(scores.query_keys):
        if query in eligible and scores.pair_digests[index] in true_pair_digests:
            query_true_ranks[query].append(int(scores.ranks[index]))
    best_ranks = [min(ranks) for ranks in query_true_ranks.values() if ranks]
    retrieved = len(best_ranks)
    eligible_count = len(eligible)
    recall = tuple(
        (k, 1.0 if eligible_count == 0 else sum(rank <= k for rank in best_ranks) / eligible_count)
        for k in k_values
    )
    return RankingValidationReport(
        eligible_query_count=eligible_count,
        retrieved_true_query_count=retrieved,
        top1_true_query_count=sum(rank == 1 for rank in best_ranks),
        top1_fraction=(
            1.0 if eligible_count == 0 else sum(rank == 1 for rank in best_ranks) / eligible_count
        ),
        mean_reciprocal_rank=(
            0.0 if eligible_count == 0 else sum(1.0 / rank for rank in best_ranks) / eligible_count
        ),
        mean_true_match_rank=(float(np.mean(best_ranks)) if best_ranks else 0.0),
        median_true_match_rank=(float(np.median(best_ranks)) if best_ranks else 0.0),
        recall_at_k=recall,
    )
