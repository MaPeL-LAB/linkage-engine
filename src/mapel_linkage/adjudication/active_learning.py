"""Uncertainty, margin, and committee-based active learning review prioritization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from mapel_linkage.adjudication.review_queue import ReviewQueue, ReviewQueueEntry
from mapel_linkage.domain.errors import AdjudicationError

ActiveLearningStrategy = Literal["uncertainty", "margin", "committee", "hybrid"]

_ALLOWED_STRATEGIES: frozenset[str] = frozenset({"uncertainty", "margin", "committee", "hybrid"})

_REASON_CODE_WEIGHTS: dict[str, float] = {
    "review_anchor_conflict": 1.0,
    "review_model_disagreement": 0.95,
    "review_assignment_contention": 0.90,
    "review_probability_region": 0.80,
    "review_candidate_budget_truncated": 0.70,
    "unresolved_insufficient_probability": 0.60,
}


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def calculate_uncertainty_score(probability: float | None, threshold: float = 0.5) -> float:
    """Calculate uncertainty score in [0.0, 1.0] where threshold proximity maximizes uncertainty."""
    if probability is None or math.isnan(probability):
        return 1.0
    p = max(0.0, min(1.0, float(probability)))
    t = max(0.01, min(0.99, float(threshold)))
    max_dist = max(t, 1.0 - t)
    distance = abs(p - t)
    return max(0.0, min(1.0, 1.0 - (distance / max_dist)))


def calculate_margin_score(margin: float | None) -> float:
    """Calculate margin score in [0.0, 1.0] where smaller probability margin gives higher score."""
    if margin is None or math.isnan(margin) or margin < 0.0:
        return 1.0
    m = max(0.0, min(1.0, float(margin)))
    return max(0.0, min(1.0, 1.0 - m))


def calculate_committee_disagreement(probabilities: Sequence[float]) -> float:
    """Calculate committee disagreement score in [0.0, 1.0] across multiple model predictions."""
    valid_probs = [float(p) for p in probabilities if not math.isnan(p)]
    if len(valid_probs) < 2:
        return 0.0
    spread = max(valid_probs) - min(valid_probs)
    return max(0.0, min(1.0, spread))


@dataclass(frozen=True, slots=True)
class ActiveLearningConfig:
    """Configuration for active learning prioritization strategies and weights."""

    strategy: ActiveLearningStrategy = "hybrid"
    decision_threshold: float = 0.5
    uncertainty_weight: float = 0.40
    margin_weight: float = 0.30
    committee_weight: float = 0.20
    reason_code_weight: float = 0.10
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.strategy not in _ALLOWED_STRATEGIES:
            raise AdjudicationError(
                "ML-ADJ-020", f"Unsupported active learning strategy: {self.strategy}"
            )
        if self.decision_threshold <= 0.0 or self.decision_threshold >= 1.0:
            raise AdjudicationError(
                "ML-ADJ-020", "Decision threshold must be strictly between 0.0 and 1.0."
            )
        if self.temperature <= 0.0:
            raise AdjudicationError("ML-ADJ-020", "Sampling temperature must be strictly positive.")


@dataclass(frozen=True, slots=True, repr=False)
class ActiveLearningScore:
    """Priority scoring and auditability weights for a single review queue candidate."""

    relationship_id: str
    priority_score: float
    uncertainty_score: float
    margin_score: float
    committee_score: float
    reason_score: float
    sampling_probability: float
    strategy_used: str

    def safe_summary(self) -> dict[str, object]:
        return {
            "relationship_id": self.relationship_id,
            "priority_score": round(self.priority_score, 6),
            "uncertainty_score": round(self.uncertainty_score, 6),
            "margin_score": round(self.margin_score, 6),
            "committee_score": round(self.committee_score, 6),
            "reason_score": round(self.reason_score, 6),
            "sampling_probability": round(self.sampling_probability, 6),
            "strategy_used": self.strategy_used,
        }


@dataclass(frozen=True, slots=True, repr=False)
class PrioritizedReviewQueue:
    """An ordered review queue sorted by active learning priority with audit provenance."""

    entries: tuple[ReviewQueueEntry, ...] = field(repr=False)
    scores: tuple[ActiveLearningScore, ...] = field(repr=False)
    run_id: str
    queue_digest: str
    strategy: str
    relationship_count: int

    def safe_summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "queue_digest": self.queue_digest,
            "strategy": self.strategy,
            "relationship_count": self.relationship_count,
            "top_priority_score": (round(self.scores[0].priority_score, 6) if self.scores else 0.0),
        }


def score_review_entry(
    entry: ReviewQueueEntry,
    config: ActiveLearningConfig | None = None,
    *,
    committee_probabilities: Sequence[float] = (),
) -> tuple[float, float, float, float, float]:
    """Compute component scores (priority, uncertainty, margin, committee, reason) for an entry."""
    active_config = config or ActiveLearningConfig()
    uncertainty = calculate_uncertainty_score(
        entry.calibrated_probability, threshold=active_config.decision_threshold
    )
    margin = calculate_margin_score(entry.probability_margin)
    committee = calculate_committee_disagreement(committee_probabilities)

    reason_weights = [_REASON_CODE_WEIGHTS.get(code, 0.5) for code in entry.review_reason_codes]
    reason_score = max(reason_weights) if reason_weights else 0.5

    if active_config.strategy == "uncertainty":
        priority = uncertainty
    elif active_config.strategy == "margin":
        priority = margin
    elif active_config.strategy == "committee":
        priority = committee
    elif active_config.strategy == "hybrid":
        total_w = (
            active_config.uncertainty_weight
            + active_config.margin_weight
            + active_config.committee_weight
            + active_config.reason_code_weight
        )
        if total_w <= 0.0:
            total_w = 1.0
        priority = (
            (active_config.uncertainty_weight * uncertainty)
            + (active_config.margin_weight * margin)
            + (active_config.committee_weight * committee)
            + (active_config.reason_code_weight * reason_score)
        ) / total_w
    else:
        priority = uncertainty

    return priority, uncertainty, margin, committee, reason_score


def prioritize_review_queue(
    queue: ReviewQueue,
    config: ActiveLearningConfig | None = None,
    *,
    committee_scores: Mapping[str, Sequence[float]] | None = None,
) -> PrioritizedReviewQueue:
    """Order review queue entries by active learning priority score descending."""
    active_config = config or ActiveLearningConfig()
    if not queue.entries:
        return PrioritizedReviewQueue(
            entries=(),
            scores=(),
            run_id=queue.run_id,
            queue_digest=queue.queue_digest,
            strategy=active_config.strategy,
            relationship_count=0,
        )

    committee_map = committee_scores or {}
    raw_scores: list[tuple[ReviewQueueEntry, float, float, float, float, float]] = []

    for entry in queue.entries:
        comm_probs = committee_map.get(entry.relationship_id, ())
        prio, unc, marg, comm, r_sc = score_review_entry(
            entry, active_config, committee_probabilities=comm_probs
        )
        raw_scores.append((entry, prio, unc, marg, comm, r_sc))

    # Calculate softmax / normalized sampling probabilities for auditability and inverse weighting
    # Use numerically stable softmax with temperature
    priorities = [item[1] for item in raw_scores]
    scaled = [p / active_config.temperature for p in priorities]
    max_scaled = max(scaled)
    exp_scores = [math.exp(s - max_scaled) for s in scaled]
    sum_exp = sum(exp_scores)
    sampling_probs = [val / sum_exp for val in exp_scores]

    scored_items: list[tuple[ReviewQueueEntry, ActiveLearningScore]] = []
    for idx, (entry, prio, unc, marg, comm, r_sc) in enumerate(raw_scores):
        score_obj = ActiveLearningScore(
            relationship_id=entry.relationship_id,
            priority_score=prio,
            uncertainty_score=unc,
            margin_score=marg,
            committee_score=comm,
            reason_score=r_sc,
            sampling_probability=sampling_probs[idx],
            strategy_used=active_config.strategy,
        )
        scored_items.append((entry, score_obj))

    # Deterministic sort: priority descending, relationship_id ascending as stable tie-breaker
    sorted_items = sorted(
        scored_items,
        key=lambda item: (-item[1].priority_score, item[0].relationship_id),
    )

    ordered_entries = tuple(item[0] for item in sorted_items)
    ordered_scores = tuple(item[1] for item in sorted_items)

    queue_digest = _digest(
        {
            "run_id": queue.run_id,
            "strategy": active_config.strategy,
            "entries": [e.restricted_digest_payload() for e in ordered_entries],
            "scores": [
                {
                    "relationship_id": s.relationship_id,
                    "priority_score": round(s.priority_score, 6),
                    "sampling_probability": round(s.sampling_probability, 6),
                }
                for s in ordered_scores
            ],
        }
    )

    return PrioritizedReviewQueue(
        entries=ordered_entries,
        scores=ordered_scores,
        run_id=queue.run_id,
        queue_digest=queue_digest,
        strategy=active_config.strategy,
        relationship_count=len(ordered_entries),
    )
