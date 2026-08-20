"""Unit tests for active learning queue sampling and prioritization."""

from __future__ import annotations

import pytest

from mapel_linkage.adjudication.active_learning import (
    sample_active_learning_queue,
)
from mapel_linkage.adjudication.review_queue import ReviewQueueEntry
from mapel_linkage.domain.errors import AdjudicationError


def _create_sample_entries() -> list[ReviewQueueEntry]:
    return [
        ReviewQueueEntry(
            relationship_id="rel_near_threshold",
            source_record_ref="src_1",
            target_record_ref="tgt_1",
            relationship_status="review_required",
            calibrated_probability=0.51,
            candidate_rank=1,
            probability_margin=0.02,
            review_reason_codes=("review_probability_region",),
            model_version="v1.0",
            decision_rule_id="rule_prob",
            assignment_method="ortools",
            run_id="run_test_01",
        ),
        ReviewQueueEntry(
            relationship_id="rel_high_confidence",
            source_record_ref="src_2",
            target_record_ref="tgt_2",
            relationship_status="review_required",
            calibrated_probability=0.92,
            candidate_rank=1,
            probability_margin=0.84,
            review_reason_codes=("review_assignment_contention",),
            model_version="v1.0",
            decision_rule_id="rule_prob",
            assignment_method="ortools",
            run_id="run_test_01",
        ),
        ReviewQueueEntry(
            relationship_id="rel_low_margin",
            source_record_ref="src_3",
            target_record_ref="tgt_3",
            relationship_status="unresolved",
            calibrated_probability=0.48,
            candidate_rank=2,
            probability_margin=0.01,
            review_reason_codes=("unresolved_insufficient_probability",),
            model_version="v1.0",
            decision_rule_id="rule_prob",
            assignment_method="ortools",
            run_id="run_test_01",
        ),
        ReviewQueueEntry(
            relationship_id="rel_anchor_conflict",
            source_record_ref="src_4",
            target_record_ref="tgt_4",
            relationship_status="review_required",
            calibrated_probability=0.70,
            candidate_rank=1,
            probability_margin=0.40,
            review_reason_codes=("review_anchor_conflict",),
            model_version="v1.0",
            decision_rule_id="rule_prob",
            assignment_method="ortools",
            run_id="run_test_01",
        ),
    ]


def test_sample_active_learning_uncertainty_strategy() -> None:
    entries = _create_sample_entries()
    result = sample_active_learning_queue(entries, budget=2, strategy="uncertainty")

    assert len(result.entries) == 2
    assert result.relationship_count == 2
    top_rel_ids = [e.relationship_id for e in result.entries]
    assert "rel_near_threshold" in top_rel_ids
    assert "rel_low_margin" in top_rel_ids
    assert result.entries[0].candidate_rank == 1
    assert result.entries[1].candidate_rank == 2
    assert "active_learning_uncertainty" in result.entries[0].review_reason_codes


def test_sample_active_learning_margin_strategy() -> None:
    entries = _create_sample_entries()
    result = sample_active_learning_queue(entries, budget=2, strategy="margin")

    assert len(result.entries) == 2
    assert result.entries[0].relationship_id == "rel_low_margin"
    assert result.entries[1].relationship_id == "rel_near_threshold"
    assert "active_learning_margin" in result.entries[0].review_reason_codes


def test_sample_active_learning_committee_strategy() -> None:
    entries = _create_sample_entries()
    committee = {
        "rel_near_threshold": [0.2, 0.8, 0.9],
        "rel_high_confidence": [0.91, 0.92, 0.93],
        "rel_low_margin": [0.45, 0.49, 0.50],
        "rel_anchor_conflict": [0.1, 0.9, 0.85],
    }
    result = sample_active_learning_queue(
        entries, budget=2, strategy="committee", committee_scores=committee
    )

    assert len(result.entries) == 2
    assert result.entries[0].relationship_id == "rel_anchor_conflict"
    assert result.entries[1].relationship_id == "rel_near_threshold"
    assert "active_learning_committee" in result.entries[0].review_reason_codes


def test_sample_active_learning_hybrid_strategy() -> None:
    entries = _create_sample_entries()
    result = sample_active_learning_queue(entries, budget=3, strategy="hybrid")

    assert len(result.entries) == 3
    assert all("active_learning_hybrid" in e.review_reason_codes for e in result.entries)


def test_sample_active_learning_budget_bounds_and_empty() -> None:
    entries = _create_sample_entries()

    # Zero budget
    zero_res = sample_active_learning_queue(entries, budget=0)
    assert len(zero_res.entries) == 0

    # Budget exceeding count
    over_res = sample_active_learning_queue(entries, budget=10)
    assert len(over_res.entries) == len(entries)

    # Empty queue
    empty_res = sample_active_learning_queue([], budget=5)
    assert len(empty_res.entries) == 0

    # Negative budget raises error
    with pytest.raises(AdjudicationError) as exc_info:
        sample_active_learning_queue(entries, budget=-1)
    assert exc_info.value.code == "ML-ADJ-008"


def test_sample_active_learning_determinism() -> None:
    entries = _create_sample_entries()
    res1 = sample_active_learning_queue(entries, budget=3, strategy="hybrid")
    res2 = sample_active_learning_queue(entries, budget=3, strategy="hybrid")

    assert res1.queue_digest == res2.queue_digest
    assert [e.relationship_id for e in res1.entries] == [e.relationship_id for e in res2.entries]
    assert [s.priority_score for s in res1.scores] == [s.priority_score for s in res2.scores]
