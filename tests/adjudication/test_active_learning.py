from __future__ import annotations

import hashlib
import json

import pytest

from mapel_linkage.adjudication.active_learning import (
    ActiveLearningConfig,
    calculate_committee_disagreement,
    calculate_margin_score,
    calculate_uncertainty_score,
    prioritize_review_queue,
    score_review_entry,
)
from mapel_linkage.adjudication.review_queue import ReviewQueue, ReviewQueueEntry
from mapel_linkage.domain.errors import AdjudicationError


def _make_entry(
    rel_id: str,
    left: str,
    right: str,
    prob: float | None,
    margin: float,
    reasons: tuple[str, ...] = ("review_probability_region",),
    run_id: str = "run_001",
) -> ReviewQueueEntry:
    return ReviewQueueEntry(
        relationship_id=rel_id,
        source_record_ref=left,
        target_record_ref=right,
        relationship_status="review_required",
        calibrated_probability=prob,
        candidate_rank=1 if prob is not None else None,
        probability_margin=margin,
        review_reason_codes=reasons,
        model_version="v1",
        decision_rule_id="rule_rev",
        assignment_method="ortools",
        run_id=run_id,
    )


def _make_queue(entries: tuple[ReviewQueueEntry, ...], run_id: str = "run_001") -> ReviewQueue:
    payload = {
        "run_id": run_id,
        "entries": [entry.restricted_digest_payload() for entry in entries],
    }
    queue_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    review_req = sum(1 for e in entries if e.relationship_status == "review_required")
    unresolved = sum(1 for e in entries if e.relationship_status == "unresolved")
    return ReviewQueue(
        entries=entries,
        run_id=run_id,
        queue_digest=queue_digest,
        relationship_count=len(entries),
        review_required_count=review_req,
        unresolved_count=unresolved,
    )


def test_calculate_uncertainty_score() -> None:
    # Near default threshold 0.5
    assert calculate_uncertainty_score(0.50) == pytest.approx(1.0)
    assert calculate_uncertainty_score(0.51) == pytest.approx(0.98)
    assert calculate_uncertainty_score(0.90) == pytest.approx(0.20)
    assert calculate_uncertainty_score(0.10) == pytest.approx(0.20)
    assert calculate_uncertainty_score(1.00) == pytest.approx(0.0)
    assert calculate_uncertainty_score(0.00) == pytest.approx(0.0)
    assert calculate_uncertainty_score(None) == pytest.approx(1.0)

    # Custom threshold 0.7
    assert calculate_uncertainty_score(0.70, threshold=0.70) == pytest.approx(1.0)
    assert calculate_uncertainty_score(0.00, threshold=0.70) == pytest.approx(0.0)


def test_calculate_margin_score() -> None:
    assert calculate_margin_score(0.00) == pytest.approx(1.0)
    assert calculate_margin_score(0.10) == pytest.approx(0.90)
    assert calculate_margin_score(0.50) == pytest.approx(0.50)
    assert calculate_margin_score(1.00) == pytest.approx(0.0)
    assert calculate_margin_score(None) == pytest.approx(1.0)
    assert calculate_margin_score(-0.05) == pytest.approx(1.0)


def test_calculate_committee_disagreement() -> None:
    assert calculate_committee_disagreement([]) == pytest.approx(0.0)
    assert calculate_committee_disagreement([0.8]) == pytest.approx(0.0)
    assert calculate_committee_disagreement([0.8, 0.8, 0.8]) == pytest.approx(0.0)
    assert calculate_committee_disagreement([0.2, 0.5, 0.9]) == pytest.approx(0.7)


def test_prioritize_review_queue_uncertainty_ordering() -> None:
    e_clear = _make_entry("rel_clear", "l1", "r1", 0.95, 0.40)
    e_most_uncertain = _make_entry("rel_uncertain", "l2", "r2", 0.51, 0.05)
    e_mid = _make_entry("rel_mid", "l3", "r3", 0.70, 0.20)

    queue = _make_queue((e_clear, e_most_uncertain, e_mid))

    prioritized = prioritize_review_queue(
        queue, config=ActiveLearningConfig(strategy="uncertainty")
    )

    assert prioritized.relationship_count == 3
    assert prioritized.strategy == "uncertainty"
    assert [e.relationship_id for e in prioritized.entries] == [
        "rel_uncertain",
        "rel_mid",
        "rel_clear",
    ]
    assert prioritized.scores[0].uncertainty_score > prioritized.scores[1].uncertainty_score
    assert prioritized.scores[1].uncertainty_score > prioritized.scores[2].uncertainty_score


def test_prioritize_review_queue_margin_ordering() -> None:
    e_low_margin = _make_entry("rel_tight", "l1", "r1", 0.80, 0.01)
    e_high_margin = _make_entry("rel_wide", "l2", "r2", 0.80, 0.50)

    queue = _make_queue((e_high_margin, e_low_margin))

    prioritized = prioritize_review_queue(queue, config=ActiveLearningConfig(strategy="margin"))

    assert [e.relationship_id for e in prioritized.entries] == ["rel_tight", "rel_wide"]


def test_prioritize_review_queue_committee_ordering() -> None:
    e1 = _make_entry("rel_agree", "l1", "r1", 0.80, 0.10)
    e2 = _make_entry("rel_disagree", "l2", "r2", 0.80, 0.10)

    queue = _make_queue((e1, e2))

    committee_scores = {
        "rel_agree": [0.80, 0.81, 0.79],
        "rel_disagree": [0.10, 0.90, 0.45],
    }

    prioritized = prioritize_review_queue(
        queue,
        config=ActiveLearningConfig(strategy="committee"),
        committee_scores=committee_scores,
    )

    assert [e.relationship_id for e in prioritized.entries] == ["rel_disagree", "rel_agree"]


def test_prioritize_review_queue_deterministic_tie_breaking() -> None:
    e_b = _make_entry("rel_b", "l1", "r1", 0.50, 0.0)
    e_a = _make_entry("rel_a", "l2", "r2", 0.50, 0.0)

    queue = _make_queue((e_b, e_a))

    prioritized = prioritize_review_queue(queue, config=ActiveLearningConfig())

    assert [e.relationship_id for e in prioritized.entries] == ["rel_a", "rel_b"]


def test_prioritize_empty_queue() -> None:
    queue = _make_queue(())
    prioritized = prioritize_review_queue(queue)
    assert prioritized.relationship_count == 0
    assert len(prioritized.entries) == 0
    assert len(prioritized.scores) == 0


def test_invalid_config_rejection() -> None:
    with pytest.raises(AdjudicationError) as cap1:
        ActiveLearningConfig(strategy="unsupported")  # type: ignore[arg-type]
    assert cap1.value.code == "ML-ADJ-020"

    with pytest.raises(AdjudicationError) as cap2:
        ActiveLearningConfig(decision_threshold=1.5)
    assert cap2.value.code == "ML-ADJ-020"

    with pytest.raises(AdjudicationError) as cap3:
        ActiveLearningConfig(temperature=0.0)
    assert cap3.value.code == "ML-ADJ-020"


def test_privacy_repr_hides_private_record_keys() -> None:
    sentinel_left = "SYNTHETIC-PRIVATE-LEFT-ACTIVE"
    sentinel_right = "SYNTHETIC-PRIVATE-RIGHT-ACTIVE"
    e = _make_entry("rel_priv", sentinel_left, sentinel_right, 0.55, 0.1)

    _prio, unc, _marg, _comm, _r_sc = score_review_entry(e)
    assert unc > 0.0

    rendered = repr(e)
    assert sentinel_left not in rendered
    assert sentinel_right not in rendered
