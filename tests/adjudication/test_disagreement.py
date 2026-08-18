from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from mapel_linkage.adjudication.disagreement import (
    evaluate_disagreements,
    resolve_pair_consensus,
)
from mapel_linkage.adjudication.review_import import AdjudicationOutcome, AdjudicationRecord
from mapel_linkage.domain.errors import AdjudicationError


def _digest(val: str) -> str:
    return hashlib.sha256(val.encode("utf-8")).hexdigest()


def _make_rec(
    event_id: str,
    left: str,
    right: str,
    decision: AdjudicationOutcome,
    confidence: float,
    reviewer_id: str,
    protocol: str = "v1",
) -> AdjudicationRecord:
    return AdjudicationRecord(
        event_id=event_id,
        left_record_key=left,
        right_record_key=right,
        decision=decision,
        confidence=confidence,
        reviewer_id=reviewer_id,
        timestamp=datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
        protocol_version=protocol,
        entity_component_digests=(_digest(f"entity_{left}"),),
    )


def test_single_reviewer_resolution() -> None:
    rec = _make_rec("e1", "l1", "r1", "match", 0.90, "rev_alice")
    result = resolve_pair_consensus((rec,))

    assert result.is_resolved is True
    assert result.consensus_outcome == "match"
    assert result.consensus_confidence == 0.90
    assert result.resolution_method == "single_reviewer"
    assert result.has_conflict is False
    assert result.reviewer_count == 1


def test_unanimous_double_review() -> None:
    rec1 = _make_rec("e1", "l1", "r1", "match", 0.90, "rev_alice")
    rec2 = _make_rec("e2", "l1", "r1", "match", 0.80, "rev_bob")
    result = resolve_pair_consensus((rec1, rec2))

    assert result.is_resolved is True
    assert result.consensus_outcome == "match"
    assert result.consensus_confidence == pytest.approx(0.85)
    assert result.resolution_method == "unanimous"
    assert result.has_conflict is False
    assert result.reviewer_count == 2


def test_majority_vote_resolution() -> None:
    rec1 = _make_rec("e1", "l1", "r1", "match", 0.90, "rev_alice")
    rec2 = _make_rec("e2", "l1", "r1", "match", 0.80, "rev_bob")
    rec3 = _make_rec("e3", "l1", "r1", "nonmatch", 0.70, "rev_carol")
    result = resolve_pair_consensus((rec1, rec2, rec3), policy="majority_vote")

    assert result.is_resolved is True
    assert result.consensus_outcome == "match"
    assert result.consensus_confidence == pytest.approx(0.85)
    assert result.resolution_method == "majority"
    assert result.has_conflict is True
    assert result.reviewer_count == 3


def test_tied_vote_majority_policy() -> None:
    rec1 = _make_rec("e1", "l1", "r1", "match", 0.90, "rev_alice")
    rec2 = _make_rec("e2", "l1", "r1", "nonmatch", 0.80, "rev_bob")
    result = resolve_pair_consensus((rec1, rec2), policy="majority_vote")

    assert result.is_resolved is False
    assert result.consensus_outcome is None
    assert result.consensus_confidence == 0.0
    assert result.resolution_method == "unresolved"
    assert result.has_conflict is True
    assert result.dispute_reason == "tied_vote"


def test_senior_reviewer_override() -> None:
    rec_j1 = _make_rec("e1", "l1", "r1", "match", 0.60, "rev_junior1")
    rec_j2 = _make_rec("e2", "l1", "r1", "nonmatch", 0.60, "rev_junior2")
    rec_s = _make_rec("e3", "l1", "r1", "match", 0.95, "rev_senior")

    result = resolve_pair_consensus(
        (rec_j1, rec_j2, rec_s),
        policy="senior_reviewer_override",
        senior_reviewers={"rev_senior"},
    )

    assert result.is_resolved is True
    assert result.consensus_outcome == "match"
    assert result.consensus_confidence == 0.95
    assert result.resolution_method == "senior_override"
    assert result.senior_reviewer_id == "rev_senior"
    assert result.has_conflict is True


def test_conflicting_senior_reviewers() -> None:
    rec_s1 = _make_rec("e1", "l1", "r1", "match", 0.90, "rev_senior1")
    rec_s2 = _make_rec("e2", "l1", "r1", "nonmatch", 0.90, "rev_senior2")

    result = resolve_pair_consensus(
        (rec_s1, rec_s2),
        policy="senior_reviewer_override",
        senior_reviewers={"rev_senior1", "rev_senior2"},
    )

    assert result.is_resolved is False
    assert result.consensus_outcome is None
    assert result.resolution_method == "unresolved"
    assert result.dispute_reason == "conflicting_senior_reviewers"


def test_strict_double_review_policy() -> None:
    rec = _make_rec("e1", "l1", "r1", "match", 0.90, "rev_alice")
    result = resolve_pair_consensus((rec,), policy="strict_double_review")

    assert result.is_resolved is False
    assert result.consensus_outcome is None
    assert result.resolution_method == "unresolved"
    assert result.dispute_reason == "insufficient_reviewers"


def test_unanimous_only_policy_with_conflict() -> None:
    rec1 = _make_rec("e1", "l1", "r1", "match", 0.90, "rev_alice")
    rec2 = _make_rec("e2", "l1", "r1", "nonmatch", 0.80, "rev_bob")
    result = resolve_pair_consensus((rec1, rec2), policy="unanimous_only")

    assert result.is_resolved is False
    assert result.resolution_method == "unresolved"
    assert result.dispute_reason == "unanimous_policy_conflict"


def test_evaluate_disagreements_batch() -> None:
    # Pair 1: Unanimous match
    p1_r1 = _make_rec("p1_1", "l1", "r1", "match", 0.9, "rev1")
    p1_r2 = _make_rec("p1_2", "l1", "r1", "match", 0.9, "rev2")

    # Pair 2: Majority match (2 vs 1)
    p2_r1 = _make_rec("p2_1", "l2", "r2", "match", 0.9, "rev1")
    p2_r2 = _make_rec("p2_2", "l2", "r2", "match", 0.8, "rev2")
    p2_r3 = _make_rec("p2_3", "l2", "r2", "nonmatch", 0.7, "rev3")

    # Pair 3: Tied conflict (1 vs 1)
    p3_r1 = _make_rec("p3_1", "l3", "r3", "match", 0.8, "rev1")
    p3_r2 = _make_rec("p3_2", "l3", "r3", "nonmatch", 0.8, "rev2")

    # Pair 4: Single reviewer
    p4_r1 = _make_rec("p4_1", "l4", "r4", "nonmatch", 0.85, "rev1")

    all_records = [p1_r1, p1_r2, p2_r1, p2_r2, p2_r3, p3_r1, p3_r2, p4_r1]

    results, report = evaluate_disagreements(all_records, policy="majority_vote")

    assert len(results) == 4
    assert report.total_pairs == 4
    assert report.resolved_pairs == 3
    assert report.unresolved_pairs == 1
    assert report.conflict_count == 2  # Pair 2 and Pair 3
    assert report.unanimous_count == 1  # Pair 1
    assert report.majority_count == 1  # Pair 2
    assert report.single_reviewer_count == 1  # Pair 4


def test_invalid_policy_rejection() -> None:
    rec = _make_rec("e1", "l1", "r1", "match", 0.9, "rev1")
    with pytest.raises(AdjudicationError) as captured:
        resolve_pair_consensus((rec,), policy="unsupported_policy")  # type: ignore[arg-type]
    assert captured.value.code == "ML-ADJ-016"


def test_empty_records_rejection() -> None:
    with pytest.raises(AdjudicationError) as captured:
        resolve_pair_consensus(())
    assert captured.value.code == "ML-ADJ-015"


def test_mismatched_pair_digests_rejection() -> None:
    rec1 = _make_rec("e1", "l1", "r1", "match", 0.9, "rev1")
    rec2 = _make_rec("e2", "l2", "r2", "match", 0.9, "rev2")
    with pytest.raises(AdjudicationError) as captured:
        resolve_pair_consensus((rec1, rec2))
    assert captured.value.code == "ML-ADJ-015"


def test_privacy_repr_hides_private_record_keys() -> None:
    sentinel_left = "SYNTHETIC-PRIVATE-LEFT-DISAGREE"
    sentinel_right = "SYNTHETIC-PRIVATE-RIGHT-DISAGREE"
    rec = _make_rec("e1", sentinel_left, sentinel_right, "match", 0.9, "rev1")
    result = resolve_pair_consensus((rec,))

    rendered = repr(result)
    assert sentinel_left not in rendered
    assert sentinel_right not in rendered

    summary = result.safe_summary()
    assert summary["consensus_outcome"] == "match"
    assert "pair_digest" in summary
