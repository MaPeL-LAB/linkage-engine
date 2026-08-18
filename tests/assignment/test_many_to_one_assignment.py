from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mapel_linkage.assignment import (
    AssignmentEdgeBatch,
    AssignmentPlan,
    ManyToOneAssignmentSolver,
    OneToManyAssignmentSolver,
)
from mapel_linkage.domain.errors import AssignmentError


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def many_to_one_batch() -> AssignmentEdgeBatch:
    # Both s1 and s2 have high probability to match target t1
    pairs = (
        ("s1", "t1"),
        ("s1", "t2"),
        ("s2", "t1"),
        ("s2", "t2"),
        ("s4", "t3"),
    )
    return AssignmentEdgeBatch(
        source_record_keys=("s1", "s2", "s3", "s4"),
        pair_references=pairs,
        pair_digests=tuple(digest(f"{left}\x00{right}") for left, right in pairs),
        probabilities=np.asarray([0.95, 0.70, 0.92, 0.60, 0.20], dtype=np.float64),
        candidate_ranks=np.asarray([1, 2, 1, 2, 1], dtype=np.int64),
        source_model_id="xgb_pair_classifier",
        source_model_version="v1",
        calibrator_digest=digest("calibrator"),
        ranking_model_digest=digest("ranker"),
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )


def test_many_to_one_allows_multiple_sources_matching_single_target() -> None:
    plan = AssignmentPlan(constraint="many_to_one", no_match_utility=0.0)
    result = ManyToOneAssignmentSolver.solve(many_to_one_batch(), plan)

    assigned = {item.source_record_key: item.target_record_key for item in result.assignments}
    assert assigned["s1"] == "t1"
    assert assigned["s2"] == "t1"
    assert assigned["s3"] is None
    assert assigned["s4"] is None  # prob 0.20 has utility < 0

    assert result.constraint == "many_to_one"
    assert result.constraint_violation_count == 0
    assert result.real_assignment_count == 2
    assert result.no_match_count == 2
    assert result.target_record_count == 1  # only t1 is targeted
    assert result.source_record_count == 4
    assert result.decision_authority == "none"
    assert result.assignment_authority == "global_selection_only"
    assert "s1" not in repr(result)


def test_many_to_one_respects_target_capacity_limit() -> None:
    # Restrict target capacity to 1 match per target
    plan = AssignmentPlan(
        constraint="many_to_one",
        max_matches_per_target=1,
        no_match_utility=0.0,
    )
    result = ManyToOneAssignmentSolver.solve(many_to_one_batch(), plan)

    assigned = {item.source_record_key: item.target_record_key for item in result.assignments}
    # s1 has higher prob (0.95) so gets t1; s2 (0.92) is displaced to t2 (0.60 > 0.5)
    assert assigned["s1"] == "t1"
    assert assigned["s2"] == "t2"
    assert assigned["s3"] is None
    assert assigned["s4"] is None
    assert result.target_record_count == 2


def test_one_to_many_allows_single_source_matching_multiple_targets() -> None:
    pairs = (
        ("s1", "t1"),
        ("s1", "t2"),
        ("s2", "t3"),
        ("s3", "t1"),
    )
    b = AssignmentEdgeBatch(
        source_record_keys=("s1", "s2", "s3", "s4"),
        pair_references=pairs,
        pair_digests=tuple(digest(f"{left}\x00{right}") for left, right in pairs),
        probabilities=np.asarray([0.95, 0.90, 0.85, 0.60], dtype=np.float64),
        candidate_ranks=np.asarray([1, 2, 1, 1], dtype=np.int64),
        source_model_id="model",
        source_model_version="v1",
        calibrator_digest=digest("calibrator"),
        ranking_model_digest=None,
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )
    plan = AssignmentPlan(constraint="one_to_many", no_match_utility=0.0)
    result = OneToManyAssignmentSolver.solve(b, plan)

    # s1 is best for t1 (0.95 > 0.60 from s3) and best for t2 (0.90)
    # s2 is best for t3 (0.85)
    # s3 lost t1 to s1 so has no matches -> no-match
    # s4 had no candidates -> no-match
    s1_targets = {
        item.target_record_key
        for item in result.assignments
        if item.source_record_key == "s1" and not item.selected_no_match
    }
    assert s1_targets == {"t1", "t2"}

    s2_targets = {
        item.target_record_key
        for item in result.assignments
        if item.source_record_key == "s2" and not item.selected_no_match
    }
    assert s2_targets == {"t3"}

    s3_assignments = [item for item in result.assignments if item.source_record_key == "s3"]
    assert len(s3_assignments) == 1
    assert s3_assignments[0].selected_no_match is True

    s4_assignments = [item for item in result.assignments if item.source_record_key == "s4"]
    assert len(s4_assignments) == 1
    assert s4_assignments[0].selected_no_match is True

    assert result.constraint == "one_to_many"
    assert result.constraint_violation_count == 0
    assert result.real_assignment_count == 3
    assert result.no_match_count == 2
    assert result.target_record_count == 3


def test_many_to_one_is_deterministic_under_ties() -> None:
    pairs = (("s1", "t1"), ("s2", "t1"))
    tied = AssignmentEdgeBatch(
        source_record_keys=("s1", "s2"),
        pair_references=pairs,
        pair_digests=tuple(digest(f"{left}\x00{right}") for left, right in pairs),
        probabilities=np.asarray([0.85, 0.85], dtype=np.float64),
        candidate_ranks=np.asarray([1, 1], dtype=np.int64),
        source_model_id="model",
        source_model_version="v1",
        calibrator_digest=digest("calibrator"),
        ranking_model_digest=None,
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )
    plan = AssignmentPlan(constraint="many_to_one")
    r1 = ManyToOneAssignmentSolver.solve(tied, plan)
    r2 = ManyToOneAssignmentSolver.solve(tied, plan)
    assert r1.assignment_digest == r2.assignment_digest
    assert tuple(it.target_record_key for it in r1.assignments) == tuple(
        it.target_record_key for it in r2.assignments
    )


def test_many_to_one_rejects_budget_overflow() -> None:
    b = many_to_one_batch()
    with pytest.raises(AssignmentError, match="ML-ASSIGN-019"):
        ManyToOneAssignmentSolver.solve(
            b,
            AssignmentPlan(
                constraint="many_to_one",
                maximum_candidate_edges=2,
            ),
        )
