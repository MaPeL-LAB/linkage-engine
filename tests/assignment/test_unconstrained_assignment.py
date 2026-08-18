from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mapel_linkage.assignment import (
    AssignmentEdgeBatch,
    AssignmentPlan,
    UnconstrainedAssignmentSolver,
)
from mapel_linkage.domain.errors import AssignmentError


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def unconstrained_batch() -> AssignmentEdgeBatch:
    pairs = (
        ("s1", "t1"),
        ("s1", "t2"),
        ("s2", "t1"),
        ("s2", "t2"),
        ("s3", "t3"),
    )
    return AssignmentEdgeBatch(
        source_record_keys=("s1", "s2", "s3", "s4"),
        pair_references=pairs,
        pair_digests=tuple(digest(f"{left}\x00{right}") for left, right in pairs),
        probabilities=np.asarray([0.95, 0.85, 0.90, 0.80, 0.20], dtype=np.float64),
        candidate_ranks=np.asarray([1, 2, 1, 2, 1], dtype=np.int64),
        source_model_id="xgb_pair_classifier",
        source_model_version="v1",
        calibrator_digest=digest("calibrator"),
        ranking_model_digest=digest("ranker"),
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )


def test_unconstrained_selects_all_eligible_pairs() -> None:
    plan = AssignmentPlan(constraint="unconstrained", no_match_utility=0.0)
    result = UnconstrainedAssignmentSolver.solve(unconstrained_batch(), plan)

    # s1 matched to t1 (0.95) and t2 (0.85)
    # s2 matched to t1 (0.90) and t2 (0.80)
    # s3 (0.20 <= 0.5) has utility <= 0 -> no-match
    # s4 has no candidates -> no-match
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
    assert s2_targets == {"t1", "t2"}

    s3_assignments = [item for item in result.assignments if item.source_record_key == "s3"]
    assert len(s3_assignments) == 1
    assert s3_assignments[0].selected_no_match is True

    s4_assignments = [item for item in result.assignments if item.source_record_key == "s4"]
    assert len(s4_assignments) == 1
    assert s4_assignments[0].selected_no_match is True

    assert result.constraint == "unconstrained"
    assert result.constraint_violation_count == 0
    assert result.real_assignment_count == 4
    assert result.no_match_count == 2
    assert result.target_record_count == 2
    assert result.source_record_count == 4
    assert "s1" not in repr(result)


def test_unconstrained_respects_capacity_limits() -> None:
    # Cap source matches to 1 per source
    plan = AssignmentPlan(
        constraint="unconstrained",
        max_matches_per_source=1,
        max_matches_per_target=10_000_000,
        no_match_utility=0.0,
    )
    result = UnconstrainedAssignmentSolver.solve(unconstrained_batch(), plan)

    # s1 only matches t1 (best prob 0.95)
    # s2 only matches t1 (best prob 0.90)
    s1_targets = {
        item.target_record_key
        for item in result.assignments
        if item.source_record_key == "s1" and not item.selected_no_match
    }
    assert s1_targets == {"t1"}

    s2_targets = {
        item.target_record_key
        for item in result.assignments
        if item.source_record_key == "s2" and not item.selected_no_match
    }
    assert s2_targets == {"t1"}
    assert result.real_assignment_count == 2


def test_unconstrained_rejects_budget_overflow() -> None:
    b = unconstrained_batch()
    with pytest.raises(AssignmentError, match="ML-ASSIGN-019"):
        UnconstrainedAssignmentSolver.solve(
            b,
            AssignmentPlan(
                constraint="unconstrained",
                maximum_candidate_edges=2,
            ),
        )


def test_unconstrained_deterministic_under_ties() -> None:
    pairs = (("s1", "t1"), ("s1", "t2"))
    tied = AssignmentEdgeBatch(
        source_record_keys=("s1",),
        pair_references=pairs,
        pair_digests=tuple(digest(f"{left}\x00{right}") for left, right in pairs),
        probabilities=np.asarray([0.85, 0.85], dtype=np.float64),
        candidate_ranks=np.asarray([1, 2], dtype=np.int64),
        source_model_id="model",
        source_model_version="v1",
        calibrator_digest=digest("calibrator"),
        ranking_model_digest=None,
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )
    plan = AssignmentPlan(constraint="unconstrained")
    r1 = UnconstrainedAssignmentSolver.solve(tied, plan)
    r2 = UnconstrainedAssignmentSolver.solve(tied, plan)
    assert r1.assignment_digest == r2.assignment_digest
    assert len(r1.assignments) == len(r2.assignments)
