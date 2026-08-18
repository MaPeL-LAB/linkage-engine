from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from mapel_linkage.assignment import (
    AssignmentEdgeBatch,
    AssignmentPlan,
    ScipyOneToOneAssignmentSolver,
)
from mapel_linkage.domain.errors import AssignmentError


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def batch() -> AssignmentEdgeBatch:
    pairs = (("s1", "t1"), ("s1", "t2"), ("s2", "t1"), ("s2", "t2"), ("s4", "t3"))
    return AssignmentEdgeBatch(
        source_record_keys=("s1", "s2", "s3", "s4"),
        pair_references=pairs,
        pair_digests=tuple(digest(f"{left}\x00{right}") for left, right in pairs),
        probabilities=np.asarray([0.95, 0.80, 0.94, 0.10, 0.20], dtype=np.float64),
        candidate_ranks=np.asarray([1, 2, 1, 2, 1], dtype=np.int64),
        source_model_id="xgb_pair_classifier",
        source_model_version="v1",
        calibrator_digest=digest("calibrator"),
        ranking_model_digest=digest("ranker"),
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )


def test_global_assignment_enforces_one_to_one_and_explicit_no_match() -> None:
    result = ScipyOneToOneAssignmentSolver.solve(
        batch(), AssignmentPlan(solver="scipy_linear_sum_assignment")
    )
    selected = {item.source_record_key: item.target_record_key for item in result.assignments}
    assert selected == {"s1": "t2", "s2": "t1", "s3": None, "s4": None}
    assert result.constraint_violation_count == 0
    assert result.no_match_count == 2
    assert result.changed_from_independent_top1_count == 1
    assert result.decision_authority == "none"
    assert "s1" not in repr(result)


def test_assignment_is_deterministic_under_ties() -> None:
    pairs = (("s1", "t1"), ("s1", "t2"), ("s2", "t1"), ("s2", "t2"))
    tied = AssignmentEdgeBatch(
        source_record_keys=("s1", "s2"),
        pair_references=pairs,
        pair_digests=tuple(digest(f"{left}\x00{right}") for left, right in pairs),
        probabilities=np.asarray([0.8, 0.8, 0.8, 0.8], dtype=np.float64),
        candidate_ranks=np.asarray([1, 2, 1, 2], dtype=np.int64),
        source_model_id="model",
        source_model_version="v1",
        calibrator_digest=digest("calibrator"),
        ranking_model_digest=None,
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )
    plan = AssignmentPlan(solver="scipy_linear_sum_assignment")
    first = ScipyOneToOneAssignmentSolver.solve(tied, plan)
    second = ScipyOneToOneAssignmentSolver.solve(tied, plan)
    assert first.assignment_digest == second.assignment_digest
    assert tuple(item.target_record_key for item in first.assignments) == tuple(
        item.target_record_key for item in second.assignments
    )


def test_assignment_rejects_budget_overflow_without_pair_values() -> None:
    sentinel = "PRIVATE-SOURCE-IDENTIFIER"
    protected = batch()
    protected = AssignmentEdgeBatch(
        source_record_keys=(*protected.source_record_keys, sentinel),
        pair_references=protected.pair_references,
        pair_digests=protected.pair_digests,
        probabilities=protected.probabilities,
        candidate_ranks=protected.candidate_ranks,
        source_model_id=protected.source_model_id,
        source_model_version=protected.source_model_version,
        calibrator_digest=protected.calibrator_digest,
        ranking_model_digest=protected.ranking_model_digest,
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )
    with pytest.raises(AssignmentError) as captured:
        ScipyOneToOneAssignmentSolver.solve(
            protected,
            AssignmentPlan(solver="scipy_linear_sum_assignment", maximum_candidate_edges=2),
        )
    assert sentinel not in str(captured.value)


def test_assignment_rejects_pair_digest_mismatch() -> None:
    protected = batch()
    with pytest.raises(AssignmentError, match="ML-ASSIGN-023"):
        replace(protected, pair_digests=(digest("wrong"), *protected.pair_digests[1:]))


def test_assignment_rejects_complete_and_truncated_search() -> None:
    with pytest.raises(AssignmentError, match="ML-ASSIGN-026"):
        replace(batch(), candidate_search_truncated=True)
