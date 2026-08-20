"""Integration tests for assignment constraint solvers within workflow runners."""

from __future__ import annotations

import numpy as np

from mapel_linkage.assignment.contracts import (
    AssignmentEdgeBatch,
    AssignmentPlan,
    pair_digest,
)
from mapel_linkage.assignment.solvers import (
    ManyToOneAssignmentSolver,
    OneToManyAssignmentSolver,
    OrToolsOneToOneAssignmentSolver,
    ScipyOneToOneAssignmentSolver,
    UnconstrainedAssignmentSolver,
)


def _build_test_edge_batch() -> AssignmentEdgeBatch:
    source_record_keys = ("s1", "s2", "s3")
    pair_references = (
        ("s1", "t1"),
        ("s1", "t2"),
        ("s2", "t1"),
        ("s2", "t2"),
        ("s3", "t2"),
    )
    pair_digests = tuple(pair_digest(left, right) for left, right in pair_references)
    probabilities = np.array([0.90, 0.60, 0.80, 0.40, 0.70], dtype=np.float64)
    # Ranks per source: s1 has ranks [1, 2], s2 has ranks [1, 2], s3 has rank [1]
    candidate_ranks = np.array([1, 2, 1, 2, 1], dtype=np.int64)

    return AssignmentEdgeBatch(
        source_record_keys=source_record_keys,
        pair_references=pair_references,
        pair_digests=pair_digests,
        probabilities=probabilities,
        candidate_ranks=candidate_ranks,
        source_model_id="test_model",
        source_model_version="v1.0",
        calibrator_digest="a" * 64,
        ranking_model_digest=None,
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )


def test_ortools_one_to_one_assignment() -> None:
    batch = _build_test_edge_batch()
    plan = AssignmentPlan(constraint="one_to_one", solver="ortools_linear_sum_assignment")
    result = OrToolsOneToOneAssignmentSolver.solve(batch, plan)

    assert result.constraint == "one_to_one"
    assert result.source_record_count == 3
    assert result.real_assignment_count <= 2
    assigned_targets = [
        item.target_record_key for item in result.assignments if not item.selected_no_match
    ]
    assert len(assigned_targets) == len(set(assigned_targets))


def test_scipy_one_to_one_assignment() -> None:
    batch = _build_test_edge_batch()
    plan = AssignmentPlan(constraint="one_to_one", solver="scipy_linear_sum_assignment")
    result = ScipyOneToOneAssignmentSolver.solve(batch, plan)

    assert result.constraint == "one_to_one"
    assert result.source_record_count == 3
    assert result.real_assignment_count <= 2
    assigned_targets = [
        item.target_record_key for item in result.assignments if not item.selected_no_match
    ]
    assert len(assigned_targets) == len(set(assigned_targets))


def test_many_to_one_assignment() -> None:
    batch = _build_test_edge_batch()
    plan = AssignmentPlan(constraint="many_to_one", solver="greedy_many_to_one")
    result = ManyToOneAssignmentSolver.solve(batch, plan)

    assert result.constraint == "many_to_one"
    assert result.source_record_count == 3
    # Each source gets at most one match, multiple sources may match the same target
    real_assignments = [item for item in result.assignments if not item.selected_no_match]
    assert len(real_assignments) == 3


def test_one_to_many_assignment() -> None:
    batch = _build_test_edge_batch()
    plan = AssignmentPlan(constraint="one_to_many", solver="greedy_one_to_many")
    result = OneToManyAssignmentSolver.solve(batch, plan)

    assert result.constraint == "one_to_many"
    assert result.source_record_count == 3
    assigned_targets = [
        item.target_record_key for item in result.assignments if not item.selected_no_match
    ]
    # Each target receives at most one source
    assert len(assigned_targets) == len(set(assigned_targets))


def test_unconstrained_assignment() -> None:
    batch = _build_test_edge_batch()
    plan = AssignmentPlan(constraint="unconstrained", solver="threshold_unconstrained")
    result = UnconstrainedAssignmentSolver.solve(batch, plan)

    assert result.constraint == "unconstrained"
    assert result.source_record_count == 3
    real_assignments = [item for item in result.assignments if not item.selected_no_match]
    # 4 candidate edges with calibrated probability > 0.50 (above no-match utility) are preserved
    assert len(real_assignments) == 4
