"""Explicit no-match assignment solvers."""

from mapel_linkage.assignment.contracts import (
    AssignedEdge,
    AssignmentEdgeBatch,
    AssignmentPlan,
    AssignmentResult,
    pair_digest,
)
from mapel_linkage.assignment.solvers import (
    OrToolsOneToOneAssignmentSolver,
    ScipyOneToOneAssignmentSolver,
    UnconstrainedAssignmentSolver,
)

__all__ = [
    "AssignedEdge",
    "AssignmentEdgeBatch",
    "AssignmentPlan",
    "AssignmentResult",
    "OrToolsOneToOneAssignmentSolver",
    "ScipyOneToOneAssignmentSolver",
    "UnconstrainedAssignmentSolver",
    "pair_digest",
]
