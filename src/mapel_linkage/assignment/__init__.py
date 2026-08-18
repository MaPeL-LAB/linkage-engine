"""Explicit no-match assignment solvers and deduplication."""

from mapel_linkage.assignment.contracts import (
    AssignedEdge,
    AssignmentEdgeBatch,
    AssignmentPlan,
    AssignmentResult,
    pair_digest,
)
from mapel_linkage.assignment.deduplication import (
    DeduplicationPlan,
    DeduplicationResult,
    DuplicateCluster,
    IntraSourceDeduplicator,
    LinkAndDedupeResolver,
    LinkAndDedupeResult,
)
from mapel_linkage.assignment.solvers import (
    ManyToOneAssignmentSolver,
    OneToManyAssignmentSolver,
    OrToolsOneToOneAssignmentSolver,
    ScipyOneToOneAssignmentSolver,
    UnconstrainedAssignmentSolver,
)

__all__ = [
    "AssignedEdge",
    "AssignmentEdgeBatch",
    "AssignmentPlan",
    "AssignmentResult",
    "DeduplicationPlan",
    "DeduplicationResult",
    "DuplicateCluster",
    "IntraSourceDeduplicator",
    "LinkAndDedupeResolver",
    "LinkAndDedupeResult",
    "ManyToOneAssignmentSolver",
    "OneToManyAssignmentSolver",
    "OrToolsOneToOneAssignmentSolver",
    "ScipyOneToOneAssignmentSolver",
    "UnconstrainedAssignmentSolver",
    "pair_digest",
]
