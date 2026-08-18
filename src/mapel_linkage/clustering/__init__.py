"""Multi-source entity resolution, graph clustering, and BCubed evaluation metrics."""

from __future__ import annotations

from mapel_linkage.clustering.contracts import (
    CandidateEdge,
    ClusterEntity,
    ClusteringPlan,
    ClusteringResult,
    GlobalCrosswalkEntry,
    MultiSourceGraph,
    MultiSourceResolutionResult,
    build_cluster_entity,
    pair_digest,
)
from mapel_linkage.clustering.metrics import (
    BCubedMetrics,
    ClusterPurityMetrics,
    ConstraintViolationMetrics,
    MultiSourceEvaluationReport,
    PairwiseClusterMetrics,
    calculate_bcubed_metrics,
    calculate_cluster_purity,
    calculate_constraint_violations,
    calculate_pairwise_metrics,
    evaluate_multisource_clustering,
)
from mapel_linkage.clustering.resolver import MultiSourceEntityResolver
from mapel_linkage.clustering.solvers import (
    ConnectedComponentsSolver,
    ConstrainedAgglomerativeSolver,
    CorrelationClusteringSolver,
    MultiSourceClusterer,
)

__all__ = [
    "BCubedMetrics",
    "CandidateEdge",
    "ClusterEntity",
    "ClusterPurityMetrics",
    "ClusteringPlan",
    "ClusteringResult",
    "ConnectedComponentsSolver",
    "ConstrainedAgglomerativeSolver",
    "ConstraintViolationMetrics",
    "CorrelationClusteringSolver",
    "GlobalCrosswalkEntry",
    "MultiSourceClusterer",
    "MultiSourceEntityResolver",
    "MultiSourceEvaluationReport",
    "MultiSourceGraph",
    "MultiSourceResolutionResult",
    "PairwiseClusterMetrics",
    "build_cluster_entity",
    "calculate_bcubed_metrics",
    "calculate_cluster_purity",
    "calculate_constraint_violations",
    "calculate_pairwise_metrics",
    "evaluate_multisource_clustering",
    "pair_digest",
]
