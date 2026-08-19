"""Multi-source graph data structures and clustering contracts."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from mapel_linkage.domain.errors import ClusteringError

_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")


def _canonical_edge(u: str, v: str) -> tuple[str, str]:
    """Return canonical undirected edge pair (min, max)."""
    return (u, v) if u < v else (v, u)


def _compute_graph_digest(
    nodes: Mapping[str, str],
    edges: Mapping[tuple[str, str], float],
    cannot_link: frozenset[tuple[str, str]],
) -> str:
    """Compute deterministic SHA-256 digest of multi-source graph."""
    hasher = hashlib.sha256()
    hasher.update(b"NODES\n")
    for k, ds in sorted(nodes.items()):
        hasher.update(f"{k}\x00{ds}\n".encode())
    hasher.update(b"EDGES\n")
    for (u, v), w in sorted(edges.items()):
        hasher.update(f"{u}\x00{v}\x00{w:.8f}\n".encode())
    hasher.update(b"CANNOT_LINK\n")
    for u, v in sorted(cannot_link):
        hasher.update(f"{u}\x00{v}\n".encode())
    return hasher.hexdigest()


def _compute_entity_digest(
    member_record_keys: tuple[str, ...],
    nodes: Mapping[str, str],
) -> str:
    """Compute deterministic SHA-256 digest of a cluster entity."""
    hasher = hashlib.sha256()
    for k in sorted(member_record_keys):
        ds = nodes.get(k, "")
        hasher.update(f"{k}\x00{ds}\n".encode())
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class MultiSourceGraph:
    """Multi-source graph with dataset provenance, weighted edges, and cannot-links."""

    nodes: dict[str, str] = field(repr=False)
    edges: dict[tuple[str, str], float] = field(repr=False)
    cannot_link: frozenset[tuple[str, str]] = field(repr=False)
    graph_digest: str

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ClusteringError(
                "ML-CLUSTER-001", "The multi-source graph node set cannot be empty."
            )
        if _DIGEST_PATTERN.fullmatch(self.graph_digest) is None:
            raise ClusteringError("ML-CLUSTER-013", "The graph digest is invalid.")

        node_set = set(self.nodes)
        for u, v in self.edges:
            if u == v:
                raise ClusteringError(
                    "ML-CLUSTER-004", "Self-loop edge detected in candidate graph."
                )
            if u not in node_set or v not in node_set:
                raise ClusteringError(
                    "ML-CLUSTER-002", "Candidate edge references an unknown record key."
                )

        for w in self.edges.values():
            if not math.isfinite(w) or not (0.0 <= w <= 1.0):
                raise ClusteringError(
                    "ML-CLUSTER-005",
                    "Candidate edge weight must be a finite probability in [0, 1].",
                )

        for u, v in self.cannot_link:
            if u == v:
                raise ClusteringError(
                    "ML-CLUSTER-004", "Self-loop cannot-link constraint detected."
                )
            if u not in node_set or v not in node_set:
                raise ClusteringError(
                    "ML-CLUSTER-003", "Cannot-link constraint references an unknown record key."
                )

    @classmethod
    def build(
        cls,
        nodes: Mapping[str, str] | Iterable[tuple[str, str]],
        edges: Mapping[tuple[str, str], float] | Iterable[tuple[str, str, float]] | None = None,
        cannot_link: Iterable[tuple[str, str]] | None = None,
    ) -> MultiSourceGraph:
        """Construct and validate a MultiSourceGraph with canonicalised components."""
        node_map = dict(nodes)

        if not node_map:
            raise ClusteringError(
                "ML-CLUSTER-001", "The multi-source graph node set cannot be empty."
            )

        edge_map: dict[tuple[str, str], float] = {}
        if edges is not None:
            if isinstance(edges, Mapping):
                for (u, v), w in edges.items():
                    if u == v:
                        raise ClusteringError(
                            "ML-CLUSTER-004", "Self-loop edge detected in candidate graph."
                        )
                    edge_map[_canonical_edge(u, v)] = float(w)
            else:
                for u, v, w in edges:
                    if u == v:
                        raise ClusteringError(
                            "ML-CLUSTER-004", "Self-loop edge detected in candidate graph."
                        )
                    edge_map[_canonical_edge(u, v)] = float(w)

        cl_set: set[tuple[str, str]] = set()
        if cannot_link is not None:
            for u, v in cannot_link:
                if u == v:
                    raise ClusteringError(
                        "ML-CLUSTER-004", "Self-loop cannot-link constraint detected."
                    )
                cl_set.add(_canonical_edge(u, v))

        cl_frozen = frozenset(cl_set)
        digest = _compute_graph_digest(node_map, edge_map, cl_frozen)
        return cls(
            nodes=node_map,
            edges=edge_map,
            cannot_link=cl_frozen,
            graph_digest=digest,
        )

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def cannot_link_count(self) -> int:
        return len(self.cannot_link)

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.nodes.values())))

    def get_dataset(self, record_key: str) -> str:
        dataset = self.nodes.get(record_key)
        if dataset is None:
            raise ClusteringError("ML-CLUSTER-002", "Record key not found in graph nodes.")
        return dataset

    def get_edge_weight(self, u: str, v: str) -> float | None:
        if u == v:
            return 1.0
        return self.edges.get(_canonical_edge(u, v))

    def has_edge(self, u: str, v: str) -> bool:
        if u == v:
            return False
        return _canonical_edge(u, v) in self.edges

    def is_cannot_link(self, u: str, v: str) -> bool:
        if u == v:
            return False
        return _canonical_edge(u, v) in self.cannot_link

    def neighbors(self, node: str) -> dict[str, float]:
        """Return dict of neighbor node keys and edge weights for given node."""
        if node not in self.nodes:
            raise ClusteringError("ML-CLUSTER-002", "Record key not found in graph nodes.")
        result: dict[str, float] = {}
        for (u, v), w in self.edges.items():
            if u == node:
                result[v] = w
            elif v == node:
                result[u] = w
        return result

    def adjacency_list(self) -> dict[str, dict[str, float]]:
        """Return full adjacency mapping {node: {neighbor: weight}}."""
        adj: dict[str, dict[str, float]] = {k: {} for k in self.nodes}
        for (u, v), w in self.edges.items():
            adj[u][v] = w
            adj[v][u] = w
        return adj

    def cannot_link_adjacency(self) -> dict[str, set[str]]:
        """Return mapping {node: set(cannot_linked_nodes)}."""
        cl_adj: dict[str, set[str]] = {k: set() for k in self.nodes}
        for u, v in self.cannot_link:
            cl_adj[u].add(v)
            cl_adj[v].add(u)
        return cl_adj

    def __repr__(self) -> str:
        return (
            f"MultiSourceGraph(node_count={self.node_count}, "
            f"edge_count={self.edge_count}, "
            f"cannot_link_count={self.cannot_link_count}, "
            f"datasets={sorted(self.dataset_ids)!r}, "
            f"graph_digest={self.graph_digest!r})"
        )

    def safe_summary(self) -> dict[str, Any]:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "cannot_link_count": self.cannot_link_count,
            "dataset_count": len(self.dataset_ids),
            "datasets": list(self.dataset_ids),
            "graph_digest": self.graph_digest,
        }


@dataclass(frozen=True, slots=True)
class ClusteringPlan:
    """Configuration plan for multi-source graph clustering solvers."""

    algorithm: Literal[
        "correlation_clustering",
        "constrained_agglomerative",
        "connected_components",
    ] = "correlation_clustering"
    threshold: float = 0.50
    positive_weight_multiplier: float = 1.0
    negative_weight_multiplier: float = 1.0
    max_cluster_size: int = 10_000
    cannot_link_strictness: Literal["hard", "soft"] = "hard"
    cannot_link_same_dataset: bool = False
    deterministic_tie_breaking: Literal[True] = True
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not (0.0 <= self.threshold <= 1.0) or not math.isfinite(self.threshold):
            raise ClusteringError("ML-CLUSTER-006", "Clustering threshold is invalid.")
        if self.positive_weight_multiplier < 0.0 or not math.isfinite(
            self.positive_weight_multiplier
        ):
            raise ClusteringError("ML-CLUSTER-018", "Positive weight multiplier is invalid.")
        if self.negative_weight_multiplier < 0.0 or not math.isfinite(
            self.negative_weight_multiplier
        ):
            raise ClusteringError("ML-CLUSTER-018", "Negative weight multiplier is invalid.")
        if self.max_cluster_size < 1:
            raise ClusteringError("ML-CLUSTER-007", "Max cluster size must be at least 1.")
        if self.cannot_link_strictness not in ("hard", "soft"):
            raise ClusteringError("ML-CLUSTER-008", "Cannot-link strictness is invalid.")
        if self.algorithm not in (
            "correlation_clustering",
            "constrained_agglomerative",
            "connected_components",
        ):
            raise ClusteringError("ML-CLUSTER-008", "Unsupported clustering algorithm.")


@dataclass(frozen=True, slots=True, repr=False)
class ClusterEntity:
    """Immutable resolved multi-source cluster with member keys, distribution, and digest."""

    entity_digest: str
    canonical_record_key: str
    member_record_keys: tuple[str, ...]
    dataset_distribution: dict[str, int]
    mean_edge_weight: float
    edge_count: int
    is_singleton: bool
    cluster_id: str | None = None

    def __post_init__(self) -> None:
        if not self.member_record_keys:
            raise ClusteringError("ML-CLUSTER-009", "Cluster member keys cannot be empty.")
        if self.canonical_record_key not in self.member_record_keys:
            raise ClusteringError(
                "ML-CLUSTER-010", "Canonical record key must be a member of the cluster."
            )
        if self.is_singleton != (len(self.member_record_keys) == 1):
            raise ClusteringError("ML-CLUSTER-011", "Singleton status does not match member count.")
        if not (0.0 <= self.mean_edge_weight <= 1.0) or not math.isfinite(self.mean_edge_weight):
            raise ClusteringError("ML-CLUSTER-012", "Cluster mean edge weight is invalid.")
        if self.edge_count < 0:
            raise ClusteringError("ML-CLUSTER-012", "Cluster edge count cannot be negative.")
        if _DIGEST_PATTERN.fullmatch(self.entity_digest) is None:
            raise ClusteringError("ML-CLUSTER-013", "Cluster entity digest is invalid.")
        if sum(self.dataset_distribution.values()) != len(self.member_record_keys):
            raise ClusteringError(
                "ML-CLUSTER-014", "Dataset distribution count does not match member count."
            )
        if self.cluster_id is None:
            object.__setattr__(self, "cluster_id", self.entity_digest)
        elif _DIGEST_PATTERN.fullmatch(self.cluster_id) is None:
            raise ClusteringError("ML-CLUSTER-013", "Cluster ID digest is invalid.")

    def __repr__(self) -> str:
        return (
            f"ClusterEntity(entity_digest={self.entity_digest!r}, "
            f"size={len(self.member_record_keys)}, "
            f"is_singleton={self.is_singleton}, "
            f"datasets={self.dataset_distribution!r}, "
            f"edge_count={self.edge_count}, "
            f"mean_edge_weight={self.mean_edge_weight:.4f})"
        )

    def safe_summary(self) -> dict[str, Any]:
        return {
            "entity_digest": self.entity_digest,
            "cluster_id": self.cluster_id,
            "size": len(self.member_record_keys),
            "is_singleton": self.is_singleton,
            "dataset_distribution": dict(self.dataset_distribution),
            "edge_count": self.edge_count,
            "mean_edge_weight": self.mean_edge_weight,
        }


def _compute_clustering_digest(
    graph_digest: str,
    algorithm: str,
    threshold: float,
    clusters: tuple[ClusterEntity, ...],
) -> str:
    """Compute deterministic SHA-256 digest of clustering result."""
    hasher = hashlib.sha256()
    hasher.update(f"{graph_digest}\x00{algorithm}\x00{threshold:.8f}\n".encode())
    for c in sorted(clusters, key=lambda x: x.entity_digest):
        hasher.update(f"{c.entity_digest}\n".encode())
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class ClusteringResult:
    """Complete multi-source clustering outcome with aggregate distributions and digests."""

    clusters: tuple[ClusterEntity, ...] = field(repr=False)
    record_to_cluster: dict[str, str] = field(repr=False)
    algorithm: str
    threshold: float
    total_records: int
    total_clusters: int
    singleton_count: int
    cluster_size_distribution: dict[int, int]
    constraint_violation_count: int
    graph_digest: str
    clustering_digest: str

    def __post_init__(self) -> None:
        if sum(len(c.member_record_keys) for c in self.clusters) != self.total_records:
            raise ClusteringError(
                "ML-CLUSTER-015", "Cluster member counts do not match total records."
            )
        if len(self.clusters) != self.total_clusters:
            raise ClusteringError("ML-CLUSTER-015", "Total clusters count is inconsistent.")
        if sum(c.is_singleton for c in self.clusters) != self.singleton_count:
            raise ClusteringError("ML-CLUSTER-015", "Singleton count is inconsistent.")
        if len(self.record_to_cluster) != self.total_records:
            raise ClusteringError(
                "ML-CLUSTER-015", "Record-to-cluster mapping size is inconsistent."
            )
        expected_dist: dict[int, int] = {}
        for c in self.clusters:
            sz = len(c.member_record_keys)
            expected_dist[sz] = expected_dist.get(sz, 0) + 1
        if self.cluster_size_distribution != expected_dist:
            raise ClusteringError(
                "ML-CLUSTER-016", "Cluster size distribution is inconsistent with clusters."
            )
        if self.constraint_violation_count < 0:
            raise ClusteringError(
                "ML-CLUSTER-015", "Constraint violation count cannot be negative."
            )
        if _DIGEST_PATTERN.fullmatch(self.graph_digest) is None:
            raise ClusteringError("ML-CLUSTER-013", "Graph digest is invalid.")
        if _DIGEST_PATTERN.fullmatch(self.clustering_digest) is None:
            raise ClusteringError("ML-CLUSTER-013", "Clustering digest is invalid.")

    def __repr__(self) -> str:
        return (
            f"ClusteringResult(algorithm={self.algorithm!r}, "
            f"total_records={self.total_records}, "
            f"total_clusters={self.total_clusters}, "
            f"singleton_count={self.singleton_count}, "
            f"constraint_violation_count={self.constraint_violation_count}, "
            f"clustering_digest={self.clustering_digest!r})"
        )

    def safe_summary(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "threshold": self.threshold,
            "total_records": self.total_records,
            "total_clusters": self.total_clusters,
            "singleton_count": self.singleton_count,
            "cluster_size_distribution": dict(self.cluster_size_distribution),
            "constraint_violation_count": self.constraint_violation_count,
            "graph_digest": self.graph_digest,
            "clustering_digest": self.clustering_digest,
        }


def build_cluster_entity(
    member_keys: Iterable[str],
    graph: MultiSourceGraph,
) -> ClusterEntity:
    """Construct a ClusterEntity from a collection of member keys and graph context."""
    members = tuple(sorted(member_keys))
    if not members:
        raise ClusteringError("ML-CLUSTER-009", "Cluster member keys cannot be empty.")

    canonical_key = members[0]
    dist = dict(Counter(graph.get_dataset(k) for k in members))

    internal_weights: list[float] = []
    n = len(members)
    for i in range(n):
        for j in range(i + 1, n):
            w = graph.get_edge_weight(members[i], members[j])
            if w is not None:
                internal_weights.append(w)

    edge_cnt = len(internal_weights)
    if n == 1:
        mean_w = 1.0
    elif edge_cnt > 0:
        mean_w = sum(internal_weights) / edge_cnt
    else:
        mean_w = 0.0

    digest = _compute_entity_digest(members, graph.nodes)

    return ClusterEntity(
        entity_digest=digest,
        canonical_record_key=canonical_key,
        member_record_keys=members,
        dataset_distribution=dist,
        mean_edge_weight=mean_w,
        edge_count=edge_cnt,
        is_singleton=(n == 1),
        cluster_id=digest,
    )


def pair_digest(left: str, right: str) -> str:
    """Return deterministic SHA-256 pair digest."""
    return hashlib.sha256(f"{left}\x00{right}".encode()).hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class CandidateEdge:
    """A weighted candidate edge between two records."""

    source_record_key: str = field(repr=False)
    source_dataset_id: str
    target_record_key: str = field(repr=False)
    target_dataset_id: str
    probability: float
    pair_digest: str | None = None
    candidate_rank: int | None = None
    edge_source_model: str | None = None

    def __post_init__(self) -> None:
        if not self.source_record_key or not self.source_dataset_id:
            raise ClusteringError("ML-CLUSTER-001", "Source record reference is incomplete.")
        if not self.target_record_key or not self.target_dataset_id:
            raise ClusteringError("ML-CLUSTER-001", "Target record reference is incomplete.")
        if not math.isfinite(self.probability) or not (0.0 <= self.probability <= 1.0):
            raise ClusteringError("ML-CLUSTER-005", "Candidate edge probability is invalid.")
        if self.pair_digest is not None and _DIGEST_PATTERN.fullmatch(self.pair_digest) is None:
            raise ClusteringError("ML-CLUSTER-013", "Candidate edge pair digest is invalid.")
        if self.candidate_rank is not None and self.candidate_rank < 1:
            raise ClusteringError("ML-CLUSTER-005", "Candidate rank must be positive.")


@dataclass(frozen=True, slots=True, repr=False)
class GlobalCrosswalkEntry:
    """A single record entry in the resolved global entity crosswalk table."""

    record_key: str = field(repr=False)
    dataset_id: str
    global_entity_id: str
    is_canonical: bool
    cluster_size: int
    confidence: float

    def __post_init__(self) -> None:
        if not self.record_key or not self.dataset_id:
            raise ClusteringError("ML-CLUSTER-001", "Crosswalk record or dataset key is empty.")
        if _DIGEST_PATTERN.fullmatch(self.global_entity_id) is None:
            raise ClusteringError("ML-CLUSTER-013", "Global entity ID digest is invalid.")
        if self.cluster_size < 1:
            raise ClusteringError("ML-CLUSTER-007", "Cluster size must be at least 1.")
        if not math.isfinite(self.confidence) or not (0.0 <= self.confidence <= 1.0):
            raise ClusteringError("ML-CLUSTER-012", "Confidence must be a finite float in [0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_key": self.record_key,
            "dataset_id": self.dataset_id,
            "global_entity_id": self.global_entity_id,
            "is_canonical": self.is_canonical,
            "cluster_size": self.cluster_size,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True, repr=False)
class MultiSourceResolutionResult:
    """Resolved multi-source entity resolution outcome."""

    clustering_result: ClusteringResult = field(repr=False)
    crosswalk_entries: tuple[GlobalCrosswalkEntry, ...] = field(repr=False)
    graph: MultiSourceGraph = field(repr=False)
    total_records: int
    total_clusters: int
    singleton_count: int
    multi_record_cluster_count: int
    resolution_digest: str
    algorithm: str
    threshold: float

    @property
    def clusters(self) -> tuple[ClusterEntity, ...]:
        return self.clustering_result.clusters

    @property
    def record_to_cluster(self) -> dict[str, str]:
        return self.clustering_result.record_to_cluster

    @property
    def cannot_link_violations(self) -> int:
        return self.clustering_result.constraint_violation_count

    @property
    def max_cluster_size(self) -> int:
        return max((len(c.member_record_keys) for c in self.clusters), default=0)

    @property
    def source_collision_count(self) -> int:
        count = 0
        for c in self.clusters:
            if any(cnt > 1 for cnt in c.dataset_distribution.values()):
                count += 1
        return count

    def to_crosswalk_records(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.crosswalk_entries]

    def __repr__(self) -> str:
        return (
            f"MultiSourceResolutionResult(algorithm={self.algorithm!r}, "
            f"total_records={self.total_records}, "
            f"total_clusters={self.total_clusters}, "
            f"singleton_count={self.singleton_count}, "
            f"multi_record_cluster_count={self.multi_record_cluster_count}, "
            f"cannot_link_violations={self.cannot_link_violations}, "
            f"source_collision_count={self.source_collision_count}, "
            f"resolution_digest={self.resolution_digest!r})"
        )

    def safe_summary(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "threshold": self.threshold,
            "total_records": self.total_records,
            "total_clusters": self.total_clusters,
            "singleton_count": self.singleton_count,
            "multi_record_cluster_count": self.multi_record_cluster_count,
            "cannot_link_violations": self.cannot_link_violations,
            "source_collision_count": self.source_collision_count,
            "max_cluster_size": self.max_cluster_size,
            "resolution_digest": self.resolution_digest,
        }
