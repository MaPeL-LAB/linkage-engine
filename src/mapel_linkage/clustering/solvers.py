"""Multi-source graph clustering solvers with hard constraint enforcement."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Literal

from mapel_linkage.clustering.contracts import (
    ClusterEntity,
    ClusteringPlan,
    ClusteringResult,
    MultiSourceGraph,
    _compute_clustering_digest,
    build_cluster_entity,
)
from mapel_linkage.domain.errors import ClusteringError


class _DisjointSetUnion:
    """Disjoint Set Union (Union-Find) with path compression and rank heuristic."""

    __slots__ = ("parent", "rank")

    def __init__(self, elements: Iterable[str]) -> None:
        self.parent: dict[str, str] = {elem: elem for elem in elements}
        self.rank: dict[str, int] = {elem: 0 for elem in elements}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        curr = item
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, item_a: str, item_b: str) -> str:
        root_a = self.find(item_a)
        root_b = self.find(item_b)
        if root_a == root_b:
            return root_a
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
            return root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
            return root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1
            return root_a


def _assemble_clustering_result(
    graph: MultiSourceGraph,
    plan: ClusteringPlan,
    cluster_entities: list[ClusterEntity],
    constraint_violation_count: int = 0,
) -> ClusteringResult:
    """Assemble, validate, and compute digest for a ClusteringResult."""
    sorted_clusters = tuple(
        sorted(
            cluster_entities,
            key=lambda c: (-len(c.member_record_keys), c.canonical_record_key),
        )
    )

    record_to_cluster: dict[str, str] = {}
    for c in sorted_clusters:
        for k in c.member_record_keys:
            record_to_cluster[k] = c.entity_digest

    total_records = len(graph.nodes)
    total_clusters = len(sorted_clusters)
    singleton_count = sum(c.is_singleton for c in sorted_clusters)

    cluster_size_distribution: dict[int, int] = {}
    for c in sorted_clusters:
        sz = len(c.member_record_keys)
        cluster_size_distribution[sz] = cluster_size_distribution.get(sz, 0) + 1

    clustering_digest = _compute_clustering_digest(
        graph_digest=graph.graph_digest,
        algorithm=plan.algorithm,
        threshold=plan.threshold,
        clusters=sorted_clusters,
    )

    return ClusteringResult(
        clusters=sorted_clusters,
        record_to_cluster=record_to_cluster,
        algorithm=plan.algorithm,
        threshold=plan.threshold,
        total_records=total_records,
        total_clusters=total_clusters,
        singleton_count=singleton_count,
        cluster_size_distribution=cluster_size_distribution,
        constraint_violation_count=constraint_violation_count,
        graph_digest=graph.graph_digest,
        clustering_digest=clustering_digest,
    )


class CorrelationClusteringSolver:
    """Ailon-Charikar-Newman pivoting correlation clustering with hard constraints."""

    @classmethod
    def solve(
        cls,
        graph: MultiSourceGraph,
        plan: ClusteringPlan | None = None,
    ) -> ClusteringResult:
        plan = plan or ClusteringPlan(algorithm="correlation_clustering")
        threshold = plan.threshold
        pos_mult = plan.positive_weight_multiplier
        neg_mult = plan.negative_weight_multiplier
        max_size = plan.max_cluster_size
        same_ds_cannot_link = plan.cannot_link_same_dataset

        unassigned: set[str] = set(graph.nodes)
        clusters: list[ClusterEntity] = []

        while unassigned:
            # Deterministic pivot selection: node in unassigned with maximum positive score
            best_pivot = None
            best_score: tuple[int, float, str] = (-1, -1.0, "")

            for candidate in sorted(unassigned):
                pos_degree = 0
                pos_weight_sum = 0.0
                for neighbor, weight in graph.neighbors(candidate).items():
                    if neighbor in unassigned and weight >= threshold:
                        pos_degree += 1
                        pos_weight_sum += (weight - threshold) * pos_mult
                candidate_score = (pos_degree, pos_weight_sum, candidate)
                if best_pivot is None or candidate_score > best_score:
                    best_pivot = candidate
                    best_score = candidate_score

            pivot = best_pivot if best_pivot is not None else next(iter(sorted(unassigned)))

            # Form cluster starting with pivot
            cluster_members: list[str] = [pivot]
            cluster_datasets: set[str] = {graph.get_dataset(pivot)}

            # Candidate neighbors connected to pivot with weight >= threshold
            candidate_neighbors: list[tuple[float, str]] = []
            for neighbor, weight in graph.neighbors(pivot).items():
                if neighbor in unassigned and neighbor != pivot and weight >= threshold:
                    candidate_neighbors.append((weight, neighbor))

            # Sort candidate neighbors by weight descending, tie-broken by key
            candidate_neighbors.sort(key=lambda item: (-item[0], item[1]))

            for _edge_weight, neighbor in candidate_neighbors:
                if len(cluster_members) >= max_size:
                    break

                # Check same dataset constraint
                neighbor_ds = graph.get_dataset(neighbor)
                if same_ds_cannot_link and neighbor_ds in cluster_datasets:
                    continue

                # Check hard cannot-link constraints against all current cluster members
                has_conflict = False
                for member in cluster_members:
                    if graph.is_cannot_link(member, neighbor):
                        has_conflict = True
                        break
                if has_conflict:
                    continue

                # Check correlation objective net agreement with current cluster members
                net_agreement = 0.0
                for member in cluster_members:
                    w = graph.get_edge_weight(member, neighbor)
                    if w is not None and w >= threshold:
                        net_agreement += (w - threshold) * pos_mult
                    elif w is not None:
                        net_agreement -= (threshold - w) * neg_mult
                    else:
                        net_agreement -= threshold * neg_mult

                if net_agreement >= 0.0 or len(cluster_members) == 1:
                    cluster_members.append(neighbor)
                    cluster_datasets.add(neighbor_ds)

            # Build cluster entity
            entity = build_cluster_entity(cluster_members, graph)
            clusters.append(entity)

            # Remove clustered nodes from unassigned
            for member in cluster_members:
                unassigned.remove(member)

        return _assemble_clustering_result(
            graph=graph,
            plan=plan,
            cluster_entities=clusters,
            constraint_violation_count=0,
        )


class ConstrainedAgglomerativeSolver:
    """Greedy agglomerative clustering with strict cannot-link and capacity constraints."""

    @classmethod
    def solve(
        cls,
        graph: MultiSourceGraph,
        plan: ClusteringPlan | None = None,
    ) -> ClusteringResult:
        plan = plan or ClusteringPlan(algorithm="constrained_agglomerative")
        threshold = plan.threshold
        max_size = plan.max_cluster_size
        same_ds_cannot_link = plan.cannot_link_same_dataset

        dsu = _DisjointSetUnion(graph.nodes)
        cluster_members: dict[str, list[str]] = {k: [k] for k in graph.nodes}
        cluster_datasets: dict[str, Counter[str]] = {
            k: Counter([graph.get_dataset(k)]) for k in graph.nodes
        }

        # Filter candidate edges with weight >= threshold
        candidate_edges: list[tuple[float, str, str]] = []
        for (u, v), w in graph.edges.items():
            if w >= threshold:
                candidate_edges.append((w, u, v))

        # Sort candidate edges descending by weight, tie-broken canonically
        candidate_edges.sort(key=lambda e: (-e[0], e[1], e[2]))

        for _weight, u, v in candidate_edges:
            root_u = dsu.find(u)
            root_v = dsu.find(v)

            if root_u == root_v:
                continue

            members_u = cluster_members[root_u]
            members_v = cluster_members[root_v]

            # Check capacity constraint
            if len(members_u) + len(members_v) > max_size:
                continue

            # Check same-dataset cannot-link constraint
            if same_ds_cannot_link:
                ds_u = cluster_datasets[root_u]
                ds_v = cluster_datasets[root_v]
                if any(ds in ds_v for ds in ds_u):
                    continue

            # Check pairwise cannot-link constraints
            has_conflict = False
            for x in members_u:
                for y in members_v:
                    if graph.is_cannot_link(x, y):
                        has_conflict = True
                        break
                if has_conflict:
                    break
            if has_conflict:
                continue

            # Perform merge
            new_root = dsu.union(root_u, root_v)
            old_root = root_v if new_root == root_u else root_u

            cluster_members[new_root].extend(cluster_members[old_root])
            cluster_datasets[new_root].update(cluster_datasets[old_root])

            del cluster_members[old_root]
            del cluster_datasets[old_root]

        # Construct cluster entities from final clusters
        clusters: list[ClusterEntity] = [
            build_cluster_entity(members, graph) for members in cluster_members.values()
        ]

        return _assemble_clustering_result(
            graph=graph,
            plan=plan,
            cluster_entities=clusters,
            constraint_violation_count=0,
        )


class ConnectedComponentsSolver:
    """Baseline connected components solver with constraint violation tracking."""

    @classmethod
    def solve(
        cls,
        graph: MultiSourceGraph,
        plan: ClusteringPlan | None = None,
    ) -> ClusteringResult:
        plan = plan or ClusteringPlan(algorithm="connected_components")
        threshold = plan.threshold
        same_ds_cannot_link = plan.cannot_link_same_dataset

        dsu = _DisjointSetUnion(graph.nodes)

        # Merge all edges >= threshold
        for (u, v), w in graph.edges.items():
            if w >= threshold:
                dsu.union(u, v)

        # Group members by root
        grouped: dict[str, list[str]] = {}
        for node in graph.nodes:
            root = dsu.find(node)
            grouped.setdefault(root, []).append(node)

        clusters: list[ClusterEntity] = []
        total_violations = 0

        for members in grouped.values():
            entity = build_cluster_entity(members, graph)
            clusters.append(entity)

            # Count cannot-link violations in this component
            n = len(members)
            for i in range(n):
                for j in range(i + 1, n):
                    u = members[i]
                    v = members[j]
                    if graph.is_cannot_link(u, v) or (
                        same_ds_cannot_link and graph.get_dataset(u) == graph.get_dataset(v)
                    ):
                        total_violations += 1

        return _assemble_clustering_result(
            graph=graph,
            plan=plan,
            cluster_entities=clusters,
            constraint_violation_count=total_violations,
        )


class MultiSourceClusterer:
    """Unified dispatcher for multi-source graph clustering solvers."""

    @classmethod
    def solve(
        cls,
        graph: MultiSourceGraph,
        plan: ClusteringPlan | None = None,
    ) -> ClusteringResult:
        plan = plan or ClusteringPlan()
        algorithm: Literal[
            "correlation_clustering",
            "constrained_agglomerative",
            "connected_components",
        ] = plan.algorithm

        if algorithm == "correlation_clustering":
            return CorrelationClusteringSolver.solve(graph, plan)
        elif algorithm == "constrained_agglomerative":
            return ConstrainedAgglomerativeSolver.solve(graph, plan)
        elif algorithm == "connected_components":
            return ConnectedComponentsSolver.solve(graph, plan)
        else:
            raise ClusteringError("ML-CLUSTER-008", f"Unsupported algorithm: {algorithm}")
