"""Graph-based deduplication and intra-source duplicate clustering."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mapel_linkage.assignment.contracts import (
    AssignmentEdgeBatch,
    AssignmentPlan,
    AssignmentResult,
    pair_digest,
)
from mapel_linkage.assignment.solvers import (
    ManyToOneAssignmentSolver,
    OneToManyAssignmentSolver,
    OrToolsOneToOneAssignmentSolver,
    ScipyOneToOneAssignmentSolver,
    UnconstrainedAssignmentSolver,
    _canonical_digest,
    _utility,
)
from mapel_linkage.domain.errors import AssignmentError

_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True, slots=True)
class DeduplicationPlan:
    """Configuration plan for intra-source deduplication clustering."""

    algorithm: Literal["connected_components", "clique", "thresholded_clique"] = (
        "connected_components"
    )
    threshold: float = 0.5
    no_match_utility: float = 0.0
    min_cluster_size: int = 1
    max_cluster_size: int = 10_000
    maximum_candidate_edges: int = 10_000_000
    deterministic_tie_breaking: Literal[True] = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.threshold <= 1.0) or not math.isfinite(self.threshold):
            raise AssignmentError("ML-ASSIGN-005", "Deduplication threshold is invalid.")
        if not math.isfinite(self.no_match_utility):
            raise AssignmentError("ML-ASSIGN-009", "The no-match utility is invalid.")
        if self.min_cluster_size < 1:
            raise AssignmentError("ML-ASSIGN-030", "Minimum cluster size must be at least 1.")
        if self.max_cluster_size < self.min_cluster_size:
            raise AssignmentError(
                "ML-ASSIGN-031", "Maximum cluster size cannot be smaller than minimum."
            )
        if self.maximum_candidate_edges <= 0:
            raise AssignmentError("ML-ASSIGN-011", "The assignment edge budget is invalid.")


@dataclass(frozen=True, slots=True, repr=False)
class DuplicateCluster:
    """A cluster of mutually duplicate records within a single source."""

    cluster_id: str
    canonical_record_key: str
    member_record_keys: tuple[str, ...]
    edge_count: int
    mean_probability: float
    is_singleton: bool
    dataset_id: str | None = None

    def __post_init__(self) -> None:
        if not self.member_record_keys:
            raise AssignmentError("ML-ASSIGN-030", "A duplicate cluster cannot be empty.")
        if self.canonical_record_key not in self.member_record_keys:
            raise AssignmentError(
                "ML-ASSIGN-031", "Canonical record key must be a member of the cluster."
            )
        if self.is_singleton != (len(self.member_record_keys) == 1):
            raise AssignmentError("ML-ASSIGN-032", "Singleton status does not match member count.")
        if self.edge_count < 0:
            raise AssignmentError("ML-ASSIGN-033", "Cluster edge count cannot be negative.")
        if not (0.0 <= self.mean_probability <= 1.0) or not math.isfinite(self.mean_probability):
            raise AssignmentError("ML-ASSIGN-034", "Cluster mean probability is invalid.")
        if _DIGEST_PATTERN.fullmatch(self.cluster_id) is None:
            raise AssignmentError("ML-ASSIGN-035", "Cluster digest ID is invalid.")


@dataclass(frozen=True, slots=True, repr=False)
class DeduplicationResult:
    """Result of intra-source duplicate clustering."""

    clusters: tuple[DuplicateCluster, ...] = field(repr=False)
    record_to_cluster: dict[str, str] = field(repr=False)
    mode: Literal["dedupe_only", "link_and_dedupe"]
    algorithm: str
    threshold: float
    total_records: int
    total_clusters: int
    singleton_count: int
    duplicate_record_count: int
    max_cluster_size: int
    deduplication_digest: str

    def __post_init__(self) -> None:
        if sum(len(c.member_record_keys) for c in self.clusters) != self.total_records:
            raise AssignmentError(
                "ML-ASSIGN-036", "Cluster member counts do not match total records."
            )
        if len(self.clusters) != self.total_clusters:
            raise AssignmentError("ML-ASSIGN-037", "Total clusters count is inconsistent.")
        if sum(c.is_singleton for c in self.clusters) != self.singleton_count:
            raise AssignmentError("ML-ASSIGN-038", "Singleton count is inconsistent.")
        if self.total_records - self.singleton_count != self.duplicate_record_count:
            raise AssignmentError("ML-ASSIGN-039", "Duplicate record count is inconsistent.")
        expected_max = max((len(c.member_record_keys) for c in self.clusters), default=0)
        if expected_max != self.max_cluster_size:
            raise AssignmentError("ML-ASSIGN-040", "Max cluster size is inconsistent.")
        if len(self.record_to_cluster) != self.total_records:
            raise AssignmentError(
                "ML-ASSIGN-041", "Record-to-cluster mapping size is inconsistent."
            )
        if _DIGEST_PATTERN.fullmatch(self.deduplication_digest) is None:
            raise AssignmentError("ML-ASSIGN-042", "Deduplication digest is invalid.")

    def safe_summary(self) -> dict[str, int | float | str]:
        return {
            "mode": self.mode,
            "algorithm": self.algorithm,
            "threshold": self.threshold,
            "total_records": self.total_records,
            "total_clusters": self.total_clusters,
            "singleton_count": self.singleton_count,
            "duplicate_record_count": self.duplicate_record_count,
            "max_cluster_size": self.max_cluster_size,
            "deduplication_digest": self.deduplication_digest,
        }


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

    def union(self, item_a: str, item_b: str) -> None:
        root_a = self.find(item_a)
        root_b = self.find(item_b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


class IntraSourceDeduplicator:
    """Graph-based deduplication within a single dataset."""

    @classmethod
    def cluster_batch(
        cls,
        batch: AssignmentEdgeBatch,
        plan: DeduplicationPlan | None = None,
        *,
        dataset_id: str | None = None,
    ) -> DeduplicationResult:
        """Cluster records from an AssignmentEdgeBatch representation."""
        plan_obj = plan or DeduplicationPlan()
        return cls.cluster(
            record_keys=batch.source_record_keys,
            pair_references=batch.pair_references,
            probabilities=batch.probabilities,
            pair_digests=batch.pair_digests,
            plan=plan_obj,
            dataset_id=dataset_id,
        )

    @classmethod
    def cluster(
        cls,
        record_keys: tuple[str, ...],
        pair_references: tuple[tuple[str, str], ...],
        probabilities: NDArray[np.float64] | list[float] | tuple[float, ...],
        pair_digests: tuple[str, ...] | None = None,
        plan: DeduplicationPlan | None = None,
        *,
        dataset_id: str | None = None,
        mode: Literal["dedupe_only", "link_and_dedupe"] = "dedupe_only",
    ) -> DeduplicationResult:
        """Perform graph-based intra-source duplicate clustering."""
        plan_obj = plan or DeduplicationPlan()
        probs = np.asarray(probabilities, dtype=np.float64)

        if not record_keys or len(set(record_keys)) != len(record_keys):
            raise AssignmentError("ML-ASSIGN-001", "The assignment source universe is invalid.")
        if len(pair_references) > plan_obj.maximum_candidate_edges:
            raise AssignmentError("ML-ASSIGN-019", "The assignment candidate budget was exceeded.")
        if len(pair_references) != len(probs):
            raise AssignmentError("ML-ASSIGN-002", "Assignment candidate coverage is invalid.")
        if np.any(probs < 0.0) or np.any(probs > 1.0) or not np.all(np.isfinite(probs)):
            raise AssignmentError("ML-ASSIGN-005", "Assignment probabilities are invalid.")

        digests = pair_digests or tuple(pair_digest(left, right) for left, right in pair_references)
        if len(digests) != len(pair_references):
            raise AssignmentError("ML-ASSIGN-002", "Assignment candidate coverage is invalid.")

        if plan_obj.algorithm == "connected_components":
            return cls._cluster_connected_components(
                record_keys=record_keys,
                pair_references=pair_references,
                probabilities=probs,
                pair_digests=digests,
                plan=plan_obj,
                dataset_id=dataset_id,
                mode=mode,
            )
        elif plan_obj.algorithm in ("clique", "thresholded_clique"):
            return cls._cluster_clique(
                record_keys=record_keys,
                pair_references=pair_references,
                probabilities=probs,
                pair_digests=digests,
                plan=plan_obj,
                dataset_id=dataset_id,
                mode=mode,
            )
        else:
            raise AssignmentError("ML-ASSIGN-022", f"Unknown algorithm {plan_obj.algorithm}")

    @staticmethod
    def _cluster_connected_components(
        record_keys: tuple[str, ...],
        pair_references: tuple[tuple[str, str], ...],
        probabilities: NDArray[np.float64],
        pair_digests: tuple[str, ...],
        plan: DeduplicationPlan,
        dataset_id: str | None,
        mode: Literal["dedupe_only", "link_and_dedupe"],
    ) -> DeduplicationResult:
        dsu = _DisjointSetUnion(record_keys)
        eligible_edges: dict[frozenset[str], float] = {}

        for index, (left, right) in enumerate(pair_references):
            prob = float(probabilities[index])
            util = _utility(prob)
            if prob >= plan.threshold and util > plan.no_match_utility:
                dsu.union(left, right)
                eligible_edges[frozenset((left, right))] = prob

        components: dict[str, list[str]] = defaultdict(list)
        for key in record_keys:
            root = dsu.find(key)
            components[root].append(key)

        clusters: list[DuplicateCluster] = []
        record_to_cluster: dict[str, str] = {}

        for _root, members in sorted(components.items(), key=lambda item: sorted(item[1])[0]):
            sorted_members = tuple(sorted(members))
            size = len(sorted_members)
            if size > plan.max_cluster_size:
                raise AssignmentError(
                    "ML-ASSIGN-032",
                    "Deduplication cluster size exceeds configured maximum limit.",
                )

            canonical = sorted_members[0]
            cluster_id_str = f"{dataset_id or 'default'}\x00{','.join(sorted_members)}"
            cluster_id = hashlib.sha256(cluster_id_str.encode("utf-8")).hexdigest()

            if size == 1:
                cluster = DuplicateCluster(
                    cluster_id=cluster_id,
                    canonical_record_key=canonical,
                    member_record_keys=sorted_members,
                    edge_count=0,
                    mean_probability=1.0,
                    is_singleton=True,
                    dataset_id=dataset_id,
                )
            else:
                intra_probs: list[float] = []
                for i in range(size):
                    for j in range(i + 1, size):
                        pair_key = frozenset((sorted_members[i], sorted_members[j]))
                        if pair_key in eligible_edges:
                            intra_probs.append(eligible_edges[pair_key])
                mean_p = float(np.mean(intra_probs)) if intra_probs else 1.0
                cluster = DuplicateCluster(
                    cluster_id=cluster_id,
                    canonical_record_key=canonical,
                    member_record_keys=sorted_members,
                    edge_count=len(intra_probs),
                    mean_probability=mean_p,
                    is_singleton=False,
                    dataset_id=dataset_id,
                )

            clusters.append(cluster)
            for m in sorted_members:
                record_to_cluster[m] = cluster_id

        clusters_tuple = tuple(clusters)
        digest = _canonical_digest(
            {
                "mode": mode,
                "algorithm": "connected_components",
                "threshold": plan.threshold,
                "dataset_id": dataset_id,
                "clusters": [
                    {
                        "cluster_id": c.cluster_id,
                        "canonical": c.canonical_record_key,
                        "size": len(c.member_record_keys),
                        "members_digest": hashlib.sha256(
                            ",".join(c.member_record_keys).encode("utf-8")
                        ).hexdigest(),
                    }
                    for c in clusters_tuple
                ],
            }
        )

        singletons = sum(c.is_singleton for c in clusters_tuple)
        duplicates = len(record_keys) - singletons
        max_size = max((len(c.member_record_keys) for c in clusters_tuple), default=0)

        return DeduplicationResult(
            clusters=clusters_tuple,
            record_to_cluster=record_to_cluster,
            mode=mode,
            algorithm="connected_components",
            threshold=plan.threshold,
            total_records=len(record_keys),
            total_clusters=len(clusters_tuple),
            singleton_count=singletons,
            duplicate_record_count=duplicates,
            max_cluster_size=max_size,
            deduplication_digest=digest,
        )

    @staticmethod
    def _cluster_clique(
        record_keys: tuple[str, ...],
        pair_references: tuple[tuple[str, str], ...],
        probabilities: NDArray[np.float64],
        pair_digests: tuple[str, ...],
        plan: DeduplicationPlan,
        dataset_id: str | None,
        mode: Literal["dedupe_only", "link_and_dedupe"],
    ) -> DeduplicationResult:
        eligible_edges: dict[frozenset[str], float] = {}
        sorted_pairs: list[tuple[float, str, str, str]] = []

        for index, (left, right) in enumerate(pair_references):
            prob = float(probabilities[index])
            util = _utility(prob)
            if prob >= plan.threshold and util > plan.no_match_utility:
                pair_set = frozenset((left, right))
                eligible_edges[pair_set] = prob
                sorted_pairs.append((-prob, pair_digests[index], left, right))

        # Sort candidate edges by descending probability with deterministic tie-breaking
        sorted_pairs.sort()

        # Each record starts in its own cluster
        cluster_of: dict[str, set[str]] = {k: {k} for k in record_keys}

        for _neg_prob, _pdigest, left, right in sorted_pairs:
            cluster_l = cluster_of[left]
            cluster_r = cluster_of[right]
            if cluster_l is cluster_r:
                continue
            if len(cluster_l) + len(cluster_r) > plan.max_cluster_size:
                continue

            # Verify clique property: every pair across cluster_l and
            # cluster_r must have an eligible edge
            is_clique = True
            for u in cluster_l:
                for v in cluster_r:
                    if frozenset((u, v)) not in eligible_edges:
                        is_clique = False
                        break
                if not is_clique:
                    break

            if is_clique:
                merged = cluster_l | cluster_r
                for member in merged:
                    cluster_of[member] = merged

        # Group distinct clusters
        seen_cluster_ids: set[int] = set()
        unique_clusters: list[set[str]] = []
        for key in sorted(record_keys):
            c_set = cluster_of[key]
            c_id = id(c_set)
            if c_id not in seen_cluster_ids:
                seen_cluster_ids.add(c_id)
                unique_clusters.append(c_set)

        clusters: list[DuplicateCluster] = []
        record_to_cluster: dict[str, str] = {}

        for members_set in sorted(unique_clusters, key=lambda s: sorted(s)[0]):
            sorted_members = tuple(sorted(members_set))
            size = len(sorted_members)
            canonical = sorted_members[0]
            cluster_id_str = f"{dataset_id or 'default'}\x00{','.join(sorted_members)}"
            cluster_id = hashlib.sha256(cluster_id_str.encode("utf-8")).hexdigest()

            if size == 1:
                cluster = DuplicateCluster(
                    cluster_id=cluster_id,
                    canonical_record_key=canonical,
                    member_record_keys=sorted_members,
                    edge_count=0,
                    mean_probability=1.0,
                    is_singleton=True,
                    dataset_id=dataset_id,
                )
            else:
                intra_probs: list[float] = []
                for i in range(size):
                    for j in range(i + 1, size):
                        pair_key = frozenset((sorted_members[i], sorted_members[j]))
                        if pair_key in eligible_edges:
                            intra_probs.append(eligible_edges[pair_key])
                mean_p = float(np.mean(intra_probs)) if intra_probs else 1.0
                cluster = DuplicateCluster(
                    cluster_id=cluster_id,
                    canonical_record_key=canonical,
                    member_record_keys=sorted_members,
                    edge_count=len(intra_probs),
                    mean_probability=mean_p,
                    is_singleton=False,
                    dataset_id=dataset_id,
                )

            clusters.append(cluster)
            for m in sorted_members:
                record_to_cluster[m] = cluster_id

        clusters_tuple = tuple(clusters)
        digest = _canonical_digest(
            {
                "mode": mode,
                "algorithm": "clique",
                "threshold": plan.threshold,
                "dataset_id": dataset_id,
                "clusters": [
                    {
                        "cluster_id": c.cluster_id,
                        "canonical": c.canonical_record_key,
                        "size": len(c.member_record_keys),
                        "members_digest": hashlib.sha256(
                            ",".join(c.member_record_keys).encode("utf-8")
                        ).hexdigest(),
                    }
                    for c in clusters_tuple
                ],
            }
        )

        singletons = sum(c.is_singleton for c in clusters_tuple)
        duplicates = len(record_keys) - singletons
        max_size = max((len(c.member_record_keys) for c in clusters_tuple), default=0)

        return DeduplicationResult(
            clusters=clusters_tuple,
            record_to_cluster=record_to_cluster,
            mode=mode,
            algorithm="clique",
            threshold=plan.threshold,
            total_records=len(record_keys),
            total_clusters=len(clusters_tuple),
            singleton_count=singletons,
            duplicate_record_count=duplicates,
            max_cluster_size=max_size,
            deduplication_digest=digest,
        )


@dataclass(frozen=True, slots=True, repr=False)
class LinkAndDedupeResult:
    """Result of cross-dataset linkage with internal intra-source duplicate resolution."""

    source_a_deduplication: DeduplicationResult = field(repr=False)
    source_b_deduplication: DeduplicationResult = field(repr=False)
    cross_assignment: AssignmentResult = field(repr=False)
    resolved_record_pairs: tuple[tuple[str, str], ...] = field(repr=False)
    source_a_record_count: int
    source_b_record_count: int
    source_a_cluster_count: int
    source_b_cluster_count: int
    assigned_cluster_pair_count: int
    resolved_record_pair_count: int
    link_and_dedupe_digest: str
    mode: Literal["link_and_dedupe"] = "link_and_dedupe"

    def __post_init__(self) -> None:
        if self.source_a_record_count != self.source_a_deduplication.total_records:
            raise AssignmentError("ML-ASSIGN-043", "Source A record count is inconsistent.")
        if self.source_b_record_count != self.source_b_deduplication.total_records:
            raise AssignmentError("ML-ASSIGN-044", "Source B record count is inconsistent.")
        if self.source_a_cluster_count != self.source_a_deduplication.total_clusters:
            raise AssignmentError("ML-ASSIGN-045", "Source A cluster count is inconsistent.")
        if self.source_b_cluster_count != self.source_b_deduplication.total_clusters:
            raise AssignmentError("ML-ASSIGN-046", "Source B cluster count is inconsistent.")
        if self.assigned_cluster_pair_count != self.cross_assignment.real_assignment_count:
            raise AssignmentError("ML-ASSIGN-047", "Assigned cluster pair count is inconsistent.")
        if len(self.resolved_record_pairs) != self.resolved_record_pair_count:
            raise AssignmentError("ML-ASSIGN-048", "Resolved record pair count is inconsistent.")
        if _DIGEST_PATTERN.fullmatch(self.link_and_dedupe_digest) is None:
            raise AssignmentError("ML-ASSIGN-049", "Link and dedupe digest is invalid.")

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "mode": self.mode,
            "source_a_record_count": self.source_a_record_count,
            "source_b_record_count": self.source_b_record_count,
            "source_a_cluster_count": self.source_a_cluster_count,
            "source_b_cluster_count": self.source_b_cluster_count,
            "assigned_cluster_pair_count": self.assigned_cluster_pair_count,
            "resolved_record_pair_count": self.resolved_record_pair_count,
            "link_and_dedupe_digest": self.link_and_dedupe_digest,
        }


class LinkAndDedupeResolver:
    """Orchestrates combined cross-source linkage and intra-source duplicate clustering."""

    @staticmethod
    def resolve(
        *,
        source_a_keys: tuple[str, ...],
        source_b_keys: tuple[str, ...],
        cross_candidates: AssignmentEdgeBatch,
        cross_plan: AssignmentPlan | None = None,
        assignment_plan: AssignmentPlan | None = None,
        intra_a_candidates: AssignmentEdgeBatch | None = None,
        intra_b_candidates: AssignmentEdgeBatch | None = None,
        deduplication_plan: DeduplicationPlan | None = None,
        dedupe_plan: DeduplicationPlan | None = None,
        dataset_a_id: str = "source_a",
        dataset_b_id: str = "source_b",
    ) -> LinkAndDedupeResult:
        """Resolve combined linkage and deduplication."""
        d_plan = deduplication_plan or dedupe_plan or DeduplicationPlan()
        a_plan = cross_plan or assignment_plan or AssignmentPlan()

        # 1. Deduplicate Source A
        if intra_a_candidates is not None:
            dedupe_a = IntraSourceDeduplicator.cluster_batch(
                intra_a_candidates, d_plan, dataset_id=dataset_a_id
            )
        else:
            dedupe_a = IntraSourceDeduplicator.cluster(
                source_a_keys,
                (),
                [],
                plan=d_plan,
                dataset_id=dataset_a_id,
                mode="link_and_dedupe",
            )

        # 2. Deduplicate Source B
        if intra_b_candidates is not None:
            dedupe_b = IntraSourceDeduplicator.cluster_batch(
                intra_b_candidates, d_plan, dataset_id=dataset_b_id
            )
        else:
            dedupe_b = IntraSourceDeduplicator.cluster(
                source_b_keys,
                (),
                [],
                plan=d_plan,
                dataset_id=dataset_b_id,
                mode="link_and_dedupe",
            )

        # Map records to canonical cluster representatives
        cluster_by_id_a = {c.cluster_id: c for c in dedupe_a.clusters}
        cluster_by_id_b = {c.cluster_id: c for c in dedupe_b.clusters}
        canonical_by_record_a = {
            rec: cluster_by_id_a[c_id].canonical_record_key
            for rec, c_id in dedupe_a.record_to_cluster.items()
        }
        canonical_by_record_b = {
            rec: cluster_by_id_b[c_id].canonical_record_key
            for rec, c_id in dedupe_b.record_to_cluster.items()
        }

        # 3. Aggregate cross-dataset candidate pairs to canonical cluster representatives
        cluster_pairs: dict[tuple[str, str], list[float]] = defaultdict(list)
        for index, (left, right) in enumerate(cross_candidates.pair_references):
            c_left = canonical_by_record_a[left]
            c_right = canonical_by_record_b[right]
            prob = float(cross_candidates.probabilities[index])
            cluster_pairs[(c_left, c_right)].append(prob)

        cluster_sources = tuple(sorted({c.canonical_record_key for c in dedupe_a.clusters}))
        raw_pairs = sorted(cluster_pairs.keys())
        pair_references_tuple = tuple(raw_pairs)
        pair_digests_tuple = tuple(
            pair_digest(left, right) for left, right in pair_references_tuple
        )
        probabilities_list = [max(cluster_pairs[pair]) for pair in pair_references_tuple]

        # Compute candidate ranks per canonical cluster source
        ranks_by_source: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for idx, (c_src, _) in enumerate(pair_references_tuple):
            ranks_by_source[c_src].append((probabilities_list[idx], idx))

        ranks_array = np.zeros(len(pair_references_tuple), dtype=np.int64)
        for _src, items in ranks_by_source.items():
            sorted_items = sorted(items, key=lambda it: -it[0])
            for rank_num, (_p, original_idx) in enumerate(sorted_items, start=1):
                ranks_array[original_idx] = rank_num

        cluster_batch = AssignmentEdgeBatch(
            source_record_keys=cluster_sources,
            pair_references=pair_references_tuple,
            pair_digests=pair_digests_tuple,
            probabilities=np.asarray(probabilities_list, dtype=np.float64),
            candidate_ranks=ranks_array,
            source_model_id=cross_candidates.source_model_id,
            source_model_version=cross_candidates.source_model_version,
            calibrator_digest=cross_candidates.calibrator_digest,
            ranking_model_digest=cross_candidates.ranking_model_digest,
            candidate_search_complete=cross_candidates.candidate_search_complete,
            candidate_search_truncated=cross_candidates.candidate_search_truncated,
        )

        # 4. Perform assignment on cluster representatives
        if a_plan.constraint == "one_to_one":
            if a_plan.solver == "scipy_linear_sum_assignment":
                assign_result = ScipyOneToOneAssignmentSolver.solve(cluster_batch, a_plan)
            else:
                assign_result = OrToolsOneToOneAssignmentSolver.solve(cluster_batch, a_plan)
        elif a_plan.constraint == "many_to_one":
            assign_result = ManyToOneAssignmentSolver.solve(cluster_batch, a_plan)
        elif a_plan.constraint == "one_to_many":
            assign_result = OneToManyAssignmentSolver.solve(cluster_batch, a_plan)
        elif a_plan.constraint == "unconstrained":
            assign_result = UnconstrainedAssignmentSolver.solve(cluster_batch, a_plan)
        else:
            raise AssignmentError(
                "ML-ASSIGN-022", f"Unsupported assignment constraint {a_plan.constraint}"
            )

        # 5. Expand matched cluster pairs to member record pairs
        canonical_to_cluster_a = {c.canonical_record_key: c for c in dedupe_a.clusters}
        canonical_to_cluster_b = {c.canonical_record_key: c for c in dedupe_b.clusters}

        resolved_pairs: list[tuple[str, str]] = []
        for assigned in assign_result.assignments:
            if not assigned.selected_no_match and assigned.target_record_key is not None:
                c_a = canonical_to_cluster_a[assigned.source_record_key]
                c_b = canonical_to_cluster_b[assigned.target_record_key]
                for rec_a in sorted(c_a.member_record_keys):
                    for rec_b in sorted(c_b.member_record_keys):
                        resolved_pairs.append((rec_a, rec_b))

        resolved_pairs_tuple = tuple(sorted(resolved_pairs))

        link_and_dedupe_digest = _canonical_digest(
            {
                "mode": "link_and_dedupe",
                "dedupe_a_digest": dedupe_a.deduplication_digest,
                "dedupe_b_digest": dedupe_b.deduplication_digest,
                "cross_assignment_digest": assign_result.assignment_digest,
                "resolved_record_pair_count": len(resolved_pairs_tuple),
            }
        )

        return LinkAndDedupeResult(
            source_a_deduplication=dedupe_a,
            source_b_deduplication=dedupe_b,
            cross_assignment=assign_result,
            resolved_record_pairs=resolved_pairs_tuple,
            mode="link_and_dedupe",
            source_a_record_count=dedupe_a.total_records,
            source_b_record_count=dedupe_b.total_records,
            source_a_cluster_count=dedupe_a.total_clusters,
            source_b_cluster_count=dedupe_b.total_clusters,
            assigned_cluster_pair_count=assign_result.real_assignment_count,
            resolved_record_pair_count=len(resolved_pairs_tuple),
            link_and_dedupe_digest=link_and_dedupe_digest,
        )
