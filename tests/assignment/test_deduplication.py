from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mapel_linkage.assignment import (
    AssignmentEdgeBatch,
    AssignmentPlan,
    DeduplicationPlan,
    DuplicateCluster,
    IntraSourceDeduplicator,
    LinkAndDedupeResolver,
)
from mapel_linkage.domain.errors import AssignmentError


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_connected_components_clustering_dedupe_only() -> None:
    records = ("r1", "r2", "r3", "r4", "r5")
    pairs = (
        ("r1", "r2"),
        ("r2", "r3"),
        ("r4", "r5"),
    )
    probs = [0.95, 0.90, 0.30]  # r4-r5 below threshold 0.50
    digests = tuple(digest(f"{k_l}\x00{k_r}") for k_l, k_r in pairs)

    plan = DeduplicationPlan(algorithm="connected_components", threshold=0.50)
    result = IntraSourceDeduplicator.cluster(
        record_keys=records,
        pair_references=pairs,
        probabilities=probs,
        pair_digests=digests,
        plan=plan,
        dataset_id="test_ds",
    )

    assert result.total_records == 5
    assert result.total_clusters == 3  # {r1, r2, r3}, {r4}, {r5}
    assert result.singleton_count == 2
    assert result.duplicate_record_count == 3
    assert result.max_cluster_size == 3
    assert result.algorithm == "connected_components"

    # Find the multi-record cluster
    multi_clusters = [c for c in result.clusters if not c.is_singleton]
    assert len(multi_clusters) == 1
    multi = multi_clusters[0]
    assert multi.canonical_record_key == "r1"
    assert multi.member_record_keys == ("r1", "r2", "r3")
    assert multi.edge_count == 2
    assert 0.90 <= multi.mean_probability <= 0.95

    # Check mapping
    assert result.record_to_cluster["r1"] == multi.cluster_id
    assert result.record_to_cluster["r2"] == multi.cluster_id
    assert result.record_to_cluster["r3"] == multi.cluster_id
    assert result.record_to_cluster["r4"] != multi.cluster_id
    assert result.record_to_cluster["r5"] != multi.cluster_id
    assert "r1" not in repr(result)


def test_clique_clustering_prevents_transitive_chaining() -> None:
    # r1-r2 connected (0.95), r2-r3 connected (0.95), but NO edge between r1 and r3
    records = ("r1", "r2", "r3")
    pairs = (
        ("r1", "r2"),
        ("r2", "r3"),
    )
    probs = [0.95, 0.95]
    digests = tuple(digest(f"{k_l}\x00{k_r}") for k_l, k_r in pairs)

    # 1. Connected components would merge all 3:
    cc_res = IntraSourceDeduplicator.cluster(
        record_keys=records,
        pair_references=pairs,
        probabilities=probs,
        pair_digests=digests,
        plan=DeduplicationPlan(algorithm="connected_components", threshold=0.50),
    )
    assert cc_res.max_cluster_size == 3

    # 2. Clique clustering prevents merging because (r1, r3) is missing:
    clique_res = IntraSourceDeduplicator.cluster(
        record_keys=records,
        pair_references=pairs,
        probabilities=probs,
        pair_digests=digests,
        plan=DeduplicationPlan(algorithm="clique", threshold=0.50),
    )
    # Since r1-r2 is processed first, {r1, r2} forms a clique, r3 cannot join
    assert clique_res.max_cluster_size == 2
    assert clique_res.total_clusters == 2  # {r1, r2} and {r3}
    assert clique_res.singleton_count == 1
    assert clique_res.duplicate_record_count == 2


def test_clique_clustering_merges_full_triangle() -> None:
    # Full triangle: r1-r2 (0.95), r2-r3 (0.95), r1-r3 (0.90)
    records = ("r1", "r2", "r3")
    pairs = (
        ("r1", "r2"),
        ("r2", "r3"),
        ("r1", "r3"),
    )
    probs = [0.95, 0.95, 0.90]
    digests = tuple(digest(f"{k_l}\x00{k_r}") for k_l, k_r in pairs)

    res = IntraSourceDeduplicator.cluster(
        record_keys=records,
        pair_references=pairs,
        probabilities=probs,
        pair_digests=digests,
        plan=DeduplicationPlan(algorithm="clique", threshold=0.50),
    )
    assert res.total_clusters == 1
    assert res.clusters[0].member_record_keys == ("r1", "r2", "r3")
    assert res.clusters[0].edge_count == 3


def test_link_and_dedupe_resolver_cross_dataset() -> None:
    # Source A: {a1, a2} duplicate, {a3} singleton
    src_a_records = ("a1", "a2", "a3")
    intra_a_pairs = (("a1", "a2"),)
    intra_a_batch = AssignmentEdgeBatch(
        source_record_keys=src_a_records,
        pair_references=intra_a_pairs,
        pair_digests=tuple(digest(f"{k_l}\x00{k_r}") for k_l, k_r in intra_a_pairs),
        probabilities=np.asarray([0.95], dtype=np.float64),
        candidate_ranks=np.asarray([1], dtype=np.int64),
        source_model_id="intra_model",
        source_model_version="v1",
        calibrator_digest=digest("cal_a"),
        ranking_model_digest=None,
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )

    # Source B: {b1, b2} duplicate, {b3} singleton
    src_b_records = ("b1", "b2", "b3")
    intra_b_pairs = (("b1", "b2"),)
    intra_b_batch = AssignmentEdgeBatch(
        source_record_keys=src_b_records,
        pair_references=intra_b_pairs,
        pair_digests=tuple(digest(f"{k_l}\x00{k_r}") for k_l, k_r in intra_b_pairs),
        probabilities=np.asarray([0.92], dtype=np.float64),
        candidate_ranks=np.asarray([1], dtype=np.int64),
        source_model_id="intra_model",
        source_model_version="v1",
        calibrator_digest=digest("cal_b"),
        ranking_model_digest=None,
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )

    # Cross-source candidates
    cross_pairs = (
        ("a1", "b1"),
        ("a2", "b2"),
        ("a3", "b3"),
    )
    cross_batch = AssignmentEdgeBatch(
        source_record_keys=src_a_records,
        pair_references=cross_pairs,
        pair_digests=tuple(digest(f"{k_l}\x00{k_r}") for k_l, k_r in cross_pairs),
        probabilities=np.asarray([0.94, 0.91, 0.88], dtype=np.float64),
        candidate_ranks=np.asarray([1, 1, 1], dtype=np.int64),
        source_model_id="cross_model",
        source_model_version="v1",
        calibrator_digest=digest("cal_cross"),
        ranking_model_digest=None,
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )

    result = LinkAndDedupeResolver.resolve(
        source_a_keys=src_a_records,
        source_b_keys=src_b_records,
        cross_candidates=cross_batch,
        intra_a_candidates=intra_a_batch,
        intra_b_candidates=intra_b_batch,
        dedupe_plan=DeduplicationPlan(threshold=0.50),
        assignment_plan=AssignmentPlan(
            constraint="one_to_one", solver="scipy_linear_sum_assignment"
        ),
        dataset_a_id="src_a",
        dataset_b_id="src_b",
    )

    assert result.source_a_record_count == 3
    assert result.source_b_record_count == 3
    assert result.source_a_cluster_count == 2
    assert result.source_b_cluster_count == 2
    assert result.assigned_cluster_pair_count == 2
    # {a1, a2} matches {b1, b2} -> generates 4 pairs: (a1,b1), (a1,b2), (a2,b1), (a2,b2)
    # {a3} matches {b3} -> generates 1 pair: (a3,b3)
    # Total = 5 resolved record pairs
    assert result.resolved_record_pair_count == 5
    assert ("a1", "b1") in result.resolved_record_pairs
    assert ("a1", "b2") in result.resolved_record_pairs
    assert ("a2", "b1") in result.resolved_record_pairs
    assert ("a2", "b2") in result.resolved_record_pairs
    assert ("a3", "b3") in result.resolved_record_pairs


def test_deduplication_validations_and_errors() -> None:
    # Invalid threshold
    with pytest.raises(AssignmentError, match="ML-ASSIGN-005"):
        DeduplicationPlan(threshold=-0.1)

    # Empty universe
    with pytest.raises(AssignmentError, match="ML-ASSIGN-001"):
        IntraSourceDeduplicator.cluster(
            record_keys=(),
            pair_references=(),
            probabilities=[],
        )

    # Duplicate keys in universe
    with pytest.raises(AssignmentError, match="ML-ASSIGN-001"):
        IntraSourceDeduplicator.cluster(
            record_keys=("r1", "r1"),
            pair_references=(),
            probabilities=[],
        )

    # Candidate budget exceeded
    with pytest.raises(AssignmentError, match="ML-ASSIGN-019"):
        IntraSourceDeduplicator.cluster(
            record_keys=("r1", "r2", "r3"),
            pair_references=(("r1", "r2"), ("r2", "r3")),
            probabilities=[0.9, 0.9],
            plan=DeduplicationPlan(maximum_candidate_edges=1),
        )

    # Max cluster size exceeded
    with pytest.raises(AssignmentError, match="ML-ASSIGN-032"):
        IntraSourceDeduplicator.cluster(
            record_keys=("r1", "r2", "r3"),
            pair_references=(("r1", "r2"), ("r2", "r3")),
            probabilities=[0.9, 0.9],
            plan=DeduplicationPlan(max_cluster_size=2),
        )

    # Cluster invariant: canonical record key not in members
    with pytest.raises(AssignmentError, match="ML-ASSIGN-031"):
        DuplicateCluster(
            cluster_id=digest("c1"),
            canonical_record_key="other_key",
            member_record_keys=("r1", "r2"),
            edge_count=1,
            mean_probability=0.9,
            is_singleton=False,
        )
