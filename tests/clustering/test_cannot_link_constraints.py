"""Unit tests for cannot-link hard constraints, same-dataset constraints, and value safety."""

from __future__ import annotations

import contextlib
import re
from typing import Any

try:
    import pytest
except ImportError:

    class _PytestShim:
        @staticmethod
        def approx(expected: float, abs: float | None = None) -> Any:
            class _Approx:
                def __eq__(self, actual: Any) -> bool:
                    tolerance = abs if abs is not None else 1e-6
                    import math

                    return math.isclose(actual, expected, abs_tol=tolerance)

            return _Approx()

        @staticmethod
        @contextlib.contextmanager
        def raises(expected_exception: type[BaseException], match: str | None = None) -> Any:
            class _ExcInfo:
                value: BaseException

            exc_info = _ExcInfo()
            try:
                yield exc_info
            except expected_exception as e:
                exc_info.value = e
                if match and not re.search(match, str(e)):
                    raise AssertionError(f"Pattern {match!r} did not match {str(e)!r}") from e
            else:
                raise AssertionError(
                    f"Expected exception {expected_exception.__name__} was not raised"
                )

    pytest = _PytestShim()  # type: ignore[assignment]

from mapel_linkage.clustering import (
    ClusteringPlan,
    ConnectedComponentsSolver,
    ConstrainedAgglomerativeSolver,
    CorrelationClusteringSolver,
    MultiSourceGraph,
    build_cluster_entity,
)
from mapel_linkage.domain.errors import ClusteringError


def test_explicit_cannot_link_in_agglomerative_solver() -> None:
    # Nodes A, B, C with candidate edges (A, B)=0.95, (B, C)=0.90, (A, C)=0.85
    # Cannot-link constraint between (A, C)
    nodes = {"rec_a": "ds1", "rec_b": "ds2", "rec_c": "ds3"}
    edges = {
        ("rec_a", "rec_b"): 0.95,
        ("rec_b", "rec_c"): 0.90,
        ("rec_a", "rec_c"): 0.85,
    }
    cannot_link = [("rec_a", "rec_c")]

    graph = MultiSourceGraph.build(nodes=nodes, edges=edges, cannot_link=cannot_link)
    plan = ClusteringPlan(algorithm="constrained_agglomerative", threshold=0.50)

    result = ConstrainedAgglomerativeSolver.solve(graph, plan)

    # A and B merge first. Merging {A, B} with C is rejected due to cannot-link (A, C).
    assert result.total_clusters == 2
    assert result.constraint_violation_count == 0

    c_ab = next(c for c in result.clusters if len(c.member_record_keys) == 2)
    assert set(c_ab.member_record_keys) == {"rec_a", "rec_b"}

    c_c = next(c for c in result.clusters if len(c.member_record_keys) == 1)
    assert set(c_c.member_record_keys) == {"rec_c"}


def test_explicit_cannot_link_in_correlation_clustering_solver() -> None:
    nodes = {"rec_a": "ds1", "rec_b": "ds2", "rec_c": "ds3"}
    edges = {
        ("rec_a", "rec_b"): 0.95,
        ("rec_b", "rec_c"): 0.90,
        ("rec_a", "rec_c"): 0.85,
    }
    cannot_link = [("rec_a", "rec_c")]

    graph = MultiSourceGraph.build(nodes=nodes, edges=edges, cannot_link=cannot_link)
    plan = ClusteringPlan(algorithm="correlation_clustering", threshold=0.50)

    result = CorrelationClusteringSolver.solve(graph, plan)

    assert result.total_clusters == 2
    assert result.constraint_violation_count == 0

    # Ensure rec_a and rec_c are not in the same cluster
    cluster_a = result.record_to_cluster["rec_a"]
    cluster_c = result.record_to_cluster["rec_c"]
    assert cluster_a != cluster_c


def test_transitive_cannot_link_prevents_cluster_merging() -> None:
    # Cluster 1: {a1, a2}, Cluster 2: {b1, b2}
    # Edge between a2 and b1 is 0.88, but cannot-link between a1 and b2
    nodes = {"a1": "ds1", "a2": "ds2", "b1": "ds3", "b2": "ds4"}
    edges = {
        ("a1", "a2"): 0.98,
        ("b1", "b2"): 0.95,
        ("a2", "b1"): 0.88,
    }
    cannot_link = [("a1", "b2")]

    graph = MultiSourceGraph.build(nodes=nodes, edges=edges, cannot_link=cannot_link)
    plan = ClusteringPlan(algorithm="constrained_agglomerative", threshold=0.50)

    result = ConstrainedAgglomerativeSolver.solve(graph, plan)

    assert result.total_clusters == 2
    assert result.constraint_violation_count == 0
    assert {c.member_record_keys for c in result.clusters} == {("a1", "a2"), ("b1", "b2")}


def test_cannot_link_same_dataset_constraint() -> None:
    # 4 records: two from source_1, one from source_2, one from source_3
    nodes = {
        "s1_rec1": "source_1",
        "s1_rec2": "source_1",
        "s2_rec1": "source_2",
        "s3_rec1": "source_3",
    }
    edges = {
        ("s1_rec1", "s2_rec1"): 0.95,
        ("s2_rec1", "s3_rec1"): 0.90,
        ("s1_rec1", "s3_rec1"): 0.88,
        ("s1_rec2", "s2_rec1"): 0.85,  # s1_rec2 also links to s2_rec1
        ("s1_rec1", "s1_rec2"): 0.80,  # intra-source link
    }
    graph = MultiSourceGraph.build(nodes=nodes, edges=edges)

    # 1. With cannot_link_same_dataset = True, agglomerative clustering strictly isolates s1_rec2
    plan_agg = ClusteringPlan(
        algorithm="constrained_agglomerative",
        threshold=0.50,
        cannot_link_same_dataset=True,
    )
    result_agg = ConstrainedAgglomerativeSolver.solve(graph, plan_agg)

    assert result_agg.total_clusters == 2
    assert result_agg.constraint_violation_count == 0

    c3 = next(c for c in result_agg.clusters if len(c.member_record_keys) == 3)
    assert set(c3.member_record_keys) == {"s1_rec1", "s2_rec1", "s3_rec1"}
    assert c3.dataset_distribution == {"source_1": 1, "source_2": 1, "source_3": 1}

    c1 = next(c for c in result_agg.clusters if len(c.member_record_keys) == 1)
    assert set(c1.member_record_keys) == {"s1_rec2"}

    # 2. Correlation clustering with cannot_link_same_dataset = True
    plan_cc = ClusteringPlan(
        algorithm="correlation_clustering",
        threshold=0.50,
        cannot_link_same_dataset=True,
    )
    result_cc = CorrelationClusteringSolver.solve(graph, plan_cc)
    assert result_cc.total_clusters == 2
    assert result_cc.constraint_violation_count == 0
    assert all(max(c.dataset_distribution.values()) <= 1 for c in result_cc.clusters)


def test_connected_components_tracks_violations() -> None:
    nodes = {"n1": "ds1", "n2": "ds2", "n3": "ds3"}
    edges = {
        ("n1", "n2"): 0.90,
        ("n2", "n3"): 0.85,
    }
    cannot_link = [("n1", "n3")]
    graph = MultiSourceGraph.build(nodes=nodes, edges=edges, cannot_link=cannot_link)

    plan = ClusteringPlan(algorithm="connected_components", threshold=0.50)
    result = ConnectedComponentsSolver.solve(graph, plan)

    # Baseline connected components merges all 3 nodes, but tracks 1 violation
    assert result.total_clusters == 1
    assert result.constraint_violation_count == 1


def test_connected_components_tracks_same_dataset_violations() -> None:
    nodes = {"a1": "ds1", "a2": "ds1", "b1": "ds2"}
    edges = {
        ("a1", "b1"): 0.90,
        ("a2", "b1"): 0.85,
    }
    graph = MultiSourceGraph.build(nodes=nodes, edges=edges)

    plan = ClusteringPlan(
        algorithm="connected_components",
        threshold=0.50,
        cannot_link_same_dataset=True,
    )
    result = ConnectedComponentsSolver.solve(graph, plan)

    # Baseline connected components merges all 3, tracking 1 same-dataset violation (a1, a2)
    assert result.total_clusters == 1
    assert result.constraint_violation_count == 1


def test_privacy_and_value_safety_reprs() -> None:
    sentinel_node = "SENTINEL-NODE-RECORD-KEY-DO-NOT-PRINT"
    sentinel_target = "SENTINEL-TARGET-RECORD-KEY-DO-NOT-PRINT"

    nodes = {sentinel_node: "ds1", sentinel_target: "ds2"}
    edges = {(sentinel_node, sentinel_target): 0.95}
    cannot_link = [(sentinel_node, sentinel_target)]

    # MultiSourceGraph repr
    graph = MultiSourceGraph.build(nodes=nodes, edges=edges, cannot_link=cannot_link)
    rendered_graph = repr(graph)
    assert sentinel_node not in rendered_graph
    assert sentinel_target not in rendered_graph

    # ClusterEntity repr
    entity = build_cluster_entity([sentinel_node, sentinel_target], graph)
    rendered_entity = repr(entity)
    assert sentinel_node not in rendered_entity
    assert sentinel_target not in rendered_entity

    # ClusteringResult repr
    plan = ClusteringPlan(algorithm="constrained_agglomerative")
    result = ConstrainedAgglomerativeSolver.solve(graph, plan)
    rendered_result = repr(result)
    assert sentinel_node not in rendered_result
    assert sentinel_target not in rendered_result

    # ClusteringError message safety
    with pytest.raises(ClusteringError) as exc_info:
        MultiSourceGraph.build(
            nodes={sentinel_node: "ds1"},
            edges={(sentinel_node, sentinel_target): 0.95},
        )
    err_str = str(exc_info.value)
    assert sentinel_node not in err_str
    assert sentinel_target not in err_str
