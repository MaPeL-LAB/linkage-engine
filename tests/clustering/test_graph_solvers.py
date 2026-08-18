"""Unit tests for multi-source graph data structures and clustering solvers."""

from __future__ import annotations

import contextlib
import math
import re
from typing import Any, Literal

try:
    import pytest
except ImportError:

    class _PytestShim:
        @staticmethod
        def approx(expected: float, abs: float | None = None) -> Any:
            class _Approx:
                def __eq__(self, actual: Any) -> bool:
                    tolerance = abs if abs is not None else 1e-6
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
    ClusterEntity,
    ClusteringPlan,
    ClusteringResult,
    ConnectedComponentsSolver,
    ConstrainedAgglomerativeSolver,
    CorrelationClusteringSolver,
    MultiSourceClusterer,
    MultiSourceGraph,
    build_cluster_entity,
)
from mapel_linkage.domain.errors import ClusteringError


def test_multi_source_graph_construction_and_properties() -> None:
    nodes = {"rec_a1": "ds_a", "rec_b1": "ds_b", "rec_c1": "ds_c", "rec_d1": "ds_d"}
    edges = {
        ("rec_a1", "rec_b1"): 0.95,
        ("rec_b1", "rec_c1"): 0.88,
        ("rec_a1", "rec_c1"): 0.92,
    }
    cannot_link = [("rec_a1", "rec_d1")]

    graph = MultiSourceGraph.build(nodes=nodes, edges=edges, cannot_link=cannot_link)

    assert graph.node_count == 4
    assert graph.edge_count == 3
    assert graph.cannot_link_count == 1
    assert set(graph.dataset_ids) == {"ds_a", "ds_b", "ds_c", "ds_d"}
    assert graph.get_dataset("rec_a1") == "ds_a"
    assert graph.get_edge_weight("rec_a1", "rec_b1") == 0.95
    assert graph.get_edge_weight("rec_b1", "rec_a1") == 0.95
    assert graph.get_edge_weight("rec_a1", "rec_d1") is None
    assert graph.has_edge("rec_a1", "rec_b1") is True
    assert graph.has_edge("rec_a1", "rec_d1") is False
    assert graph.is_cannot_link("rec_a1", "rec_d1") is True
    assert graph.is_cannot_link("rec_d1", "rec_a1") is True
    assert graph.is_cannot_link("rec_a1", "rec_b1") is False

    neighbors_a = graph.neighbors("rec_a1")
    assert neighbors_a == {"rec_b1": 0.95, "rec_c1": 0.92}

    summary = graph.safe_summary()
    assert summary["node_count"] == 4
    assert summary["edge_count"] == 3
    assert summary["cannot_link_count"] == 1
    assert summary["dataset_count"] == 4


def test_multi_source_graph_validation_errors() -> None:
    with pytest.raises(ClusteringError, match="ML-CLUSTER-001"):
        MultiSourceGraph.build(nodes={})

    with pytest.raises(ClusteringError, match="ML-CLUSTER-002"):
        MultiSourceGraph.build(
            nodes={"n1": "ds1"},
            edges={("n1", "n_missing"): 0.9},
        )

    with pytest.raises(ClusteringError, match="ML-CLUSTER-003"):
        MultiSourceGraph.build(
            nodes={"n1": "ds1"},
            cannot_link=[("n1", "n_missing")],
        )

    with pytest.raises(ClusteringError, match="ML-CLUSTER-004"):
        MultiSourceGraph.build(
            nodes={"n1": "ds1"},
            edges={("n1", "n1"): 0.9},
        )

    with pytest.raises(ClusteringError, match="ML-CLUSTER-005"):
        MultiSourceGraph.build(
            nodes={"n1": "ds1", "n2": "ds2"},
            edges={("n1", "n2"): 1.5},
        )


def test_clustering_plan_validation() -> None:
    plan = ClusteringPlan()
    assert plan.algorithm == "correlation_clustering"
    assert plan.threshold == 0.50
    assert plan.positive_weight_multiplier == 1.0
    assert plan.negative_weight_multiplier == 1.0
    assert plan.max_cluster_size == 10_000
    assert plan.cannot_link_strictness == "hard"
    assert plan.cannot_link_same_dataset is False

    with pytest.raises(ClusteringError, match="ML-CLUSTER-006"):
        ClusteringPlan(threshold=-0.1)

    with pytest.raises(ClusteringError, match="ML-CLUSTER-006"):
        ClusteringPlan(threshold=1.5)

    with pytest.raises(ClusteringError, match="ML-CLUSTER-018"):
        ClusteringPlan(positive_weight_multiplier=-1.0)

    with pytest.raises(ClusteringError, match="ML-CLUSTER-018"):
        ClusteringPlan(negative_weight_multiplier=-0.5)

    with pytest.raises(ClusteringError, match="ML-CLUSTER-007"):
        ClusteringPlan(max_cluster_size=0)

    with pytest.raises(ClusteringError, match="ML-CLUSTER-008"):
        ClusteringPlan(cannot_link_strictness="invalid")  # type: ignore[arg-type]

    with pytest.raises(ClusteringError, match="ML-CLUSTER-008"):
        ClusteringPlan(algorithm="unknown_algorithm")  # type: ignore[arg-type]


def test_cluster_entity_and_result_contracts() -> None:
    nodes = {"r1": "d1", "r2": "d2", "r3": "d3"}
    edges = {("r1", "r2"): 0.90, ("r2", "r3"): 0.80, ("r1", "r3"): 0.85}
    graph = MultiSourceGraph.build(nodes=nodes, edges=edges)

    entity = build_cluster_entity(["r1", "r2", "r3"], graph)
    assert isinstance(entity, ClusterEntity)
    assert entity.canonical_record_key == "r1"
    assert entity.member_record_keys == ("r1", "r2", "r3")
    assert entity.dataset_distribution == {"d1": 1, "d2": 1, "d3": 1}
    assert entity.is_singleton is False
    assert entity.edge_count == 3
    assert pytest.approx(entity.mean_edge_weight, 0.001) == 0.85

    summary = entity.safe_summary()
    assert summary["size"] == 3
    assert summary["is_singleton"] is False

    # Singleton entity
    singleton = build_cluster_entity(["r1"], graph)
    assert singleton.is_singleton is True
    assert singleton.mean_edge_weight == 1.0
    assert singleton.edge_count == 0


def test_correlation_clustering_solver_triad() -> None:
    nodes = {
        "srcA_1": "src_a",
        "srcB_1": "src_b",
        "srcC_1": "src_c",
        "srcA_2": "src_a",
        "srcB_2": "src_b",
    }
    edges = {
        ("srcA_1", "srcB_1"): 0.95,
        ("srcB_1", "srcC_1"): 0.92,
        ("srcA_1", "srcC_1"): 0.90,
        ("srcA_2", "srcB_2"): 0.88,
        ("srcA_1", "srcA_2"): 0.20,  # Weak cross-entity link below threshold
    }
    graph = MultiSourceGraph.build(nodes=nodes, edges=edges)
    plan = ClusteringPlan(algorithm="correlation_clustering", threshold=0.50)

    result = CorrelationClusteringSolver.solve(graph, plan)
    assert isinstance(result, ClusteringResult)
    assert result.total_records == 5
    assert result.total_clusters == 2
    assert result.singleton_count == 0
    assert result.constraint_violation_count == 0
    assert result.cluster_size_distribution == {3: 1, 2: 1}

    # Find the 3-source cluster
    c3 = next(c for c in result.clusters if len(c.member_record_keys) == 3)
    assert set(c3.member_record_keys) == {"srcA_1", "srcB_1", "srcC_1"}
    assert c3.dataset_distribution == {"src_a": 1, "src_b": 1, "src_c": 1}

    # Find the 2-source cluster
    c2 = next(c for c in result.clusters if len(c.member_record_keys) == 2)
    assert set(c2.member_record_keys) == {"srcA_2", "srcB_2"}


def test_constrained_agglomerative_solver_merging() -> None:
    nodes = {"a": "ds1", "b": "ds2", "c": "ds3", "d": "ds4", "e": "ds5"}
    edges = {
        ("a", "b"): 0.99,
        ("b", "c"): 0.95,
        ("d", "e"): 0.85,
        ("c", "d"): 0.40,  # Below threshold 0.50
    }
    graph = MultiSourceGraph.build(nodes=nodes, edges=edges)
    plan = ClusteringPlan(algorithm="constrained_agglomerative", threshold=0.50)

    result = ConstrainedAgglomerativeSolver.solve(graph, plan)
    assert isinstance(result, ClusteringResult)
    assert result.total_records == 5
    assert result.total_clusters == 2  # {a, b, c} and {d, e}
    assert result.cluster_size_distribution == {3: 1, 2: 1}
    assert result.constraint_violation_count == 0


def test_max_cluster_size_constraint() -> None:
    nodes = {"n1": "ds1", "n2": "ds2", "n3": "ds3", "n4": "ds4"}
    edges = {
        ("n1", "n2"): 0.95,
        ("n2", "n3"): 0.90,
        ("n3", "n4"): 0.85,
        ("n1", "n3"): 0.80,
    }
    graph = MultiSourceGraph.build(nodes=nodes, edges=edges)

    # With max_cluster_size = 2, agglomerative clustering must not form clusters of size > 2
    plan = ClusteringPlan(
        algorithm="constrained_agglomerative",
        threshold=0.50,
        max_cluster_size=2,
    )
    result = ConstrainedAgglomerativeSolver.solve(graph, plan)

    assert all(len(c.member_record_keys) <= 2 for c in result.clusters)
    assert result.total_records == 4

    # Correlation clustering with max_cluster_size = 2
    plan_cc = ClusteringPlan(
        algorithm="correlation_clustering",
        threshold=0.50,
        max_cluster_size=2,
    )
    result_cc = CorrelationClusteringSolver.solve(graph, plan_cc)
    assert all(len(c.member_record_keys) <= 2 for c in result_cc.clusters)
    assert result_cc.total_records == 4


def test_connected_components_baseline_solver() -> None:
    nodes = {"x1": "ds1", "x2": "ds2", "x3": "ds3", "y1": "ds1", "z_isolated": "ds9"}
    edges = {
        ("x1", "x2"): 0.90,
        ("x2", "x3"): 0.85,
        ("y1", "x3"): 0.20,  # Below threshold
    }
    graph = MultiSourceGraph.build(nodes=nodes, edges=edges)
    plan = ClusteringPlan(algorithm="connected_components", threshold=0.50)

    result = ConnectedComponentsSolver.solve(graph, plan)
    assert isinstance(result, ClusteringResult)
    assert result.total_records == 5
    assert result.total_clusters == 3  # {x1, x2, x3}, {y1}, {z_isolated}
    assert result.singleton_count == 2
    assert result.cluster_size_distribution == {3: 1, 1: 2}


def test_multi_source_clusterer_dispatcher() -> None:
    nodes = {"a": "ds1", "b": "ds2"}
    edges = {("a", "b"): 0.90}
    graph = MultiSourceGraph.build(nodes=nodes, edges=edges)

    algos: tuple[
        Literal["correlation_clustering", "constrained_agglomerative", "connected_components"], ...
    ] = (
        "correlation_clustering",
        "constrained_agglomerative",
        "connected_components",
    )
    for algo in algos:
        plan = ClusteringPlan(algorithm=algo, threshold=0.50)
        res = MultiSourceClusterer.solve(graph, plan)
        assert res.total_records == 2
        assert res.total_clusters == 1
        assert res.algorithm == algo
