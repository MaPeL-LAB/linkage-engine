"""Unit tests for MultiSourceEntityResolver, graph construction, solvers, and crosswalk export."""

from __future__ import annotations

import contextlib
import hashlib
import math
import re
import tempfile
from pathlib import Path
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
    CandidateEdge,
    ClusteringPlan,
    MultiSourceEntityResolver,
    MultiSourceResolutionResult,
    pair_digest,
)
from mapel_linkage.domain.errors import ClusteringError


def test_multisource_resolver_three_sources_resolution(tmp_path: Path | None = None) -> None:
    path_dir = tmp_path if tmp_path is not None else Path(tempfile.mkdtemp())
    # 3 datasets: Source A, Source B, Source C
    # True entity 1: {a1, b1, c1}
    # True entity 2: {a2, b2, c2}
    # Singletons: {a3}, {b3}, {c3}
    datasets = {
        "source_a": ("a1", "a2", "a3"),
        "source_b": ("b1", "b2", "b3"),
        "source_c": ("c1", "c2", "c3"),
    }

    edges = [
        CandidateEdge("a1", "source_a", "b1", "source_b", 0.95, pair_digest("a1", "b1")),
        CandidateEdge("b1", "source_b", "c1", "source_c", 0.92, pair_digest("b1", "c1")),
        CandidateEdge("a1", "source_a", "c1", "source_c", 0.90, pair_digest("a1", "c1")),
        CandidateEdge("a2", "source_a", "b2", "source_b", 0.88, pair_digest("a2", "b2")),
        CandidateEdge("b2", "source_b", "c2", "source_c", 0.85, pair_digest("b2", "c2")),
        CandidateEdge("a2", "source_a", "c2", "source_c", 0.83, pair_digest("a2", "c2")),
    ]

    plan = ClusteringPlan(
        algorithm="constrained_agglomerative",
        threshold=0.50,
        cannot_link_same_dataset=True,
    )

    result = MultiSourceEntityResolver.resolve(
        datasets=datasets,
        candidate_edges=edges,
        plan=plan,
        min_datasets=3,
    )

    assert isinstance(result, MultiSourceResolutionResult)
    assert result.total_records == 9
    assert result.total_clusters == 5  # {a1, b1, c1}, {a2, b2, c2}, {a3}, {b3}, {c3}
    assert result.singleton_count == 3
    assert result.multi_record_cluster_count == 2
    assert result.source_collision_count == 0
    assert result.cannot_link_violations == 0
    assert result.max_cluster_size == 3

    # Check crosswalk export
    crosswalk_json = path_dir / "crosswalk.json"
    records = MultiSourceEntityResolver.export_crosswalk(result, crosswalk_json)
    assert len(records) == 9
    assert crosswalk_json.exists()

    # Check crosswalk contents
    e1_records = [r for r in records if r["record_key"] in ("a1", "b1", "c1")]
    assert len(e1_records) == 3
    e1_id = e1_records[0]["global_entity_id"]
    assert all(r["global_entity_id"] == e1_id for r in e1_records)
    assert all(r["cluster_size"] == 3 for r in e1_records)
    # Exactly one record is canonical
    assert sum(r["is_canonical"] for r in e1_records) == 1

    # Check safe summary and representation
    summary = result.safe_summary()
    assert summary["total_records"] == 9
    assert summary["total_clusters"] == 5
    assert "a1" not in repr(result)


def test_multisource_resolver_enforces_source_uniqueness() -> None:
    # If edge exists between two records of the same dataset (e.g. a1 and a2 in source_a),
    # or transitive link a1-b1 and a2-b1 would merge a1 and a2 into the same cluster:
    # source uniqueness prevents merging!
    datasets = {
        "source_a": ("a1", "a2"),
        "source_b": ("b1", "b2"),
        "source_c": ("c1", "c2"),
    }

    # a1 links to b1 (0.95), and a2 ALSO links to b1 (0.90)
    edges = [
        CandidateEdge("a1", "source_a", "b1", "source_b", 0.95, pair_digest("a1", "b1")),
        CandidateEdge("a2", "source_a", "b1", "source_b", 0.90, pair_digest("a2", "b1")),
    ]

    # 1. With cannot_link_same_dataset = True:
    # (a1, b1) processed first and merged into {a1, b1}.
    # (a2, b1) rejected because b1 is in a cluster with a1 (both from source_a)!
    result = MultiSourceEntityResolver.resolve(
        datasets=datasets,
        candidate_edges=edges,
        plan=ClusteringPlan(cannot_link_same_dataset=True),
        min_datasets=3,
    )
    assert result.source_collision_count == 0
    # a1 and b1 are clustered together; a2 remains in its own cluster
    cluster_a1 = result.record_to_cluster["a1"]
    cluster_b1 = result.record_to_cluster["b1"]
    cluster_a2 = result.record_to_cluster["a2"]
    assert cluster_a1 == cluster_b1
    assert cluster_a2 != cluster_a1


def test_multisource_resolver_explicit_cannot_link_constraint() -> None:
    datasets = {
        "source_a": ("a1",),
        "source_b": ("b1",),
        "source_c": ("c1",),
    }

    edges = [
        CandidateEdge("a1", "source_a", "b1", "source_b", 0.95, pair_digest("a1", "b1")),
        CandidateEdge("b1", "source_b", "c1", "source_c", 0.92, pair_digest("b1", "c1")),
    ]

    # Explicit cannot-link constraint between a1 and c1
    cannot_link = [("a1", "c1")]

    result = MultiSourceEntityResolver.resolve(
        datasets=datasets,
        candidate_edges=edges,
        cannot_link_pairs=cannot_link,
        plan=ClusteringPlan(algorithm="constrained_agglomerative"),
        min_datasets=3,
    )

    # (a1, b1) merged first. Then (b1, c1) cannot merge because {a1, b1} has a1 (cannot link c1)
    cluster_a1 = result.record_to_cluster["a1"]
    cluster_b1 = result.record_to_cluster["b1"]
    cluster_c1 = result.record_to_cluster["c1"]
    assert cluster_a1 == cluster_b1
    assert cluster_c1 != cluster_a1
    assert result.cannot_link_violations == 0


def test_solvers_variety() -> None:
    datasets = {
        "source_a": ("a1", "a2"),
        "source_b": ("b1", "b2"),
        "source_c": ("c1", "c2"),
    }
    edges = [
        CandidateEdge("a1", "source_a", "b1", "source_b", 0.95, pair_digest("a1", "b1")),
        CandidateEdge("b1", "source_b", "c1", "source_c", 0.92, pair_digest("b1", "c1")),
        CandidateEdge("a1", "source_a", "c1", "source_c", 0.90, pair_digest("a1", "c1")),
    ]

    algos: tuple[
        Literal["correlation_clustering", "constrained_agglomerative", "connected_components"], ...
    ] = (
        "correlation_clustering",
        "constrained_agglomerative",
        "connected_components",
    )
    for algo in algos:
        res = MultiSourceEntityResolver.resolve(
            datasets=datasets,
            candidate_edges=edges,
            plan=ClusteringPlan(algorithm=algo, threshold=0.50),
            min_datasets=3,
        )
        assert res.total_records == 6
        # All algorithms successfully form {a1, b1, c1} for the full triangle
        assert res.max_cluster_size == 3


def test_assignment_batch_ingestion() -> None:
    try:
        import numpy as np

        from mapel_linkage.assignment.contracts import AssignmentEdgeBatch
    except ImportError:
        return

    datasets = {
        "ds_a": ("a1", "a2"),
        "ds_b": ("b1", "b2"),
        "ds_c": ("c1", "c2"),
    }

    pairs_ab = (("a1", "b1"),)
    batch_ab = AssignmentEdgeBatch(
        source_record_keys=("a1", "a2"),
        pair_references=pairs_ab,
        pair_digests=tuple(pair_digest(left_k, right_k) for left_k, right_k in pairs_ab),
        probabilities=np.asarray([0.95], dtype=np.float64),
        candidate_ranks=np.asarray([1], dtype=np.int64),
        source_model_id="m1",
        source_model_version="v1",
        calibrator_digest=hashlib.sha256(b"cal1").hexdigest(),
        ranking_model_digest=None,
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )

    pairs_bc = (("b1", "c1"),)
    batch_bc = AssignmentEdgeBatch(
        source_record_keys=("b1", "b2"),
        pair_references=pairs_bc,
        pair_digests=tuple(pair_digest(left_k, right_k) for left_k, right_k in pairs_bc),
        probabilities=np.asarray([0.90], dtype=np.float64),
        candidate_ranks=np.asarray([1], dtype=np.int64),
        source_model_id="m1",
        source_model_version="v1",
        calibrator_digest=hashlib.sha256(b"cal2").hexdigest(),
        ranking_model_digest=None,
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )

    result = MultiSourceEntityResolver.resolve(
        datasets=datasets,
        candidate_batches={
            ("ds_a", "ds_b"): batch_ab,
            ("ds_b", "ds_c"): batch_bc,
        },
        plan=ClusteringPlan(algorithm="constrained_agglomerative"),
        min_datasets=3,
    )

    assert result.total_records == 6
    assert result.max_cluster_size == 3  # {a1, b1, c1}


def test_resolver_validation_errors() -> None:
    # Insufficient dataset count
    with pytest.raises(ClusteringError, match="ML-CLUSTER-001"):
        MultiSourceEntityResolver.resolve(
            datasets={"ds1": ("r1",)},
            min_datasets=3,
        )

    # Empty datasets
    with pytest.raises(ClusteringError, match="ML-CLUSTER-001"):
        MultiSourceEntityResolver.resolve(
            datasets={},
        )

    # Candidate edge references unknown node
    with pytest.raises(ClusteringError, match="ML-CLUSTER-002"):
        MultiSourceEntityResolver.resolve(
            datasets={"ds1": ("r1",), "ds2": ("r2",), "ds3": ("r3",)},
            candidate_edges=[
                CandidateEdge("r1", "ds1", "unknown_rec", "ds2", 0.9, pair_digest("r1", "unknown"))
            ],
            min_datasets=3,
        )
