from __future__ import annotations

from pathlib import Path

import numpy as np

from mapel_linkage.assignment.contracts import (
    AssignmentEdgeBatch,
    pair_digest,
)
from mapel_linkage.assignment.deduplication import DeduplicationPlan
from mapel_linkage.pipeline.deduplication_runner import (
    DeduplicationWorkflowRunner,
    LinkAndDedupeSolver,
    SingleSourceDeduplicationSolver,
)


def test_deduplication_runner_single_source_connected_components(tmp_path: Path) -> None:
    record_keys = ("rec_1", "rec_2", "rec_3", "rec_4")
    pair_references = (("rec_1", "rec_2"), ("rec_2", "rec_3"))
    probabilities = [0.95, 0.90]

    plan = DeduplicationPlan(algorithm="connected_components", threshold=0.5)
    manifest_path = tmp_path / "clusters.json"

    result = DeduplicationWorkflowRunner.run_dedupe_only(
        record_keys=record_keys,
        pair_references=pair_references,
        probabilities=probabilities,
        plan=plan,
        dataset_id="test_source",
        output_manifest_path=manifest_path,
    )

    assert result.mode == "dedupe_only"
    assert result.deduplication_result is not None
    assert result.total_records == 4
    assert result.total_clusters == 2
    assert result.singleton_count == 1
    assert result.duplicate_record_count == 3
    assert result.max_cluster_size == 3
    assert manifest_path.is_file()

    summary = result.safe_summary()
    assert summary["mode"] == "dedupe_only"
    assert summary["total_records"] == 4
    assert summary["total_clusters"] == 2
    assert summary["manifest_written"] is True


def test_deduplication_runner_single_source_clique(tmp_path: Path) -> None:
    record_keys = ("r1", "r2", "r3", "r4")
    pair_references = (("r1", "r2"), ("r2", "r3"))
    probabilities = [0.9, 0.85]

    plan = DeduplicationPlan(algorithm="clique", threshold=0.5)
    manifest_path = tmp_path / "clusters.csv"

    result = DeduplicationWorkflowRunner.run_dedupe_only(
        record_keys=record_keys,
        pair_references=pair_references,
        probabilities=probabilities,
        plan=plan,
        dataset_id="ds1",
        output_manifest_path=manifest_path,
    )

    assert result.mode == "dedupe_only"
    assert result.deduplication_result is not None
    assert result.total_records == 4
    assert manifest_path.is_file()


def test_deduplication_runner_link_and_dedupe(tmp_path: Path) -> None:
    source_a_keys = ("a1", "a2", "a3")
    source_b_keys = ("b1", "b2", "b3")

    cross_pairs = (("a1", "b1"), ("a2", "b2"))
    cross_probs = np.array([0.92, 0.88], dtype=np.float64)
    cross_ranks = np.array([1, 1], dtype=np.int64)

    cross_batch = AssignmentEdgeBatch(
        source_record_keys=source_a_keys,
        pair_references=cross_pairs,
        pair_digests=tuple(pair_digest(p[0], p[1]) for p in cross_pairs),
        probabilities=cross_probs,
        candidate_ranks=cross_ranks,
        source_model_id="xgb_model",
        source_model_version="v1",
        calibrator_digest="0" * 64,
        ranking_model_digest=None,
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )

    manifest_path = tmp_path / "link_dedupe_manifest.json"

    result = DeduplicationWorkflowRunner.run_link_and_dedupe(
        source_a_keys=source_a_keys,
        source_b_keys=source_b_keys,
        cross_candidates=cross_batch,
        output_manifest_path=manifest_path,
    )

    assert result.mode == "link_and_dedupe"
    assert result.link_and_dedupe_result is not None
    assert result.link_and_dedupe_result.source_a_record_count == 3
    assert result.link_and_dedupe_result.source_b_record_count == 3
    assert result.link_and_dedupe_result.assigned_cluster_pair_count == 2
    assert manifest_path.is_file()

    summary = result.safe_summary()
    assert summary["mode"] == "link_and_dedupe"
    assert summary["assigned_cluster_pair_count"] == 2
    assert summary["manifest_written"] is True


def test_solver_aliases_are_present() -> None:
    assert SingleSourceDeduplicationSolver is not None
    assert LinkAndDedupeSolver is not None
