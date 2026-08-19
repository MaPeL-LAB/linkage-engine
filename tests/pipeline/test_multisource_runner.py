from __future__ import annotations

import json
from pathlib import Path

from mapel_linkage.clustering.contracts import CandidateEdge, ClusteringPlan
from mapel_linkage.pipeline.multisource_runner import (
    MultiSourceWorkflowRunner,
)


def test_multisource_workflow_runner_3_datasets(tmp_path: Path) -> None:
    datasets = {
        "ds_a": ("a1", "a2", "a3"),
        "ds_b": ("b1", "b2", "b3"),
        "ds_c": ("c1", "c2", "c3"),
    }

    candidate_edges = [
        CandidateEdge(
            source_record_key="a1",
            source_dataset_id="ds_a",
            target_record_key="b1",
            target_dataset_id="ds_b",
            probability=0.95,
        ),
        CandidateEdge(
            source_record_key="b1",
            source_dataset_id="ds_b",
            target_record_key="c1",
            target_dataset_id="ds_c",
            probability=0.92,
        ),
        CandidateEdge(
            source_record_key="a2",
            source_dataset_id="ds_a",
            target_record_key="b2",
            target_dataset_id="ds_b",
            probability=0.88,
        ),
    ]

    cannot_link = [("a1", "a2"), ("b1", "b2")]

    crosswalk_dest = tmp_path / "global_crosswalk.csv"
    report_dest = tmp_path / "eval_report.json"

    # Ground truth mapping for evaluation
    true_clusters = {
        "entity_1": ("a1", "b1", "c1"),
        "entity_2": ("a2", "b2"),
        "entity_3": ("a3",),
        "entity_4": ("b3",),
        "entity_5": ("c2",),
        "entity_6": ("c3",),
    }

    result = MultiSourceWorkflowRunner.run(
        datasets=datasets,
        candidate_edges=candidate_edges,
        cannot_link=cannot_link,
        plan=ClusteringPlan(algorithm="correlation_clustering", threshold=0.5),
        min_datasets=3,
        true_clusters=true_clusters,
        output_crosswalk_path=crosswalk_dest,
        output_report_path=report_dest,
    )

    assert result.total_records == 9
    assert result.total_clusters >= 2
    assert result.crosswalk_path is not None and crosswalk_dest.is_file()
    assert result.evaluation_report is not None and report_dest.is_file()
    assert result.evaluation_report.bcubed_f1 > 0.0
    assert result.evaluation_report.cluster_purity > 0.0

    summary = result.safe_summary()
    assert summary["total_records"] == 9
    assert summary["crosswalk_written"] is True
    assert summary["evaluation_available"] is True
    assert "bcubed_f1" in summary


def test_multisource_workflow_runner_connected_components(tmp_path: Path) -> None:
    datasets = {
        "src1": ("s1", "s2"),
        "src2": ("t1", "t2"),
        "src3": ("u1", "u2"),
    }

    edges = {
        ("s1", "t1"): 0.9,
        ("t1", "u1"): 0.85,
    }

    crosswalk_dest = tmp_path / "crosswalk.json"

    result = MultiSourceWorkflowRunner.run(
        datasets=datasets,
        edges=edges,
        plan=ClusteringPlan(algorithm="connected_components", threshold=0.5),
        min_datasets=3,
        output_crosswalk_path=crosswalk_dest,
    )

    assert result.total_records == 6
    assert crosswalk_dest.is_file()
    entries = json.loads(crosswalk_dest.read_text(encoding="utf-8"))
    assert len(entries) == 6
