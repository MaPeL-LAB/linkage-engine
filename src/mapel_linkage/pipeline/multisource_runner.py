"""Multi-source N-dataset entity resolution workflow runner with crosswalk and evaluation export."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mapel_linkage.assignment.contracts import AssignmentEdgeBatch
from mapel_linkage.clustering.contracts import (
    CandidateEdge,
    ClusteringPlan,
    MultiSourceResolutionResult,
)
from mapel_linkage.clustering.metrics import (
    MultiSourceEvaluationReport,
    evaluate_multisource_clustering,
)
from mapel_linkage.clustering.resolver import MultiSourceEntityResolver
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.paths import PathPolicy


@dataclass(frozen=True, slots=True, repr=False)
class MultiSourceWorkflowResult:
    """Outcome of an orchestrated multi-source entity resolution workflow run."""

    resolution_result: MultiSourceResolutionResult
    evaluation_report: MultiSourceEvaluationReport | None = None
    crosswalk_path: Path | None = field(default=None, repr=False)
    evaluation_report_path: Path | None = field(default=None, repr=False)
    workflow_digest: str = ""

    def __post_init__(self) -> None:
        if not self.workflow_digest:
            object.__setattr__(self, "workflow_digest", self.resolution_result.resolution_digest)

    @property
    def total_records(self) -> int:
        return self.resolution_result.total_records

    @property
    def total_clusters(self) -> int:
        return self.resolution_result.total_clusters

    @property
    def singleton_count(self) -> int:
        return self.resolution_result.singleton_count

    @property
    def multi_record_cluster_count(self) -> int:
        return self.resolution_result.multi_record_cluster_count

    def safe_summary(self) -> dict[str, Any]:
        """Return aggregate summary without row-level keys."""
        summary = self.resolution_result.safe_summary()
        summary["crosswalk_written"] = self.crosswalk_path is not None
        summary["evaluation_available"] = self.evaluation_report is not None
        if self.evaluation_report is not None:
            eval_summary = self.evaluation_report.safe_summary()
            summary["bcubed_f1"] = eval_summary["bcubed_f1"]
            summary["cluster_purity"] = eval_summary["cluster_purity"]
            summary["cannot_link_violations"] = eval_summary["cannot_link_violations"]
        return summary


class MultiSourceWorkflowRunner:
    """Orchestrates N >= 3 dataset entity resolution, graph clustering, and evaluation."""

    @classmethod
    def run(
        cls,
        *,
        datasets: Mapping[str, Iterable[str]],
        candidate_batches: Sequence[AssignmentEdgeBatch]
        | Mapping[tuple[str, str], AssignmentEdgeBatch]
        | None = None,
        candidate_edges: Iterable[CandidateEdge] | None = None,
        edges: Mapping[tuple[str, str], float] | Iterable[tuple[str, str, float]] | None = None,
        cannot_link_pairs: Iterable[tuple[str, str]] | None = None,
        cannot_link: Iterable[tuple[str, str]] | None = None,
        plan: ClusteringPlan | None = None,
        min_datasets: int = 1,
        true_clusters: Mapping[str, str]
        | Mapping[str, Iterable[str]]
        | Sequence[Iterable[str]]
        | None = None,
        must_link_pairs: Iterable[tuple[str, str]] = (),
        output_crosswalk_path: str | Path | None = None,
        output_report_path: str | Path | None = None,
        policy: PathPolicy | None = None,
    ) -> MultiSourceWorkflowResult:
        """Resolve multi-source entities, generate crosswalk, and compute BCubed evaluation."""
        clustering_plan = plan or ClusteringPlan()

        # 1. Resolve multi-source graph entities
        resolution_result = MultiSourceEntityResolver.resolve(
            datasets=datasets,
            candidate_batches=candidate_batches,
            candidate_edges=candidate_edges,
            edges=edges,
            cannot_link_pairs=cannot_link_pairs,
            cannot_link=cannot_link,
            plan=clustering_plan,
            min_datasets=min_datasets,
        )

        # 2. Export global crosswalk if requested
        crosswalk_file: Path | None = None
        if output_crosswalk_path is not None:
            crosswalk_file = cls.export_crosswalk(
                result=resolution_result,
                output_path=output_crosswalk_path,
                policy=policy,
            )

        # 3. Evaluate against ground truth if provided
        evaluation_report: MultiSourceEvaluationReport | None = None
        report_file: Path | None = None

        if true_clusters is not None:
            # Map record to predicted cluster
            pred_mapping = resolution_result.record_to_cluster
            cl_list = list(cannot_link_pairs or ()) + list(cannot_link or ())
            rec_to_ds = resolution_result.graph.nodes

            evaluation_report = evaluate_multisource_clustering(
                true_clusters=true_clusters,
                predicted_clusters=pred_mapping,
                cannot_link_pairs=cl_list,
                must_link_pairs=must_link_pairs,
                record_to_dataset=rec_to_ds,
            )

            if output_report_path is not None:
                report_file = cls.export_evaluation_report(
                    report=evaluation_report,
                    output_path=output_report_path,
                    policy=policy,
                )

        return MultiSourceWorkflowResult(
            resolution_result=resolution_result,
            evaluation_report=evaluation_report,
            crosswalk_path=crosswalk_file,
            evaluation_report_path=report_file,
            workflow_digest=resolution_result.resolution_digest,
        )

    @staticmethod
    def export_crosswalk(
        *,
        result: MultiSourceResolutionResult,
        output_path: str | Path,
        policy: PathPolicy | None = None,
    ) -> Path:
        """Write global crosswalk table to CSV or JSON."""
        dest = policy.resolve_output(str(output_path)) if policy is not None else Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        MultiSourceEntityResolver.export_crosswalk(result, destination_path=dest)
        return dest

    @staticmethod
    def export_evaluation_report(
        *,
        report: MultiSourceEvaluationReport,
        output_path: str | Path,
        policy: PathPolicy | None = None,
    ) -> Path:
        """Write evaluation report to JSON."""
        dest = policy.resolve_output(str(output_path)) if policy is not None else Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload = report.safe_summary()
        atomic_write_text(dest, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return dest


__all__ = [
    "MultiSourceWorkflowResult",
    "MultiSourceWorkflowRunner",
]
