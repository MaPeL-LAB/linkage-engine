"""Deduplication workflow runner supporting single-source deduplication and link-and-dedupe."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mapel_linkage.assignment.contracts import AssignmentEdgeBatch, AssignmentPlan
from mapel_linkage.assignment.deduplication import (
    DeduplicationPlan,
    DeduplicationResult,
    DuplicateCluster,
    IntraSourceDeduplicator,
    LinkAndDedupeResolver,
    LinkAndDedupeResult,
)
from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.paths import PathPolicy

# Explicit solver aliases as required by the milestone specification
SingleSourceDeduplicationSolver = IntraSourceDeduplicator
LinkAndDedupeSolver = LinkAndDedupeResolver


@dataclass(frozen=True, slots=True, repr=False)
class DeduplicationWorkflowResult:
    """Outcome of an orchestrated deduplication workflow run."""

    mode: Literal["dedupe_only", "link_and_dedupe"]
    deduplication_result: DeduplicationResult | None = None
    link_and_dedupe_result: LinkAndDedupeResult | None = None
    manifest_path: Path | None = field(default=None, repr=False)
    workflow_digest: str = ""

    def __post_init__(self) -> None:
        if self.mode == "dedupe_only" and self.deduplication_result is None:
            raise PipelineError(
                "ML-PIPE-030", "Dedupe-only workflow result requires DeduplicationResult."
            )
        if self.mode == "link_and_dedupe" and self.link_and_dedupe_result is None:
            raise PipelineError(
                "ML-PIPE-031", "Link-and-dedupe workflow result requires LinkAndDedupeResult."
            )
        if not self.workflow_digest:
            digest = (
                self.deduplication_result.deduplication_digest
                if self.deduplication_result is not None
                else self.link_and_dedupe_result.link_and_dedupe_digest  # type: ignore[union-attr]
            )
            object.__setattr__(self, "workflow_digest", digest)

    @property
    def total_records(self) -> int:
        if self.deduplication_result is not None:
            return self.deduplication_result.total_records
        assert self.link_and_dedupe_result is not None
        return (
            self.link_and_dedupe_result.source_a_record_count
            + self.link_and_dedupe_result.source_b_record_count
        )

    @property
    def total_clusters(self) -> int:
        if self.deduplication_result is not None:
            return self.deduplication_result.total_clusters
        assert self.link_and_dedupe_result is not None
        return (
            self.link_and_dedupe_result.source_a_deduplication.total_clusters
            + self.link_and_dedupe_result.source_b_deduplication.total_clusters
        )

    @property
    def singleton_count(self) -> int:
        if self.deduplication_result is not None:
            return self.deduplication_result.singleton_count
        assert self.link_and_dedupe_result is not None
        return (
            self.link_and_dedupe_result.source_a_deduplication.singleton_count
            + self.link_and_dedupe_result.source_b_deduplication.singleton_count
        )

    @property
    def duplicate_record_count(self) -> int:
        if self.deduplication_result is not None:
            return self.deduplication_result.duplicate_record_count
        assert self.link_and_dedupe_result is not None
        return (
            self.link_and_dedupe_result.source_a_deduplication.duplicate_record_count
            + self.link_and_dedupe_result.source_b_deduplication.duplicate_record_count
        )

    @property
    def max_cluster_size(self) -> int:
        if self.deduplication_result is not None:
            return self.deduplication_result.max_cluster_size
        assert self.link_and_dedupe_result is not None
        return max(
            self.link_and_dedupe_result.source_a_deduplication.max_cluster_size,
            self.link_and_dedupe_result.source_b_deduplication.max_cluster_size,
        )

    def safe_summary(self) -> dict[str, Any]:
        """Return aggregate summary without row-level keys."""
        if self.mode == "dedupe_only" and self.deduplication_result is not None:
            summary: dict[str, Any] = dict(self.deduplication_result.safe_summary())
            summary["mode"] = self.mode
            summary["workflow_digest"] = self.workflow_digest
            summary["manifest_written"] = self.manifest_path is not None
            return summary
        elif self.link_and_dedupe_result is not None:
            summary_ld: dict[str, Any] = dict(self.link_and_dedupe_result.safe_summary())
            summary_ld["mode"] = self.mode
            summary_ld["workflow_digest"] = self.workflow_digest
            summary_ld["manifest_written"] = self.manifest_path is not None
            return summary_ld
        return {"mode": self.mode, "workflow_digest": self.workflow_digest}


class DeduplicationWorkflowRunner:
    """Orchestrates single-source deduplication and combined link-and-dedupe workflows."""

    @classmethod
    def run_dedupe_only(
        cls,
        *,
        record_keys: tuple[str, ...],
        candidate_batch: AssignmentEdgeBatch | None = None,
        pair_references: tuple[tuple[str, str], ...] | None = None,
        probabilities: Sequence[float] | None = None,
        pair_digests: tuple[str, ...] | None = None,
        plan: DeduplicationPlan | None = None,
        dataset_id: str | None = None,
        output_manifest_path: str | Path | None = None,
        policy: PathPolicy | None = None,
    ) -> DeduplicationWorkflowResult:
        """Run single-source deduplication using SingleSourceDeduplicationSolver."""
        dedupe_plan = plan or DeduplicationPlan()

        if candidate_batch is not None:
            result = SingleSourceDeduplicationSolver.cluster_batch(
                candidate_batch,
                plan=dedupe_plan,
                dataset_id=dataset_id,
            )
        elif pair_references is not None and probabilities is not None:
            result = SingleSourceDeduplicationSolver.cluster(
                record_keys=record_keys,
                pair_references=pair_references,
                probabilities=list(probabilities),
                pair_digests=pair_digests,
                plan=dedupe_plan,
                dataset_id=dataset_id,
                mode="dedupe_only",
            )
        else:
            result = SingleSourceDeduplicationSolver.cluster(
                record_keys=record_keys,
                pair_references=(),
                probabilities=[],
                plan=dedupe_plan,
                dataset_id=dataset_id,
                mode="dedupe_only",
            )

        manifest_file: Path | None = None
        if output_manifest_path is not None:
            manifest_file = cls.export_cluster_manifest(
                result=result,
                output_path=output_manifest_path,
                policy=policy,
            )

        return DeduplicationWorkflowResult(
            mode="dedupe_only",
            deduplication_result=result,
            manifest_path=manifest_file,
            workflow_digest=result.deduplication_digest,
        )

    @classmethod
    def run_link_and_dedupe(
        cls,
        *,
        source_a_keys: tuple[str, ...],
        source_b_keys: tuple[str, ...],
        cross_candidates: AssignmentEdgeBatch,
        cross_plan: AssignmentPlan | None = None,
        intra_a_candidates: AssignmentEdgeBatch | None = None,
        intra_b_candidates: AssignmentEdgeBatch | None = None,
        deduplication_plan: DeduplicationPlan | None = None,
        dataset_a_id: str = "source_a",
        dataset_b_id: str = "source_b",
        output_manifest_path: str | Path | None = None,
        policy: PathPolicy | None = None,
    ) -> DeduplicationWorkflowResult:
        """Run combined two-source linkage and intra-source deduplication."""
        result = LinkAndDedupeSolver.resolve(
            source_a_keys=source_a_keys,
            source_b_keys=source_b_keys,
            cross_candidates=cross_candidates,
            cross_plan=cross_plan,
            intra_a_candidates=intra_a_candidates,
            intra_b_candidates=intra_b_candidates,
            deduplication_plan=deduplication_plan,
            dataset_a_id=dataset_a_id,
            dataset_b_id=dataset_b_id,
        )

        manifest_file: Path | None = None
        if output_manifest_path is not None:
            manifest_file = cls.export_link_and_dedupe_manifest(
                result=result,
                output_path=output_manifest_path,
                policy=policy,
            )

        return DeduplicationWorkflowResult(
            mode="link_and_dedupe",
            link_and_dedupe_result=result,
            manifest_path=manifest_file,
            workflow_digest=result.link_and_dedupe_digest,
        )

    @staticmethod
    def export_cluster_manifest(
        *,
        result: DeduplicationResult,
        output_path: str | Path,
        policy: PathPolicy | None = None,
    ) -> Path:
        """Write cluster assignments to CSV or JSON."""
        dest = policy.resolve_output(str(output_path)) if policy is not None else Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        for cluster in result.clusters:
            for member_key in cluster.member_record_keys:
                rows.append(
                    {
                        "record_key": member_key,
                        "cluster_id": cluster.cluster_id,
                        "canonical_record_key": cluster.canonical_record_key,
                        "is_singleton": cluster.is_singleton,
                        "cluster_size": len(cluster.member_record_keys),
                        "mean_probability": cluster.mean_probability,
                        "dataset_id": cluster.dataset_id or "default",
                    }
                )

        rows.sort(key=lambda r: (r["cluster_id"], r["record_key"]))

        if dest.suffix.lower() == ".json":
            atomic_write_text(dest, json.dumps(rows, indent=2, sort_keys=True) + "\n")
        else:
            with dest.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "record_key",
                        "cluster_id",
                        "canonical_record_key",
                        "is_singleton",
                        "cluster_size",
                        "mean_probability",
                        "dataset_id",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)
        return dest

    @staticmethod
    def export_link_and_dedupe_manifest(
        *,
        result: LinkAndDedupeResult,
        output_path: str | Path,
        policy: PathPolicy | None = None,
    ) -> Path:
        """Write combined link-and-dedupe cluster and edge assignments to CSV or JSON."""
        dest = policy.resolve_output(str(output_path)) if policy is not None else Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        resolved_rows: list[dict[str, Any]] = [
            {"source_a_record_key": a, "source_b_record_key": b}
            for a, b in result.resolved_record_pairs
        ]
        payload = {
            "mode": "link_and_dedupe",
            "cluster_count": result.source_a_cluster_count + result.source_b_cluster_count,
            "resolved_pair_count": len(result.resolved_record_pairs),
            "resolved_record_pairs": resolved_rows,
        }

        if dest.suffix.lower() == ".json":
            atomic_write_text(dest, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            with dest.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["source_a_record_key", "source_b_record_key"],
                )
                writer.writeheader()
                writer.writerows(resolved_rows)
        return dest


__all__ = [
    "DeduplicationWorkflowResult",
    "DeduplicationWorkflowRunner",
    "DuplicateCluster",
    "LinkAndDedupeSolver",
    "SingleSourceDeduplicationSolver",
]
