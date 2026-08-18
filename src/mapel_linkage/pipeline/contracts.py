"""Synthetic vertical-slice pipeline contracts and safe stage summaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class StageSummary:
    """Aggregate-only status for one package-owned pipeline stage."""

    stage: str
    status: str
    counts: Mapping[str, int] = field(default_factory=dict)
    digests: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(self, "digests", MappingProxyType(dict(self.digests)))

    def safe_summary(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "status": self.status,
            "counts": dict(self.counts),
            "digests": dict(self.digests),
        }


@dataclass(frozen=True, slots=True, repr=False)
class SyntheticVerticalSliceResult:
    """Complete synthetic run result with all row-bearing paths hidden."""

    run_id: str
    configuration_digest: str
    package_version: str
    stage_summaries: tuple[StageSummary, ...]
    relationship_status_counts: Mapping[str, int]
    selected_model_family: str
    selected_model_id: str
    calibrator_method: str
    calibrator_digest: str
    ranking_model_digest: str
    assignment_digest: str
    review_queue_count: int
    aggregate_report_path: Path = field(repr=False)
    relationship_output_path: Path = field(repr=False)
    review_queue_path: Path = field(repr=False)
    run_manifest_path: Path = field(repr=False)
    evaluation_scope: str = "synthetic_mechanical_evaluation"
    real_data_validation_status: str = "not_established"
    merge_authority: str = "none"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relationship_status_counts",
            MappingProxyType(dict(self.relationship_status_counts)),
        )

    def safe_summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "configuration_digest": self.configuration_digest,
            "package_version": self.package_version,
            "stage_count": len(self.stage_summaries),
            "relationship_status_counts": dict(self.relationship_status_counts),
            "selected_model_family": self.selected_model_family,
            "selected_model_id": self.selected_model_id,
            "calibrator_method": self.calibrator_method,
            "calibrator_digest": self.calibrator_digest,
            "ranking_model_digest": self.ranking_model_digest,
            "assignment_digest": self.assignment_digest,
            "review_queue_count": self.review_queue_count,
            "evaluation_scope": self.evaluation_scope,
            "real_data_validation_status": self.real_data_validation_status,
            "merge_authority": self.merge_authority,
        }
