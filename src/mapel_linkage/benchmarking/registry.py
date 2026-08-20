"""Deterministic construction and file-backed persistence of benchmark-registry snapshots."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mapel_linkage.benchmarking.contracts import (
    BenchmarkAggregateMetrics,
    BenchmarkEvidenceScope,
    BenchmarkFailureRecord,
    BenchmarkRegistrySnapshot,
    BenchmarkRunRecord,
    BenchmarkRunStatus,
    CoverageSummaryReport,
    ScenarioFamilyManifest,
    ScenarioInstanceManifest,
)


def build_registry_snapshot(
    *,
    snapshot_id: str,
    records: Iterable[BenchmarkRunRecord],
    evidence_scope: BenchmarkEvidenceScope = BenchmarkEvidenceScope.GLOBAL_SYNTHETIC,
) -> BenchmarkRegistrySnapshot:
    """Create a deterministic registry snapshot without row-level material."""

    ordered = tuple(sorted(records, key=lambda item: item.run_id))
    return BenchmarkRegistrySnapshot(
        snapshot_id=snapshot_id,
        records=ordered,
        evidence_scope=evidence_scope,
    )


class BenchmarkRegistry:
    """File-backed repository for benchmark manifests, run evidence, and coverage reports."""

    def __init__(self, root_directory: Path | str) -> None:
        self.root_directory = Path(root_directory)
        if self.root_directory.is_symlink():
            raise ValueError("Benchmark registry roots cannot be symbolic links.")
        if self.root_directory.exists() and not self.root_directory.is_dir():
            raise ValueError("Benchmark registry roots must be directories.")
        self.families_dir = self.root_directory / "families"
        self.instances_dir = self.root_directory / "instances"
        self.runs_dir = self.root_directory / "runs"
        self.metrics_dir = self.root_directory / "metrics"
        self.failures_dir = self.root_directory / "failures"
        self.snapshots_dir = self.root_directory / "snapshots"
        self.reports_dir = self.root_directory / "reports"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        for d in (
            self.families_dir,
            self.instances_dir,
            self.runs_dir,
            self.metrics_dir,
            self.failures_dir,
            self.snapshots_dir,
            self.reports_dir,
        ):
            if d.is_symlink():
                raise ValueError("Benchmark registry managed directories cannot be symbolic links.")
            if d.exists() and not d.is_dir():
                raise ValueError("Benchmark registry managed paths must be directories.")
            d.mkdir(parents=True, exist_ok=True)

    def save_family(self, manifest: ScenarioFamilyManifest) -> Path:
        self.families_dir.mkdir(parents=True, exist_ok=True)
        dest = self.families_dir / f"{manifest.family_id}.json"
        if dest.exists():
            existing = ScenarioFamilyManifest.model_validate_json(dest.read_text(encoding="utf-8"))
            if existing.family_digest != manifest.family_digest:
                raise FileExistsError(
                    "A different scenario-family manifest already exists for this ID."
                )
            return dest
        dest.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return dest

    def load_family(self, family_id: str) -> ScenarioFamilyManifest:
        path = self.families_dir / f"{family_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Scenario family not found: {family_id} at {path}")
        return ScenarioFamilyManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def list_families(self) -> tuple[ScenarioFamilyManifest, ...]:
        if not self.families_dir.exists():
            return ()
        manifests = []
        for file in sorted(self.families_dir.glob("*.json")):
            manifests.append(
                ScenarioFamilyManifest.model_validate_json(file.read_text(encoding="utf-8"))
            )
        return tuple(sorted(manifests, key=lambda m: m.family_id))

    def save_instance(self, manifest: ScenarioInstanceManifest) -> Path:
        self.instances_dir.mkdir(parents=True, exist_ok=True)
        dest = self.instances_dir / f"{manifest.instance_id}.json"
        if dest.exists():
            existing = ScenarioInstanceManifest.model_validate_json(
                dest.read_text(encoding="utf-8")
            )
            if existing.instance_digest != manifest.instance_digest:
                raise FileExistsError(
                    "A different scenario-instance manifest already exists for this ID."
                )
            return dest
        dest.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return dest

    def load_instance(self, instance_id: str) -> ScenarioInstanceManifest:
        path = self.instances_dir / f"{instance_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Scenario instance not found: {instance_id} at {path}")
        return ScenarioInstanceManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def list_instances(self, family_id: str | None = None) -> tuple[ScenarioInstanceManifest, ...]:
        if not self.instances_dir.exists():
            return ()
        manifests = []
        for file in sorted(self.instances_dir.glob("*.json")):
            inst = ScenarioInstanceManifest.model_validate_json(file.read_text(encoding="utf-8"))
            if family_id is None or inst.family_id == family_id:
                manifests.append(inst)
        return tuple(sorted(manifests, key=lambda m: m.instance_id))

    def save_run_record(
        self,
        record: BenchmarkRunRecord,
        metrics: BenchmarkAggregateMetrics | dict[str, Any] | None = None,
        failure: BenchmarkFailureRecord | None = None,
    ) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        dest = self.runs_dir / f"{record.run_id}.json"
        if dest.exists():
            raise FileExistsError("A benchmark run record already exists for this run ID.")
        metrics_dest = self.metrics_dir / f"{record.run_id}.json"
        failure_dest = self.failures_dir / f"{record.run_id}.json"
        if metrics is not None and metrics_dest.exists():
            raise FileExistsError("Aggregate metrics already exist for this benchmark run ID.")
        if failure is not None and failure_dest.exists():
            raise FileExistsError("Failure evidence already exists for this benchmark run ID.")
        dest.write_text(
            json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if metrics is not None:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            m_dest = self.metrics_dir / f"{record.run_id}.json"
            if isinstance(metrics, BenchmarkAggregateMetrics):
                m_payload = metrics.model_dump(mode="json")
            else:
                m_payload = metrics
            m_dest.write_text(
                json.dumps(m_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        if failure is not None:
            self.save_failure_record(failure)

        return dest

    def load_run_record(self, run_id: str) -> BenchmarkRunRecord:
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Benchmark run record not found: {run_id} at {path}")
        return BenchmarkRunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def load_metrics(self, run_id: str) -> BenchmarkAggregateMetrics | None:
        path = self.metrics_dir / f"{run_id}.json"
        if not path.exists():
            return None
        return BenchmarkAggregateMetrics.model_validate_json(path.read_text(encoding="utf-8"))

    def list_run_records(
        self, family_id: str | None = None, instance_id: str | None = None
    ) -> tuple[BenchmarkRunRecord, ...]:
        if not self.runs_dir.exists():
            return ()
        records = []
        for file in sorted(self.runs_dir.glob("*.json")):
            rec = BenchmarkRunRecord.model_validate_json(file.read_text(encoding="utf-8"))
            if family_id is not None and rec.family_id != family_id:
                continue
            if instance_id is not None and rec.instance_id != instance_id:
                continue
            records.append(rec)
        return tuple(sorted(records, key=lambda r: r.run_id))

    def save_failure_record(self, failure: BenchmarkFailureRecord) -> Path:
        self.failures_dir.mkdir(parents=True, exist_ok=True)
        dest = self.failures_dir / f"{failure.run_id}.json"
        if dest.exists():
            raise FileExistsError("Failure evidence already exists for this benchmark run ID.")
        dest.write_text(
            json.dumps(failure.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return dest

    def load_failure_record(self, run_id: str) -> BenchmarkFailureRecord | None:
        path = self.failures_dir / f"{run_id}.json"
        if not path.exists():
            return None
        return BenchmarkFailureRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_failure_records(self) -> tuple[BenchmarkFailureRecord, ...]:
        if not self.failures_dir.exists():
            return ()
        failures = []
        for file in sorted(self.failures_dir.glob("*.json")):
            failures.append(
                BenchmarkFailureRecord.model_validate_json(file.read_text(encoding="utf-8"))
            )
        return tuple(sorted(failures, key=lambda f: f.run_id))

    def build_snapshot(
        self, snapshot_id: str = "snapshot.global_synthetic"
    ) -> BenchmarkRegistrySnapshot:
        records = self.list_run_records()
        snapshot = build_registry_snapshot(snapshot_id=snapshot_id, records=records)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        dest = self.snapshots_dir / f"{snapshot_id}.json"
        dest.write_text(
            json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return snapshot

    def generate_coverage_report(
        self,
        snapshot_id: str = "snapshot.global_synthetic",
        report_id: str = "coverage_report_v1",
    ) -> CoverageSummaryReport:
        snapshot = self.build_snapshot(snapshot_id)
        families = self.list_families()
        instances = self.list_instances()

        family_count = len(families)
        instance_count = len(instances)
        run_count = len(snapshot.records)

        replicates_set = {(record.instance_id, record.replicate_id) for record in snapshot.records}
        replicate_count = len(replicates_set)

        status_counts = {status.value: 0 for status in BenchmarkRunStatus}
        successful_count = 0
        failed_count = 0

        recipe_families: dict[str, set[str]] = defaultdict(set)
        recipe_runs: dict[str, int] = Counter()
        recipe_fails: dict[str, int] = Counter()

        # Pairwise recipe comparisons tracking: (instance_id, replicate_id) -> set of recipes
        recipes_per_replicate: dict[tuple[str, str], set[str]] = defaultdict(set)

        for rec in snapshot.records:
            status_counts[rec.status.value] += 1
            if rec.status == BenchmarkRunStatus.SUCCESS:
                successful_count += 1
            else:
                failed_count += 1
                recipe_fails[rec.pipeline_recipe_digest] += 1

            recipe_runs[rec.pipeline_recipe_digest] += 1
            recipe_families[rec.pipeline_recipe_digest].add(rec.family_id)
            recipes_per_replicate[(rec.instance_id, rec.replicate_id)].add(
                rec.pipeline_recipe_digest
            )

        pairwise_counts: Counter[str] = Counter()
        for rec_set in recipes_per_replicate.values():
            sorted_recipes = sorted(rec_set)
            for i in range(len(sorted_recipes)):
                for j in range(i + 1, len(sorted_recipes)):
                    pair_key = f"{sorted_recipes[i][:8]}--{sorted_recipes[j][:8]}"
                    pairwise_counts[pair_key] += 1

        held_out_count = sum(fam.prospectively_held_out for fam in families)

        recipe_coverage = {
            recipe_id: tuple(sorted(fams)) for recipe_id, fams in recipe_families.items()
        }
        failure_rates = {
            recipe_id: (
                recipe_fails[recipe_id] / recipe_runs[recipe_id] if recipe_runs[recipe_id] else 0.0
            )
            for recipe_id in recipe_runs
        }

        report = CoverageSummaryReport(
            report_id=report_id,
            snapshot_digest=snapshot.registry_digest,
            family_count=family_count,
            instance_count=instance_count,
            replicate_count=replicate_count,
            run_count=run_count,
            successful_run_count=successful_count,
            failed_run_count=failed_count,
            status_counts=status_counts,
            recipe_by_family_coverage=recipe_coverage,
            pairwise_comparison_counts=dict(sorted(pairwise_counts.items())),
            held_out_mechanism_count=held_out_count,
            failure_rates_by_recipe=failure_rates,
        )

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        dest = self.reports_dir / f"{report_id}.json"
        dest.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report


__all__ = [
    "BenchmarkRegistry",
    "build_registry_snapshot",
]
