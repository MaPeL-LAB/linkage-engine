"""Deterministic construction of aggregate benchmark-registry snapshots."""

from __future__ import annotations

from collections.abc import Iterable

from mapel_linkage.benchmarking.contracts import (
    BenchmarkEvidenceScope,
    BenchmarkRegistrySnapshot,
    BenchmarkRunRecord,
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


__all__ = ["build_registry_snapshot"]
