"""Public benchmarking tools, contracts, and registry helpers."""

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
from mapel_linkage.benchmarking.generator import (
    BenchmarkScenarioBundle,
    BenchmarkScenarioGenerator,
    ScenarioLatentSpec,
)
from mapel_linkage.benchmarking.registry import (
    BenchmarkRegistry,
    build_registry_snapshot,
)
from mapel_linkage.benchmarking.runner import (
    BenchmarkPortfolioRunner,
    BenchmarkRecipe,
    BenchmarkRunResult,
)

__all__ = [
    "BenchmarkAggregateMetrics",
    "BenchmarkEvidenceScope",
    "BenchmarkFailureRecord",
    "BenchmarkPortfolioRunner",
    "BenchmarkRecipe",
    "BenchmarkRegistry",
    "BenchmarkRegistrySnapshot",
    "BenchmarkRunRecord",
    "BenchmarkRunResult",
    "BenchmarkRunStatus",
    "BenchmarkScenarioBundle",
    "BenchmarkScenarioGenerator",
    "CoverageSummaryReport",
    "ScenarioFamilyManifest",
    "ScenarioInstanceManifest",
    "ScenarioLatentSpec",
    "build_registry_snapshot",
]
