"""Public benchmarking tools, contracts, runner, and seed corpus generation."""

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
from mapel_linkage.benchmarking.seed_corpus import (
    generate_and_run_seed_corpus,
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
    "generate_and_run_seed_corpus",
]
