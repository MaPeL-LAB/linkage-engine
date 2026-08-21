"""Public benchmarking tools, contracts, runner, and seed corpus generation."""

from mapel_linkage.benchmarking.advisor_catalogue import (
    AdvisorCorpusDesignManifest,
    AdvisorCorpusReadinessManifest,
    BenchmarkShard,
    BenchmarkShardPlan,
    build_advisor_corpus_design,
    build_advisor_corpus_readiness,
    build_advisor_v2_generator,
    build_benchmark_shard_plan,
)
from mapel_linkage.benchmarking.advisor_execution import (
    CorpusExecutionApproval,
    CorpusShardExecutionReport,
    audit_advisor_corpus,
    execute_advisor_corpus_shard,
)
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
    "AdvisorCorpusDesignManifest",
    "AdvisorCorpusReadinessManifest",
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
    "BenchmarkShard",
    "BenchmarkShardPlan",
    "CorpusExecutionApproval",
    "CorpusShardExecutionReport",
    "CoverageSummaryReport",
    "ScenarioFamilyManifest",
    "ScenarioInstanceManifest",
    "ScenarioLatentSpec",
    "audit_advisor_corpus",
    "build_advisor_corpus_design",
    "build_advisor_corpus_readiness",
    "build_advisor_v2_generator",
    "build_benchmark_shard_plan",
    "build_registry_snapshot",
    "execute_advisor_corpus_shard",
    "generate_and_run_seed_corpus",
]
