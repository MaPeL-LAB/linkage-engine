"""Canonical seed corpus generation and execution for Linkage Engine synthetic benchmarks."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from mapel_linkage.benchmarking.generator import (
    BenchmarkScenarioGenerator,
)
from mapel_linkage.benchmarking.registry import (
    BenchmarkRegistry,
)
from mapel_linkage.benchmarking.runner import (
    BenchmarkPortfolioRunner,
    BenchmarkRecipe,
)


def generate_and_run_seed_corpus(
    registry_directory: Path | str,
    *,
    generator: BenchmarkScenarioGenerator | None = None,
    runner: BenchmarkPortfolioRunner | None = None,
    families: Iterable[str] | None = None,
    instances: Iterable[str] | None = None,
    recipes: Iterable[BenchmarkRecipe] | None = None,
    replicates: int = 1,
    base_seed: int = 20260816,
    snapshot_id: str = "snapshot.seed_corpus",
    report_id: str = "coverage_report_seed_corpus",
) -> BenchmarkRegistry:
    """Generate baseline scenario families, execute recipes, and persist registry records."""
    reg_path = Path(registry_directory)
    registry = BenchmarkRegistry(reg_path)
    gen = generator or BenchmarkScenarioGenerator()
    run_engine = runner or BenchmarkPortfolioRunner()

    all_families = gen.list_families()
    target_family_ids = (
        set(families) if families is not None else {f.family_id for f in all_families}
    )
    for fam in all_families:
        if fam.family_id in target_family_ids:
            registry.save_family(fam)

    all_instances = gen.list_instances()
    target_instance_ids = set(instances) if instances is not None else None
    for inst in all_instances:
        if inst.family_id in target_family_ids and (
            target_instance_ids is None or inst.instance_id in target_instance_ids
        ):
            registry.save_instance(inst)

    run_results = run_engine.run_portfolio(
        gen,
        families=families,
        instances=instances,
        recipes=recipes,
        replicates=replicates,
        base_seed=base_seed,
    )

    for res in run_results:
        registry.save_run_record(
            record=res.record,
            metrics=res.metrics,
            failure=res.failure,
        )

    registry.build_snapshot(snapshot_id=snapshot_id)
    registry.generate_coverage_report(snapshot_id=snapshot_id, report_id=report_id)

    return registry


__all__ = [
    "generate_and_run_seed_corpus",
]
