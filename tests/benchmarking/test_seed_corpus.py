from __future__ import annotations

from pathlib import Path

from mapel_linkage.benchmarking.contracts import (
    BenchmarkRunStatus,
)
from mapel_linkage.benchmarking.generator import (
    BenchmarkScenarioGenerator,
)
from mapel_linkage.benchmarking.runner import (
    BenchmarkPortfolioRunner,
)
from mapel_linkage.benchmarking.seed_corpus import (
    generate_and_run_seed_corpus,
)


def test_generate_and_run_seed_corpus_execution(tmp_path: Path) -> None:
    gen = BenchmarkScenarioGenerator()
    runner = BenchmarkPortfolioRunner()

    # Run on a subset for fast unit test execution
    families = ("family.typo_stress", "family.missingness_regime")
    instances = ("instance.typo_low", "instance.missing_zero")

    registry = generate_and_run_seed_corpus(
        registry_directory=tmp_path,
        generator=gen,
        runner=runner,
        families=families,
        instances=instances,
        replicates=1,
        base_seed=20260816,
    )

    # 1. Manifests persisted
    saved_fams = registry.list_families()
    assert len(saved_fams) == 2
    assert {f.family_id for f in saved_fams} == set(families)

    saved_insts = registry.list_instances()
    assert len(saved_insts) == 2
    assert {i.instance_id for i in saved_insts} == set(instances)

    # 2. Run records persisted
    records = registry.list_run_records()
    assert len(records) > 0

    # 3. Metrics and failures
    success_runs = [r for r in records if r.status == BenchmarkRunStatus.SUCCESS]
    assert len(success_runs) > 0
    for r in success_runs:
        m = registry.load_metrics(r.run_id)
        assert m is not None
        assert 0.0 <= m.candidate_recall <= 1.0
        assert 0.0 <= m.positive_predictive_value <= 1.0
        assert 0.0 <= m.brier_score <= 1.0

    # 4. Snapshot and Coverage Report
    snapshot = registry.build_snapshot("snapshot.seed_corpus")
    assert snapshot.registry_digest is not None
    assert len(snapshot.records) == len(records)

    report = registry.generate_coverage_report(
        snapshot_id="snapshot.seed_corpus",
        report_id="coverage_report_seed_corpus",
    )
    assert report.family_count == 2
    assert report.instance_count == 2
    assert report.run_count == len(records)
