from __future__ import annotations

import pytest
from pydantic import ValidationError

from mapel_linkage.benchmarking import (
    BenchmarkRegistrySnapshot,
    BenchmarkRunRecord,
    BenchmarkRunStatus,
    ScenarioFamilyManifest,
    ScenarioInstanceManifest,
    build_registry_snapshot,
)

_DIGEST = "a" * 64
_OTHER_DIGEST = "b" * 64


def _success_record(run_id: str = "run.one") -> BenchmarkRunRecord:
    return BenchmarkRunRecord(
        run_id=run_id,
        family_id="family.typo",
        instance_id="instance.low",
        replicate_id="replicate.one",
        task_profile_digest=_DIGEST,
        pipeline_recipe_digest=_OTHER_DIGEST,
        engine_commit="1" * 40,
        dependency_lock_digest=_DIGEST,
        environment_digest=_OTHER_DIGEST,
        random_seed=42,
        status=BenchmarkRunStatus.SUCCESS,
        aggregate_metrics_digest=_DIGEST,
        stage_artifact_manifest_digest=_OTHER_DIGEST,
        runtime_ms=120,
        peak_memory_mb=256,
    )


def test_scenario_family_instance_and_run_contracts_are_digest_linked() -> None:
    family = ScenarioFamilyManifest(
        family_id="family.typo",
        mechanism_tags=("character_substitution", "source_specific"),
        latent_scenario_manifest_digest=_DIGEST,
        prospectively_held_out=True,
    )
    instance = ScenarioInstanceManifest(
        family_id=family.family_id,
        instance_id="instance.low",
        family_digest=family.family_digest,
        latent_parameter_manifest_digest=_OTHER_DIGEST,
        observable_profile_digest=_DIGEST,
        planned_replicates=5,
    )
    run = _success_record()

    assert len(family.family_digest) == 64
    assert len(instance.instance_digest) == 64
    assert len(run.run_digest) == 64
    assert run.safe_summary()["contains_record_values"] is False
    assert run.safe_summary()["contains_candidate_pairs"] is False


def test_unsuccessful_runs_are_retained_with_stable_failure_codes() -> None:
    failed = BenchmarkRunRecord(
        run_id="run.failed",
        family_id="family.typo",
        instance_id="instance.high",
        replicate_id="replicate.two",
        task_profile_digest=_DIGEST,
        pipeline_recipe_digest=_OTHER_DIGEST,
        engine_commit="2" * 40,
        dependency_lock_digest=_DIGEST,
        environment_digest=_OTHER_DIGEST,
        random_seed=43,
        status=BenchmarkRunStatus.CANDIDATE_BUDGET_FAILURE,
        failure_code="ML-BENCH-PAIR-BUDGET",
    )
    snapshot = build_registry_snapshot(snapshot_id="snapshot.one", records=[failed])

    summary = snapshot.safe_summary()
    status_counts = summary["status_counts"]
    assert isinstance(status_counts, dict)
    assert status_counts["candidate_budget_failure"] == 1
    assert summary["run_count"] == 1


def test_success_requires_metrics_and_failure_requires_failure_code() -> None:
    with pytest.raises(ValidationError):
        BenchmarkRunRecord(
            run_id="run.bad.success",
            family_id="family.typo",
            instance_id="instance.low",
            replicate_id="replicate.one",
            task_profile_digest=_DIGEST,
            pipeline_recipe_digest=_OTHER_DIGEST,
            engine_commit="3" * 40,
            dependency_lock_digest=_DIGEST,
            environment_digest=_OTHER_DIGEST,
            random_seed=44,
            status=BenchmarkRunStatus.SUCCESS,
        )
    with pytest.raises(ValidationError):
        BenchmarkRunRecord(
            run_id="run.bad.failure",
            family_id="family.typo",
            instance_id="instance.low",
            replicate_id="replicate.one",
            task_profile_digest=_DIGEST,
            pipeline_recipe_digest=_OTHER_DIGEST,
            engine_commit="4" * 40,
            dependency_lock_digest=_DIGEST,
            environment_digest=_OTHER_DIGEST,
            random_seed=45,
            status=BenchmarkRunStatus.TIMEOUT,
        )


def test_registry_rejects_duplicate_runs() -> None:
    record = _success_record()
    with pytest.raises(ValidationError):
        BenchmarkRegistrySnapshot(snapshot_id="snapshot.duplicate", records=(record, record))
