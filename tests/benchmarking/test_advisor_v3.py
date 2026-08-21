from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Literal, cast

import pytest

from mapel_linkage.benchmarking.advisor_catalogue import (
    build_advisor_corpus_design,
    build_benchmark_shard_plan,
)
from mapel_linkage.benchmarking.advisor_v3_catalogue import (
    AdvisorV3CorpusDesignManifest,
    advisor_v3_family_roles,
    build_advisor_v3_corpus_design,
    build_advisor_v3_geometry_coherence,
    build_advisor_v3_preregistration,
    build_advisor_v3_shard_plan,
    validate_advisor_v3_geometry_coherence,
)
from mapel_linkage.benchmarking.advisor_v3_execution import (
    AdvisorV3CorpusExecutionApproval,
    _shard_lock,
    advisor_v3_execution_provenance_digest,
    audit_advisor_v3_corpus,
    build_advisor_v3_execution_approval,
    execute_advisor_v3_corpus_shard,
    load_committed_advisor_v3_preregistration,
    prepare_advisor_v3_execution,
)
from mapel_linkage.benchmarking.advisor_v3_label_budget import (
    advisor_v3_label_budget_policy_digest,
    apply_advisor_v3_training_label_budget,
)
from mapel_linkage.benchmarking.registry import BenchmarkRegistry
from mapel_linkage.benchmarking.runner import BenchmarkPortfolioRunner
from mapel_linkage.cli.main import main
from mapel_linkage.governance.labels import VerifiedLabelBatch, VerifiedPairLabel
from mapel_linkage.recommendation.qualification import AdvisorQualificationPolicy
from tests.benchmarking.test_advisor_execution import _AggregateOnlyFakeRunner
from tests.helpers import ROOT


def _approval(
    runner: BenchmarkPortfolioRunner | None = None,
) -> AdvisorV3CorpusExecutionApproval:
    return build_advisor_v3_execution_approval(
        approval_reference="synthetic-v3-owner-approval",
        runner=runner,
    )


def _hold_shard_lock(registry_path: str, entered: multiprocessing.Queue[float]) -> None:
    with _shard_lock(BenchmarkRegistry(Path(registry_path)), 0):
        entered.put(time.monotonic())
        time.sleep(0.3)


def test_v2_digest_sensitive_contracts_remain_unchanged() -> None:
    assert build_advisor_corpus_design().design_digest == (
        "f8ade27029cc667cbc4e23293615d68a294ef5cc3c0e82bfdfc62520bddbf8e9"
    )
    assert build_benchmark_shard_plan(shard_count=32).plan_digest == (
        "212664884dfd36a07a8929e14259dfa131631c9bc546ae630414ea94cdeda00a"
    )
    assert AdvisorQualificationPolicy().policy_digest == (
        "42800754c4b6862bef2fd554bb030f13a958525e2ff9bc916c2002edd144362c"
    )


def test_v3_design_is_new_family_disjoint_and_whole_family_sharded() -> None:
    design = build_advisor_v3_corpus_design()
    roles = advisor_v3_family_roles()
    counts = Counter(role for _, role in roles)
    plan = build_advisor_v3_shard_plan()

    assert design.family_count == 84
    assert design.instance_count == 336
    assert counts == {
        "meta_training": 48,
        "conformal": 12,
        "locked_evaluation": 12,
        "ood_holdout": 12,
    }
    assert all(family_id.startswith("family.advisor_v3.") for family_id, _ in roles)
    assert not any("advisor_v2" in family_id for family_id, _ in roles)
    assert len(plan.shards) == 42
    assert {len(shard.families) for shard in plan.shards} == {2}
    assert {len(shard.instance_ids) for shard in plan.shards} == {8}
    assert len({family.family_id for shard in plan.shards for family in shard.families}) == 84

    drifted = design.model_dump(mode="json")
    drifted["catalogue_manifest_digest"] = "0" * 64
    with pytest.raises(ValueError, match="catalogue binding is stale"):
        AdvisorV3CorpusDesignManifest.model_validate(drifted)


def test_v3_preregistration_is_canonical_outcome_free_and_bound() -> None:
    path = ROOT / "docs" / "evidence" / "advisor_v3_preregistration_20260821.json"
    loaded = load_committed_advisor_v3_preregistration(path)
    expected = build_advisor_v3_preregistration()

    assert loaded == expected
    assert loaded.expected_run_count == 11_760
    assert loaded.expected_required_success_run_count == 5_040
    assert loaded.expected_ineligible_run_count == 6_720
    assert loaded.outcome_fields_present is False
    assert loaded.qualification_evaluation_accessed is False
    assert loaded.operational_validity == "not_established"
    assert json.loads(path.read_text(encoding="utf-8")) == expected.model_dump(mode="json")


def test_v3_geometry_is_deterministic_and_structurally_coherent() -> None:
    fixed = build_advisor_v3_geometry_coherence()
    observed = validate_advisor_v3_geometry_coherence()

    assert observed == fixed
    assert observed.locked_above_threshold_count == 0
    assert observed.ood_above_threshold_count == 12
    assert observed.maximum_locked_nearest_training_distance < (
        observed.selected_distance_threshold
    )
    assert observed.minimum_ood_nearest_training_distance > observed.selected_distance_threshold


def test_v3_training_label_budget_changes_supervised_count_not_fs_surface() -> None:
    from mapel_linkage.benchmarking.advisor_v3_catalogue import build_advisor_v3_generator

    generator = build_advisor_v3_generator()
    runner = BenchmarkPortfolioRunner()
    high = generator.generate("instance.advisor_v3.f029.p01", seed=20260816)
    low = generator.generate("instance.advisor_v3.f029.p04", seed=20260816)
    high_pairs, _ = runner._candidate_pairs(high.datasets["source_a"], high.datasets["source_b"])
    low_pairs, _ = runner._candidate_pairs(low.datasets["source_a"], low.datasets["source_b"])
    high_batches, high_full_fs, high_budget_digest = runner._protected_label_batches(
        bundle=high,
        candidate_pairs=high_pairs,
        seed=20260816,
    )
    low_batches, low_full_fs, low_budget_digest = runner._protected_label_batches(
        bundle=low,
        candidate_pairs=low_pairs,
        seed=20260816,
    )
    high_by_partition = {batch.partition: batch for batch in high_batches}
    low_by_partition = {batch.partition: batch for batch in low_batches}

    assert high.datasets == low.datasets
    assert high_pairs == low_pairs
    assert high_full_fs == low_full_fs
    assert len(high_full_fs) > len(high_by_partition["training"].labels)
    assert len(low_full_fs) > len(low_by_partition["training"].labels)
    assert len(high_by_partition["training"].labels) > len(low_by_partition["training"].labels)
    protected_partitions: tuple[Literal["validation", "calibration", "decision", "test"], ...] = (
        "validation",
        "calibration",
        "decision",
        "test",
    )
    assert all(
        high_by_partition[name].label_authority_digest
        == low_by_partition[name].label_authority_digest
        for name in protected_partitions
    )
    assert high_budget_digest != low_budget_digest
    assert advisor_v3_label_budget_policy_digest() == (
        "5db284af17f2b8acd3118c507a5c3192b893ca6d33750c53dfe1a857e44e6972"
    )


def test_v3_label_budget_round_half_even_and_authority_derivation_are_frozen() -> None:
    def label(index: int, value: Literal[0, 1]) -> VerifiedPairLabel:
        return VerifiedPairLabel(
            left_record_key=f"left-{index}",
            right_record_key=f"right-{index}",
            label=value,
            entity_component_digests=(f"{index + 1:064x}",),
        )

    training_labels = tuple(label(index, 1 if index < 3 else 0) for index in range(6))
    partitions: tuple[Literal["training", "validation", "calibration", "decision", "test"], ...] = (
        "training",
        "validation",
        "calibration",
        "decision",
        "test",
    )
    batches = tuple(
        VerifiedLabelBatch(
            source_kind="synthetic_truth",
            verification_protocol="synthetic_v3_freeze",
            source_digest=f"{partition_index + 100:064x}",
            partition=partition,
            labels=(
                label(
                    partition_index + 10,
                    cast(Literal[0, 1], partition_index % 2),
                ),
            )
            if partition != "training"
            else training_labels,
        )
        for partition_index, partition in enumerate(partitions)
    )

    retained, report = apply_advisor_v3_training_label_budget(
        batches,
        planned_training_label_budget=5,
        random_seed=20260816,
    )
    repeated, repeated_report = apply_advisor_v3_training_label_budget(
        batches,
        planned_training_label_budget=5,
        random_seed=20260816,
    )

    assert report.retained_training_label_count == 5
    assert report.retained_positive_count == 2
    assert report.retained_negative_count == 3
    assert retained == repeated
    assert report == repeated_report


def test_v3_governance_prepare_resume_and_tamper_fail_closed(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path / "registry")
    runner = _AggregateOnlyFakeRunner()
    approval = _approval(runner)
    prepare_advisor_v3_execution(
        registry=registry,
        committed_preregistration_path=(
            ROOT / "docs" / "evidence" / "advisor_v3_preregistration_20260821.json"
        ),
        approval=approval,
        runner=runner,
    )
    first = execute_advisor_v3_corpus_shard(
        registry=registry,
        shard_index=0,
        approval=approval,
        runner=runner,
    )
    second = execute_advisor_v3_corpus_shard(
        registry=registry,
        shard_index=0,
        approval=approval,
        runner=runner,
    )
    assert first.newly_persisted_run_count == 280
    assert first.successful_run_count == 120
    assert second.newly_persisted_run_count == 0
    assert second.resumed_run_count == 280

    audit_advisor_v3_corpus(registry=registry, approval=approval, runner=runner)
    metric_path = next(registry.metrics_dir.glob("*.json"))
    metric_text = metric_path.read_text(encoding="utf-8")
    metric_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        audit_advisor_v3_corpus(registry=registry, approval=approval, runner=runner)
    metric_path.write_text(metric_text, encoding="utf-8")

    failure_path = next(registry.failures_dir.glob("*.json"))
    failure_text = failure_path.read_text(encoding="utf-8")
    failure_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        audit_advisor_v3_corpus(registry=registry, approval=approval, runner=runner)
    failure_path.write_text(failure_text, encoding="utf-8")

    governance = next((registry.root_directory / "governance").glob("design.v3.*.json"))
    governance.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="conflicts"):
        execute_advisor_v3_corpus_shard(
            registry=registry,
            shard_index=1,
            approval=approval,
            runner=runner,
        )


@pytest.mark.parametrize(
    "attribute",
    ("_engine_commit", "_dependency_lock_digest", "_environment_digest"),
)
def test_v3_execution_approval_rejects_provenance_substitution(
    tmp_path: Path, attribute: str
) -> None:
    approved_runner = _AggregateOnlyFakeRunner()
    approval = _approval(approved_runner)
    drifted_runner = _AggregateOnlyFakeRunner()
    setattr(drifted_runner, attribute, "0" * 64)

    with pytest.raises(ValueError, match="readiness and provenance"):
        prepare_advisor_v3_execution(
            registry=BenchmarkRegistry(tmp_path / "registry"),
            committed_preregistration_path=(
                ROOT / "docs" / "evidence" / "advisor_v3_preregistration_20260821.json"
            ),
            approval=approval,
            runner=drifted_runner,
        )


def test_v3_execution_approval_rejects_stale_readiness_and_hides_provenance_values(
    tmp_path: Path,
) -> None:
    frozen_runner = _AggregateOnlyFakeRunner()
    frozen_runner._engine_commit = "1" * 64
    frozen_runner._dependency_lock_digest = "2" * 64
    frozen_runner._environment_digest = "3" * 64
    assert advisor_v3_execution_provenance_digest(frozen_runner) == (
        "39fc4d493cf2410330e163317e6504fad9f9b8fbd091a2277fa1657addd6a1df"
    )

    runner = _AggregateOnlyFakeRunner()
    approval = _approval(runner)
    provenance = runner.provenance_summary()
    summary_text = json.dumps(approval.safe_summary(), sort_keys=True)
    assert approval.execution_provenance_digest == advisor_v3_execution_provenance_digest(runner)
    assert all(value not in summary_text for value in provenance.values())

    payload = approval.model_dump(mode="json")
    payload["readiness_digest"] = "0" * 64
    stale = AdvisorV3CorpusExecutionApproval.model_validate(payload)
    with pytest.raises(ValueError, match="readiness and provenance"):
        prepare_advisor_v3_execution(
            registry=BenchmarkRegistry(tmp_path / "registry"),
            committed_preregistration_path=(
                ROOT / "docs" / "evidence" / "advisor_v3_preregistration_20260821.json"
            ),
            approval=stale,
            runner=runner,
        )


def test_v3_workers_and_audit_recompute_execution_provenance(tmp_path: Path) -> None:
    approved_runner = _AggregateOnlyFakeRunner()
    approval = _approval(approved_runner)
    registry = BenchmarkRegistry(tmp_path / "registry")
    prepare_advisor_v3_execution(
        registry=registry,
        committed_preregistration_path=(
            ROOT / "docs" / "evidence" / "advisor_v3_preregistration_20260821.json"
        ),
        approval=approval,
        runner=approved_runner,
    )

    source_drifted = _AggregateOnlyFakeRunner()
    source_drifted._engine_commit = "0" * 64
    with pytest.raises(ValueError, match="stale for current readiness or provenance"):
        execute_advisor_v3_corpus_shard(
            registry=registry,
            shard_index=0,
            approval=approval,
            runner=source_drifted,
        )

    environment_drifted = _AggregateOnlyFakeRunner()
    environment_drifted._environment_digest = "0" * 64
    with pytest.raises(ValueError, match="stale for current readiness or provenance"):
        audit_advisor_v3_corpus(
            registry=registry,
            approval=approval,
            runner=environment_drifted,
        )


def test_v3_shard_lock_serializes_processes(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path / "registry")
    lock_directory = registry.root_directory / "governance" / "locks.v3"
    lock_directory.mkdir(parents=True)
    context = multiprocessing.get_context("fork")
    entered: multiprocessing.Queue[float] = context.Queue()
    first = context.Process(target=_hold_shard_lock, args=(str(registry.root_directory), entered))
    second = context.Process(target=_hold_shard_lock, args=(str(registry.root_directory), entered))
    first.start()
    first_time = entered.get(timeout=3)
    second.start()
    second_time = entered.get(timeout=3)
    first.join(timeout=3)
    second.join(timeout=3)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert second_time - first_time >= 0.25


def test_v3_preregistration_rejects_noncanonical_and_symlink_paths(tmp_path: Path) -> None:
    expected = build_advisor_v3_preregistration()
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(expected.model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        load_committed_advisor_v3_preregistration(noncanonical)

    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps(expected.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    linked = tmp_path / "linked.json"
    linked.symlink_to(canonical)
    with pytest.raises(FileNotFoundError, match="unavailable"):
        load_committed_advisor_v3_preregistration(linked)


def test_v3_driver_rejects_fake_python_and_dry_run_is_private_path_safe() -> None:
    script = ROOT / "scripts" / "run_advisor_v3_corpus.sh"
    syntax = subprocess.run(
        ["bash", "-n", str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0
    fake = subprocess.run(
        ["bash", str(script), "--python", "/usr/bin/true", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert fake.returncode == 2
    assert "not Python 3.12" in fake.stderr

    environment = {**os.environ, "MAPEL_TEST_DATA_POLICY": "synthetic_only"}
    dry = subprocess.run(
        ["bash", str(script), "--python", sys.executable, "--dry-run"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert dry.returncode == 0
    assert "Changed: none (dry-run planning only)." in dry.stdout
    assert "ten workers by default" not in dry.stderr
    assert "private/benchmark_registry" not in dry.stdout + dry.stderr
    source = script.read_text(encoding="utf-8")
    assert "WORKERS=10" in source
    assert "completed evidence remains resumable" in source
    assert 'OMP_NUM_THREADS="1"' in source


def test_v3_cli_rejects_registry_outside_ignored_private_envelope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        (
            "prepare-advisor-v3-corpus",
            "--project-root",
            str(ROOT),
            "--registry-dir",
            "artifacts/advisor-v3",
            "--approve-execution",
            "--approval-reference",
            "synthetic-v3-path-negative",
        )
    )

    assert exit_code == 2
    assert "governance preparation failed closed" in capsys.readouterr().err
