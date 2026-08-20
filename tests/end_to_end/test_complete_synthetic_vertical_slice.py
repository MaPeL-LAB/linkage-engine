from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mapel_linkage.configuration import LinkageConfig, compile_config, load_config
from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.pipeline import SyntheticVerticalSliceRunner
from mapel_linkage.synthetic import SyntheticGenerationConfig
from tests.helpers import EXAMPLE_CONFIG, ROOT


def _has_complete_runtime() -> bool:
    required = ("duckdb", "ortools.graph.python.min_cost_flow", "splink")
    if any(importlib.util.find_spec(name) is None for name in required):
        return False
    import duckdb

    return hasattr(duckdb, "connect")


@pytest.mark.skipif(not _has_complete_runtime(), reason="complete scientific runtime unavailable")
def test_complete_synthetic_vertical_slice_is_deterministic_and_privacy_bounded(
    tmp_path: Path,
) -> None:
    loaded = load_config(EXAMPLE_CONFIG)
    plan = compile_config(loaded.config, project_root=tmp_path)
    generation = SyntheticGenerationConfig(
        seed=plan.random_seed,
        entity_count=120,
        left_only_count=8,
        right_only_count=8,
        duplicate_count=8,
        competing_candidate_count=20,
        source_a_missing_rate=0.05,
        source_b_missing_rate=0.20,
        source_b_typo_rate=0.35,
        source_b_date_shift_rate=0.20,
    )

    first = SyntheticVerticalSliceRunner.run(plan, generation=generation)
    first_report = first.aggregate_report_path.read_bytes()
    first_relationships = first.relationship_output_path.read_bytes()
    second = SyntheticVerticalSliceRunner.run(plan, generation=generation)

    assert first.safe_summary() == second.safe_summary()
    assert first_report == second.aggregate_report_path.read_bytes()
    assert first_relationships == second.relationship_output_path.read_bytes()
    assert first.merge_authority == "none"
    assert first.real_data_validation_status == "not_established"
    assert sum(first.relationship_status_counts.values()) == 136
    assert set(first.relationship_status_counts) == {
        "confirmed",
        "review_required",
        "unresolved",
        "no_match",
    }

    stage_names = tuple(stage.stage for stage in first.stage_summaries)
    assert stage_names == (
        "synthetic_generation",
        "canonical_preprocessing",
        "candidate_generation",
        "comparison_and_anchor_evidence",
        "protected_label_partitions",
        "pair_model_training_and_scoring",
        "champion_selection_and_calibration",
        "candidate_ranking_and_assignment",
        "relationship_decisions_and_review",
        "synthetic_evaluation",
    )
    candidate_stage = next(
        stage for stage in first.stage_summaries if stage.stage == "candidate_generation"
    )
    model_stage = next(
        stage for stage in first.stage_summaries if stage.stage == "pair_model_training_and_scoring"
    )
    assert (
        model_stage.counts["fs_native_training_candidate_pair_count"]
        == model_stage.counts["fs_native_scored_pair_count"]
    )
    assert (
        model_stage.digests["fs_native_training_candidate_pair_set_digest"]
        == candidate_stage.digests["candidate_pair_set_digest"]
        == model_stage.digests["fs_native_scoring_candidate_pair_set_digest"]
    )
    assert model_stage.digests["fs_native_decision_authority"] == "evidence_only"
    assert model_stage.digests["fs_native_relationship_authority"] == "none"
    assert model_stage.digests["fs_native_assignment_authority"] == "none"
    assert model_stage.digests["fs_native_merge_authority"] == "none"
    assert model_stage.digests["fs_native_operational_validation"] == "not_established"

    report = json.loads(first.aggregate_report_path.read_text(encoding="utf-8"))
    benchmark = json.loads(
        (ROOT / "tests" / "benchmarks" / "synthetic_mvp_v1.json").read_text(encoding="utf-8")
    )
    candidate_report = report["reports"]["candidate_retrieval"]
    assignment_report = report["reports"]["assignment"]
    pair_report = report["reports"]["calibrated_test_pairs"]
    assert candidate_report["candidate_recall"] >= benchmark["minimum_candidate_recall"]
    assert assignment_report["assignment_accuracy"] >= benchmark["minimum_assignment_accuracy"]
    assert pair_report["pair_count"] >= benchmark["minimum_test_pair_count"]
    assert (
        assignment_report["constraint_violation_count"]
        <= benchmark["maximum_constraint_violation_count"]
    )
    assert set(first.relationship_status_counts) == set(benchmark["required_relationship_statuses"])
    assert report["evaluation_scope"] == "synthetic_mechanical_evaluation"
    assert report["real_data_validation_status"] == "not_established"
    assert report["reports"]["configured_decision_thresholds"]["test_partition_used"] is False
    assert (
        report["reports"]["configured_decision_thresholds"]["threshold_authority"]
        == "synthetic_benchmark_only"
    )
    assert "Synthetic testing establishes software behaviour only" in report["warning"]

    unrestricted = b"\n".join(
        (
            first.aggregate_report_path.read_bytes(),
            first.run_manifest_path.read_bytes(),
        )
    )
    assert b"source_a_" not in unrestricted
    assert b"source_b_" not in unrestricted
    assert str(tmp_path).encode("utf-8") not in unrestricted


def test_synthetic_slice_rejects_non_fixture_input_before_dataset_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_config(EXAMPLE_CONFIG)
    payload = loaded.config.model_dump(mode="json")
    payload["datasets"][0]["path"] = "private/operational-shaped-input.jsonl"
    plan = compile_config(LinkageConfig.model_validate(payload), project_root=tmp_path)
    accessed = False

    def forbidden_prepare(*args: object, **kwargs: object) -> None:
        nonlocal accessed
        accessed = True
        raise AssertionError("dataset access must not occur")

    monkeypatch.setattr(
        "mapel_linkage.pipeline.synthetic_vertical_slice.ConfiguredDatasetPreparer.prepare_all",
        forbidden_prepare,
    )
    with pytest.raises(PipelineError, match="ML-PIPE-020"):
        SyntheticVerticalSliceRunner.run(plan)
    assert accessed is False
    assert not (tmp_path / "data" / "synthetic").exists()


def test_synthetic_slice_rejects_generation_seed_mismatch_before_writing(
    tmp_path: Path,
) -> None:
    loaded = load_config(EXAMPLE_CONFIG)
    plan = compile_config(loaded.config, project_root=tmp_path)
    generation = SyntheticGenerationConfig(seed=plan.random_seed + 1, entity_count=120)

    with pytest.raises(PipelineError, match="ML-PIPE-022"):
        SyntheticVerticalSliceRunner.run(plan, generation=generation)
    assert not (tmp_path / "data" / "synthetic").exists()
