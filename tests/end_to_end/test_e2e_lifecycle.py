from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.e2e_linkage_lifecycle import (
    CONFIG_PATH,
    ROOT,
    SEED,
    LifecycleArtifacts,
    run_lifecycle,
    write_aggregate_outputs,
)
from mapel_linkage.adjudication import AdjudicationWorkflowRunner
from mapel_linkage.domain.errors import LabelProvenanceError, PipelineError
from mapel_linkage.pipeline import (
    OperationalValidationStatus,
    RecipeApprovalStatus,
    RecipeExecutionMode,
    deserialize_pipeline_recipe,
)


@pytest.fixture(scope="module")
def lifecycle(tmp_path_factory: pytest.TempPathFactory) -> LifecycleArtifacts:
    return run_lifecycle(project_root=tmp_path_factory.mktemp("e2e_lifecycle_project"))


def test_example_is_import_safe() -> None:
    environment = os.environ.copy()
    environment["MAPEL_TEST_DATA_POLICY"] = "synthetic_only"
    completed = subprocess.run(
        [sys.executable, "-c", "import examples.e2e_linkage_lifecycle"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_external_runner_is_syntax_valid_and_has_a_no_write_dry_run(
    tmp_path: Path,
) -> None:
    runner = ROOT / "scripts" / "run_e2e_lifecycle.sh"
    syntax = subprocess.run(
        ["bash", "-n", str(runner)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    output_directory = tmp_path / "must_not_exist"
    dry_run = subprocess.run(
        [
            "bash",
            str(runner),
            "--dry-run",
            "--python",
            sys.executable,
            "--output-dir",
            str(output_directory),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert "Changed: nothing (dry run)." in dry_run.stdout
    assert "MAPEL_TEST_DATA_POLICY" not in dry_run.stderr
    assert str(ROOT) not in dry_run.stdout
    assert str(output_directory) not in dry_run.stdout
    assert sys.executable not in dry_run.stdout
    assert not output_directory.exists()

    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "symlink_parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    rejected = subprocess.run(
        [
            "bash",
            str(runner),
            "--dry-run",
            "--python",
            sys.executable,
            "--output-dir",
            str(symlink_parent / "nested"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "symbolic link" in rejected.stderr
    assert str(ROOT) not in rejected.stderr
    assert str(symlink_parent) not in rejected.stderr


def test_complete_lifecycle_is_deterministic_and_aggregate_only(
    lifecycle: LifecycleArtifacts,
    tmp_path: Path,
) -> None:
    second = run_lifecycle(project_root=tmp_path / "second_project")

    assert lifecycle.summary == second.summary
    assert lifecycle.summary["seed"] == SEED
    assert lifecycle.summary["data_policy"] == "synthetic_only"
    assert len(str(lifecycle.summary["lifecycle_lineage_digest"])) == 64
    assert lifecycle.profile.contains_record_values is False
    assert lifecycle.profile.contains_source_field_names is False
    assert CONFIG_PATH.is_file()

    serialized = json.dumps(lifecycle.summary, sort_keys=True)
    for forbidden in (
        "A000000",
        "B000000",
        "C000000",
        str(tmp_path),
    ):
        assert forbidden not in serialized


def test_advisor_tournament_and_calibration_authority_boundaries(
    lifecycle: LifecycleArtifacts,
) -> None:
    advisor = lifecycle.advisor_report
    assert advisor.recommendation_authority == "advisory_only"
    assert advisor.decision_authority == "none"
    assert advisor.assignment_authority == "none"
    assert advisor.merge_authority == "none"
    assert advisor.automatic_promotion == "prohibited"
    assert advisor.operational_validity == "not_established"
    assert advisor.fallback_to_similarity is False
    assert advisor.meta_model_type == "ridge_meta_ranker_v1"
    assert advisor.meta_model_trained_runs > 0
    assert advisor.predicted_candidate_utilities

    tournament = lifecycle.tournament
    assert tournament.recipe.approval_status is RecipeApprovalStatus.SYNTHETIC_VALIDATED
    assert tournament.recipe.operational_validation is OperationalValidationStatus.NOT_ESTABLISHED
    assert tournament.recipe.decision_authority == "explicit_policy_only"
    assert tournament.recipe.merge_authority == "none"
    assert tournament.calibrator_artifact.calibration_status == "calibrated_on_protected_partition"
    assert len(tournament.oof_manifests) == 1
    assert tournament.oof_manifests[0].model_id == "xgb_pair_classifier"
    assert all(manifest.partition == "training_oof" for manifest in tournament.oof_manifests)
    assert all(not manifest.test_partition_used for manifest in tournament.oof_manifests)
    assert all(not manifest.calibration_partition_used for manifest in tournament.oof_manifests)
    assert all(not manifest.decision_partition_used for manifest in tournament.oof_manifests)
    assert all(
        manifest.decision_authority == "evidence_only" and manifest.merge_authority == "none"
        for manifest in tournament.oof_manifests
    )
    assert all(
        candidate.decision_authority == "evidence_only" and candidate.merge_authority == "none"
        for candidate in tournament.portfolio.pair_candidates
    )
    summary = lifecycle.summary
    generation = summary["synthetic_generation"]
    portfolio = summary["portfolio_tournament"]
    assert isinstance(generation, dict)
    assert isinstance(portfolio, dict)
    assert generation["bundle_digest"] == portfolio["synthetic_bundle_digest"]


def test_review_ordering_consensus_and_partition_disjoint_promotion(
    lifecycle: LifecycleArtifacts,
) -> None:
    queue = lifecycle.review_queue
    assert queue.relationship_count == 3
    assert queue.strategy == "hybrid"
    assert [entry.candidate_rank for entry in queue.entries] == [1, 2, 3]
    assert all(not hasattr(entry, "label") for entry in queue.entries)
    assert all("active_learning_hybrid" in entry.review_reason_codes for entry in queue.entries)
    active_learning = lifecycle.summary["active_learning"]
    assert isinstance(active_learning, dict)
    assert lifecycle.review_inference.inference_digest == active_learning["source_inference_digest"]

    consensus = lifecycle.consensus
    assert consensus.disagreement_report.total_pairs == 3
    assert consensus.disagreement_report.resolved_pairs == 3
    assert consensus.disagreement_report.unresolved_pairs == 0
    assert all(item.reviewer_count == 2 for item in consensus.results)
    assert all(item.resolution_method == "unanimous" for item in consensus.results)

    promotion = lifecycle.promotion
    assert promotion.verified_batch.source_kind == "verified_human_adjudication"
    assert promotion.verified_batch.partition == "training"
    assert promotion.promotion_summary.eligible_count == 3
    assert promotion.retraining_triggered is False
    assert promotion.disjointness_report is not None
    assert promotion.disjointness_report.partition_count == 3

    with pytest.raises(LabelProvenanceError, match="ML-LABEL-016"):
        AdjudicationWorkflowRunner.promote_to_verified_labels(
            consensus,
            target_partition="validation",
            min_confidence=0.90,
            require_consensus=True,
            require_double_review=True,
            minimum_reviewers=2,
            allowed_protocols=frozenset({"synthetic_double_review_v1"}),
            verification_protocol="synthetic_double_review_v1",
            existing_partition_batches=(promotion.verified_batch,),
        )


def test_synthetic_recipe_cannot_authorize_operational_inference(
    lifecycle: LifecycleArtifacts,
) -> None:
    recipe = deserialize_pipeline_recipe(lifecycle.recipe_payload)

    with pytest.raises(PipelineError, match="ML-RECIPE-014"):
        recipe.assert_usable_for(RecipeExecutionMode.SYNTHETIC_INFERENCE)
    with pytest.raises(PipelineError, match="ML-RECIPE-005"):
        recipe.assert_usable_for(RecipeExecutionMode.INFERENCE)

    assert recipe.approval_status is RecipeApprovalStatus.SYNTHETIC_VALIDATED
    assert recipe.operational_validation is OperationalValidationStatus.NOT_ESTABLISHED
    assert lifecycle.inference.execution_mode is RecipeExecutionMode.SYNTHETIC_INFERENCE
    assert lifecycle.inference.synthetic_attestation_digest is not None
    assert len(lifecycle.inference.synthetic_attestation_digest) == 64
    assert lifecycle.inference.assignment_result.assignment_authority == "global_selection_only"
    assert lifecycle.inference.assignment_result.decision_authority == "none"
    assert all(
        decision.decision_authority == "policy_classification"
        and decision.merge_authority == "none"
        for decision in lifecycle.inference.decisions
    )


def test_multisource_evaluation_is_aggregate_and_never_writes_a_crosswalk(
    lifecycle: LifecycleArtifacts,
) -> None:
    result = lifecycle.multisource

    assert result.total_records == 9
    assert 1 <= result.total_clusters <= result.total_records
    assert result.resolution_result.source_collision_count == 0
    assert result.resolution_result.cannot_link_violations == 0
    assert result.crosswalk_path is None
    assert result.evaluation_report_path is None
    assert result.evaluation_report is not None
    assert 0.0 <= result.evaluation_report.bcubed_f1 <= 1.0
    assert result.evaluation_report.dataset_collisions == 0


def test_explicit_outputs_are_idempotent_aggregate_only_and_symlink_safe(
    lifecycle: LifecycleArtifacts,
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "aggregate_outputs"
    write_aggregate_outputs(output_directory, lifecycle)
    write_aggregate_outputs(output_directory, lifecycle)

    summary_path = output_directory / "lifecycle_summary.json"
    recipe_path = output_directory / "pipeline_recipe.json"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == lifecycle.summary
    assert deserialize_pipeline_recipe(recipe_path.read_text(encoding="utf-8"))

    output_text = summary_path.read_text(encoding="utf-8") + recipe_path.read_text(encoding="utf-8")
    assert "A000000" not in output_text
    assert "B000000" not in output_text
    assert str(tmp_path) not in output_text

    real_directory = tmp_path / "real_output"
    real_directory.mkdir()
    symlink_directory = tmp_path / "symlink_output"
    symlink_directory.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symbolic link"):
        write_aggregate_outputs(symlink_directory, lifecycle)

    symlink_parent = tmp_path / "symlink_parent"
    symlink_parent.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symbolic link"):
        write_aggregate_outputs(symlink_parent / "nested", lifecycle)


def test_non_synthetic_policy_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAPEL_TEST_DATA_POLICY", "operational")
    with pytest.raises(RuntimeError, match="synthetic-only"):
        run_lifecycle()
