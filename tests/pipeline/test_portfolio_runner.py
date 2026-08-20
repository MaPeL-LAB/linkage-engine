from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from mapel_linkage.assignment.contracts import pair_digest
from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.governance.labels import (
    PartitionDisjointnessReport,
)
from mapel_linkage.io.duckdb_store import DuckDBStore
from mapel_linkage.models.boosted import (
    BoostedFeatureMatrix,
    BoostedLabelledMatrix,
    XGBoostModelArtifact,
)
from mapel_linkage.pipeline.inference_runner import (
    attest_generated_synthetic_inference,
    infer_with_approved_recipe,
)
from mapel_linkage.pipeline.model_portfolio import (
    ModelPortfolioDeclaration,
    PairModelCandidateDeclaration,
)
from mapel_linkage.pipeline.portfolio_runner import (
    ModelPortfolioRunner,
    StackingInferenceArtifactBundle,
)
from mapel_linkage.pipeline.recipes import (
    RecipeApprovalStatus,
    RecipeExecutionMode,
)
from mapel_linkage.synthetic import SyntheticGenerationConfig, generate_synthetic_bundle


def _make_labelled_matrix(
    *,
    n_pairs: int,
    n_features: int,
    partition: str,
    random_seed: int = 42,
) -> BoostedLabelledMatrix:
    rng = np.random.default_rng(random_seed)
    features = rng.uniform(0.0, 1.0, size=(n_pairs, n_features))
    labels = (features[:, 0] + features[:, 1] > 1.0).astype(np.int8)

    # Ensure both classes exist
    if np.sum(labels == 1) == 0:
        labels[0] = 1
    if np.sum(labels == 0) == 0:
        labels[1] = 0

    feature_names = tuple(f"feature_{i}" for i in range(n_features))
    pairs = tuple((f"left_{partition}_{i}", f"right_{partition}_{i}") for i in range(n_pairs))
    pair_digests = tuple(pair_digest(left_k, right_k) for left_k, right_k in pairs)

    return BoostedLabelledMatrix(
        features=features,
        labels=labels,
        feature_names=feature_names,
        pair_references=pairs,
        pair_digests=pair_digests,
        feature_schema_digest="a" * 64,
        label_authority_digest=hashlib.sha256(f"label_{partition}".encode()).hexdigest(),
        selection_digest="b" * 64,
        label_source_kind="synthetic_truth",
        partition=partition,  # type: ignore[arg-type]
        positive_count=int(np.sum(labels == 1)),
        negative_count=int(np.sum(labels == 0)),
        hard_negative_count=0,
    )


def _stacking_portfolio() -> ModelPortfolioDeclaration:
    return ModelPortfolioDeclaration(
        portfolio_id="tournament_demo",
        pair_candidates=(
            PairModelCandidateDeclaration(
                model_id="fs_baseline",
                family="fellegi_sunter",
                implementation="mapel_reference_fellegi_sunter",
                role="baseline",
                require_verified_labels=False,
                artifact_format="package_json",
            ),
            PairModelCandidateDeclaration(
                model_id="xgb_challenger",
                family="xgboost",
                implementation="xgboost_classifier",
                role="challenger",
                require_verified_labels=True,
                artifact_format="xgboost_json",
            ),
            PairModelCandidateDeclaration(
                model_id="stacked_model",
                family="stacking",
                implementation="stacking_logistic",
                role="ensemble",
                require_verified_labels=True,
                artifact_format="package_json",
                base_model_ids=("fs_baseline", "xgb_challenger"),
            ),
        ),
        mandatory_baseline_id="fs_baseline",
        maximum_challengers=2,
    )


def test_portfolio_runner_tournament_and_stacking() -> None:
    train_mat = _make_labelled_matrix(n_pairs=60, n_features=4, partition="training", random_seed=1)
    val_mat = _make_labelled_matrix(n_pairs=30, n_features=4, partition="validation", random_seed=2)
    cal_mat = _make_labelled_matrix(
        n_pairs=30, n_features=4, partition="calibration", random_seed=3
    )

    disjointness = PartitionDisjointnessReport(
        partition_count=3,
        entity_component_count=120,
        household_component_count=0,
        manifest_digest="0" * 64,
        partition_authority_digests=(
            ("training", train_mat.label_authority_digest),
            ("validation", val_mat.label_authority_digest),
            ("calibration", cal_mat.label_authority_digest),
        ),
    )

    portfolio = _stacking_portfolio()

    with DuckDBStore() as store:
        runner = ModelPortfolioRunner(store)
        result = runner.run_tournament(
            portfolio=portfolio,
            training_matrix=train_mat,
            validation_matrix=val_mat,
            calibration_matrix=cal_mat,
            disjointness=disjointness,
            split_manifest_digest="c" * 64,
            configuration_digest="d" * 64,
            candidate_plan_digest="e" * 64,
            feature_schema_digest="a" * 64,
            decision_policy_digest="f" * 64,
            random_seed=42,
            k_folds=3,
        )

    assert result.portfolio.portfolio_id == "tournament_demo"
    assert result.champion_selection is not None
    assert result.champion_selection.selected_model_id in (
        "fs_baseline",
        "xgb_challenger",
        "stacked_model",
    )
    assert result.calibrator_artifact is not None
    assert len(result.oof_manifests) == 2  # FS and XGB base models
    assert all(m.partition == "training_oof" for m in result.oof_manifests)
    assert all(not m.test_partition_used for m in result.oof_manifests)
    assert all(not m.calibration_partition_used for m in result.oof_manifests)
    assert result.recipe is not None
    assert result.recipe.approval_status == RecipeApprovalStatus.DRAFT

    summary = result.safe_summary()
    assert summary["portfolio_id"] == "tournament_demo"
    assert summary["oof_manifest_count"] == 2
    assert summary["candidate_count"] == 3
    assert len(summary["tournament_digest"]) == 64


def test_stacking_champion_bundle_scores_raw_features_and_rejects_substitution() -> None:
    train_mat = _make_labelled_matrix(n_pairs=60, n_features=4, partition="training", random_seed=1)
    val_mat = _make_labelled_matrix(n_pairs=30, n_features=4, partition="validation", random_seed=2)
    cal_mat = _make_labelled_matrix(
        n_pairs=30, n_features=4, partition="calibration", random_seed=3
    )
    disjointness = PartitionDisjointnessReport(
        partition_count=3,
        entity_component_count=120,
        household_component_count=0,
        manifest_digest="0" * 64,
        partition_authority_digests=(
            ("training", train_mat.label_authority_digest),
            ("validation", val_mat.label_authority_digest),
            ("calibration", cal_mat.label_authority_digest),
        ),
    )
    with DuckDBStore() as store:
        result = ModelPortfolioRunner(store).run_tournament(
            portfolio=_stacking_portfolio(),
            training_matrix=train_mat,
            validation_matrix=val_mat,
            calibration_matrix=cal_mat,
            disjointness=disjointness,
            split_manifest_digest="c" * 64,
            configuration_digest="d" * 64,
            candidate_plan_digest="e" * 64,
            feature_schema_digest="a" * 64,
            decision_policy_digest="f" * 64,
            random_seed=42,
            k_folds=3,
            calibrator_methods=("sigmoid",),
            approval_status=RecipeApprovalStatus.SYNTHETIC_VALIDATED,
        )

    artifact_bundle = result.champion_model_artifact
    assert result.champion_selection.selected_model_id == "stacked_model"
    assert isinstance(artifact_bundle, StackingInferenceArtifactBundle)
    assert result.recipe.champion_artifact_digest == artifact_bundle.bundle_digest
    assert "base_artifacts" not in artifact_bundle.safe_summary()
    assert "XGBoostModelArtifact" not in repr(artifact_bundle)

    synthetic_bundle = generate_synthetic_bundle(SyntheticGenerationConfig(seed=20260816))
    source_keys = tuple(record.record_key for record in synthetic_bundle.source_a[:2])
    pair_references = (
        (source_keys[0], synthetic_bundle.source_b[0].record_key),
        (source_keys[1], synthetic_bundle.source_b[1].record_key),
    )
    feature_matrix = BoostedFeatureMatrix(
        features=np.asarray(
            (
                (0.92, 0.84, 0.77, 0.68),
                (0.18, 0.24, 0.31, 0.27),
            ),
            dtype=np.float64,
        ),
        pair_references=pair_references,
        pair_digests=tuple(pair_digest(left, right) for left, right in pair_references),
        feature_names=train_mat.feature_names,
        feature_schema_digest=train_mat.feature_schema_digest,
    )
    attestation = attest_generated_synthetic_inference(
        bundle=synthetic_bundle,
        source_record_keys=source_keys,
        pair_references=pair_references,
        feature_matrix=feature_matrix,
    )
    inference = infer_with_approved_recipe(
        recipe=result.recipe,
        source_record_keys=source_keys,
        pair_references=pair_references,
        feature_matrix=feature_matrix,
        champion_model_artifact=artifact_bundle,
        calibrator_artifact=result.calibrator_artifact,
        execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
        synthetic_attestation=attestation,
        synthetic_bundle=synthetic_bundle,
        source_dataset_id="source_a",
        target_dataset_id="source_b",
    )
    assert inference.pair_count == len(pair_references)
    assert inference.synthetic_attestation_digest == attestation.attestation_digest
    assert all(decision.merge_authority == "none" for decision in inference.decisions)

    with pytest.raises(PipelineError, match="ML-PIPE-064"):
        infer_with_approved_recipe(
            recipe=result.recipe,
            source_record_keys=source_keys,
            pair_references=pair_references,
            feature_matrix=feature_matrix,
            champion_model_artifact=artifact_bundle.stacking_artifact,
            calibrator_artifact=result.calibrator_artifact,
            execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
            synthetic_attestation=attestation,
            synthetic_bundle=synthetic_bundle,
            source_dataset_id="source_a",
            target_dataset_id="source_b",
        )

    xgboost_artifact = next(
        item for item in artifact_bundle.base_artifacts if isinstance(item, XGBoostModelArtifact)
    )
    substituted_payload = b"{}"
    substituted_xgboost = replace(
        xgboost_artifact,
        model_json=substituted_payload,
        model_digest=hashlib.sha256(substituted_payload).hexdigest(),
    )
    substituted_bases = tuple(
        substituted_xgboost if item is xgboost_artifact else item
        for item in artifact_bundle.base_artifacts
    )
    substituted_bundle = StackingInferenceArtifactBundle(
        stacking_artifact=artifact_bundle.stacking_artifact,
        base_artifacts=substituted_bases,
        feature_schema_digest=artifact_bundle.feature_schema_digest,
    )
    assert substituted_bundle.bundle_digest != artifact_bundle.bundle_digest
    with pytest.raises(PipelineError, match="ML-PIPE-064"):
        infer_with_approved_recipe(
            recipe=result.recipe,
            source_record_keys=source_keys,
            pair_references=pair_references,
            feature_matrix=feature_matrix,
            champion_model_artifact=substituted_bundle,
            calibrator_artifact=result.calibrator_artifact,
            execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
            synthetic_attestation=attestation,
            synthetic_bundle=synthetic_bundle,
            source_dataset_id="source_a",
            target_dataset_id="source_b",
        )

    with pytest.raises(PipelineError, match="ML-PIPE-066"):
        StackingInferenceArtifactBundle(
            stacking_artifact=artifact_bundle.stacking_artifact,
            base_artifacts=artifact_bundle.base_artifacts[:-1],
            feature_schema_digest=artifact_bundle.feature_schema_digest,
        )
