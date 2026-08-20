from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest

from mapel_linkage.assignment.contracts import pair_digest
from mapel_linkage.calibration import (
    ChampionChallengerSelector,
    ChampionSelection,
    ModelEvaluationCandidate,
)
from mapel_linkage.configuration.models import ModelsConfig, ModelSelectionConfig
from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.governance.labels import (
    PartitionDisjointnessReport,
    VerifiedLabelBatch,
    VerifiedPairLabel,
)
from mapel_linkage.io.duckdb_store import DuckDBStore
from mapel_linkage.models.boosted import (
    BoostedFeatureMatrix,
    BoostedLabelledMatrix,
    XGBoostModelArtifact,
)
from mapel_linkage.pipeline.inference_runner import (
    ApprovedRecipeInferenceRunner,
    attest_generated_synthetic_inference,
    infer_with_approved_recipe,
)
from mapel_linkage.pipeline.model_portfolio import (
    ModelPortfolioDeclaration,
    PairModelCandidateDeclaration,
)
from mapel_linkage.pipeline.portfolio_runner import (
    ModelPortfolioRunner,
    ReferenceFeatureScoreArtifact,
    StackingInferenceArtifactBundle,
    _grouped_oof_folds,
)
from mapel_linkage.pipeline.recipes import (
    RecipeApprovalStatus,
    RecipeExecutionMode,
)
from mapel_linkage.pipeline.score_evidence import PairScoreEvidenceBatch
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
                model_id="xgb_challenger_two",
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
                base_model_ids=("xgb_challenger", "xgb_challenger_two"),
            ),
        ),
        mandatory_baseline_id="fs_baseline",
        maximum_challengers=3,
    )


def _training_batch(matrix: BoostedLabelledMatrix) -> VerifiedLabelBatch:
    return VerifiedLabelBatch(
        source_kind="synthetic_truth",
        verification_protocol="synthetic_oof_test_v1",
        source_digest="8" * 64,
        partition="training",
        labels=tuple(
            VerifiedPairLabel(
                left_record_key=left,
                right_record_key=right,
                label=int(label),  # type: ignore[arg-type]
                entity_component_digests=(hashlib.sha256(left.encode()).hexdigest(),),
                household_component_digests=(hashlib.sha256(right.encode()).hexdigest(),),
            )
            for (left, right), label in zip(matrix.pair_references, matrix.labels, strict=True)
        ),
    )


def _source_groups(batch: VerifiedLabelBatch) -> dict[str, tuple[str, ...]]:
    return {
        item.left_record_key: (
            *item.entity_component_digests,
            *item.household_component_digests,
        )
        for item in batch.labels
    }


def test_grouped_oof_keeps_related_source_records_in_one_holdout_fold() -> None:
    matrix = _make_labelled_matrix(
        n_pairs=18,
        n_features=4,
        partition="training",
        random_seed=11,
    )
    shared_entity = "7" * 64
    batch = VerifiedLabelBatch(
        source_kind="synthetic_truth",
        verification_protocol="synthetic_oof_related_v1",
        source_digest="8" * 64,
        partition="training",
        labels=tuple(
            VerifiedPairLabel(
                left_record_key=left,
                right_record_key=right,
                label=int(label),  # type: ignore[arg-type]
                entity_component_digests=(
                    shared_entity if index < 2 else hashlib.sha256(left.encode()).hexdigest(),
                ),
                household_component_digests=(hashlib.sha256(right.encode()).hexdigest(),),
            )
            for index, ((left, right), label) in enumerate(
                zip(matrix.pair_references, matrix.labels, strict=True)
            )
        ),
    )
    matrix = replace(matrix, label_authority_digest=batch.label_authority_digest)
    folds, group_count, _ = _grouped_oof_folds(
        matrix=matrix,
        labels=batch,
        source_group_digests=_source_groups(batch),
        fold_count=3,
        random_seed=20260816,
    )

    assert group_count == matrix.pair_count - 1
    assert any(0 in fold.tolist() and 1 in fold.tolist() for fold in folds)
    with pytest.raises(PipelineError, match="ML-PIPE-090"):
        _grouped_oof_folds(
            matrix=matrix,
            labels=batch,
            source_group_digests={
                key: value
                for key, value in _source_groups(batch).items()
                if key != "left_training_0"
            },
            fold_count=3,
            random_seed=20260816,
        )


def _models_config() -> ModelsConfig:
    return ModelsConfig.model_validate(
        {
            "fellegi_sunter": {
                "implementation": "splink_duckdb",
                "model_id": "fs_baseline",
            },
            "boosted_trees": [
                {
                    "implementation": "xgboost_classifier",
                    "model_id": "xgb_challenger",
                    "n_estimators": 20,
                    "max_depth": 2,
                    "learning_rate": 0.2,
                    "maximum_training_pairs": 1000,
                },
                {
                    "implementation": "xgboost_classifier",
                    "model_id": "xgb_challenger_two",
                    "n_estimators": 50,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "maximum_training_pairs": 1000,
                },
            ],
            "ensembles": [
                {
                    "enabled": True,
                    "implementation": "stacking_logistic",
                    "model_id": "stacked_model",
                    "base_model_ids": ["xgb_challenger", "xgb_challenger_two"],
                    "maximum_training_pairs": 1000,
                }
            ],
        }
    )


def _fs_inputs(
    *matrices: BoostedLabelledMatrix,
) -> tuple[ReferenceFeatureScoreArtifact, tuple[PairScoreEvidenceBatch, ...]]:
    artifact = ReferenceFeatureScoreArtifact.create(
        model_id="fs_baseline",
        feature_schema_digest="a" * 64,
        configuration_digest="d" * 64,
        source_evidence_digest="9" * 64,
        scoring_rule="external_scores_only",
    )
    evidence = tuple(
        PairScoreEvidenceBatch._issue(
            pair_digests=matrix.pair_digests,
            scores=np.full(matrix.pair_count, 0.5, dtype=np.float64),
            champion_model_id=artifact.model_id,
            champion_model_version=artifact.model_version,
            champion_artifact_digest=artifact.model_digest,
            configuration_digest=artifact.configuration_digest,
            feature_schema_digest=artifact.feature_schema_digest,
            probability_status="model_score_uncalibrated",
        )
        for matrix in matrices
    )
    return artifact, evidence


def test_portfolio_runner_tournament_and_stacking() -> None:
    train_mat = _make_labelled_matrix(n_pairs=60, n_features=4, partition="training", random_seed=1)
    val_mat = _make_labelled_matrix(n_pairs=30, n_features=4, partition="validation", random_seed=2)
    cal_mat = _make_labelled_matrix(
        n_pairs=30, n_features=4, partition="calibration", random_seed=3
    )
    training_batch = _training_batch(train_mat)
    train_mat = replace(train_mat, label_authority_digest=training_batch.label_authority_digest)

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
    fs_artifact, fs_evidence = _fs_inputs(train_mat, val_mat, cal_mat)

    with DuckDBStore() as store:
        runner = ModelPortfolioRunner(store)
        result = runner.run_tournament(
            portfolio=portfolio,
            models_config=_models_config(),
            training_label_batch=training_batch,
            training_source_group_digests=_source_groups(training_batch),
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
            fs_model_artifact=fs_artifact,
            fs_training_evidence=fs_evidence[0],
            fs_validation_evidence=fs_evidence[1],
            fs_calibration_evidence=fs_evidence[2],
        )

    assert result.portfolio.portfolio_id == "tournament_demo"
    assert result.champion_selection is not None
    assert result.champion_selection.selected_model_id in (
        "fs_baseline",
        "xgb_challenger",
        "xgb_challenger_two",
        "stacked_model",
    )
    assert result.calibrator_artifact is not None
    assert len(result.oof_manifests) == 2  # Two supervised XGBoost base models only
    assert all(m.partition == "training_oof" for m in result.oof_manifests)
    assert all(not m.test_partition_used for m in result.oof_manifests)
    assert all(not m.calibration_partition_used for m in result.oof_manifests)
    assert all(m.group_count == train_mat.pair_count for m in result.oof_manifests)
    assert all(
        m.grouping_method == "source_entity_household_connected_components"
        for m in result.oof_manifests
    )
    assert result.recipe is not None
    assert result.recipe.approval_status == RecipeApprovalStatus.DRAFT

    summary = result.safe_summary()
    assert summary["portfolio_id"] == "tournament_demo"
    assert summary["oof_manifest_count"] == 2
    assert summary["candidate_count"] == 4
    assert len(summary["tournament_digest"]) == 64


def test_stacking_champion_bundle_scores_raw_features_and_rejects_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_select = ChampionChallengerSelector.select

    def force_stacking(
        candidates: Sequence[ModelEvaluationCandidate],
        config: ModelSelectionConfig,
    ) -> ChampionSelection:
        adjusted = tuple(
            replace(
                candidate,
                average_precision=1.0 if candidate.model_family == "stacking" else 0.0,
                brier_score=0.0 if candidate.model_family == "stacking" else 1.0,
            )
            for candidate in candidates
        )
        return original_select(adjusted, config)

    monkeypatch.setattr(ChampionChallengerSelector, "select", force_stacking)
    train_mat = _make_labelled_matrix(n_pairs=60, n_features=4, partition="training", random_seed=1)
    val_mat = _make_labelled_matrix(n_pairs=30, n_features=4, partition="validation", random_seed=2)
    cal_mat = _make_labelled_matrix(
        n_pairs=30, n_features=4, partition="calibration", random_seed=3
    )
    training_batch = _training_batch(train_mat)
    train_mat = replace(train_mat, label_authority_digest=training_batch.label_authority_digest)
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
    fs_artifact, fs_evidence = _fs_inputs(train_mat, val_mat, cal_mat)
    with DuckDBStore() as store:
        result = ModelPortfolioRunner(store).run_tournament(
            portfolio=_stacking_portfolio(),
            models_config=_models_config(),
            training_label_batch=training_batch,
            training_source_group_digests=_source_groups(training_batch),
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
            fs_model_artifact=fs_artifact,
            fs_training_evidence=fs_evidence[0],
            fs_validation_evidence=fs_evidence[1],
            fs_calibration_evidence=fs_evidence[2],
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
    recomputed_scores = ApprovedRecipeInferenceRunner._score_with_model(
        feature_matrix=feature_matrix,
        model_artifact=artifact_bundle,
    )
    score_evidence = PairScoreEvidenceBatch._issue(
        pair_digests=feature_matrix.pair_digests,
        scores=recomputed_scores,
        champion_model_id=artifact_bundle.model_id,
        champion_model_version=artifact_bundle.model_version,
        champion_artifact_digest=artifact_bundle.model_digest,
        configuration_digest=artifact_bundle.configuration_digest,
        feature_schema_digest=artifact_bundle.feature_schema_digest,
        probability_status=artifact_bundle.probability_status,
    )
    inference = infer_with_approved_recipe(
        recipe=result.recipe,
        source_record_keys=source_keys,
        pair_references=pair_references,
        feature_matrix=feature_matrix,
        champion_model_artifact=artifact_bundle,
        score_evidence=score_evidence,
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

    substituted_evidence = PairScoreEvidenceBatch._issue(
        pair_digests=feature_matrix.pair_digests,
        scores=np.asarray((0.5, 0.5), dtype=np.float64),
        champion_model_id=artifact_bundle.model_id,
        champion_model_version=artifact_bundle.model_version,
        champion_artifact_digest=artifact_bundle.model_digest,
        configuration_digest=artifact_bundle.configuration_digest,
        feature_schema_digest=artifact_bundle.feature_schema_digest,
        probability_status=artifact_bundle.probability_status,
    )
    with pytest.raises(PipelineError, match="ML-PIPE-068"):
        infer_with_approved_recipe(
            recipe=result.recipe,
            source_record_keys=source_keys,
            pair_references=pair_references,
            feature_matrix=feature_matrix,
            champion_model_artifact=artifact_bundle,
            score_evidence=substituted_evidence,
            calibrator_artifact=result.calibrator_artifact,
            execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
            synthetic_attestation=attestation,
            synthetic_bundle=synthetic_bundle,
            source_dataset_id="source_a",
            target_dataset_id="source_b",
        )

    reordered_evidence = PairScoreEvidenceBatch._issue(
        pair_digests=tuple(reversed(feature_matrix.pair_digests)),
        scores=np.asarray(tuple(reversed(recomputed_scores)), dtype=np.float64),
        champion_model_id=artifact_bundle.model_id,
        champion_model_version=artifact_bundle.model_version,
        champion_artifact_digest=artifact_bundle.model_digest,
        configuration_digest=artifact_bundle.configuration_digest,
        feature_schema_digest=artifact_bundle.feature_schema_digest,
        probability_status=artifact_bundle.probability_status,
    )
    with pytest.raises(PipelineError, match="ML-PIPE-068"):
        infer_with_approved_recipe(
            recipe=result.recipe,
            source_record_keys=source_keys,
            pair_references=pair_references,
            feature_matrix=feature_matrix,
            champion_model_artifact=artifact_bundle,
            score_evidence=reordered_evidence,
            calibrator_artifact=result.calibrator_artifact,
            execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
            synthetic_attestation=attestation,
            synthetic_bundle=synthetic_bundle,
            source_dataset_id="source_a",
            target_dataset_id="source_b",
        )

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
