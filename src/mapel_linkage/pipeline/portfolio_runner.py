"""Multi-model portfolio tournament runner with zero-leakage out-of-fold stacking."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mapel_linkage.calibration import (
    ChampionCalibratorSelector,
    ChampionChallengerSelector,
    ModelEvaluationCandidate,
    PairScoreBatch,
)
from mapel_linkage.calibration.contracts import (
    CalibrationMethod,
    CalibratorArtifact,
    ChampionSelection,
)
from mapel_linkage.configuration.models import (
    BoostedTreeModelConfig,
    ModelSelectionConfig,
    NeuralModelConfig,
)
from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.governance.labels import PartitionDisjointnessReport
from mapel_linkage.io.duckdb_store import DuckDBStore
from mapel_linkage.models.boosted import (
    BoostedFeatureMatrix,
    BoostedLabelledMatrix,
    LightGBMPairClassifier,
    XGBoostPairClassifier,
)
from mapel_linkage.models.ensembles import (
    StackingPairClassifier,
)
from mapel_linkage.models.neural import (
    PyTorchPairMatcher,
)
from mapel_linkage.pipeline.model_portfolio import (
    ModelPortfolioDeclaration,
)
from mapel_linkage.pipeline.recipes import (
    AssignmentConstraint,
    LinkageMode,
    OperationalValidationStatus,
    PipelineRecipeArtifact,
    RecipeApprovalStatus,
)
from mapel_linkage.pipeline.stage_artifacts import OutOfFoldPredictionManifest
from mapel_linkage.validation import PairValidationReport, evaluate_binary_scores


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class PortfolioTournamentResult:
    """Outcome of side-by-side model portfolio tournament execution."""

    portfolio: ModelPortfolioDeclaration
    champion_selection: ChampionSelection
    champion_model_artifact: Any
    calibrator_artifact: CalibratorArtifact
    oof_manifests: tuple[OutOfFoldPredictionManifest, ...]
    validation_reports: dict[str, PairValidationReport]
    recipe: PipelineRecipeArtifact
    ranking_artifact: Any | None = None
    tournament_digest: str = ""

    def __post_init__(self) -> None:
        if not self.tournament_digest:
            payload = {
                "portfolio_digest": self.portfolio.portfolio_digest,
                "selection_digest": self.champion_selection.selection_digest,
                "calibrator_digest": self.calibrator_artifact.calibrator_digest,
                "recipe_digest": self.recipe.recipe_digest,
            }
            object.__setattr__(self, "tournament_digest", _canonical_digest(payload))

    def safe_summary(self) -> dict[str, Any]:
        """Return aggregate summary without row-level data or private paths."""
        return {
            "portfolio_id": self.portfolio.portfolio_id,
            "champion_model_id": self.champion_selection.selected_model_id,
            "champion_model_family": self.champion_selection.selected_model_family,
            "champion_model_version": self.champion_selection.selected_model_version,
            "calibrator_method": self.calibrator_artifact.method,
            "calibrator_digest": self.calibrator_artifact.calibrator_digest,
            "oof_manifest_count": len(self.oof_manifests),
            "candidate_count": len(self.validation_reports),
            "recipe_id": self.recipe.recipe_id,
            "recipe_digest": self.recipe.recipe_digest,
            "approval_status": self.recipe.approval_status.value,
            "tournament_digest": self.tournament_digest,
        }


class ModelPortfolioRunner:
    """Orchestrates side-by-side portfolio training, out-of-fold stacking, and recipe creation."""

    def __init__(self, store: DuckDBStore | None = None) -> None:
        self._store = store

    def run_tournament(
        self,
        *,
        portfolio: ModelPortfolioDeclaration,
        training_matrix: BoostedLabelledMatrix,
        validation_matrix: BoostedLabelledMatrix,
        calibration_matrix: BoostedLabelledMatrix,
        disjointness: PartitionDisjointnessReport,
        split_manifest_digest: str,
        configuration_digest: str,
        candidate_plan_digest: str,
        feature_schema_digest: str,
        decision_policy_digest: str,
        random_seed: int = 42,
        k_folds: int = 5,
        linkage_mode: LinkageMode = "link_only",
        assignment_constraint: AssignmentConstraint = "one_to_one",
        selection_config: ModelSelectionConfig | None = None,
        ranking_artifact: Any | None = None,
        ranking_artifact_digest: str | None = None,
        calibrator_methods: Sequence[CalibrationMethod] = ("sigmoid", "isotonic", "beta"),
        approval_status: RecipeApprovalStatus = RecipeApprovalStatus.DRAFT,
        operational_validation: OperationalValidationStatus = (
            OperationalValidationStatus.NOT_ESTABLISHED
        ),
        fs_training_scores: NDArray[np.float64] | None = None,
        fs_validation_scores: NDArray[np.float64] | None = None,
        fs_calibration_scores: NDArray[np.float64] | None = None,
        fs_evidence_digest: str | None = None,
    ) -> PortfolioTournamentResult:
        """Run complete side-by-side tournament across all portfolio candidates."""
        if k_folds < 2:
            raise PipelineError("ML-PIPE-040", "Out-of-fold generation requires at least 2 folds.")
        if training_matrix.partition != "training":
            raise PipelineError("ML-PIPE-041", "Training matrix must belong to training partition.")
        if validation_matrix.partition != "validation":
            raise PipelineError(
                "ML-PIPE-042", "Validation matrix must belong to validation partition."
            )
        if calibration_matrix.partition != "calibration":
            raise PipelineError(
                "ML-PIPE-043", "Calibration matrix must belong to calibration partition."
            )

        sel_config = selection_config or ModelSelectionConfig(
            primary_metric="average_precision",
        )

        active_store = self._store or DuckDBStore()

        fitted_models: dict[str, Any] = {}
        oof_scores: dict[str, NDArray[np.float64]] = {}
        oof_manifests: list[OutOfFoldPredictionManifest] = []
        validation_scores: dict[str, NDArray[np.float64]] = {}
        calibration_scores: dict[str, NDArray[np.float64]] = {}

        # 1. K-Fold Out-of-fold partitioning on the training split
        n_samples = training_matrix.pair_count
        indices = np.arange(n_samples)
        rng = np.random.default_rng(random_seed)
        shuffled_indices = rng.permutation(indices)
        fold_slices = np.array_split(shuffled_indices, k_folds)

        # Base candidate training and OOF score collection
        for candidate in portfolio.pair_candidates:
            if not candidate.enabled:
                continue

            if candidate.family == "fellegi_sunter":
                # Fellegi-Sunter baseline scores
                if fs_training_scores is not None and fs_validation_scores is not None:
                    train_sc = np.asarray(fs_training_scores, dtype=np.float64)
                    val_sc = np.asarray(fs_validation_scores, dtype=np.float64)
                    cal_sc = (
                        np.asarray(fs_calibration_scores, dtype=np.float64)
                        if fs_calibration_scores is not None
                        else np.zeros(calibration_matrix.pair_count, dtype=np.float64)
                    )
                else:
                    # Synthetic/feature-based proxy for reference baseline
                    train_sc = np.clip(np.mean(training_matrix.features, axis=1), 0.0, 1.0)
                    val_sc = np.clip(np.mean(validation_matrix.features, axis=1), 0.0, 1.0)
                    cal_sc = np.clip(np.mean(calibration_matrix.features, axis=1), 0.0, 1.0)

                fs_digest = fs_evidence_digest or _canonical_digest(
                    {"model_id": candidate.model_id, "family": "fellegi_sunter"}
                )
                fitted_models[candidate.model_id] = {
                    "model_id": candidate.model_id,
                    "family": "fellegi_sunter",
                    "model_digest": fs_digest,
                    "model_version": "v1",
                }
                oof_scores[candidate.model_id] = train_sc
                validation_scores[candidate.model_id] = val_sc
                calibration_scores[candidate.model_id] = cal_sc

                oof_manifests.append(
                    OutOfFoldPredictionManifest(
                        model_id=candidate.model_id,
                        model_version="v1",
                        model_artifact_digest=fs_digest,
                        feature_schema_digest=feature_schema_digest,
                        label_authority_digest=training_matrix.label_authority_digest,
                        split_manifest_digest=split_manifest_digest,
                        fold_count=k_folds,
                        pair_count=n_samples,
                        prediction_digest=hashlib.sha256(train_sc.tobytes()).hexdigest(),
                    )
                )

            elif candidate.family == "xgboost":
                xgb_classifier = XGBoostPairClassifier(active_store)
                xgb_config = BoostedTreeModelConfig(
                    model_id=candidate.model_id,
                    implementation="xgboost_classifier",
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                )

                # Generate out-of-fold predictions
                oof_vec = np.zeros(n_samples, dtype=np.float64)
                for fold_idx in range(k_folds):
                    val_idx = fold_slices[fold_idx]
                    train_idx = np.setdiff1d(shuffled_indices, val_idx)

                    fold_train_features = training_matrix.features[train_idx]
                    fold_train_labels = training_matrix.labels[train_idx]
                    fold_val_features = training_matrix.features[val_idx]

                    fold_train_mat = BoostedLabelledMatrix(
                        features=fold_train_features,
                        labels=fold_train_labels,
                        feature_names=training_matrix.feature_names,
                        pair_references=tuple(
                            training_matrix.pair_references[i] for i in train_idx
                        ),
                        pair_digests=tuple(training_matrix.pair_digests[i] for i in train_idx),
                        feature_schema_digest=training_matrix.feature_schema_digest,
                        label_authority_digest=training_matrix.label_authority_digest,
                        selection_digest=_canonical_digest(
                            {"fold": fold_idx, "train_count": len(train_idx)}
                        ),
                        label_source_kind=training_matrix.label_source_kind,
                        partition="training",
                        positive_count=int(np.sum(fold_train_labels == 1)),
                        negative_count=int(np.sum(fold_train_labels == 0)),
                        hard_negative_count=0,
                    )
                    fold_val_feat_mat = BoostedFeatureMatrix(
                        features=fold_val_features,
                        feature_names=training_matrix.feature_names,
                        pair_references=tuple(training_matrix.pair_references[i] for i in val_idx),
                        pair_digests=tuple(training_matrix.pair_digests[i] for i in val_idx),
                        feature_schema_digest=training_matrix.feature_schema_digest,
                    )

                    fold_xgb_model = xgb_classifier.fit(
                        matrix=fold_train_mat,
                        model=xgb_config,
                        random_seed=random_seed + fold_idx,
                        configuration_digest=configuration_digest,
                    )
                    oof_vec[val_idx] = xgb_classifier._predict(
                        matrix=fold_val_feat_mat, model=fold_xgb_model
                    )

                oof_scores[candidate.model_id] = oof_vec

                # Fit on 100% of training data
                full_xgb_model = xgb_classifier.fit(
                    matrix=training_matrix,
                    model=xgb_config,
                    random_seed=random_seed,
                    configuration_digest=configuration_digest,
                )
                fitted_models[candidate.model_id] = full_xgb_model

                oof_manifests.append(
                    OutOfFoldPredictionManifest(
                        model_id=candidate.model_id,
                        model_version=full_xgb_model.model_version,
                        model_artifact_digest=full_xgb_model.model_digest,
                        feature_schema_digest=feature_schema_digest,
                        label_authority_digest=training_matrix.label_authority_digest,
                        split_manifest_digest=split_manifest_digest,
                        fold_count=k_folds,
                        pair_count=n_samples,
                        prediction_digest=hashlib.sha256(oof_vec.tobytes()).hexdigest(),
                    )
                )

                # Validation & Calibration scoring
                val_feat_mat = BoostedFeatureMatrix(
                    features=validation_matrix.features,
                    feature_names=validation_matrix.feature_names,
                    pair_references=validation_matrix.pair_references,
                    pair_digests=validation_matrix.pair_digests,
                    feature_schema_digest=validation_matrix.feature_schema_digest,
                )
                cal_feat_mat = BoostedFeatureMatrix(
                    features=calibration_matrix.features,
                    feature_names=calibration_matrix.feature_names,
                    pair_references=calibration_matrix.pair_references,
                    pair_digests=calibration_matrix.pair_digests,
                    feature_schema_digest=calibration_matrix.feature_schema_digest,
                )
                validation_scores[candidate.model_id] = xgb_classifier._predict(
                    matrix=val_feat_mat, model=full_xgb_model
                )
                calibration_scores[candidate.model_id] = xgb_classifier._predict(
                    matrix=cal_feat_mat, model=full_xgb_model
                )

            elif candidate.family == "lightgbm":
                lgb_classifier = LightGBMPairClassifier(self._store)
                lgb_config = BoostedTreeModelConfig(
                    model_id=candidate.model_id,
                    implementation="lightgbm_classifier",
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                )

                oof_vec = np.zeros(n_samples, dtype=np.float64)
                for fold_idx in range(k_folds):
                    val_idx = fold_slices[fold_idx]
                    train_idx = np.setdiff1d(shuffled_indices, val_idx)

                    fold_train_features = training_matrix.features[train_idx]
                    fold_train_labels = training_matrix.labels[train_idx]
                    fold_val_features = training_matrix.features[val_idx]

                    fold_train_mat = BoostedLabelledMatrix(
                        features=fold_train_features,
                        labels=fold_train_labels,
                        feature_names=training_matrix.feature_names,
                        pair_references=tuple(
                            training_matrix.pair_references[i] for i in train_idx
                        ),
                        pair_digests=tuple(training_matrix.pair_digests[i] for i in train_idx),
                        feature_schema_digest=training_matrix.feature_schema_digest,
                        label_authority_digest=training_matrix.label_authority_digest,
                        selection_digest=_canonical_digest(
                            {"fold": fold_idx, "train_count": len(train_idx)}
                        ),
                        label_source_kind=training_matrix.label_source_kind,
                        partition="training",
                        positive_count=int(np.sum(fold_train_labels == 1)),
                        negative_count=int(np.sum(fold_train_labels == 0)),
                        hard_negative_count=0,
                    )
                    fold_val_feat_mat = BoostedFeatureMatrix(
                        features=fold_val_features,
                        feature_names=training_matrix.feature_names,
                        pair_references=tuple(training_matrix.pair_references[i] for i in val_idx),
                        pair_digests=tuple(training_matrix.pair_digests[i] for i in val_idx),
                        feature_schema_digest=training_matrix.feature_schema_digest,
                    )

                    fold_lgb_model = lgb_classifier.fit(
                        matrix=fold_train_mat,
                        model=lgb_config,
                        random_seed=random_seed + fold_idx,
                        configuration_digest=configuration_digest,
                    )
                    oof_vec[val_idx] = lgb_classifier._predict(
                        matrix=fold_val_feat_mat, model=fold_lgb_model
                    )

                oof_scores[candidate.model_id] = oof_vec
                full_lgb_model = lgb_classifier.fit(
                    matrix=training_matrix,
                    model=lgb_config,
                    random_seed=random_seed,
                    configuration_digest=configuration_digest,
                )
                fitted_models[candidate.model_id] = full_lgb_model

                oof_manifests.append(
                    OutOfFoldPredictionManifest(
                        model_id=candidate.model_id,
                        model_version=full_lgb_model.model_version,
                        model_artifact_digest=full_lgb_model.model_digest,
                        feature_schema_digest=feature_schema_digest,
                        label_authority_digest=training_matrix.label_authority_digest,
                        split_manifest_digest=split_manifest_digest,
                        fold_count=k_folds,
                        pair_count=n_samples,
                        prediction_digest=hashlib.sha256(oof_vec.tobytes()).hexdigest(),
                    )
                )

                val_feat_mat = BoostedFeatureMatrix(
                    features=validation_matrix.features,
                    feature_names=validation_matrix.feature_names,
                    pair_references=validation_matrix.pair_references,
                    pair_digests=validation_matrix.pair_digests,
                    feature_schema_digest=validation_matrix.feature_schema_digest,
                )
                cal_feat_mat = BoostedFeatureMatrix(
                    features=calibration_matrix.features,
                    feature_names=calibration_matrix.feature_names,
                    pair_references=calibration_matrix.pair_references,
                    pair_digests=calibration_matrix.pair_digests,
                    feature_schema_digest=calibration_matrix.feature_schema_digest,
                )
                validation_scores[candidate.model_id] = lgb_classifier._predict(
                    matrix=val_feat_mat, model=full_lgb_model
                )
                calibration_scores[candidate.model_id] = lgb_classifier._predict(
                    matrix=cal_feat_mat, model=full_lgb_model
                )

            elif candidate.family == "pytorch":
                pt_matcher = PyTorchPairMatcher(self._store)
                pt_config = NeuralModelConfig(
                    model_id=candidate.model_id,
                    implementation="pytorch_pair_mlp",
                )

                oof_vec = np.zeros(n_samples, dtype=np.float64)
                for fold_idx in range(k_folds):
                    val_idx = fold_slices[fold_idx]
                    train_idx = np.setdiff1d(shuffled_indices, val_idx)

                    fold_train_features = training_matrix.features[train_idx]
                    fold_train_labels = training_matrix.labels[train_idx]
                    fold_val_features = training_matrix.features[val_idx]

                    fold_train_mat = BoostedLabelledMatrix(
                        features=fold_train_features,
                        labels=fold_train_labels,
                        feature_names=training_matrix.feature_names,
                        pair_references=tuple(
                            training_matrix.pair_references[i] for i in train_idx
                        ),
                        pair_digests=tuple(training_matrix.pair_digests[i] for i in train_idx),
                        feature_schema_digest=training_matrix.feature_schema_digest,
                        label_authority_digest=training_matrix.label_authority_digest,
                        selection_digest=_canonical_digest(
                            {"fold": fold_idx, "train_count": len(train_idx)}
                        ),
                        label_source_kind=training_matrix.label_source_kind,
                        partition="training",
                        positive_count=int(np.sum(fold_train_labels == 1)),
                        negative_count=int(np.sum(fold_train_labels == 0)),
                        hard_negative_count=0,
                    )
                    fold_val_feat_mat = BoostedFeatureMatrix(
                        features=fold_val_features,
                        feature_names=training_matrix.feature_names,
                        pair_references=tuple(training_matrix.pair_references[i] for i in val_idx),
                        pair_digests=tuple(training_matrix.pair_digests[i] for i in val_idx),
                        feature_schema_digest=training_matrix.feature_schema_digest,
                    )

                    fold_pt_model = pt_matcher.fit(
                        matrix=fold_train_mat,
                        model=pt_config,
                        random_seed=random_seed + fold_idx,
                        configuration_digest=configuration_digest,
                    )
                    oof_vec[val_idx] = pt_matcher._predict(
                        matrix=fold_val_feat_mat, model=fold_pt_model
                    )

                oof_scores[candidate.model_id] = oof_vec
                full_pt_model = pt_matcher.fit(
                    matrix=training_matrix,
                    model=pt_config,
                    random_seed=random_seed,
                    configuration_digest=configuration_digest,
                )
                fitted_models[candidate.model_id] = full_pt_model

                oof_manifests.append(
                    OutOfFoldPredictionManifest(
                        model_id=candidate.model_id,
                        model_version=full_pt_model.model_version,
                        model_artifact_digest=full_pt_model.model_digest,
                        feature_schema_digest=feature_schema_digest,
                        label_authority_digest=training_matrix.label_authority_digest,
                        split_manifest_digest=split_manifest_digest,
                        fold_count=k_folds,
                        pair_count=n_samples,
                        prediction_digest=hashlib.sha256(oof_vec.tobytes()).hexdigest(),
                    )
                )

                val_feat_mat = BoostedFeatureMatrix(
                    features=validation_matrix.features,
                    feature_names=validation_matrix.feature_names,
                    pair_references=validation_matrix.pair_references,
                    pair_digests=validation_matrix.pair_digests,
                    feature_schema_digest=validation_matrix.feature_schema_digest,
                )
                cal_feat_mat = BoostedFeatureMatrix(
                    features=calibration_matrix.features,
                    feature_names=calibration_matrix.feature_names,
                    pair_references=calibration_matrix.pair_references,
                    pair_digests=calibration_matrix.pair_digests,
                    feature_schema_digest=calibration_matrix.feature_schema_digest,
                )
                validation_scores[candidate.model_id] = pt_matcher._predict(
                    matrix=val_feat_mat, model=full_pt_model
                )
                calibration_scores[candidate.model_id] = pt_matcher._predict(
                    matrix=cal_feat_mat, model=full_pt_model
                )

        # 2. Train Stacking ensemble if present in portfolio
        for candidate in portfolio.pair_candidates:
            if candidate.enabled and candidate.family == "stacking":
                stacking_classifier = StackingPairClassifier(self._store)
                base_ids = candidate.base_model_ids
                for b_id in base_ids:
                    if b_id not in oof_scores:
                        raise PipelineError(
                            "ML-PIPE-044",
                            f"Base model {b_id} OOF predictions missing for stacking ensemble.",
                        )

                stacking_model = stacking_classifier.fit(
                    base_scores=oof_scores,
                    labels=training_matrix.labels,
                    base_model_ids=base_ids,
                    random_seed=random_seed,
                    model_id=candidate.model_id,
                    configuration_digest=configuration_digest,
                    label_authority_digest=training_matrix.label_authority_digest,
                    selection_digest=training_matrix.selection_digest,
                    label_source_kind=training_matrix.label_source_kind,
                )
                fitted_models[candidate.model_id] = stacking_model

                val_base = {b_id: validation_scores[b_id] for b_id in base_ids}
                cal_base = {b_id: calibration_scores[b_id] for b_id in base_ids}
                validation_scores[candidate.model_id] = stacking_classifier.predict(
                    base_scores=val_base, model=stacking_model
                )
                calibration_scores[candidate.model_id] = stacking_classifier.predict(
                    base_scores=cal_base, model=stacking_model
                )

        # 3. Evaluate all enabled candidates on the validation partition
        eval_candidates: list[ModelEvaluationCandidate] = []
        val_reports: dict[str, PairValidationReport] = {}

        for candidate in portfolio.pair_candidates:
            if not candidate.enabled:
                continue

            v_scores = validation_scores[candidate.model_id]
            report = evaluate_binary_scores(
                labels=validation_matrix.labels,
                scores=v_scores,
                diagnostic_threshold=0.5,
                evaluation_scope="synthetic_mechanical_evaluation",
                partition_manifest_digest=disjointness.manifest_digest,
            )
            val_reports[candidate.model_id] = report

            model_obj = fitted_models[candidate.model_id]
            model_ver = getattr(model_obj, "model_version", "v1")
            model_dig = getattr(
                model_obj,
                "model_digest",
                _canonical_digest({"model_id": candidate.model_id}),
            )
            if isinstance(model_obj, dict):
                model_ver = model_obj.get("model_version", "v1")
                model_dig = model_obj.get("model_digest", "")

            eval_candidates.append(
                ModelEvaluationCandidate(
                    model_family=candidate.family,
                    model_id=candidate.model_id,
                    model_version=model_ver,
                    evidence_digest=model_dig,
                    feature_schema_digest=feature_schema_digest,
                    validation_label_authority_digest=validation_matrix.label_authority_digest,
                    partition_manifest_digest=disjointness.manifest_digest,
                    average_precision=report.average_precision,
                    brier_score=report.brier_score,
                    pair_count=report.pair_count,
                    training_label_authority_digest=training_matrix.label_authority_digest,
                )
            )

        # 4. Champion selection using protected validation partition
        champion_selection = ChampionChallengerSelector.select(
            candidates=eval_candidates,
            config=sel_config,
        )

        champ_id = champion_selection.selected_model_id
        champ_artifact = fitted_models[champ_id]

        # 5. Calibration fitting on protected calibration partition
        cal_score_batch = PairScoreBatch(
            pair_references=calibration_matrix.pair_references,
            pair_digests=calibration_matrix.pair_digests,
            scores=calibration_scores[champ_id],
            labels=calibration_matrix.labels,
            partition="calibration",
            source_model_family=champion_selection.selected_model_family,
            source_model_id=champ_id,
            source_model_version=champion_selection.selected_model_version,
            source_evidence_digest=champion_selection.selected_evidence_digest,
            feature_schema_digest=feature_schema_digest,
            label_authority_digest=calibration_matrix.label_authority_digest,
            partition_manifest_digest=disjointness.manifest_digest,
            champion_selection_digest=champion_selection.selection_digest,
        )

        calibrator_artifact = ChampionCalibratorSelector.select(
            batch=cal_score_batch,
            selection=champion_selection,
            methods=calibrator_methods,
        )

        # 6. Emitting immutable pipeline recipe artifact
        recipe_id = f"recipe_{portfolio.portfolio_id}"
        recipe = PipelineRecipeArtifact(
            recipe_id=recipe_id,
            recipe_version="v1.0.0",
            linkage_mode=linkage_mode,
            assignment_constraint=assignment_constraint,
            configuration_digest=configuration_digest,
            candidate_plan_digest=candidate_plan_digest,
            feature_schema_digest=feature_schema_digest,
            champion_model_id=champion_selection.selected_model_id,
            champion_model_version=champion_selection.selected_model_version,
            champion_artifact_digest=champion_selection.selected_evidence_digest,
            calibrator_digest=calibrator_artifact.calibrator_digest,
            ranking_artifact_digest=ranking_artifact_digest,
            decision_policy_digest=decision_policy_digest,
            validation_evidence_digest=champion_selection.selection_digest,
            approval_status=approval_status,
            operational_validation=operational_validation,
            decision_authority="explicit_policy_only",
            merge_authority="none",
        )

        return PortfolioTournamentResult(
            portfolio=portfolio,
            champion_selection=champion_selection,
            champion_model_artifact=champ_artifact,
            calibrator_artifact=calibrator_artifact,
            oof_manifests=tuple(oof_manifests),
            validation_reports=val_reports,
            recipe=recipe,
            ranking_artifact=ranking_artifact,
        )


__all__ = [
    "ModelPortfolioRunner",
    "PortfolioTournamentResult",
]
