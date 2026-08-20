"""Multi-model portfolio tournament with group-protected out-of-fold stacking."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from mapel_linkage.calibration import (
    BetaCalibrator,
    ChampionCalibratorSelector,
    ChampionChallengerSelector,
    IsotonicCalibrator,
    ModelEvaluationCandidate,
    PairScoreBatch,
    SigmoidCalibrator,
)
from mapel_linkage.calibration.contracts import (
    CalibrationMethod,
    CalibratorArtifact,
    ChampionSelection,
)
from mapel_linkage.configuration.models import ModelsConfig, ModelSelectionConfig
from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.governance.labels import PartitionDisjointnessReport, VerifiedLabelBatch
from mapel_linkage.io.duckdb_store import DuckDBStore
from mapel_linkage.models.boosted import (
    BoostedFeatureMatrix,
    BoostedLabelledMatrix,
    LightGBMModelArtifact,
    LightGBMPairClassifier,
    XGBoostModelArtifact,
    XGBoostPairClassifier,
)
from mapel_linkage.models.ensembles import (
    StackingModelArtifact,
    StackingPairClassifier,
)
from mapel_linkage.models.fellegi_sunter import SplinkNativeModelArtifact
from mapel_linkage.models.neural import (
    PyTorchModelArtifact,
    PyTorchPairMatcher,
)
from mapel_linkage.models.ranking import (
    LightGBMRanker,
    LightGBMRankingArtifact,
    XGBoostCandidateRanker,
    XGBoostRankingArtifact,
    build_ranking_matrix,
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
from mapel_linkage.pipeline.score_evidence import PairScoreEvidenceBatch
from mapel_linkage.pipeline.stage_artifacts import OutOfFoldPredictionManifest
from mapel_linkage.validation import (
    PairValidationReport,
    RankingValidationReport,
    evaluate_binary_scores,
    evaluate_ranking,
)


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_digest(value: str, *, code: str = "ML-PIPE-064") -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PipelineError(code, "A fitted inference artifact digest is invalid.")


def _feature_view(matrix: BoostedLabelledMatrix) -> BoostedFeatureMatrix:
    return BoostedFeatureMatrix(
        features=matrix.features,
        feature_names=matrix.feature_names,
        pair_references=matrix.pair_references,
        pair_digests=matrix.pair_digests,
        feature_schema_digest=matrix.feature_schema_digest,
    )


def _grouped_oof_folds(
    *,
    matrix: BoostedLabelledMatrix,
    labels: VerifiedLabelBatch,
    source_group_digests: Mapping[str, tuple[str, ...]],
    fold_count: int,
    random_seed: int,
) -> tuple[tuple[NDArray[np.int64], ...], int, str]:
    """Create folds protected by source-side entity/household connected components."""

    if (
        labels.partition != "training"
        or labels.label_authority_digest != matrix.label_authority_digest
        or len(labels.labels) != matrix.pair_count
    ):
        raise PipelineError("ML-PIPE-090", "Grouped OOF label provenance is incompatible.")
    by_pair = {(item.left_record_key, item.right_record_key): item for item in labels.labels}
    if set(by_pair) != set(matrix.pair_references):
        raise PipelineError("ML-PIPE-090", "Grouped OOF label provenance is incompatible.")
    ordered_labels = tuple(by_pair[pair] for pair in matrix.pair_references)
    if any(
        int(item.label) != int(label)
        for item, label in zip(ordered_labels, matrix.labels, strict=True)
    ):
        raise PipelineError("ML-PIPE-090", "Grouped OOF label provenance is incompatible.")

    sources = {left for left, _ in matrix.pair_references}
    components_by_source: dict[str, set[str]] = {source: set() for source in sources}
    for (source, _), item in zip(matrix.pair_references, ordered_labels, strict=True):
        components_by_source[source].update(item.entity_component_digests)
        components_by_source[source].update(item.household_component_digests)
    if set(source_group_digests) != sources:
        raise PipelineError("ML-PIPE-090", "Grouped OOF label provenance is incompatible.")
    for source, components in source_group_digests.items():
        if (
            not components
            or len(components) != len(set(components))
            or any(component not in components_by_source[source] for component in components)
        ):
            raise PipelineError("ML-PIPE-090", "Grouped OOF label provenance is incompatible.")
        for component in components:
            _require_digest(component, code="ML-PIPE-090")

    parent = list(range(matrix.pair_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owner_by_source: dict[str, int] = {}
    owner_by_component: dict[str, int] = {}
    for index, (source, _) in enumerate(matrix.pair_references):
        union(index, owner_by_source.setdefault(source, index))
        for component in source_group_digests[source]:
            union(index, owner_by_component.setdefault(component, index))

    grouped: dict[int, list[int]] = {}
    for index in range(matrix.pair_count):
        grouped.setdefault(find(index), []).append(index)
    groups = list(grouped.values())
    if len(groups) < fold_count:
        raise PipelineError("ML-PIPE-090", "Grouped OOF requires at least one group per fold.")

    rng = np.random.default_rng(random_seed)
    shuffled = [groups[index] for index in rng.permutation(len(groups))]
    shuffled.sort(key=len, reverse=True)
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for group in shuffled:
        destination = min(range(fold_count), key=lambda index: (len(folds[index]), index))
        folds[destination].extend(group)
    fold_arrays = tuple(np.asarray(sorted(fold), dtype=np.int64) for fold in folds)
    all_indices = np.arange(matrix.pair_count, dtype=np.int64)
    for holdout in fold_arrays:
        train = np.setdiff1d(all_indices, holdout)
        if len(holdout) == 0 or len(train) == 0 or len(np.unique(matrix.labels[train])) != 2:
            raise PipelineError("ML-PIPE-090", "Grouped OOF cannot preserve training classes.")

    group_assignment_digest = _canonical_digest(
        [
            {
                "fold": fold_index,
                "source_group_digests": sorted(
                    {
                        component
                        for row_index in holdout.tolist()
                        for component in source_group_digests[matrix.pair_references[row_index][0]]
                    }
                ),
            }
            for fold_index, holdout in enumerate(fold_arrays)
        ]
    )
    return fold_arrays, len(groups), group_assignment_digest


def _score_fitted_model(
    *,
    matrix: BoostedFeatureMatrix,
    artifact: FittedModelArtifact,
    fitted_models: dict[str, FittedModelArtifact],
) -> NDArray[np.float64]:
    if isinstance(artifact, XGBoostModelArtifact):
        return XGBoostPairClassifier._predict(matrix=matrix, model=artifact)
    if isinstance(artifact, LightGBMModelArtifact):
        return LightGBMPairClassifier._predict(matrix=matrix, model=artifact)
    if isinstance(artifact, PyTorchModelArtifact):
        return PyTorchPairMatcher._predict(matrix=matrix, model=artifact)
    if isinstance(artifact, StackingModelArtifact):
        base_scores: dict[str, NDArray[np.float64]] = {}
        for model_id in artifact.base_model_ids:
            base = fitted_models.get(model_id)
            if base is None or isinstance(
                base,
                (ReferenceFeatureScoreArtifact, SplinkNativeModelArtifact),
            ):
                raise PipelineError(
                    "ML-PIPE-065",
                    "A stacking inference bundle lacks a replayable base artifact.",
                )
            base_scores[model_id] = _score_fitted_model(
                matrix=matrix,
                artifact=base,
                fitted_models=fitted_models,
            )
        return StackingPairClassifier().predict(base_scores=base_scores, model=artifact)
    raise PipelineError(
        "ML-PIPE-075",
        "Native baseline scoring requires exact score evidence from the native scorer.",
    )


def _apply_calibrator(
    scores: NDArray[np.float64], artifact: CalibratorArtifact
) -> NDArray[np.float64]:
    if artifact.method == "sigmoid":
        return SigmoidCalibrator.apply(scores, artifact)
    if artifact.method == "beta":
        return BetaCalibrator.apply(scores, artifact)
    return IsotonicCalibrator.apply(scores, artifact)


@dataclass(frozen=True, slots=True, repr=False)
class ReferenceFeatureScoreArtifact:
    """Immutable replay contract for the feature-based reference-score baseline."""

    model_id: str
    model_version: Literal["m2-reference-feature-score-v1"]
    model_digest: str
    feature_schema_digest: str
    configuration_digest: str
    source_evidence_digest: str
    scoring_rule: Literal["mean_feature_clip", "external_scores_only"]
    probability_status: Literal["model_score_uncalibrated"] = "model_score_uncalibrated"
    calibration_status: Literal["not_calibrated"] = "not_calibrated"
    decision_authority: Literal["evidence_only"] = "evidence_only"
    merge_authority: Literal["none"] = "none"
    real_data_validation_status: Literal["not_established"] = "not_established"

    def __post_init__(self) -> None:
        for digest in (
            self.feature_schema_digest,
            self.configuration_digest,
            self.source_evidence_digest,
            self.model_digest,
        ):
            _require_digest(digest)
        expected = _canonical_digest(
            {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "feature_schema_digest": self.feature_schema_digest,
                "configuration_digest": self.configuration_digest,
                "source_evidence_digest": self.source_evidence_digest,
                "scoring_rule": self.scoring_rule,
                "decision_authority": self.decision_authority,
                "merge_authority": self.merge_authority,
            }
        )
        if self.model_digest != expected:
            raise PipelineError("ML-PIPE-064", "A fitted inference artifact digest is invalid.")

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        feature_schema_digest: str,
        configuration_digest: str,
        source_evidence_digest: str,
        scoring_rule: Literal["mean_feature_clip", "external_scores_only"],
    ) -> ReferenceFeatureScoreArtifact:
        model_version: Literal["m2-reference-feature-score-v1"] = "m2-reference-feature-score-v1"
        payload = {
            "model_id": model_id,
            "model_version": model_version,
            "feature_schema_digest": feature_schema_digest,
            "configuration_digest": configuration_digest,
            "source_evidence_digest": source_evidence_digest,
            "scoring_rule": scoring_rule,
            "decision_authority": "evidence_only",
            "merge_authority": "none",
        }
        return cls(
            model_id=model_id,
            model_version=model_version,
            model_digest=_canonical_digest(payload),
            feature_schema_digest=feature_schema_digest,
            configuration_digest=configuration_digest,
            source_evidence_digest=source_evidence_digest,
            scoring_rule=scoring_rule,
        )

    def safe_summary(self) -> dict[str, str]:
        """Return aggregate metadata without feature values or record references."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_digest": self.model_digest,
            "feature_schema_digest": self.feature_schema_digest,
            "scoring_rule": self.scoring_rule,
            "decision_authority": self.decision_authority,
            "merge_authority": self.merge_authority,
            "real_data_validation_status": self.real_data_validation_status,
        }

    def __repr__(self) -> str:
        return "<ReferenceFeatureScoreArtifact aggregate-only>"


type FittedBaseArtifact = (
    ReferenceFeatureScoreArtifact
    | SplinkNativeModelArtifact
    | XGBoostModelArtifact
    | LightGBMModelArtifact
    | PyTorchModelArtifact
)
type FittedModelArtifact = FittedBaseArtifact | StackingModelArtifact


def _base_artifact_descriptor(artifact: FittedBaseArtifact) -> dict[str, str]:
    if not isinstance(
        artifact,
        (
            ReferenceFeatureScoreArtifact,
            SplinkNativeModelArtifact,
            XGBoostModelArtifact,
            LightGBMModelArtifact,
            PyTorchModelArtifact,
        ),
    ):
        raise PipelineError(
            "ML-PIPE-065",
            "A stacking inference bundle contains an unsupported base artifact.",
        )
    if artifact.decision_authority != "evidence_only":
        raise PipelineError(
            "ML-PIPE-065",
            "A stacking inference bundle contains an unsupported base artifact.",
        )
    return {
        "model_id": artifact.model_id,
        "model_version": artifact.model_version,
        "model_digest": (
            artifact.artifact_digest
            if isinstance(artifact, SplinkNativeModelArtifact)
            else artifact.model_digest
        ),
        "feature_schema_digest": artifact.feature_schema_digest,
        "configuration_digest": artifact.configuration_digest,
        "artifact_type": type(artifact).__name__,
    }


@dataclass(frozen=True, slots=True, repr=False)
class StackingInferenceArtifactBundle:
    """Exact fitted stacking and base artifacts, hidden behind aggregate provenance."""

    stacking_artifact: StackingModelArtifact = field(repr=False)
    base_artifacts: tuple[FittedBaseArtifact, ...] = field(repr=False)
    feature_schema_digest: str
    bundle_digest: str = ""
    probability_status: Literal["model_score_uncalibrated"] = "model_score_uncalibrated"
    calibration_status: Literal["not_calibrated"] = "not_calibrated"
    decision_authority: Literal["evidence_only"] = "evidence_only"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    def __post_init__(self) -> None:
        _require_digest(self.feature_schema_digest, code="ML-PIPE-066")
        if self.stacking_artifact.decision_authority != "evidence_only":
            raise PipelineError("ML-PIPE-066", "A stacking inference artifact bundle is invalid.")
        descriptors = tuple(_base_artifact_descriptor(item) for item in self.base_artifacts)
        base_ids = tuple(item["model_id"] for item in descriptors)
        if base_ids != self.stacking_artifact.base_model_ids:
            raise PipelineError("ML-PIPE-066", "A stacking inference artifact bundle is invalid.")
        if any(
            item["feature_schema_digest"] != self.feature_schema_digest
            or item["configuration_digest"] != self.stacking_artifact.configuration_digest
            for item in descriptors
        ):
            raise PipelineError("ML-PIPE-066", "A stacking inference artifact bundle is invalid.")
        if any(
            isinstance(item, SplinkNativeModelArtifact)
            or (
                isinstance(item, ReferenceFeatureScoreArtifact)
                and item.scoring_rule != "mean_feature_clip"
            )
            for item in self.base_artifacts
        ):
            raise PipelineError(
                "ML-PIPE-066",
                "A stacking inference artifact bundle lacks a replayable base artifact.",
            )
        expected = _canonical_digest(
            {
                "stacking_model_id": self.stacking_artifact.model_id,
                "stacking_model_version": self.stacking_artifact.model_version,
                "stacking_model_digest": self.stacking_artifact.model_digest,
                "configuration_digest": self.stacking_artifact.configuration_digest,
                "feature_schema_digest": self.feature_schema_digest,
                "base_artifacts": descriptors,
                "probability_status": self.probability_status,
                "calibration_status": self.calibration_status,
                "decision_authority": self.decision_authority,
                "assignment_authority": self.assignment_authority,
                "merge_authority": self.merge_authority,
                "operational_validity": self.operational_validity,
            }
        )
        if self.bundle_digest and self.bundle_digest != expected:
            raise PipelineError("ML-PIPE-066", "A stacking inference artifact bundle is invalid.")
        object.__setattr__(self, "bundle_digest", expected)

    @property
    def model_id(self) -> str:
        return self.stacking_artifact.model_id

    @property
    def model_version(self) -> str:
        return self.stacking_artifact.model_version

    @property
    def model_digest(self) -> str:
        """Use the bundle digest as the recipe-bound champion artifact digest."""
        return self.bundle_digest

    @property
    def configuration_digest(self) -> str:
        return self.stacking_artifact.configuration_digest

    def safe_summary(self) -> dict[str, str | int]:
        """Return aggregate provenance without fitted model payloads or base identities."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "bundle_digest": self.bundle_digest,
            "feature_schema_digest": self.feature_schema_digest,
            "base_artifact_count": len(self.base_artifacts),
            "probability_status": self.probability_status,
            "calibration_status": self.calibration_status,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "operational_validity": self.operational_validity,
        }

    def __repr__(self) -> str:
        return "<StackingInferenceArtifactBundle aggregate-only>"


type ChampionInferenceArtifact = FittedBaseArtifact | StackingInferenceArtifactBundle
type FittedRankingArtifact = XGBoostRankingArtifact | LightGBMRankingArtifact


@dataclass(frozen=True, slots=True, repr=False)
class PortfolioTournamentResult:
    """Outcome of side-by-side model portfolio tournament execution."""

    portfolio: ModelPortfolioDeclaration
    champion_selection: ChampionSelection
    champion_model_artifact: ChampionInferenceArtifact = field(repr=False)
    calibrator_artifact: CalibratorArtifact
    oof_manifests: tuple[OutOfFoldPredictionManifest, ...]
    validation_reports: dict[str, PairValidationReport]
    recipe: PipelineRecipeArtifact
    ranking_artifact: FittedRankingArtifact | Any | None = None
    ranking_validation_reports: dict[str, RankingValidationReport] = field(default_factory=dict)
    locked_test_report: PairValidationReport | None = None
    test_partition_used_for_selection: Literal[False] = False
    test_partition_used_for_calibration: Literal[False] = False
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
            "ranking_candidate_count": len(self.ranking_validation_reports),
            "locked_test_evaluated": self.locked_test_report is not None,
            "test_partition_used_for_selection": self.test_partition_used_for_selection,
            "test_partition_used_for_calibration": self.test_partition_used_for_calibration,
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
        models_config: ModelsConfig,
        training_label_batch: VerifiedLabelBatch,
        training_source_group_digests: Mapping[str, tuple[str, ...]],
        training_matrix: BoostedLabelledMatrix,
        validation_matrix: BoostedLabelledMatrix,
        calibration_matrix: BoostedLabelledMatrix,
        locked_test_matrix: BoostedLabelledMatrix | None = None,
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
        fs_training_evidence: PairScoreEvidenceBatch | None = None,
        fs_validation_evidence: PairScoreEvidenceBatch | None = None,
        fs_calibration_evidence: PairScoreEvidenceBatch | None = None,
        fs_test_evidence: PairScoreEvidenceBatch | None = None,
        fs_model_artifact: ReferenceFeatureScoreArtifact | SplinkNativeModelArtifact | None = None,
        ranking_training_matrix: BoostedLabelledMatrix | None = None,
        ranking_validation_matrix: BoostedLabelledMatrix | None = None,
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
        if locked_test_matrix is not None and locked_test_matrix.partition != "test":
            raise PipelineError("ML-PIPE-072", "Locked test matrix must belong to test partition.")
        if ranking_training_matrix is not None and ranking_training_matrix.partition != "training":
            raise PipelineError("ML-PIPE-073", "Ranking training requires training labels.")
        if (
            ranking_validation_matrix is not None
            and ranking_validation_matrix.partition != "validation"
        ):
            raise PipelineError("ML-PIPE-074", "Ranking selection requires validation labels.")

        sel_config = selection_config or ModelSelectionConfig(
            primary_metric="average_precision",
        )

        active_store = self._store or DuckDBStore()
        boosted_configs = {model.model_id: model for model in models_config.all_boosted_trees()}
        neural_configs = {model.model_id: model for model in models_config.all_neural_models()}
        ensemble_configs = {model.model_id: model for model in models_config.ensembles}

        fitted_models: dict[str, FittedModelArtifact] = {}
        oof_scores: dict[str, NDArray[np.float64]] = {}
        oof_manifests: list[OutOfFoldPredictionManifest] = []
        validation_scores: dict[str, NDArray[np.float64]] = {}
        calibration_scores: dict[str, NDArray[np.float64]] = {}
        fs_locked_test_scores: NDArray[np.float64] | None = None

        # 1. K-Fold Out-of-fold partitioning on the training split
        n_samples = training_matrix.pair_count
        indices = np.arange(n_samples, dtype=np.int64)
        fold_slices, oof_group_count, oof_group_assignment_digest = _grouped_oof_folds(
            matrix=training_matrix,
            labels=training_label_batch,
            source_group_digests=training_source_group_digests,
            fold_count=k_folds,
            random_seed=random_seed,
        )

        # Base candidate training and OOF score collection
        for candidate in portfolio.pair_candidates:
            if not candidate.enabled:
                continue

            if candidate.family == "fellegi_sunter":
                if (
                    fs_training_evidence is None
                    or fs_validation_evidence is None
                    or fs_calibration_evidence is None
                    or fs_model_artifact is None
                ):
                    raise PipelineError(
                        "ML-PIPE-070",
                        "The portfolio baseline requires native or explicitly bound "
                        "score evidence.",
                    )
                fs_artifact_digest = (
                    fs_model_artifact.artifact_digest
                    if isinstance(fs_model_artifact, SplinkNativeModelArtifact)
                    else fs_model_artifact.model_digest
                )
                fs_partition_evidence = [
                    (fs_training_evidence, training_matrix),
                    (fs_validation_evidence, validation_matrix),
                    (fs_calibration_evidence, calibration_matrix),
                ]
                if locked_test_matrix is not None:
                    if fs_test_evidence is None:
                        raise PipelineError(
                            "ML-PIPE-070",
                            "Locked test evaluation requires native baseline score evidence.",
                        )
                    fs_partition_evidence.append((fs_test_evidence, locked_test_matrix))
                for evidence, matrix in fs_partition_evidence:
                    evidence.assert_model_binding(
                        model_id=fs_model_artifact.model_id,
                        model_version=fs_model_artifact.model_version,
                        model_artifact_digest=fs_artifact_digest,
                        configuration_digest=configuration_digest,
                        feature_schema_digest=fs_model_artifact.feature_schema_digest,
                        pair_digests=matrix.pair_digests,
                    )
                train_sc = fs_training_evidence.scores
                val_sc = fs_validation_evidence.scores
                cal_sc = fs_calibration_evidence.scores
                if fs_test_evidence is not None:
                    fs_locked_test_scores = fs_test_evidence.scores
                if (
                    train_sc.shape != (training_matrix.pair_count,)
                    or val_sc.shape != (validation_matrix.pair_count,)
                    or cal_sc.shape != (calibration_matrix.pair_count,)
                    or fs_model_artifact.model_id != candidate.model_id
                    or fs_model_artifact.configuration_digest != configuration_digest
                ):
                    raise PipelineError(
                        "ML-PIPE-070",
                        "The portfolio baseline score evidence is incompatible.",
                    )
                fitted_models[candidate.model_id] = fs_model_artifact
                validation_scores[candidate.model_id] = val_sc
                calibration_scores[candidate.model_id] = cal_sc

            elif candidate.family == "xgboost":
                xgb_classifier = XGBoostPairClassifier(active_store)
                xgb_config = boosted_configs.get(candidate.model_id)
                if xgb_config is None or xgb_config.implementation != "xgboost_classifier":
                    raise PipelineError("ML-PIPE-071", "A configured pair model is missing.")

                # Generate out-of-fold predictions
                oof_vec = np.zeros(n_samples, dtype=np.float64)
                for fold_idx in range(k_folds):
                    val_idx = fold_slices[fold_idx]
                    train_idx = np.setdiff1d(indices, val_idx)

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
                        group_count=oof_group_count,
                        pair_count=n_samples,
                        prediction_digest=hashlib.sha256(oof_vec.tobytes()).hexdigest(),
                        group_assignment_digest=oof_group_assignment_digest,
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
                lgb_config = boosted_configs.get(candidate.model_id)
                if lgb_config is None or lgb_config.implementation != "lightgbm_classifier":
                    raise PipelineError("ML-PIPE-071", "A configured pair model is missing.")

                oof_vec = np.zeros(n_samples, dtype=np.float64)
                for fold_idx in range(k_folds):
                    val_idx = fold_slices[fold_idx]
                    train_idx = np.setdiff1d(indices, val_idx)

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
                        group_count=oof_group_count,
                        pair_count=n_samples,
                        prediction_digest=hashlib.sha256(oof_vec.tobytes()).hexdigest(),
                        group_assignment_digest=oof_group_assignment_digest,
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
                pt_config = neural_configs.get(candidate.model_id)
                if pt_config is None:
                    raise PipelineError("ML-PIPE-071", "A configured pair model is missing.")

                oof_vec = np.zeros(n_samples, dtype=np.float64)
                for fold_idx in range(k_folds):
                    val_idx = fold_slices[fold_idx]
                    train_idx = np.setdiff1d(indices, val_idx)

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
                        group_count=oof_group_count,
                        pair_count=n_samples,
                        prediction_digest=hashlib.sha256(oof_vec.tobytes()).hexdigest(),
                        group_assignment_digest=oof_group_assignment_digest,
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
                stacking_config = ensemble_configs.get(candidate.model_id)
                if stacking_config is None or n_samples > stacking_config.maximum_training_pairs:
                    raise PipelineError("ML-PIPE-071", "A configured ensemble model is missing.")
                for b_id in base_ids:
                    if isinstance(
                        fitted_models.get(b_id),
                        (ReferenceFeatureScoreArtifact, SplinkNativeModelArtifact),
                    ):
                        raise PipelineError(
                            "ML-PIPE-044",
                            "A non-OOF baseline cannot be used as a stacking base model.",
                        )
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
            model_evidence_digest = (
                model_obj.artifact_digest
                if isinstance(model_obj, SplinkNativeModelArtifact)
                else model_obj.model_digest
            )
            model_feature_schema_digest = (
                model_obj.feature_schema_digest
                if isinstance(
                    model_obj,
                    (
                        ReferenceFeatureScoreArtifact,
                        SplinkNativeModelArtifact,
                        XGBoostModelArtifact,
                        LightGBMModelArtifact,
                        PyTorchModelArtifact,
                    ),
                )
                else training_matrix.feature_schema_digest
            )

            eval_candidates.append(
                ModelEvaluationCandidate(
                    model_family=candidate.family,
                    model_id=candidate.model_id,
                    model_version=model_obj.model_version,
                    evidence_digest=model_evidence_digest,
                    feature_schema_digest=model_feature_schema_digest,
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
        selected_artifact = fitted_models[champ_id]
        recipe_champion_artifact_digest = champion_selection.selected_evidence_digest
        champion_artifact: ChampionInferenceArtifact
        if isinstance(selected_artifact, StackingModelArtifact):
            base_artifacts: list[FittedBaseArtifact] = []
            for base_id in selected_artifact.base_model_ids:
                base_artifact = fitted_models.get(base_id)
                if not isinstance(
                    base_artifact,
                    (
                        ReferenceFeatureScoreArtifact,
                        SplinkNativeModelArtifact,
                        XGBoostModelArtifact,
                        LightGBMModelArtifact,
                        PyTorchModelArtifact,
                    ),
                ):
                    raise PipelineError(
                        "ML-PIPE-065",
                        "A stacking inference bundle contains an unsupported base artifact.",
                    )
                base_artifacts.append(base_artifact)
            champion_artifact = StackingInferenceArtifactBundle(
                stacking_artifact=selected_artifact,
                base_artifacts=tuple(base_artifacts),
                feature_schema_digest=feature_schema_digest,
            )
            recipe_champion_artifact_digest = champion_artifact.bundle_digest
        else:
            champion_artifact = selected_artifact

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
            feature_schema_digest=champion_selection.selected_feature_schema_digest,
            label_authority_digest=calibration_matrix.label_authority_digest,
            partition_manifest_digest=disjointness.manifest_digest,
            champion_selection_digest=champion_selection.selection_digest,
        )

        calibrator_artifact = ChampionCalibratorSelector.select(
            batch=cal_score_batch,
            selection=champion_selection,
            methods=calibrator_methods,
        )

        # 6. Evaluate the frozen champion on locked test labels. These values are
        # intentionally unavailable until after selection and calibration are complete.
        locked_test_report: PairValidationReport | None = None
        if locked_test_matrix is not None:
            if isinstance(
                selected_artifact, (ReferenceFeatureScoreArtifact, SplinkNativeModelArtifact)
            ):
                if fs_locked_test_scores is None:
                    raise PipelineError(
                        "ML-PIPE-075",
                        "Native baseline test scoring requires native score evidence.",
                    )
                locked_test_raw = fs_locked_test_scores
            else:
                locked_test_raw = _score_fitted_model(
                    matrix=_feature_view(locked_test_matrix),
                    artifact=selected_artifact,
                    fitted_models=fitted_models,
                )
            locked_test_probabilities = _apply_calibrator(
                locked_test_raw,
                calibrator_artifact,
            )
            locked_test_report = replace(
                evaluate_binary_scores(
                    labels=locked_test_matrix.labels,
                    scores=locked_test_probabilities,
                    diagnostic_threshold=0.5,
                    evaluation_scope="synthetic_mechanical_evaluation",
                    partition_manifest_digest=disjointness.manifest_digest,
                ),
                calibration_status="calibrated_on_protected_partition",
            )

        # 7. Train configured rankers on training only and select using validation only.
        ranker_reports: dict[str, RankingValidationReport] = {}
        if portfolio.ranking_candidates:
            if ranking_artifact is not None or ranking_artifact_digest is not None:
                raise PipelineError("ML-PIPE-076", "Ranking evidence path is ambiguous.")
            if ranking_training_matrix is None or ranking_validation_matrix is None:
                raise PipelineError(
                    "ML-PIPE-077",
                    "Configured rankers require protected training and validation matrices.",
                )
            ranking_configs = {
                model.model_id: model for model in models_config.all_ranking_models()
            }
            fitted_rankers: dict[str, FittedRankingArtifact] = {}
            for ranking_candidate in portfolio.ranking_candidates:
                if not ranking_candidate.enabled:
                    continue
                ranking_config = ranking_configs.get(ranking_candidate.model_id)
                if (
                    ranking_config is None
                    or ranking_config.implementation != ranking_candidate.implementation
                    or ranking_config.query_side != ranking_candidate.query_side
                    or ranking_config.top_k != ranking_candidate.top_k
                ):
                    raise PipelineError("ML-PIPE-078", "A configured ranker is missing.")
                train_rank = build_ranking_matrix(
                    ranking_training_matrix,
                    query_side=ranking_config.query_side,
                )
                validation_rank = build_ranking_matrix(
                    ranking_validation_matrix,
                    query_side=ranking_config.query_side,
                )
                fitted_ranker: FittedRankingArtifact
                if ranking_candidate.family == "xgboost":
                    fitted_xgb_ranker = XGBoostCandidateRanker.fit(
                        matrix=train_rank,
                        model=ranking_config,
                        random_seed=random_seed,
                        configuration_digest=configuration_digest,
                    )
                    rank_scores = XGBoostCandidateRanker.score(
                        matrix=validation_rank,
                        model=fitted_xgb_ranker,
                    )
                    fitted_ranker = fitted_xgb_ranker
                else:
                    fitted_lgb_ranker = LightGBMRanker.fit(
                        matrix=train_rank,
                        model=ranking_config,
                        random_seed=random_seed,
                        configuration_digest=configuration_digest,
                    )
                    rank_scores = LightGBMRanker.score(
                        matrix=validation_rank,
                        model=fitted_lgb_ranker,
                    )
                    fitted_ranker = fitted_lgb_ranker
                true_pair_digests = frozenset(
                    digest
                    for digest, relevance in zip(
                        validation_rank.pair_digests,
                        validation_rank.relevance,
                        strict=True,
                    )
                    if relevance > 0.0
                )
                eligible_query_keys = tuple(sorted(set(validation_rank.query_keys)))
                ranker_reports[ranking_candidate.model_id] = evaluate_ranking(
                    scores=rank_scores,
                    true_pair_digests=true_pair_digests,
                    eligible_query_keys=eligible_query_keys,
                    k_values=tuple(sorted({1, ranking_candidate.top_k})),
                )
                fitted_rankers[ranking_candidate.model_id] = fitted_ranker
            if not fitted_rankers:
                raise PipelineError("ML-PIPE-079", "The ranking portfolio has no candidate.")
            executable_rankers = {
                model_id: artifact
                for model_id, artifact in fitted_rankers.items()
                if artifact.query_side == "source"
            }
            if executable_rankers:
                selected_ranker_id = min(
                    executable_rankers,
                    key=lambda model_id: (
                        -ranker_reports[model_id].mean_reciprocal_rank,
                        -ranker_reports[model_id].top1_fraction,
                        model_id,
                    ),
                )
                ranking_artifact = executable_rankers[selected_ranker_id]
                ranking_artifact_digest = ranking_artifact.artifact_digest

        # 8. Emit immutable pipeline recipe artifact.
        recipe_id = f"recipe_{portfolio.portfolio_id}"
        recipe = PipelineRecipeArtifact(
            recipe_id=recipe_id,
            recipe_version="v1.0.0",
            linkage_mode=linkage_mode,
            assignment_constraint=assignment_constraint,
            configuration_digest=configuration_digest,
            candidate_plan_digest=candidate_plan_digest,
            feature_schema_digest=champion_selection.selected_feature_schema_digest,
            champion_model_id=champion_selection.selected_model_id,
            champion_model_version=champion_selection.selected_model_version,
            champion_artifact_digest=recipe_champion_artifact_digest,
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
            champion_model_artifact=champion_artifact,
            calibrator_artifact=calibrator_artifact,
            oof_manifests=tuple(oof_manifests),
            validation_reports=val_reports,
            recipe=recipe,
            ranking_artifact=ranking_artifact,
            ranking_validation_reports=ranker_reports,
            locked_test_report=locked_test_report,
        )


__all__ = [
    "ModelPortfolioRunner",
    "PortfolioTournamentResult",
    "ReferenceFeatureScoreArtifact",
    "StackingInferenceArtifactBundle",
]
