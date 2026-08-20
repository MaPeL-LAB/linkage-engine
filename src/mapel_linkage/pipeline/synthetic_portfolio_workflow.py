"""Configuration-driven all-model tournament on generated synthetic evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from mapel_linkage.calibration import (
    CalibratorArtifact,
    read_calibrator_artifact,
    write_calibrator_artifact,
)
from mapel_linkage.candidate_generation import DuckDBCandidateGenerator
from mapel_linkage.comparisons import DuckDBComparisonFeatureBuilder
from mapel_linkage.configuration import ExecutionPlan
from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.labels import VerifiedLabelBatch
from mapel_linkage.io import DuckDBStore
from mapel_linkage.models.boosted import (
    BoostedFeatureMatrix,
    BoostedLabelledMatrix,
    DuckDBVerifiedMatrixBuilder,
    LightGBMModelArtifact,
    XGBoostModelArtifact,
    read_lightgbm_artifact,
    read_xgboost_artifact,
    write_lightgbm_artifact,
    write_xgboost_artifact,
)
from mapel_linkage.models.ensembles import read_stacking_artifact, write_stacking_artifact
from mapel_linkage.models.fellegi_sunter import (
    SplinkCandidateParityChecker,
    SplinkNativeDuckDBMatcher,
    SplinkNativeModelArtifact,
    SplinkSettingsPlan,
    SplinkSettingsPlanCompiler,
    assert_splink_native_recipe_binding,
    deserialize_splink_native_model,
    serialize_splink_native_model,
)
from mapel_linkage.models.neural import (
    PyTorchModelArtifact,
    read_pytorch_artifact,
    write_pytorch_artifact,
)
from mapel_linkage.models.ranking import (
    LightGBMRankingArtifact,
    XGBoostRankingArtifact,
    read_lightgbm_ranker_artifact,
    read_ranking_artifact,
    write_lightgbm_ranker_artifact,
    write_ranking_artifact,
)
from mapel_linkage.pipeline.inference_runner import (
    ApprovedRecipeInferenceResult,
    NativeSplinkInferenceReplay,
    attest_generated_synthetic_inference,
    infer_with_approved_recipe,
)
from mapel_linkage.pipeline.model_portfolio import compile_model_portfolio
from mapel_linkage.pipeline.portfolio_runner import (
    ChampionInferenceArtifact,
    ModelPortfolioRunner,
    PortfolioTournamentResult,
    ReferenceFeatureScoreArtifact,
    StackingInferenceArtifactBundle,
)
from mapel_linkage.pipeline.recipe_io import (
    deserialize_pipeline_recipe,
    serialize_pipeline_recipe,
)
from mapel_linkage.pipeline.recipes import (
    OperationalValidationStatus,
    PipelineRecipeArtifact,
    RecipeApprovalStatus,
    RecipeExecutionMode,
)
from mapel_linkage.pipeline.score_evidence import issue_native_splink_score_evidence
from mapel_linkage.pipeline.synthetic_workflow_support import (
    candidate_snapshot,
    protected_label_batches,
    runtime_blocking_rules,
    source_target_ids,
    synthetic_fixture_directory,
    synthetic_truth_records,
)
from mapel_linkage.preprocessing import ConfiguredDatasetPreparer, surrogate_record_key
from mapel_linkage.synthetic import (
    SyntheticBundle,
    SyntheticGenerationConfig,
    generate_synthetic_bundle,
    write_synthetic_bundle,
)
from mapel_linkage.validation import PairValidationReport

_SYNTHETIC_SEED = 20260816


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _feature_view(matrix: BoostedLabelledMatrix) -> BoostedFeatureMatrix:
    return BoostedFeatureMatrix(
        features=matrix.features,
        pair_references=matrix.pair_references,
        pair_digests=matrix.pair_digests,
        feature_names=matrix.feature_names,
        feature_schema_digest=matrix.feature_schema_digest,
    )


def _read_text_artifact(path: Path, *, maximum_bytes: int) -> str:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
            raise OSError
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise PipelineError(
            "ML-PIPE-081", "A portfolio artifact could not be read safely."
        ) from None


def _persist_reload_native(
    *,
    artifact: SplinkNativeModelArtifact,
    path: str,
    plan: ExecutionPlan,
    settings_plan: SplinkSettingsPlan,
) -> SplinkNativeModelArtifact:
    destination = plan.path_policy.resolve_output(path)
    if destination.suffix != ".json" or destination.is_symlink():
        raise PipelineError("ML-PIPE-081", "A portfolio artifact path is invalid.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, serialize_splink_native_model(artifact))
    return deserialize_splink_native_model(
        _read_text_artifact(destination, maximum_bytes=16 * 1024 * 1024),
        settings_plan=settings_plan,
        model=plan.config.models.fellegi_sunter,
        configuration_digest=plan.configuration_digest,
        feature_schema_digest=artifact.feature_schema_digest,
        random_seed=plan.random_seed,
    )


def _persist_reload_base(
    *,
    artifact: XGBoostModelArtifact | LightGBMModelArtifact | PyTorchModelArtifact,
    base: str,
    plan: ExecutionPlan,
) -> XGBoostModelArtifact | LightGBMModelArtifact | PyTorchModelArtifact:
    if isinstance(artifact, XGBoostModelArtifact):
        write_xgboost_artifact(
            artifact=artifact,
            model_path=f"{base}.json",
            manifest_path=f"{base}.manifest.json",
            policy=plan.path_policy,
        )
        return read_xgboost_artifact(
            model_path=f"{base}.json",
            manifest_path=f"{base}.manifest.json",
            policy=plan.path_policy,
        )
    if isinstance(artifact, LightGBMModelArtifact):
        write_lightgbm_artifact(
            artifact=artifact,
            model_path=f"{base}.txt",
            manifest_path=f"{base}.manifest.json",
            policy=plan.path_policy,
        )
        return read_lightgbm_artifact(
            model_path=f"{base}.txt",
            manifest_path=f"{base}.manifest.json",
            policy=plan.path_policy,
        )
    write_pytorch_artifact(
        artifact=artifact,
        model_path=f"{base}.pt",
        manifest_path=f"{base}.manifest.json",
        policy=plan.path_policy,
    )
    return read_pytorch_artifact(
        model_path=f"{base}.pt",
        manifest_path=f"{base}.manifest.json",
        policy=plan.path_policy,
    )


def _persist_reload_champion(
    *,
    champion: ChampionInferenceArtifact,
    base: str,
    plan: ExecutionPlan,
    settings_plan: SplinkSettingsPlan,
) -> ChampionInferenceArtifact:
    if isinstance(champion, ReferenceFeatureScoreArtifact):
        raise PipelineError(
            "ML-PIPE-082", "Reference proxy artifacts are not persistable champions."
        )
    if isinstance(champion, SplinkNativeModelArtifact):
        return _persist_reload_native(
            artifact=champion,
            path=f"{base}/native_splink.json",
            plan=plan,
            settings_plan=settings_plan,
        )
    if isinstance(champion, (XGBoostModelArtifact, LightGBMModelArtifact, PyTorchModelArtifact)):
        return _persist_reload_base(
            artifact=champion,
            base=f"{base}/champion",
            plan=plan,
        )
    if not isinstance(champion, StackingInferenceArtifactBundle):
        raise PipelineError("ML-PIPE-082", "The champion artifact type is unsupported.")
    write_stacking_artifact(
        artifact=champion.stacking_artifact,
        model_path=f"{base}/stacking.json",
        manifest_path=f"{base}/stacking.manifest.json",
        policy=plan.path_policy,
    )
    reloaded_stacking = read_stacking_artifact(
        model_path=f"{base}/stacking.json",
        manifest_path=f"{base}/stacking.manifest.json",
        policy=plan.path_policy,
    )
    reloaded_bases = tuple(
        _persist_reload_base(
            artifact=artifact,
            base=f"{base}/base_{index}",
            plan=plan,
        )
        for index, artifact in enumerate(champion.base_artifacts)
        if isinstance(artifact, (XGBoostModelArtifact, LightGBMModelArtifact, PyTorchModelArtifact))
    )
    if len(reloaded_bases) != len(champion.base_artifacts):
        raise PipelineError("ML-PIPE-082", "A stacking champion contains a non-replayable base.")
    reloaded = StackingInferenceArtifactBundle(
        stacking_artifact=reloaded_stacking,
        base_artifacts=reloaded_bases,
        feature_schema_digest=champion.feature_schema_digest,
    )
    if reloaded.bundle_digest != champion.bundle_digest:
        raise PipelineError("ML-PIPE-082", "The stacking champion bundle failed reload integrity.")
    return reloaded


def _persist_reload_ranker(
    *,
    artifact: XGBoostRankingArtifact | LightGBMRankingArtifact | None,
    base: str,
    plan: ExecutionPlan,
) -> XGBoostRankingArtifact | LightGBMRankingArtifact | None:
    if artifact is None:
        return None
    if isinstance(artifact, XGBoostRankingArtifact):
        write_ranking_artifact(
            artifact=artifact,
            model_path=f"{base}/ranker.json",
            manifest_path=f"{base}/ranker.manifest.json",
            policy=plan.path_policy,
        )
        return read_ranking_artifact(
            model_path=f"{base}/ranker.json",
            manifest_path=f"{base}/ranker.manifest.json",
            policy=plan.path_policy,
        )
    write_lightgbm_ranker_artifact(
        artifact=artifact,
        model_path=f"{base}/ranker.txt",
        manifest_path=f"{base}/ranker.manifest.json",
        policy=plan.path_policy,
    )
    return read_lightgbm_ranker_artifact(
        model_path=f"{base}/ranker.txt",
        manifest_path=f"{base}/ranker.manifest.json",
        policy=plan.path_policy,
    )


def _persist_reload_calibrator(
    *, artifact: CalibratorArtifact, base: str, plan: ExecutionPlan
) -> CalibratorArtifact:
    write_calibrator_artifact(
        artifact=artifact,
        payload_path=f"{base}/calibrator.json",
        manifest_path=f"{base}/calibrator.manifest.json",
        policy=plan.path_policy,
    )
    return read_calibrator_artifact(
        payload_path=f"{base}/calibrator.json",
        manifest_path=f"{base}/calibrator.manifest.json",
        policy=plan.path_policy,
    )


def _persist_reload_recipe(
    *, recipe: PipelineRecipeArtifact, base: str, plan: ExecutionPlan
) -> PipelineRecipeArtifact:
    destination = plan.path_policy.resolve_output(f"{base}/recipe-v1.json")
    if destination.suffix != ".json" or destination.is_symlink():
        raise PipelineError("ML-PIPE-081", "A portfolio artifact path is invalid.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, serialize_pipeline_recipe(recipe))
    return deserialize_pipeline_recipe(_read_text_artifact(destination, maximum_bytes=262_144))


def _raw_reference_matrix(
    *,
    matrix: BoostedFeatureMatrix | BoostedLabelledMatrix,
    bundle: SyntheticBundle,
    source_id: str,
    target_id: str,
) -> BoostedFeatureMatrix:
    source_map = {
        surrogate_record_key(source_id, record.record_key): record.record_key
        for record in bundle.source_a
    }
    target_map = {
        surrogate_record_key(target_id, record.record_key): record.record_key
        for record in bundle.source_b
    }
    try:
        references = tuple(
            (source_map[left], target_map[right]) for left, right in matrix.pair_references
        )
    except KeyError:
        raise PipelineError(
            "ML-PIPE-083", "Synthetic inference pair provenance is invalid."
        ) from None
    return BoostedFeatureMatrix(
        features=matrix.features,
        pair_references=references,
        pair_digests=tuple(
            hashlib.sha256(f"{left}\x00{right}".encode()).hexdigest() for left, right in references
        ),
        feature_names=matrix.feature_names,
        feature_schema_digest=matrix.feature_schema_digest,
    )


def _disjoint_decision_evidence(
    matrix: BoostedLabelledMatrix,
) -> tuple[BoostedFeatureMatrix, BoostedFeatureMatrix]:
    """Split decision evidence by source query without touching protected test data."""

    source_queries = tuple(sorted({left for left, _ in matrix.pair_references}))
    if len(source_queries) < 2:
        raise PipelineError("ML-PIPE-091", "Disjoint synthetic inference evidence is unavailable.")
    review_sources = set(source_queries[::2])
    replay_sources = set(source_queries[1::2])

    def subset(sources: set[str]) -> BoostedFeatureMatrix:
        indices = [
            index for index, (left, _) in enumerate(matrix.pair_references) if left in sources
        ]
        if not indices:
            raise PipelineError(
                "ML-PIPE-091", "Disjoint synthetic inference evidence is unavailable."
            )
        return BoostedFeatureMatrix(
            features=matrix.features[indices],
            pair_references=tuple(matrix.pair_references[index] for index in indices),
            pair_digests=tuple(matrix.pair_digests[index] for index in indices),
            feature_names=matrix.feature_names,
            feature_schema_digest=matrix.feature_schema_digest,
        )

    review, replay = subset(review_sources), subset(replay_sources)
    if set(review.pair_digests) & set(replay.pair_digests):
        raise PipelineError("ML-PIPE-091", "Disjoint synthetic inference evidence is unavailable.")
    return review, replay


@dataclass(frozen=True, slots=True, repr=False)
class SyntheticPortfolioWorkflowResult:
    """Aggregate outcome with fitted and row-level objects hidden from repr."""

    tournament: PortfolioTournamentResult = field(repr=False)
    persisted_champion: ChampionInferenceArtifact = field(repr=False)
    persisted_calibrator: CalibratorArtifact = field(repr=False)
    persisted_recipe: PipelineRecipeArtifact = field(repr=False)
    persisted_ranker: XGBoostRankingArtifact | LightGBMRankingArtifact | None = field(repr=False)
    review_inference: ApprovedRecipeInferenceResult = field(repr=False)
    inference: ApprovedRecipeInferenceResult = field(repr=False)
    protected_label_batches: tuple[VerifiedLabelBatch, ...] = field(repr=False)
    locked_test_report: PairValidationReport
    run_id: str
    pair_candidate_count: int
    ranking_candidate_count: int
    inference_status: Literal["replayed"]
    workflow_digest: str
    operational_validity: Literal["not_established"] = "not_established"
    merge_authority: Literal["none"] = "none"

    def safe_summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "portfolio_id": self.tournament.portfolio.portfolio_id,
            "champion_model_id": self.tournament.champion_selection.selected_model_id,
            "champion_model_family": self.tournament.champion_selection.selected_model_family,
            "pair_candidate_count": self.pair_candidate_count,
            "ranking_candidate_count": self.ranking_candidate_count,
            "locked_test_pair_count": self.locked_test_report.pair_count,
            "protected_partition_count": len(self.protected_label_batches),
            "test_partition_used_for_selection": False,
            "test_partition_used_for_calibration": False,
            "inference_status": self.inference_status,
            "review_inference_pair_count": self.review_inference.pair_count,
            "inference_pair_count": self.inference.pair_count,
            "recipe_digest": self.persisted_recipe.recipe_digest,
            "workflow_digest": self.workflow_digest,
            "operational_validity": self.operational_validity,
            "merge_authority": self.merge_authority,
        }


class SyntheticPortfolioWorkflowRunner:
    """Run the configured model/ranker portfolio with protected synthetic partitions."""

    @staticmethod
    def run(
        plan: ExecutionPlan,
        *,
        generation: SyntheticGenerationConfig | None = None,
        k_folds: int = 3,
    ) -> SyntheticPortfolioWorkflowResult:
        if plan.random_seed != _SYNTHETIC_SEED:
            raise PipelineError("ML-PIPE-084", "The all-model workflow requires seed 20260816.")
        source_id, target_id = source_target_ids(plan)
        fixture_directory = synthetic_fixture_directory(
            plan,
            source_id=source_id,
            target_id=target_id,
        )
        spec = generation or SyntheticGenerationConfig(seed=_SYNTHETIC_SEED)
        if spec.seed != _SYNTHETIC_SEED:
            raise PipelineError("ML-PIPE-084", "The all-model workflow requires seed 20260816.")
        bundle = generate_synthetic_bundle(spec)
        write_synthetic_bundle(fixture_directory, bundle)
        portfolio = compile_model_portfolio(plan.config)
        run_id = _canonical_digest(
            {
                "configuration_digest": plan.configuration_digest,
                "portfolio_digest": portfolio.portfolio_digest,
                "synthetic_provenance": asdict(bundle.provenance),
            }
        )[:24]

        with DuckDBStore() as store:
            catalog = ConfiguredDatasetPreparer(store).prepare_all(plan)
            left = catalog.require(source_id)
            right = catalog.require(target_id)
            candidates = DuckDBCandidateGenerator(store).generate(
                left=left.table,
                right=right.table,
                variable_columns=left.variable_columns,
                rules=runtime_blocking_rules(plan),
                maximum_candidate_pairs=plan.config.runtime.maximum_candidate_pairs,
            )
            snapshot = candidate_snapshot(store, candidates.table.table_name)
            settings_plan = SplinkSettingsPlanCompiler().compile(
                left=left,
                right=right,
                comparisons=plan.config.comparisons,
                blocking_rules=tuple(rule.predicate for rule in plan.config.blocking.rules),
                model=plan.config.models.fellegi_sunter,
            )
            parity = SplinkCandidateParityChecker.check(
                store=store,
                left=left,
                right=right,
                settings_plan=settings_plan,
                expected_pairs=snapshot.pairs,
            )
            features = DuckDBComparisonFeatureBuilder(store).build(
                candidates=candidates.table,
                left=left,
                right=right,
                comparisons=plan.config.comparisons,
            )
            truth_records = synthetic_truth_records(
                bundle,
                source_dataset_id=source_id,
                target_dataset_id=target_id,
            )
            batches, disjointness, split_manifest_digest = protected_label_batches(
                candidate_pairs=snapshot.pairs,
                truth_records=truth_records,
                plan=plan,
            )
            matrix_builder = DuckDBVerifiedMatrixBuilder(store)
            matrices = {
                partition: matrix_builder.build_labelled(
                    features=features,
                    labels=batches[partition],
                    random_seed=plan.random_seed,
                )
                for partition in ("training", "validation", "calibration", "decision", "test")
            }
            native_matcher = SplinkNativeDuckDBMatcher(store)
            trained_native = native_matcher.fit(
                left=left,
                right=right,
                settings_plan=settings_plan,
                model=plan.config.models.fellegi_sunter,
                configuration_digest=plan.configuration_digest,
                expected_pairs=snapshot.pairs,
                maximum_candidate_pairs=plan.config.runtime.maximum_candidate_pairs,
                random_seed=plan.random_seed,
            )
            native_model = deserialize_splink_native_model(
                serialize_splink_native_model(trained_native),
                settings_plan=settings_plan,
                model=plan.config.models.fellegi_sunter,
                configuration_digest=plan.configuration_digest,
                feature_schema_digest=trained_native.feature_schema_digest,
                random_seed=plan.random_seed,
            )
            native_scores = native_matcher.score(
                left=left,
                right=right,
                settings_plan=settings_plan,
                artifact=native_model,
                expected_pairs=snapshot.pairs,
                maximum_candidate_pairs=plan.config.runtime.maximum_candidate_pairs,
            )
            native_evidence = {
                partition: issue_native_splink_score_evidence(
                    store=store,
                    score_result=native_scores,
                    model_artifact=native_model,
                    pair_references=matrices[partition].pair_references,
                    pair_digests=matrices[partition].pair_digests,
                )
                for partition in ("training", "validation", "calibration", "test")
            }
            tournament = ModelPortfolioRunner(store).run_tournament(
                portfolio=portfolio,
                models_config=plan.config.models,
                training_label_batch=batches["training"],
                training_source_group_digests={
                    record.record_key: (
                        record.entity_digest,
                        *(
                            (record.household_digest,)
                            if record.household_digest is not None
                            else ()
                        ),
                    )
                    for record in truth_records
                    if record.dataset_id == source_id
                    and record.record_key
                    in {left for left, _ in matrices["training"].pair_references}
                },
                training_matrix=matrices["training"],
                validation_matrix=matrices["validation"],
                calibration_matrix=matrices["calibration"],
                locked_test_matrix=matrices["test"],
                disjointness=disjointness,
                split_manifest_digest=split_manifest_digest,
                configuration_digest=plan.configuration_digest,
                candidate_plan_digest=parity.pair_set_digest,
                feature_schema_digest=matrices["training"].feature_schema_digest,
                decision_policy_digest=_canonical_digest(
                    plan.config.decision_policy.model_dump(mode="json")
                ),
                random_seed=plan.random_seed,
                k_folds=k_folds,
                linkage_mode=plan.config.project.linkage_mode,
                assignment_constraint=plan.config.project.assignment_constraint,
                selection_config=plan.config.model_selection,
                calibrator_methods=(plan.config.calibration.method,),
                approval_status=RecipeApprovalStatus.SYNTHETIC_VALIDATED,
                operational_validation=OperationalValidationStatus.NOT_ESTABLISHED,
                fs_training_evidence=native_evidence["training"],
                fs_validation_evidence=native_evidence["validation"],
                fs_calibration_evidence=native_evidence["calibration"],
                fs_test_evidence=native_evidence["test"],
                fs_model_artifact=native_model,
                ranking_training_matrix=matrices["training"],
                ranking_validation_matrix=matrices["validation"],
            )

        if tournament.locked_test_report is None:
            raise PipelineError("ML-PIPE-085", "Locked test evaluation was not produced.")
        artifact_base = f"artifacts/runs/{run_id}/portfolio"
        persisted_champion = _persist_reload_champion(
            champion=tournament.champion_model_artifact,
            base=artifact_base,
            plan=plan,
            settings_plan=settings_plan,
        )
        persisted_calibrator = _persist_reload_calibrator(
            artifact=tournament.calibrator_artifact,
            base=artifact_base,
            plan=plan,
        )
        ranker = tournament.ranking_artifact
        if ranker is not None and not isinstance(
            ranker, (XGBoostRankingArtifact, LightGBMRankingArtifact)
        ):
            raise PipelineError("ML-PIPE-086", "The selected ranking artifact is unsupported.")
        persisted_ranker = _persist_reload_ranker(
            artifact=ranker,
            base=artifact_base,
            plan=plan,
        )
        persisted_recipe = _persist_reload_recipe(
            recipe=tournament.recipe,
            base=artifact_base,
            plan=plan,
        )
        if (
            persisted_calibrator.calibrator_digest != persisted_recipe.calibrator_digest
            or (persisted_ranker is None and persisted_recipe.ranking_artifact_digest is not None)
            or (
                persisted_ranker is not None
                and persisted_ranker.artifact_digest != persisted_recipe.ranking_artifact_digest
            )
        ):
            raise PipelineError("ML-PIPE-087", "The persisted tournament bundle is inconsistent.")
        if isinstance(persisted_champion, SplinkNativeModelArtifact):
            assert_splink_native_recipe_binding(
                recipe=persisted_recipe,
                artifact=persisted_champion,
            )
        elif persisted_champion.model_digest != persisted_recipe.champion_artifact_digest:
            raise PipelineError("ML-PIPE-087", "The persisted tournament bundle is inconsistent.")

        review_evidence, replay_evidence = _disjoint_decision_evidence(matrices["decision"])

        def replay(prepared_matrix: BoostedFeatureMatrix) -> ApprovedRecipeInferenceResult:
            inference_matrix = _raw_reference_matrix(
                matrix=prepared_matrix,
                bundle=bundle,
                source_id=source_id,
                target_id=target_id,
            )
            source_record_keys = tuple(
                sorted({left for left, _ in inference_matrix.pair_references})
            )
            if isinstance(persisted_champion, SplinkNativeModelArtifact):
                with DuckDBStore() as replay_store:
                    replay_catalog = ConfiguredDatasetPreparer(replay_store).prepare_all(plan)
                    replay_left = replay_catalog.require(source_id)
                    replay_right = replay_catalog.require(target_id)
                    replay_candidates = DuckDBCandidateGenerator(replay_store).generate(
                        left=replay_left.table,
                        right=replay_right.table,
                        variable_columns=replay_left.variable_columns,
                        rules=runtime_blocking_rules(plan),
                        maximum_candidate_pairs=plan.config.runtime.maximum_candidate_pairs,
                    )
                    replay_snapshot = candidate_snapshot(
                        replay_store,
                        replay_candidates.table.table_name,
                    )
                    native_replay = NativeSplinkInferenceReplay(
                        store=replay_store,
                        left=replay_left,
                        right=replay_right,
                        settings_plan=settings_plan,
                        model_artifact=persisted_champion,
                        expected_prepared_pairs=replay_snapshot.pairs,
                        selected_prepared_pairs=prepared_matrix.pair_references,
                        public_pair_references=inference_matrix.pair_references,
                        maximum_candidate_pairs=plan.config.runtime.maximum_candidate_pairs,
                    )
                    attestation = attest_generated_synthetic_inference(
                        bundle=bundle,
                        source_record_keys=source_record_keys,
                        pair_references=inference_matrix.pair_references,
                        feature_matrix=inference_matrix,
                        native_splink_replay=native_replay,
                        source_dataset_id="source_a",
                        target_dataset_id="source_b",
                    )
                    return infer_with_approved_recipe(
                        recipe=persisted_recipe,
                        source_record_keys=source_record_keys,
                        pair_references=inference_matrix.pair_references,
                        feature_matrix=inference_matrix,
                        native_splink_replay=native_replay,
                        calibrator_artifact=persisted_calibrator,
                        decision_policy=plan.config.decision_policy,
                        ranking_artifact=persisted_ranker,
                        execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
                        synthetic_attestation=attestation,
                        synthetic_bundle=bundle,
                        source_dataset_id="source_a",
                        target_dataset_id="source_b",
                    )
            attestation = attest_generated_synthetic_inference(
                bundle=bundle,
                source_record_keys=source_record_keys,
                pair_references=inference_matrix.pair_references,
                feature_matrix=inference_matrix,
                source_dataset_id="source_a",
                target_dataset_id="source_b",
            )
            return infer_with_approved_recipe(
                recipe=persisted_recipe,
                source_record_keys=source_record_keys,
                pair_references=inference_matrix.pair_references,
                feature_matrix=inference_matrix,
                champion_model_artifact=persisted_champion,
                calibrator_artifact=persisted_calibrator,
                decision_policy=plan.config.decision_policy,
                ranking_artifact=persisted_ranker,
                execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
                synthetic_attestation=attestation,
                synthetic_bundle=bundle,
                source_dataset_id="source_a",
                target_dataset_id="source_b",
            )

        review_inference = replay(review_evidence)
        inference = replay(replay_evidence)
        if review_inference.synthetic_attestation_digest == inference.synthetic_attestation_digest:
            raise PipelineError(
                "ML-PIPE-091", "Disjoint synthetic inference evidence is unavailable."
            )

        workflow_digest = _canonical_digest(
            {
                "tournament_digest": tournament.tournament_digest,
                "recipe_digest": persisted_recipe.recipe_digest,
                "champion_digest": persisted_recipe.champion_artifact_digest,
                "calibrator_digest": persisted_calibrator.calibrator_digest,
                "ranker_digest": persisted_recipe.ranking_artifact_digest,
                "locked_test_pair_count": tournament.locked_test_report.pair_count,
                "review_inference_digest": review_inference.inference_digest,
                "inference_digest": inference.inference_digest,
                "inference_status": "replayed",
            }
        )
        return SyntheticPortfolioWorkflowResult(
            tournament=tournament,
            persisted_champion=persisted_champion,
            persisted_calibrator=persisted_calibrator,
            persisted_recipe=persisted_recipe,
            persisted_ranker=persisted_ranker,
            review_inference=review_inference,
            inference=inference,
            protected_label_batches=tuple(batches[name] for name in sorted(batches)),
            locked_test_report=tournament.locked_test_report,
            run_id=run_id,
            pair_candidate_count=sum(item.enabled for item in portfolio.pair_candidates),
            ranking_candidate_count=sum(item.enabled for item in portfolio.ranking_candidates),
            inference_status="replayed",
            workflow_digest=workflow_digest,
        )


__all__ = ["SyntheticPortfolioWorkflowResult", "SyntheticPortfolioWorkflowRunner"]
