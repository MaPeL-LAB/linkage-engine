#!/usr/bin/env python3
"""Run the deterministic synthetic-only Linkage Engine lifecycle demonstration.

The module composes shipped APIs without executing on import. Its command-line output is one
aggregate JSON document; record references, candidate pairs, adjudication values, and local paths
remain restricted to in-memory objects. Synthetic evidence establishes software behaviour only.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mapel_linkage.adjudication import (
    ActiveLearningConfig,
    AdjudicationOutcome,
    AdjudicationRecord,
    AdjudicationWorkflowRunner,
    ConsensusReport,
    LabelPromotionResult,
    PrioritizedReviewQueue,
    build_review_queue,
    sample_active_learning_queue,
)
from mapel_linkage.assignment.contracts import pair_digest
from mapel_linkage.benchmarking import (
    BenchmarkPortfolioRunner,
    BenchmarkRecipe,
    BenchmarkRunResult,
    BenchmarkRunStatus,
    BenchmarkScenarioBundle,
    BenchmarkScenarioGenerator,
    generate_and_run_seed_corpus,
)
from mapel_linkage.clustering import CandidateEdge, ClusteringPlan
from mapel_linkage.configuration import compile_config, load_config
from mapel_linkage.configuration.models import (
    ConfirmedDecisionConfig,
    DecisionPolicyConfig,
    NoMatchDecisionConfig,
    ReviewDecisionConfig,
    UnresolvedDecisionConfig,
)
from mapel_linkage.governance import (
    VerifiedLabelBatch,
    VerifiedPairLabel,
    assert_disjoint_label_partitions,
)
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.labels import LabelPartition
from mapel_linkage.io.duckdb_store import DuckDBStore
from mapel_linkage.models.boosted import BoostedFeatureMatrix, BoostedLabelledMatrix
from mapel_linkage.pipeline import (
    ApprovedRecipeInferenceResult,
    ModelPortfolioDeclaration,
    ModelPortfolioRunner,
    MultiSourceWorkflowResult,
    MultiSourceWorkflowRunner,
    OperationalValidationStatus,
    PairModelCandidateDeclaration,
    PortfolioTournamentResult,
    RecipeApprovalStatus,
    RecipeExecutionMode,
    deserialize_pipeline_recipe,
    infer_with_approved_recipe,
    serialize_pipeline_recipe,
)
from mapel_linkage.pipeline.inference_runner import attest_generated_synthetic_inference
from mapel_linkage.profiling import PreflightTaskProfile, build_preflight_task_profile
from mapel_linkage.recommendation import (
    AdvisorContext,
    MetaRankingAdvisoryReport,
    MetaRankingLinkageAdvisor,
    RecommendationIntent,
    RuntimeDependency,
)
from mapel_linkage.synthetic import (
    SyntheticBundle,
    SyntheticGenerationConfig,
    SyntheticRecord,
    generate_synthetic_bundle,
)

SEED = 20260816
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "examples" / "synthetic_link_only.yaml"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bundle_digest(bundle: SyntheticBundle) -> str:
    payload = {
        "provenance": asdict(bundle.provenance),
        "source_a": [record.as_mapping() for record in bundle.source_a],
        "source_b": [record.as_mapping() for record in bundle.source_b],
        "truth": [record.as_mapping() for record in bundle.truth],
    }
    return _digest(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _date_proximity(left: str | None, right: str | None) -> float:
    if left is None or right is None:
        return 0.0
    difference = abs((date.fromisoformat(left) - date.fromisoformat(right)).days)
    return max(0.0, 1.0 - (difference / 366.0))


def _comparison_features(left: SyntheticRecord, right: SyntheticRecord) -> tuple[float, ...]:
    label_similarity = (
        difflib.SequenceMatcher(None, left.label_value, right.label_value).ratio()
        if left.label_value is not None and right.label_value is not None
        else 0.0
    )
    group_agreement = float(
        left.group_value is not None
        and right.group_value is not None
        and left.group_value == right.group_value
    )
    observed_fraction = (
        sum(
            value is not None
            for value in (
                left.label_value,
                right.label_value,
                left.date_value,
                right.date_value,
                left.group_value,
                right.group_value,
            )
        )
        / 6.0
    )
    return (
        label_similarity,
        _date_proximity(left.date_value, right.date_value),
        group_agreement,
        observed_fraction,
    )


class _DeterministicEvidenceRunner(BenchmarkPortfolioRunner):
    """Run real benchmarks while normalizing non-semantic timing telemetry for this example."""

    def run_single(
        self,
        *,
        bundle: BenchmarkScenarioBundle,
        recipe: BenchmarkRecipe,
        replicate_id: str = "replicate.001",
        seed: int = SEED,
    ) -> BenchmarkRunResult:
        result = super().run_single(
            bundle=bundle,
            recipe=recipe,
            replicate_id=replicate_id,
            seed=seed,
        )
        if result.record.status is not BenchmarkRunStatus.SUCCESS or result.metrics is None:
            return result
        metrics = result.metrics.model_copy(update={"runtime_ms": 1, "peak_memory_mb": 1})
        record = result.record.model_copy(
            update={
                "aggregate_metrics_digest": metrics.metrics_digest,
                "runtime_ms": 1,
                "peak_memory_mb": 1,
            }
        )
        return BenchmarkRunResult(record=record, metrics=metrics)


@dataclass(frozen=True, slots=True, repr=False)
class _SyntheticCandidateSlice:
    matrix: BoostedFeatureMatrix = field(repr=False)
    labels_by_pair_digest: dict[str, int] = field(repr=False)
    entity_components_by_pair_digest: dict[str, tuple[str, ...]] = field(repr=False)
    household_components_by_pair_digest: dict[str, tuple[str, ...]] = field(repr=False)
    bundle_digest: str


def _make_candidate_slice(
    bundle: SyntheticBundle,
    *,
    group_index: int,
    pair_count: int,
    positive_only: bool = False,
) -> _SyntheticCandidateSlice:
    """Build comparison evidence from one household-disjoint slice of one bundle."""
    bundle_digest = _bundle_digest(bundle)
    truth = {
        (item.dataset_id, item.record_key): (item.entity_key, item.household_key)
        for item in bundle.truth
    }
    households = sorted({household for _, household in truth.values()})
    household_group = {household: index % 5 for index, household in enumerate(households)}

    source_a = tuple(
        record
        for record in bundle.source_a
        if household_group[truth[("source_a", record.record_key)][1]] == group_index
    )
    source_b = tuple(
        record
        for record in bundle.source_b
        if household_group[truth[("source_b", record.record_key)][1]] == group_index
    )
    rows: list[
        tuple[
            str,
            tuple[str, str],
            tuple[float, ...],
            int,
            tuple[str, ...],
            tuple[str, ...],
        ]
    ] = []
    for left in source_a:
        left_entity, left_household = truth[("source_a", left.record_key)]
        for right in source_b:
            right_entity, right_household = truth[("source_b", right.record_key)]
            digest = pair_digest(left.record_key, right.record_key)
            rows.append(
                (
                    digest,
                    (left.record_key, right.record_key),
                    _comparison_features(left, right),
                    int(left_entity == right_entity),
                    tuple(
                        sorted(
                            {
                                _digest(f"{bundle_digest}:entity:{left_entity}"),
                                _digest(f"{bundle_digest}:entity:{right_entity}"),
                            }
                        )
                    ),
                    tuple(
                        sorted(
                            {
                                _digest(f"{bundle_digest}:household:{left_household}"),
                                _digest(f"{bundle_digest}:household:{right_household}"),
                            }
                        )
                    ),
                )
            )

    positives = sorted((row for row in rows if row[3] == 1), key=lambda row: row[0])
    negatives = sorted((row for row in rows if row[3] == 0), key=lambda row: row[0])
    if positive_only:
        selected: list[
            tuple[str, tuple[str, str], tuple[float, ...], int, tuple[str, ...], tuple[str, ...]]
        ] = []
        used_left: set[str] = set()
        used_right: set[str] = set()
        for row in positives:
            left_key, right_key = row[1]
            if left_key in used_left or right_key in used_right:
                continue
            selected.append(row)
            used_left.add(left_key)
            used_right.add(right_key)
            if len(selected) == pair_count:
                break
    else:
        positive_count = min(len(positives), max(2, pair_count // 4))
        selected = positives[:positive_count] + negatives[: pair_count - positive_count]
        selected.sort(key=lambda row: row[0])
    if len(selected) != pair_count or {row[3] for row in selected} != (
        {1} if positive_only else {0, 1}
    ):
        raise RuntimeError("The generated bundle cannot satisfy the lifecycle candidate slice.")

    feature_names = ("label_similarity", "date_proximity", "group_agreement", "observed_fraction")
    feature_schema_digest = _digest("|".join(feature_names))
    matrix = BoostedFeatureMatrix(
        features=np.asarray([row[2] for row in selected], dtype=np.float64),
        pair_references=tuple(row[1] for row in selected),
        pair_digests=tuple(row[0] for row in selected),
        feature_names=feature_names,
        feature_schema_digest=feature_schema_digest,
    )
    return _SyntheticCandidateSlice(
        matrix=matrix,
        labels_by_pair_digest={row[0]: row[3] for row in selected},
        entity_components_by_pair_digest={row[0]: row[4] for row in selected},
        household_components_by_pair_digest={row[0]: row[5] for row in selected},
        bundle_digest=bundle_digest,
    )


def _make_partition(
    bundle: SyntheticBundle,
    *,
    partition: LabelPartition,
    pair_count: int,
    group_index: int,
) -> tuple[BoostedLabelledMatrix, VerifiedLabelBatch]:
    """Create one protected labelled matrix directly from the generated bundle."""
    candidate_slice = _make_candidate_slice(
        bundle,
        group_index=group_index,
        pair_count=pair_count,
    )
    labels = np.asarray(
        [
            candidate_slice.labels_by_pair_digest[item]
            for item in candidate_slice.matrix.pair_digests
        ],
        dtype=np.int8,
    )
    verified_labels = tuple(
        VerifiedPairLabel(
            left_record_key=left,
            right_record_key=right,
            label=int(label),  # type: ignore[arg-type]
            entity_component_digests=candidate_slice.entity_components_by_pair_digest[digest],
            household_component_digests=candidate_slice.household_components_by_pair_digest[digest],
        )
        for (left, right), digest, label in zip(
            candidate_slice.matrix.pair_references,
            candidate_slice.matrix.pair_digests,
            labels.tolist(),
            strict=True,
        )
    )
    batch = VerifiedLabelBatch(
        source_kind="synthetic_truth",
        verification_protocol="synthetic_lifecycle_v1",
        source_digest=_digest(f"{candidate_slice.bundle_digest}:{partition}:{pair_count}"),
        partition=partition,
        labels=verified_labels,
    )
    matrix = BoostedLabelledMatrix(
        features=candidate_slice.matrix.features,
        labels=labels,
        feature_names=candidate_slice.matrix.feature_names,
        pair_references=candidate_slice.matrix.pair_references,
        pair_digests=candidate_slice.matrix.pair_digests,
        feature_schema_digest=candidate_slice.matrix.feature_schema_digest,
        label_authority_digest=batch.label_authority_digest,
        selection_digest=_digest(
            f"selection:{candidate_slice.bundle_digest}:{partition}:{pair_count}"
        ),
        label_source_kind="synthetic_truth",
        partition=partition,
        positive_count=int(labels.sum()),
        negative_count=pair_count - int(labels.sum()),
        hard_negative_count=0,
    )
    return matrix, batch


def _review_lifecycle(
    *,
    review_inference: ApprovedRecipeInferenceResult,
    review_slice: _SyntheticCandidateSlice,
    validation_batch: VerifiedLabelBatch,
    calibration_batch: VerifiedLabelBatch,
) -> tuple[PrioritizedReviewQueue, ConsensusReport, LabelPromotionResult]:
    queue = build_review_queue(review_inference.decisions)
    entries = tuple(
        entry
        for entry in queue.entries
        if entry.target_record_ref is not None
        and pair_digest(entry.source_record_ref, entry.target_record_ref)
        in review_slice.labels_by_pair_digest
    )
    if len(entries) < 3:
        raise RuntimeError("Model-derived review evidence did not produce three reviewable pairs.")
    entries = entries[:3]
    original_payloads = tuple(entry.restricted_digest_payload() for entry in entries)
    committee_scores = {
        entry.relationship_id: (
            max(0.0, (entry.calibrated_probability or 0.5) - 0.1),
            min(1.0, (entry.calibrated_probability or 0.5) + 0.1),
        )
        for entry in entries
    }
    prioritized = sample_active_learning_queue(
        entries,
        budget=len(entries),
        strategy="hybrid",
        config=ActiveLearningConfig(strategy="hybrid"),
        committee_scores=committee_scores,
    )
    if tuple(entry.restricted_digest_payload() for entry in entries) != original_payloads:
        raise RuntimeError("Review prioritization changed the source review evidence.")

    timestamp = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
    reviews: list[AdjudicationRecord] = []
    for entry in prioritized.entries:
        if entry.target_record_ref is None:
            raise RuntimeError("Review promotion requires a candidate-pair reference.")
        digest = pair_digest(entry.source_record_ref, entry.target_record_ref)
        outcome: AdjudicationOutcome = (
            "match" if review_slice.labels_by_pair_digest[digest] == 1 else "nonmatch"
        )
        for reviewer in ("reviewer_alpha", "reviewer_beta"):
            reviews.append(
                AdjudicationRecord(
                    event_id=f"event_{digest[:16]}_{reviewer}",
                    left_record_key=entry.source_record_ref,
                    right_record_key=entry.target_record_ref,
                    decision=outcome,
                    confidence=0.95,
                    reviewer_id=reviewer,
                    timestamp=timestamp,
                    protocol_version="synthetic_double_review_v1",
                    entity_component_digests=review_slice.entity_components_by_pair_digest[digest],
                    household_component_digests=review_slice.household_components_by_pair_digest[
                        digest
                    ],
                )
            )

    imported = AdjudicationWorkflowRunner.import_reviews(
        reviews,
        candidate_pair_references=tuple(
            (entry.source_record_ref, entry.target_record_ref)
            for entry in prioritized.entries
            if entry.target_record_ref is not None
        ),
        strict_candidate_check=True,
    )
    consensus = AdjudicationWorkflowRunner.resolve_consensus(
        imported.imported_batch,
        policy="strict_double_review",
        min_reviewers=2,
        agreement_threshold=1.0,
    )
    promotion = AdjudicationWorkflowRunner.promote_to_verified_labels(
        consensus,
        target_partition="training",
        min_confidence=0.90,
        require_consensus=True,
        require_double_review=True,
        minimum_reviewers=2,
        allowed_protocols=frozenset({"synthetic_double_review_v1"}),
        verification_protocol="synthetic_double_review_v1",
        existing_partition_batches=(validation_batch, calibration_batch),
    )
    if promotion.retraining_triggered:
        raise RuntimeError("Label promotion must not trigger automatic retraining.")
    if promotion.disjointness_report is None:
        raise RuntimeError("Promoted labels require a partition-disjointness report.")
    return prioritized, consensus, promotion


def _run_multisource_resolution(
    bundle: SyntheticBundle,
    inference: ApprovedRecipeInferenceResult,
) -> MultiSourceWorkflowResult:
    truth = {(item.dataset_id, item.record_key): item.entity_key for item in bundle.truth}
    assigned = tuple(
        decision
        for decision in inference.decisions
        if decision.target_record_ref is not None and decision.calibrated_probability is not None
    )[:3]
    if len(assigned) != 3:
        raise RuntimeError("Synthetic inference did not produce three multi-source seed edges.")

    source_a = tuple(item.source_record_ref for item in assigned)
    source_b = tuple(
        item.target_record_ref for item in assigned if item.target_record_ref is not None
    )
    source_c = tuple(f"C{index:06d}" for index in range(len(assigned)))
    datasets = {"source_a": source_a, "source_b": source_b, "source_c": source_c}
    edges: list[CandidateEdge] = []
    true_clusters: dict[str, str] = {}
    must_link_pairs: list[tuple[str, str]] = []
    for index, decision in enumerate(assigned):
        target = decision.target_record_ref
        probability = decision.calibrated_probability
        if target is None or probability is None:
            raise RuntimeError("Assigned synthetic evidence is incomplete.")
        source_entity = truth[("source_a", decision.source_record_ref)]
        target_entity = truth[("source_b", target)]
        third = source_c[index]
        true_clusters[decision.source_record_ref] = source_entity
        true_clusters[target] = target_entity
        true_clusters[third] = source_entity
        edges.extend(
            (
                CandidateEdge(
                    decision.source_record_ref,
                    "source_a",
                    target,
                    "source_b",
                    probability,
                ),
                CandidateEdge(decision.source_record_ref, "source_a", third, "source_c", 0.95),
                CandidateEdge(
                    target,
                    "source_b",
                    third,
                    "source_c",
                    0.95 if source_entity == target_entity else 0.05,
                ),
            )
        )
        must_link_pairs.append((decision.source_record_ref, third))
        if source_entity == target_entity:
            must_link_pairs.append((target, third))
    result = MultiSourceWorkflowRunner.run(
        datasets=datasets,
        candidate_edges=tuple(edges),
        must_link_pairs=tuple(must_link_pairs),
        plan=ClusteringPlan(
            algorithm="constrained_agglomerative",
            threshold=0.50,
            cannot_link_same_dataset=True,
            random_seed=SEED,
        ),
        min_datasets=3,
        true_clusters=true_clusters,
    )
    if result.crosswalk_path is not None or result.evaluation_report_path is not None:
        raise RuntimeError("Multi-source execution wrote an unrequested row-level artifact.")
    return result


@dataclass(frozen=True, slots=True, repr=False)
class LifecycleArtifacts:
    """In-memory artifacts for automated authority and partition verification."""

    summary: dict[str, object] = field(repr=False)
    profile: PreflightTaskProfile = field(repr=False)
    advisor_report: MetaRankingAdvisoryReport = field(repr=False)
    tournament: PortfolioTournamentResult = field(repr=False)
    review_inference: ApprovedRecipeInferenceResult = field(repr=False)
    review_queue: PrioritizedReviewQueue = field(repr=False)
    consensus: ConsensusReport = field(repr=False)
    promotion: LabelPromotionResult = field(repr=False)
    recipe_payload: str = field(repr=False)
    inference: ApprovedRecipeInferenceResult = field(repr=False)
    multisource: MultiSourceWorkflowResult = field(repr=False)


def run_lifecycle(*, project_root: Path | None = None) -> LifecycleArtifacts:
    """Execute the smallest complete deterministic synthetic lifecycle in memory."""
    data_policy = os.environ.get("MAPEL_TEST_DATA_POLICY", "synthetic_only")
    if data_policy != "synthetic_only":
        raise RuntimeError("The lifecycle example requires the synthetic-only data policy.")

    loaded = load_config(CONFIG_PATH)
    plan = compile_config(loaded.config, project_root=project_root or ROOT)
    if plan.random_seed != SEED:
        raise RuntimeError("The canonical lifecycle configuration seed has changed.")

    bundle = generate_synthetic_bundle(
        SyntheticGenerationConfig(
            seed=SEED,
            entity_count=24,
            left_only_count=2,
            right_only_count=2,
            duplicate_count=2,
            competing_candidate_count=2,
            source_a_missing_rate=0.05,
            source_b_missing_rate=0.20,
            source_b_typo_rate=0.35,
            source_b_date_shift_rate=0.20,
        )
    )
    profile = build_preflight_task_profile(plan)
    with TemporaryDirectory(prefix="mapel-linkage-e2e-benchmarks-") as registry_directory:
        registry = generate_and_run_seed_corpus(
            registry_directory=Path(registry_directory),
            generator=BenchmarkScenarioGenerator(),
            runner=_DeterministicEvidenceRunner(),
            families=(
                "family.typo_stress",
                "family.missingness_regime",
                "family.date_variation",
            ),
            instances=(
                "instance.typo_low",
                "instance.missing_zero",
                "instance.date_shift_low",
            ),
            replicates=1,
            base_seed=SEED,
        )
        advisor_report = MetaRankingLinkageAdvisor(
            registry=registry,
            max_ood_distance=1.0,
        ).advise(
            plan,
            profile=profile,
            context=AdvisorContext(
                intent=RecommendationIntent.DEVELOP_NEW_RECIPE,
                verified_labels_available=True,
                approved_recipe_available=False,
                protected_out_of_fold_predictions_available=False,
                available_runtimes=(RuntimeDependency.CORE,),
            ),
        )
    if (
        advisor_report.recommendation_authority != "advisory_only"
        or advisor_report.decision_authority != "none"
        or advisor_report.assignment_authority != "none"
        or advisor_report.merge_authority != "none"
        or advisor_report.automatic_promotion != "prohibited"
        or advisor_report.operational_validity != "not_established"
        or advisor_report.fallback_to_similarity
        or advisor_report.meta_model_type != "ridge_meta_ranker_v1"
    ):
        raise RuntimeError("The strategy advisor exceeded its fixed authority boundary.")

    training_matrix, training_batch = _make_partition(
        bundle, partition="training", pair_count=48, group_index=2
    )
    validation_matrix, validation_batch = _make_partition(
        bundle, partition="validation", pair_count=24, group_index=3
    )
    calibration_matrix, calibration_batch = _make_partition(
        bundle, partition="calibration", pair_count=24, group_index=4
    )
    partition_report = assert_disjoint_label_partitions(
        (training_batch, validation_batch, calibration_batch)
    )
    portfolio = ModelPortfolioDeclaration(
        portfolio_id="synthetic_e2e_portfolio",
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
                model_id="stacked_challenger",
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
    with DuckDBStore() as store:
        tournament = ModelPortfolioRunner(store).run_tournament(
            portfolio=portfolio,
            training_matrix=training_matrix,
            validation_matrix=validation_matrix,
            calibration_matrix=calibration_matrix,
            disjointness=partition_report,
            split_manifest_digest=partition_report.manifest_digest,
            configuration_digest=plan.configuration_digest,
            candidate_plan_digest=_digest(f"{_bundle_digest(bundle)}:synthetic_e2e_candidate_plan"),
            feature_schema_digest=training_matrix.feature_schema_digest,
            decision_policy_digest=_digest("synthetic_e2e_decision_policy"),
            random_seed=SEED,
            k_folds=3,
            calibrator_methods=("sigmoid",),
            approval_status=RecipeApprovalStatus.SYNTHETIC_VALIDATED,
            operational_validation=OperationalValidationStatus.NOT_ESTABLISHED,
        )
    if any(
        manifest.partition != "training_oof"
        or manifest.test_partition_used
        or manifest.calibration_partition_used
        or manifest.decision_partition_used
        or manifest.decision_authority != "evidence_only"
        or manifest.merge_authority != "none"
        for manifest in tournament.oof_manifests
    ):
        raise RuntimeError("Protected out-of-fold stacking provenance is invalid.")

    review_slice = _make_candidate_slice(
        bundle,
        group_index=0,
        pair_count=12,
    )
    review_source_keys = tuple(sorted({left for left, _ in review_slice.matrix.pair_references}))
    review_attestation = attest_generated_synthetic_inference(
        bundle=bundle,
        source_record_keys=review_source_keys,
        pair_references=review_slice.matrix.pair_references,
        feature_matrix=review_slice.matrix,
        source_dataset_id="source_a",
        target_dataset_id="source_b",
    )
    review_inference = infer_with_approved_recipe(
        recipe=tournament.recipe,
        source_record_keys=review_source_keys,
        pair_references=review_slice.matrix.pair_references,
        feature_matrix=review_slice.matrix,
        champion_model_artifact=tournament.champion_model_artifact,
        calibrator_artifact=tournament.calibrator_artifact,
        decision_policy=DecisionPolicyConfig(
            confirmed=ConfirmedDecisionConfig(
                minimum_probability=1.0,
                minimum_probability_margin=1.0,
            ),
            review_required=ReviewDecisionConfig(minimum_probability=0.01),
            no_match=NoMatchDecisionConfig(maximum_top_probability=0.0),
            unresolved=UnresolvedDecisionConfig(),
        ),
        execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
        synthetic_attestation=review_attestation,
        synthetic_bundle=bundle,
        source_dataset_id="source_a",
        target_dataset_id="source_b",
    )
    review_queue, consensus, promotion = _review_lifecycle(
        review_inference=review_inference,
        review_slice=review_slice,
        validation_batch=validation_batch,
        calibration_batch=calibration_batch,
    )

    recipe_payload = serialize_pipeline_recipe(tournament.recipe)
    recipe = deserialize_pipeline_recipe(recipe_payload)
    if recipe != tournament.recipe:
        raise RuntimeError("The immutable pipeline recipe failed its canonical round trip.")
    new_data_slice = _make_candidate_slice(bundle, group_index=1, pair_count=12)
    new_data_source_keys = tuple(
        sorted({left for left, _ in new_data_slice.matrix.pair_references})
    )
    inference_attestation = attest_generated_synthetic_inference(
        bundle=bundle,
        source_record_keys=new_data_source_keys,
        pair_references=new_data_slice.matrix.pair_references,
        feature_matrix=new_data_slice.matrix,
        source_dataset_id="source_a",
        target_dataset_id="source_b",
    )
    recipe.assert_usable_for(
        RecipeExecutionMode.SYNTHETIC_INFERENCE,
        synthetic_attestation=inference_attestation,
    )
    inference = infer_with_approved_recipe(
        recipe=recipe_payload,
        source_record_keys=new_data_source_keys,
        pair_references=new_data_slice.matrix.pair_references,
        feature_matrix=new_data_slice.matrix,
        champion_model_artifact=tournament.champion_model_artifact,
        calibrator_artifact=tournament.calibrator_artifact,
        execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
        synthetic_attestation=inference_attestation,
        synthetic_bundle=bundle,
        source_dataset_id="source_a",
        target_dataset_id="source_b",
    )
    if (
        inference.assignment_result.assignment_authority != "global_selection_only"
        or inference.assignment_result.decision_authority != "none"
        or any(
            decision.decision_authority != "policy_classification"
            or decision.merge_authority != "none"
            for decision in inference.decisions
        )
    ):
        raise RuntimeError("Inference exceeded assignment or relationship-decision authority.")

    multisource = _run_multisource_resolution(bundle, inference)
    if multisource.evaluation_report is None:
        raise RuntimeError("Synthetic multi-source truth evaluation was not produced.")

    lineage_digest = _digest(
        ":".join(
            (
                _bundle_digest(bundle),
                profile.profile_digest,
                advisor_report.report_digest,
                tournament.tournament_digest,
                review_inference.inference_digest,
                promotion.result_digest,
                recipe.recipe_digest,
                inference.inference_digest,
                multisource.workflow_digest,
            )
        )
    )
    summary: dict[str, object] = {
        "lifecycle_schema_version": "1",
        "lifecycle_lineage_digest": lineage_digest,
        "seed": SEED,
        "data_policy": "synthetic_only",
        "synthetic_generation": {
            **asdict(bundle.provenance),
            "bundle_digest": _bundle_digest(bundle),
        },
        "preflight": profile.safe_summary(),
        "stage3_advisor": advisor_report.safe_summary(),
        "portfolio_tournament": {
            **tournament.safe_summary(),
            "protected_partitions": partition_report.safe_summary(),
            "calibration_partition": calibration_matrix.partition,
            "oof_partition": "training_oof",
            "oof_test_partition_used": False,
            "oof_calibration_partition_used": False,
            "synthetic_bundle_digest": _bundle_digest(bundle),
        },
        "active_learning": {
            "strategy": review_queue.strategy,
            "relationship_count": review_queue.relationship_count,
            "ordering_authority": "review_ordering_only",
            "labels_modified": False,
            "source_inference_digest": review_inference.inference_digest,
        },
        "adjudication": {
            "consensus": consensus.safe_summary(),
            "promotion": promotion.safe_summary(),
            "automatic_retraining": promotion.retraining_triggered,
            "partition_disjointness": promotion.disjointness_report.safe_summary()
            if promotion.disjointness_report is not None
            else None,
        },
        "recipe": recipe.safe_summary(),
        "synthetic_inference": {
            **inference.safe_summary(),
            "assignment_authority": inference.assignment_result.assignment_authority,
            "assignment_decision_authority": inference.assignment_result.decision_authority,
            "relationship_decision_authority": "policy_classification",
            "merge_authority": "none",
        },
        "multi_source": {
            **multisource.safe_summary(),
            "evaluation": multisource.evaluation_report.safe_summary(),
            "merge_authority": "none",
        },
        "authority_boundaries": {
            "candidate_retrieval": "indexing_and_blocking_only",
            "model_scores": "evidence_only",
            "calibration_partition": "calibration_only",
            "assignment": "compatible_edge_selection_only",
            "relationship_status": "decision_policy_only",
            "review_queue": "ordering_only",
            "automatic_recipe_promotion": "prohibited",
            "merge_authority": "none",
            "operational_validity": "not_established",
        },
        "warning": (
            "Synthetic testing establishes software behaviour only; operational validity "
            "is not established."
        ),
    }
    return LifecycleArtifacts(
        summary=summary,
        profile=profile,
        advisor_report=advisor_report,
        tournament=tournament,
        review_inference=review_inference,
        review_queue=review_queue,
        consensus=consensus,
        promotion=promotion,
        recipe_payload=recipe_payload,
        inference=inference,
        multisource=multisource,
    )


def _write_idempotent(path: Path, payload: str) -> None:
    try:
        if path.is_symlink():
            raise RuntimeError("A lifecycle output target cannot be a symbolic link.")
        if path.exists():
            if not path.is_file() or path.read_text(encoding="utf-8") != payload:
                raise RuntimeError("A lifecycle output already exists with different content.")
            return
        atomic_write_text(path, payload)
    except OSError:
        raise RuntimeError("A lifecycle aggregate output could not be written.") from None


def _assert_no_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    if any(component.is_symlink() for component in (absolute, *absolute.parents)):
        raise RuntimeError("The lifecycle output path cannot traverse a symbolic link.")


def write_aggregate_outputs(output_directory: Path, artifacts: LifecycleArtifacts) -> None:
    """Write only aggregate, deterministic artifacts below an explicit output directory."""
    _assert_no_symlink_components(output_directory)
    try:
        if output_directory.exists() and not output_directory.is_dir():
            raise RuntimeError("The lifecycle output directory is not a trusted directory.")
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise RuntimeError("The lifecycle output directory could not be created.") from None
    _write_idempotent(
        output_directory / "lifecycle_summary.json",
        json.dumps(artifacts.summary, indent=2, sort_keys=True) + "\n",
    )
    _write_idempotent(output_directory / "pipeline_recipe.json", artifacts.recipe_payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional explicit directory for aggregate summary and recipe artifacts.",
    )
    args = parser.parse_args()

    artifacts = run_lifecycle()
    if args.output_dir is not None:
        write_aggregate_outputs(args.output_dir, artifacts)
    print(json.dumps(artifacts.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
