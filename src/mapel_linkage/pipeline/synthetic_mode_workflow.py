"""Configuration-driven generated-synthetic linkage-mode orchestration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np

from mapel_linkage.assignment import AssignmentEdgeBatch, AssignmentPlan, DeduplicationPlan
from mapel_linkage.calibration import (
    BetaCalibrator,
    CalibratorArtifact,
    ChampionCalibratorSelector,
    ChampionSelection,
    IsotonicCalibrator,
    ModelEvaluationCandidate,
    PairScoreBatch,
    SigmoidCalibrator,
    read_calibrator_artifact,
    write_calibrator_artifact,
)
from mapel_linkage.candidate_generation import DuckDBCandidateGenerator
from mapel_linkage.comparisons import DuckDBComparisonFeatureBuilder
from mapel_linkage.configuration import ExecutionPlan
from mapel_linkage.domain.errors import CandidateBudgetExceeded, PipelineError
from mapel_linkage.domain.sql_identifiers import quote_identifier
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.labels import (
    LabelPartition,
    PartitionDisjointnessReport,
    VerifiedLabelBatch,
    assert_disjoint_label_partitions,
)
from mapel_linkage.io import DuckDBStore
from mapel_linkage.models.boosted import (
    BoostedFeatureMatrix,
    BoostedLabelledMatrix,
    DuckDBVerifiedMatrixBuilder,
    XGBoostModelArtifact,
    XGBoostPairClassifier,
    read_xgboost_artifact,
    write_xgboost_artifact,
)
from mapel_linkage.pipeline.deduplication_runner import (
    DeduplicationWorkflowResult,
    DeduplicationWorkflowRunner,
)
from mapel_linkage.pipeline.inference_runner import (
    ApprovedRecipeInferenceResult,
    attest_generated_synthetic_inference,
    infer_with_approved_recipe,
)
from mapel_linkage.pipeline.mode_artifacts import (
    SyntheticModeOrchestrationArtifact,
    SyntheticModeRunArtifact,
    deserialize_mode_orchestration_artifact,
    deserialize_mode_run_artifact,
    serialize_mode_orchestration_artifact,
    serialize_mode_run_artifact,
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
from mapel_linkage.pipeline.synthetic_workflow_support import (
    SyntheticCandidateSnapshot,
    candidate_snapshot,
    protected_label_batches,
    runtime_blocking_rules,
    synthetic_truth_records,
)
from mapel_linkage.preprocessing import ConfiguredDatasetPreparer, surrogate_record_key
from mapel_linkage.synthetic import (
    SyntheticBundle,
    SyntheticGenerationConfig,
    generate_synthetic_bundle,
    matches_synthetic_fixture_layout,
    write_synthetic_bundle,
)
from mapel_linkage.validation import (
    EntityHouseholdRecord,
    PairValidationReport,
    evaluate_binary_scores,
)

_SEED = 20260816
_LINK_CONSTRAINTS = frozenset({"many_to_one", "one_to_many", "unconstrained"})


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _bundle_digest(bundle: SyntheticBundle) -> str:
    return _canonical_digest(
        {
            "provenance": asdict(bundle.provenance),
            "source_a": [record.as_mapping() for record in bundle.source_a],
            "source_b": [record.as_mapping() for record in bundle.source_b],
            "truth": [record.as_mapping() for record in bundle.truth],
        }
    )


def _record_keys(store: DuckDBStore, table_name: str) -> tuple[str, ...]:
    key = quote_identifier("__ml_record_key")
    rows = store._fetch_model_rows(
        f"SELECT {key} FROM {quote_identifier(table_name)} ORDER BY {key}"
    )
    return tuple(str(row[0]) for row in rows)


def _link_source_target_ids(plan: ExecutionPlan) -> tuple[str, str, Path]:
    mode = plan.config.project.linkage_mode
    constraint = plan.config.project.assignment_constraint
    expected_dispatch = f"synthetic_mode_v1:link_only:{constraint}"
    sources = [item for item in plan.config.datasets if item.role == "source"]
    targets = [item for item in plan.config.datasets if item.role == "target"]
    if (
        mode != "link_only"
        or constraint not in _LINK_CONSTRAINTS
        or plan.mode_dispatch_key != expected_dispatch
        or len(plan.config.datasets) != 2
        or len(sources) != 1
        or len(targets) != 1
        or plan.random_seed != _SEED
    ):
        raise PipelineError("ML-MODE-010", "The synthetic linkage-mode dispatch is invalid.")
    if plan.config.labels is None or plan.config.labels.source.kind != "synthetic_truth":
        raise PipelineError(
            "ML-MODE-011", "Synthetic mode execution requires protected synthetic truth."
        )
    if plan.label_source_path is not None:
        raise PipelineError("ML-MODE-011", "Synthetic mode execution rejects label paths.")
    source, target = sources[0], targets[0]
    fixture_directory = plan.path_policy.resolve_input("data/synthetic")
    expected_paths = {
        source.id: (fixture_directory / "source_a.jsonl").resolve(strict=False),
        target.id: (fixture_directory / "source_b.jsonl").resolve(strict=False),
    }
    if dict(plan.dataset_paths) != expected_paths or any(
        not matches_synthetic_fixture_layout(
            source_format=item.format,
            record_id_column=item.record_id_column,
        )
        for item in (source, target)
    ):
        raise PipelineError(
            "ML-MODE-012", "Synthetic mode inputs do not match package-generated fixtures."
        )
    return source.id, target.id, fixture_directory


def _dedupe_source_id(plan: ExecutionPlan) -> tuple[str, Path]:
    expected_dispatch = "synthetic_mode_v1:dedupe_only:unconstrained"
    sources = [item for item in plan.config.datasets if item.role == "source"]
    if (
        plan.config.project.linkage_mode != "dedupe_only"
        or plan.config.project.assignment_constraint != "unconstrained"
        or plan.mode_dispatch_key != expected_dispatch
        or len(plan.config.datasets) != 1
        or len(sources) != 1
        or plan.random_seed != _SEED
    ):
        raise PipelineError("ML-MODE-010", "The synthetic linkage-mode dispatch is invalid.")
    if plan.config.labels is None or plan.config.labels.source.kind != "synthetic_truth":
        raise PipelineError(
            "ML-MODE-011", "Synthetic mode execution requires protected synthetic truth."
        )
    if plan.label_source_path is not None:
        raise PipelineError("ML-MODE-011", "Synthetic mode execution rejects label paths.")
    source = sources[0]
    fixture_directory = plan.path_policy.resolve_input("data/synthetic")
    expected = (fixture_directory / "source_a.jsonl").resolve(strict=False)
    if plan.dataset_paths.get(source.id) != expected or not matches_synthetic_fixture_layout(
        source_format=source.format,
        record_id_column=source.record_id_column,
    ):
        raise PipelineError(
            "ML-MODE-012", "Synthetic mode inputs do not match package-generated fixtures."
        )
    return source.id, fixture_directory


def _link_and_dedupe_ids(plan: ExecutionPlan) -> tuple[str, str, Path]:
    expected_dispatch = "synthetic_mode_v1:link_and_dedupe:one_to_one"
    sources = [item for item in plan.config.datasets if item.role == "source"]
    targets = [item for item in plan.config.datasets if item.role == "target"]
    if (
        plan.config.project.linkage_mode != "link_and_dedupe"
        or plan.config.project.assignment_constraint != "one_to_one"
        or plan.mode_dispatch_key != expected_dispatch
        or len(plan.config.datasets) != 2
        or len(sources) != 1
        or len(targets) != 1
        or plan.random_seed != _SEED
    ):
        raise PipelineError("ML-MODE-010", "The synthetic linkage-mode dispatch is invalid.")
    if plan.config.labels is None or plan.config.labels.source.kind != "synthetic_truth":
        raise PipelineError(
            "ML-MODE-011", "Synthetic mode execution requires protected synthetic truth."
        )
    if plan.label_source_path is not None:
        raise PipelineError("ML-MODE-011", "Synthetic mode execution rejects label paths.")
    source, target = sources[0], targets[0]
    fixture_directory = plan.path_policy.resolve_input("data/synthetic")
    expected_paths = {
        source.id: (fixture_directory / "source_a.jsonl").resolve(strict=False),
        target.id: (fixture_directory / "source_b.jsonl").resolve(strict=False),
    }
    if dict(plan.dataset_paths) != expected_paths or any(
        not matches_synthetic_fixture_layout(
            source_format=item.format,
            record_id_column=item.record_id_column,
        )
        for item in (source, target)
    ):
        raise PipelineError(
            "ML-MODE-012", "Synthetic mode inputs do not match package-generated fixtures."
        )
    return source.id, target.id, fixture_directory


def _generation_spec(
    plan: ExecutionPlan,
    supplied: SyntheticGenerationConfig | None,
) -> SyntheticGenerationConfig:
    spec = supplied or SyntheticGenerationConfig(
        seed=_SEED,
        entity_count=120,
        left_only_count=8,
        right_only_count=8,
        duplicate_count=120,
        right_duplicate_count=120,
        competing_candidate_count=20,
        source_a_missing_rate=0.05,
        source_b_missing_rate=0.20,
        source_b_typo_rate=0.35,
        source_b_date_shift_rate=0.20,
    )
    if (
        spec.seed != plan.random_seed
        or spec.entity_count < 100
        or spec.duplicate_count != spec.entity_count
        or spec.right_duplicate_count != spec.entity_count
    ):
        raise PipelineError(
            "ML-MODE-013",
            "Synthetic link modes require the fixed seed and bilateral duplicate coverage.",
        )
    return spec


def _dedupe_generation_spec(
    plan: ExecutionPlan,
    supplied: SyntheticGenerationConfig | None,
) -> SyntheticGenerationConfig:
    spec = supplied or SyntheticGenerationConfig(
        seed=_SEED,
        entity_count=120,
        left_only_count=8,
        right_only_count=8,
        duplicate_count=120,
        right_duplicate_count=0,
        competing_candidate_count=20,
        source_a_missing_rate=0.05,
        source_b_missing_rate=0.20,
        source_b_typo_rate=0.35,
        source_b_date_shift_rate=0.20,
    )
    if (
        spec.seed != plan.random_seed
        or spec.entity_count < 100
        or spec.duplicate_count != spec.entity_count
    ):
        raise PipelineError(
            "ML-MODE-013",
            "Synthetic dedupe mode requires the fixed seed and source duplicate coverage.",
        )
    return spec


def _single_source_truth_records(
    bundle: SyntheticBundle,
    *,
    source_dataset_id: str,
) -> tuple[EntityHouseholdRecord, ...]:
    records = tuple(
        EntityHouseholdRecord(
            dataset_id=source_dataset_id,
            record_key=surrogate_record_key(source_dataset_id, item.record_key),
            entity_key=item.entity_key,
            household_key=item.household_key,
        )
        for item in bundle.truth
        if item.dataset_id == "source_a"
    )
    return tuple(sorted(records, key=lambda item: item.record_key))


def _candidate_plan_digest(snapshot: SyntheticCandidateSnapshot, *, shape: str) -> str:
    return _canonical_digest(
        {
            "shape": shape,
            "pair_count": len(snapshot.pair_digests),
            "evidence": [
                {
                    "pair_digest": digest,
                    "retrieval_rule_ids": list(snapshot.rule_ids_by_digest[digest]),
                }
                for digest in snapshot.pair_digests
            ],
        }
    )


def _feature_view(matrix: BoostedLabelledMatrix) -> BoostedFeatureMatrix:
    """Discard protected labels before the recipe inference boundary."""

    return BoostedFeatureMatrix(
        features=matrix.features,
        pair_references=matrix.pair_references,
        pair_digests=matrix.pair_digests,
        feature_names=matrix.feature_names,
        feature_schema_digest=matrix.feature_schema_digest,
    )


@dataclass(frozen=True, slots=True, repr=False)
class ProtectedModeEvidenceAudit:
    """Private digest-only proof that inference uses the decision partition alone."""

    prepared_inference_pair_digests: tuple[str, ...] = field(repr=False)
    partition_pair_digests: tuple[tuple[str, tuple[str, ...]], ...] = field(repr=False)
    partition_component_digests: tuple[tuple[str, tuple[str, ...]], ...] = field(repr=False)

    def __post_init__(self) -> None:
        pairs = dict(self.partition_pair_digests)
        components = dict(self.partition_component_digests)
        required = {"training", "validation", "calibration", "decision", "test"}
        inference = set(self.prepared_inference_pair_digests)
        decision_pairs = set(pairs.get("decision", ()))
        decision_components = set(components.get("decision", ()))
        if (
            set(pairs) != required
            or set(components) != required
            or not inference
            or inference != decision_pairs
            or not decision_components
            or any(
                inference & set(pairs[name]) or decision_components & set(components[name])
                for name in required - {"decision"}
            )
        ):
            raise PipelineError(
                "ML-MODE-022", "Synthetic inference crossed a protected partition boundary."
            )

    @property
    def audit_digest(self) -> str:
        return _canonical_digest(
            {
                "prepared_inference_pair_digests": self.prepared_inference_pair_digests,
                "partition_pair_digests": self.partition_pair_digests,
                "partition_component_digests": self.partition_component_digests,
            }
        )

    def pair_digests_for(self, partition: str) -> frozenset[str]:
        return frozenset(dict(self.partition_pair_digests)[partition])

    def component_digests_for(self, partition: str) -> frozenset[str]:
        return frozenset(dict(self.partition_component_digests)[partition])

    def safe_summary(self) -> dict[str, object]:
        return {
            "audit_digest": self.audit_digest,
            "inference_partition": "decision",
            "inference_pair_count": len(self.prepared_inference_pair_digests),
            "protected_partition_count": len(self.partition_pair_digests),
            "pair_and_component_disjointness_verified": True,
        }


def _protected_evidence_audit(
    *,
    batches: dict[str, VerifiedLabelBatch],
    decision_matrix: BoostedLabelledMatrix,
) -> ProtectedModeEvidenceAudit:
    def components(batch: VerifiedLabelBatch) -> tuple[str, ...]:
        values = {
            f"entity:{digest}" for item in batch.labels for digest in item.entity_component_digests
        }
        values.update(
            f"household:{digest}"
            for item in batch.labels
            for digest in item.household_component_digests
        )
        return tuple(sorted(values))

    names: tuple[LabelPartition, ...] = (
        "training",
        "validation",
        "calibration",
        "decision",
        "test",
    )
    pair_digests = tuple(
        (
            name,
            tuple(sorted(item.pair_digest() for item in batches[name].labels)),
        )
        for name in names
    )
    component_digests = tuple((name, components(batches[name])) for name in names)
    return ProtectedModeEvidenceAudit(
        prepared_inference_pair_digests=tuple(sorted(decision_matrix.pair_digests)),
        partition_pair_digests=pair_digests,
        partition_component_digests=component_digests,
    )


def _combined_label_batches(
    surface_batches: dict[str, dict[str, VerifiedLabelBatch]],
) -> tuple[dict[str, VerifiedLabelBatch], PartitionDisjointnessReport]:
    required_surfaces = {"cross", "intra_a", "intra_b"}
    names: tuple[LabelPartition, ...] = (
        "training",
        "validation",
        "calibration",
        "decision",
        "test",
    )
    if set(surface_batches) != required_surfaces or any(
        set(batches) != set(names) for batches in surface_batches.values()
    ):
        raise PipelineError("ML-MODE-024", "Combined surface label evidence is incomplete.")
    combined: dict[str, VerifiedLabelBatch] = {}
    for partition in names:
        authorities = {
            surface: surface_batches[surface][partition].label_authority_digest
            for surface in sorted(required_surfaces)
        }
        labels = tuple(
            label
            for surface in sorted(required_surfaces)
            for label in surface_batches[surface][partition].labels
        )
        combined[partition] = VerifiedLabelBatch(
            source_kind="synthetic_truth",
            verification_protocol="synthetic_combined_surfaces_v1",
            source_digest=_canonical_digest(
                {
                    "partition": partition,
                    "surface_label_authority_digests": authorities,
                }
            ),
            partition=partition,
            labels=labels,
        )
    disjointness = assert_disjoint_label_partitions(combined.values())
    return combined, disjointness


def _combined_matrix(
    *,
    surface_matrices: dict[str, BoostedLabelledMatrix],
    combined_labels: VerifiedLabelBatch,
    random_seed: int,
) -> BoostedLabelledMatrix:
    required_surfaces = ("cross", "intra_a", "intra_b")
    if tuple(sorted(surface_matrices)) != tuple(sorted(required_surfaces)):
        raise PipelineError("ML-MODE-024", "Combined surface matrix evidence is incomplete.")
    matrices = tuple(surface_matrices[name] for name in required_surfaces)
    schemas = {matrix.feature_schema_digest for matrix in matrices}
    names = {matrix.feature_names for matrix in matrices}
    partitions = {matrix.partition for matrix in matrices}
    if len(schemas) != 1 or len(names) != 1 or partitions != {combined_labels.partition}:
        raise PipelineError(
            "ML-MODE-025", "Cross and intra-source feature contracts are incompatible."
        )
    rows: list[tuple[str, str, tuple[str, str], np.ndarray, int]] = []
    for surface in required_surfaces:
        matrix = surface_matrices[surface]
        rows.extend(
            (
                surface,
                digest,
                pair,
                matrix.features[index],
                int(matrix.labels[index]),
            )
            for index, (pair, digest) in enumerate(
                zip(matrix.pair_references, matrix.pair_digests, strict=True)
            )
        )
    rows.sort(key=lambda item: (item[0], item[1]))
    pair_digests = tuple(item[1] for item in rows)
    if len(set(pair_digests)) != len(pair_digests):
        raise PipelineError("ML-MODE-024", "Combined surface pair evidence overlaps.")
    labels = np.asarray([item[4] for item in rows], dtype=np.int8)
    selected_pairs = tuple(item[2] for item in rows)
    selection_digest = _canonical_digest(
        {
            "partition": combined_labels.partition,
            "random_seed": random_seed,
            "surface_rows": [
                {"surface": surface, "pair_digest": digest, "label": label}
                for surface, digest, _, _, label in rows
            ],
            "combined_label_authority_digest": combined_labels.label_authority_digest,
        }
    )
    positive_count = int(labels.sum())
    return BoostedLabelledMatrix(
        features=np.vstack([item[3] for item in rows]),
        pair_references=selected_pairs,
        pair_digests=pair_digests,
        feature_names=matrices[0].feature_names,
        feature_schema_digest=matrices[0].feature_schema_digest,
        labels=labels,
        partition=combined_labels.partition,
        label_source_kind="synthetic_truth",
        label_authority_digest=combined_labels.label_authority_digest,
        selection_digest=selection_digest,
        positive_count=positive_count,
        negative_count=len(labels) - positive_count,
    )


_COMBINED_BINDING_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class CombinedSurfaceCalibrationBinding:
    """Authorize only canonically derived three-surface calibrated evidence."""

    model_family: Literal["xgboost"]
    model_id: str
    model_version: str
    feature_schema_digest: str
    model_digest: str
    calibrator_digest: str
    training_label_authority_digest: str
    validation_label_authority_digest: str
    calibration_label_authority_digest: str
    partition_manifest_digest: str
    surface_label_authority_digests: tuple[tuple[str, str, str, str], ...]
    combined_label_authority_digests: tuple[tuple[str, str], ...]
    combined_matrix_selection_digests: tuple[tuple[str, str], ...]
    champion_selection_digest: str
    authority_construction_digest: str
    _factory_token: InitVar[object] = None

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _COMBINED_BINDING_FACTORY_TOKEN:
            raise PipelineError("ML-MODE-024", "Combined calibration scope is invalid.")
        if self.model_family != "xgboost" or not self.model_id or not self.model_version:
            raise PipelineError("ML-MODE-024", "Combined calibration scope is invalid.")
        if tuple(item[0] for item in self.surface_label_authority_digests) != (
            "cross",
            "intra_a",
            "intra_b",
        ):
            raise PipelineError("ML-MODE-024", "Combined calibration scope is invalid.")
        if tuple(item[0] for item in self.combined_label_authority_digests) != (
            "calibration",
            "training",
            "validation",
        ) or tuple(item[0] for item in self.combined_matrix_selection_digests) != (
            "calibration",
            "training",
            "validation",
        ):
            raise PipelineError("ML-MODE-024", "Combined calibration scope is invalid.")
        for digest in (
            self.feature_schema_digest,
            self.model_digest,
            self.calibrator_digest,
            self.training_label_authority_digest,
            self.validation_label_authority_digest,
            self.calibration_label_authority_digest,
            self.partition_manifest_digest,
            self.champion_selection_digest,
            self.authority_construction_digest,
            *(digest for item in self.surface_label_authority_digests for digest in item[1:]),
            *(digest for _, digest in self.combined_label_authority_digests),
            *(digest for _, digest in self.combined_matrix_selection_digests),
        ):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise PipelineError("ML-MODE-024", "Combined calibration scope is invalid.")
        combined = dict(self.combined_label_authority_digests)
        if (
            combined["training"] != self.training_label_authority_digest
            or combined["validation"] != self.validation_label_authority_digest
            or combined["calibration"] != self.calibration_label_authority_digest
            or self.authority_construction_digest != self._expected_construction_digest()
        ):
            raise PipelineError("ML-MODE-024", "Combined calibration scope is invalid.")

    @property
    def surfaces(self) -> tuple[str, str, str]:
        return ("cross", "intra_a", "intra_b")

    def _expected_construction_digest(self) -> str:
        return _canonical_digest(
            {
                "model_family": self.model_family,
                "model_id": self.model_id,
                "model_version": self.model_version,
                "model_digest": self.model_digest,
                "feature_schema_digest": self.feature_schema_digest,
                "calibrator_digest": self.calibrator_digest,
                "surface_label_authority_digests": self.surface_label_authority_digests,
                "combined_label_authority_digests": self.combined_label_authority_digests,
                "combined_matrix_selection_digests": self.combined_matrix_selection_digests,
                "champion_selection_digest": self.champion_selection_digest,
                "partition_manifest_digest": self.partition_manifest_digest,
            }
        )

    @classmethod
    def from_protected_evidence(
        cls,
        *,
        surface_batches: dict[str, dict[str, VerifiedLabelBatch]],
        combined_batches: dict[str, VerifiedLabelBatch],
        combined_matrices: dict[str, BoostedLabelledMatrix],
        model: XGBoostModelArtifact,
        calibrator: CalibratorArtifact,
        selection: ChampionSelection,
        partition_manifest_digest: str,
    ) -> CombinedSurfaceCalibrationBinding:
        required_surfaces = ("cross", "intra_a", "intra_b")
        required_partitions = ("calibration", "training", "validation")
        if (
            tuple(sorted(surface_batches)) != required_surfaces
            or any(
                not set(required_partitions).issubset(surface_batches[surface])
                for surface in required_surfaces
            )
            or tuple(sorted(combined_matrices)) != required_partitions
            or not set(required_partitions).issubset(combined_batches)
        ):
            raise PipelineError("ML-MODE-024", "Combined surface label evidence is incomplete.")
        derived_batches, _ = _combined_label_batches(surface_batches)
        combined_authorities = tuple(
            (partition, combined_batches[partition].label_authority_digest)
            for partition in required_partitions
        )
        if any(
            derived_batches[partition].label_authority_digest
            != combined_batches[partition].label_authority_digest
            or combined_matrices[partition].label_authority_digest
            != combined_batches[partition].label_authority_digest
            for partition in required_partitions
        ):
            raise PipelineError("ML-MODE-024", "Combined surface label evidence is invalid.")
        matrix_selections = tuple(
            (partition, combined_matrices[partition].selection_digest)
            for partition in required_partitions
        )
        combined = dict(combined_authorities)
        selections = dict(matrix_selections)
        if (
            model.label_authority_digest != combined["training"]
            or model.training_selection_digest != selections["training"]
            or selection.selected_model_family != "xgboost"
            or selection.selected_model_id != model.model_id
            or selection.selected_model_version != model.model_version
            or selection.selected_evidence_digest != model.model_digest
            or selection.selected_feature_schema_digest != model.feature_schema_digest
            or selection.selected_training_label_authority_digest != combined["training"]
            or selection.validation_label_authority_digest != combined["validation"]
            or selection.selection_digest != calibrator.champion_selection_digest
            or calibrator.source_model_family != "xgboost"
            or calibrator.source_model_id != model.model_id
            or calibrator.source_model_version != model.model_version
            or calibrator.source_evidence_digest != model.model_digest
            or calibrator.feature_schema_digest != model.feature_schema_digest
            or calibrator.validation_label_authority_digest != combined["validation"]
            or calibrator.calibration_label_authority_digest != combined["calibration"]
            or selection.partition_manifest_digest != partition_manifest_digest
            or calibrator.partition_manifest_digest != partition_manifest_digest
        ):
            raise PipelineError("ML-MODE-024", "Combined calibration evidence is invalid.")
        surface_authorities = tuple(
            (
                surface,
                surface_batches[surface]["training"].label_authority_digest,
                surface_batches[surface]["validation"].label_authority_digest,
                surface_batches[surface]["calibration"].label_authority_digest,
            )
            for surface in required_surfaces
        )
        construction_digest = _canonical_digest(
            {
                "model_family": "xgboost",
                "model_id": model.model_id,
                "model_version": model.model_version,
                "model_digest": model.model_digest,
                "feature_schema_digest": model.feature_schema_digest,
                "calibrator_digest": calibrator.calibrator_digest,
                "surface_label_authority_digests": surface_authorities,
                "combined_label_authority_digests": combined_authorities,
                "combined_matrix_selection_digests": matrix_selections,
                "champion_selection_digest": selection.selection_digest,
                "partition_manifest_digest": partition_manifest_digest,
            }
        )
        return cls(
            model_family="xgboost",
            model_id=model.model_id,
            model_version=model.model_version,
            feature_schema_digest=model.feature_schema_digest,
            model_digest=model.model_digest,
            calibrator_digest=calibrator.calibrator_digest,
            training_label_authority_digest=combined["training"],
            validation_label_authority_digest=combined["validation"],
            calibration_label_authority_digest=combined["calibration"],
            partition_manifest_digest=partition_manifest_digest,
            surface_label_authority_digests=surface_authorities,
            combined_label_authority_digests=combined_authorities,
            combined_matrix_selection_digests=matrix_selections,
            champion_selection_digest=selection.selection_digest,
            authority_construction_digest=construction_digest,
            _factory_token=_COMBINED_BINDING_FACTORY_TOKEN,
        )

    @property
    def binding_digest(self) -> str:
        return _canonical_digest({**asdict(self), "surfaces": self.surfaces})

    def assert_complete(self) -> None:
        if self.authority_construction_digest != self._expected_construction_digest():
            raise PipelineError(
                "ML-MODE-026", "Cross-only calibration cannot authorize intra-source scoring."
            )

    def assert_authorizes(
        self,
        *,
        surface: str,
        matrix: BoostedFeatureMatrix,
        model: XGBoostModelArtifact,
        calibrator: CalibratorArtifact,
    ) -> None:
        if (
            surface not in self.surfaces
            or self.model_family != "xgboost"
            or model.model_id != self.model_id
            or model.model_version != self.model_version
            or matrix.feature_schema_digest != self.feature_schema_digest
            or model.model_digest != self.model_digest
            or calibrator.calibrator_digest != self.calibrator_digest
            or calibrator.source_model_family != self.model_family
            or calibrator.source_model_id != model.model_id
            or calibrator.source_model_version != model.model_version
            or calibrator.source_evidence_digest != model.model_digest
            or calibrator.feature_schema_digest != model.feature_schema_digest
            or model.label_authority_digest != self.training_label_authority_digest
            or calibrator.validation_label_authority_digest
            != self.validation_label_authority_digest
            or calibrator.calibration_label_authority_digest
            != self.calibration_label_authority_digest
            or calibrator.partition_manifest_digest != self.partition_manifest_digest
            or model.training_selection_digest
            != dict(self.combined_matrix_selection_digests)["training"]
            or calibrator.champion_selection_digest != self.champion_selection_digest
            or self.authority_construction_digest != self._expected_construction_digest()
        ):
            raise PipelineError(
                "ML-MODE-026", "Cross-only calibration cannot authorize intra-source scoring."
            )


def _selection(
    *,
    artifact: XGBoostModelArtifact,
    validation_matrix: BoostedLabelledMatrix,
    report: PairValidationReport,
    partition_manifest_digest: str,
    primary_metric: Literal["average_precision", "brier_score"],
) -> ChampionSelection:
    candidate = ModelEvaluationCandidate(
        model_family="xgboost",
        model_id=artifact.model_id,
        model_version=artifact.model_version,
        evidence_digest=artifact.model_digest,
        feature_schema_digest=artifact.feature_schema_digest,
        validation_label_authority_digest=validation_matrix.label_authority_digest,
        partition_manifest_digest=partition_manifest_digest,
        average_precision=report.average_precision,
        brier_score=report.brier_score,
        pair_count=report.pair_count,
        training_label_authority_digest=artifact.label_authority_digest,
    )
    secondary: Literal["average_precision", "brier_score"] = (
        "brier_score" if primary_metric == "average_precision" else "average_precision"
    )
    summaries = (candidate.safe_summary(),)
    digest = _canonical_digest(
        {
            "selection_kind": "configured_single_champion",
            "selected": candidate.safe_summary(),
            "selected_training_label_authority_digest": artifact.label_authority_digest,
            "validation_label_authority_digest": validation_matrix.label_authority_digest,
            "partition_manifest_digest": partition_manifest_digest,
            "primary_metric": primary_metric,
            "secondary_metric": secondary,
            "candidates": summaries,
            "test_partition_used": False,
            "calibration_partition_used": False,
        }
    )
    return ChampionSelection(
        selected_model_family="xgboost",
        selected_model_id=artifact.model_id,
        selected_model_version=artifact.model_version,
        selected_evidence_digest=artifact.model_digest,
        selected_feature_schema_digest=artifact.feature_schema_digest,
        selected_training_label_authority_digest=artifact.label_authority_digest,
        validation_label_authority_digest=validation_matrix.label_authority_digest,
        partition_manifest_digest=partition_manifest_digest,
        primary_metric=primary_metric,
        secondary_metric=secondary,
        selection_digest=digest,
        candidate_summaries=summaries,
    )


def _persist_reload_model(
    *, artifact: XGBoostModelArtifact, base: str, plan: ExecutionPlan
) -> XGBoostModelArtifact:
    try:
        write_xgboost_artifact(
            artifact=artifact,
            model_path=f"{base}/champion.json",
            manifest_path=f"{base}/champion.manifest.json",
            policy=plan.path_policy,
        )
        reloaded = read_xgboost_artifact(
            model_path=f"{base}/champion.json",
            manifest_path=f"{base}/champion.manifest.json",
            policy=plan.path_policy,
        )
    except (OSError, UnicodeError):
        raise PipelineError(
            "ML-MODE-014", "The fitted pair model could not be persisted and reloaded."
        ) from None
    if reloaded != artifact or artifact.configuration_digest != plan.configuration_digest:
        raise PipelineError("ML-MODE-014", "The fitted pair model failed strict reload.")
    return reloaded


def _persist_reload_calibrator(
    *, artifact: CalibratorArtifact, base: str, plan: ExecutionPlan
) -> CalibratorArtifact:
    try:
        write_calibrator_artifact(
            artifact=artifact,
            payload_path=f"{base}/calibrator.json",
            manifest_path=f"{base}/calibrator.manifest.json",
            policy=plan.path_policy,
        )
        reloaded = read_calibrator_artifact(
            payload_path=f"{base}/calibrator.json",
            manifest_path=f"{base}/calibrator.manifest.json",
            policy=plan.path_policy,
        )
    except (OSError, UnicodeError):
        raise PipelineError(
            "ML-MODE-015", "The probability calibrator could not be persisted and reloaded."
        ) from None
    if reloaded != artifact:
        raise PipelineError("ML-MODE-015", "The probability calibrator failed strict reload.")
    return reloaded


def _persist_reload_recipe(
    *, recipe: PipelineRecipeArtifact, base: str, plan: ExecutionPlan
) -> PipelineRecipeArtifact:
    try:
        destination = plan.path_policy.resolve_output(f"{base}/recipe-v1.json")
        if destination.suffix != ".json" or destination.is_symlink():
            raise PipelineError("ML-MODE-016", "The synthetic mode artifact path is invalid.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(destination, serialize_pipeline_recipe(recipe))
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.stat().st_size > 262_144
        ):
            raise OSError
        payload = destination.read_text(encoding="utf-8")
    except PipelineError:
        raise
    except (OSError, UnicodeError):
        raise PipelineError(
            "ML-MODE-016", "The synthetic mode artifact could not be persisted and reloaded."
        ) from None
    reloaded = deserialize_pipeline_recipe(payload)
    if reloaded.recipe_digest != recipe.recipe_digest:
        raise PipelineError("ML-MODE-017", "The pipeline recipe failed strict reload.")
    return reloaded


def _public_matrix(
    *, matrix: BoostedFeatureMatrix, bundle: SyntheticBundle, source_id: str, target_id: str
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
        raise PipelineError("ML-MODE-018", "Synthetic pair provenance is invalid.") from None
    return BoostedFeatureMatrix(
        features=matrix.features,
        pair_references=references,
        pair_digests=tuple(
            hashlib.sha256(f"{left}\x00{right}".encode()).hexdigest() for left, right in references
        ),
        feature_names=matrix.feature_names,
        feature_schema_digest=matrix.feature_schema_digest,
    )


def _assignment_batch(
    *,
    matrix: BoostedFeatureMatrix,
    record_keys: tuple[str, ...],
    model: XGBoostModelArtifact,
    calibrator: CalibratorArtifact,
    surface: str | None = None,
    calibration_binding: CombinedSurfaceCalibrationBinding | None = None,
) -> tuple[AssignmentEdgeBatch, str]:
    if (surface is None) != (calibration_binding is None):
        raise PipelineError("ML-MODE-026", "Calibrated surface binding is incomplete.")
    if surface is not None and calibration_binding is not None:
        calibration_binding.assert_authorizes(
            surface=surface,
            matrix=matrix,
            model=model,
            calibrator=calibrator,
        )
    raw_scores = XGBoostPairClassifier._predict(matrix=matrix, model=model)
    if calibrator.method == "sigmoid":
        probabilities = SigmoidCalibrator.apply(raw_scores, calibrator)
    elif calibrator.method == "isotonic":
        probabilities = IsotonicCalibrator.apply(raw_scores, calibrator)
    elif calibrator.method == "beta":
        probabilities = BetaCalibrator.apply(raw_scores, calibrator)
    else:  # pragma: no cover - CalibratorArtifact already rejects this state.
        raise PipelineError("ML-MODE-015", "The probability calibrator is unsupported.")
    ranks = np.zeros(matrix.pair_count, dtype=np.int64)
    by_source: dict[str, list[tuple[float, str, int]]] = {}
    for index, ((left, _), digest) in enumerate(
        zip(matrix.pair_references, matrix.pair_digests, strict=True)
    ):
        by_source.setdefault(left, []).append((float(probabilities[index]), digest, index))
    for items in by_source.values():
        for rank, (_, _, index) in enumerate(
            sorted(items, key=lambda item: (-item[0], item[1])), start=1
        ):
            ranks[index] = rank
    batch = AssignmentEdgeBatch(
        source_record_keys=record_keys,
        pair_references=matrix.pair_references,
        pair_digests=matrix.pair_digests,
        probabilities=probabilities,
        candidate_ranks=ranks,
        source_model_id=model.model_id,
        source_model_version=model.model_version,
        calibrator_digest=calibrator.calibrator_digest,
        ranking_model_digest=None,
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )
    values = np.asarray(probabilities, dtype="<f8")
    evidence = hashlib.sha256()
    evidence.update(b"calibrated_assignment_evidence\x00")
    evidence.update(model.model_digest.encode("ascii"))
    evidence.update(calibrator.calibrator_digest.encode("ascii"))
    evidence.update(matrix.feature_schema_digest.encode("ascii"))
    evidence.update((surface or "single_surface").encode("ascii"))
    if calibration_binding is not None:
        evidence.update(calibration_binding.binding_digest.encode("ascii"))
    evidence.update("\x00".join(matrix.pair_digests).encode("ascii"))
    evidence.update(values.tobytes(order="C"))
    return batch, evidence.hexdigest()


def _persist_reload_mode_artifacts(
    *,
    orchestration: SyntheticModeOrchestrationArtifact,
    run: SyntheticModeRunArtifact,
    base: str,
    plan: ExecutionPlan,
) -> tuple[SyntheticModeOrchestrationArtifact, SyntheticModeRunArtifact]:
    try:
        orchestration_path = plan.path_policy.resolve_output(f"{base}/mode-orchestration-v1.json")
        run_path = plan.path_policy.resolve_output(f"{base}/mode-run-v1.json")
        if (
            orchestration_path.suffix != ".json"
            or run_path.suffix != ".json"
            or orchestration_path == run_path
            or orchestration_path.is_symlink()
            or run_path.is_symlink()
        ):
            raise PipelineError("ML-MODE-016", "The synthetic mode artifact path is invalid.")
        orchestration_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            orchestration_path,
            serialize_mode_orchestration_artifact(orchestration),
        )
        atomic_write_text(run_path, serialize_mode_run_artifact(run))
        if any(
            path.is_symlink() or not path.is_file() or path.stat().st_size > 262_144
            for path in (orchestration_path, run_path)
        ):
            raise OSError
        reloaded_orchestration = deserialize_mode_orchestration_artifact(
            orchestration_path.read_text(encoding="utf-8")
        )
        reloaded_run = deserialize_mode_run_artifact(run_path.read_text(encoding="utf-8"))
    except PipelineError:
        raise
    except (OSError, UnicodeError):
        raise PipelineError(
            "ML-MODE-016", "The synthetic mode artifact could not be persisted and reloaded."
        ) from None
    if (
        reloaded_orchestration.artifact_digest != orchestration.artifact_digest
        or reloaded_run.run_digest != run.run_digest
        or reloaded_run.orchestration_artifact_digest != reloaded_orchestration.artifact_digest
    ):
        raise PipelineError("ML-MODE-017", "The synthetic mode artifacts failed reload.")
    return reloaded_orchestration, reloaded_run


@dataclass(frozen=True, slots=True, repr=False)
class SyntheticModeWorkflowResult:
    """Aggregate-only outcome of a bounded generated-synthetic mode run."""

    linkage_mode: Literal["link_only"]
    assignment_constraint: Literal["many_to_one", "one_to_many", "unconstrained"]
    run_id: str
    candidate_pair_count: int
    protected_partition_count: int
    validation_pair_count: int
    locked_test_pair_count: int
    recipe: PipelineRecipeArtifact = field(repr=False)
    model_artifact: XGBoostModelArtifact = field(repr=False)
    calibrator_artifact: CalibratorArtifact = field(repr=False)
    inference: ApprovedRecipeInferenceResult = field(repr=False)
    evidence_audit: ProtectedModeEvidenceAudit = field(repr=False)
    workflow_digest: str
    operational_validation: Literal["not_established"] = "not_established"
    merge_authority: Literal["none"] = "none"

    def safe_summary(self) -> dict[str, object]:
        return {
            "linkage_mode": self.linkage_mode,
            "assignment_constraint": self.assignment_constraint,
            "run_id": self.run_id,
            "candidate_pair_count": self.candidate_pair_count,
            "protected_partition_count": self.protected_partition_count,
            "validation_pair_count": self.validation_pair_count,
            "locked_test_pair_count": self.locked_test_pair_count,
            "recipe_digest": self.recipe.recipe_digest,
            "inference": self.inference.safe_summary(),
            "protected_inference_boundary": self.evidence_audit.safe_summary(),
            "workflow_digest": self.workflow_digest,
            "operational_validation": self.operational_validation,
            "merge_authority": self.merge_authority,
        }


@dataclass(frozen=True, slots=True, repr=False)
class SyntheticDedupeModeWorkflowResult:
    """Aggregate-only outcome of one protected synthetic dedupe run."""

    run_id: str
    candidate_pair_count: int
    decision_pair_count: int
    input_record_count: int
    model_artifact: XGBoostModelArtifact = field(repr=False)
    calibrator_artifact: CalibratorArtifact = field(repr=False)
    orchestration_artifact: SyntheticModeOrchestrationArtifact = field(repr=False)
    run_artifact: SyntheticModeRunArtifact = field(repr=False)
    workflow: DeduplicationWorkflowResult = field(repr=False)
    evidence_audit: ProtectedModeEvidenceAudit = field(repr=False)
    qualification_digest: str
    operational_validation: Literal["not_established"] = "not_established"
    decision_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"

    def safe_summary(self) -> dict[str, object]:
        return {
            "linkage_mode": "dedupe_only",
            "assignment_constraint": "unconstrained",
            "run_id": self.run_id,
            "candidate_pair_count": self.candidate_pair_count,
            "decision_pair_count": self.decision_pair_count,
            "input_record_count": self.input_record_count,
            "orchestration": self.orchestration_artifact.safe_summary(),
            "run": self.run_artifact.safe_summary(),
            "workflow": self.workflow.safe_summary(),
            "protected_inference_boundary": self.evidence_audit.safe_summary(),
            "qualification_digest": self.qualification_digest,
            "operational_validation": self.operational_validation,
            "decision_authority": self.decision_authority,
            "merge_authority": self.merge_authority,
        }


@dataclass(frozen=True, slots=True, repr=False)
class SyntheticLinkAndDedupeWorkflowResult:
    """Aggregate-only outcome of protected two-source link-and-dedupe."""

    run_id: str
    candidate_pair_count: int
    decision_pair_count: int
    input_record_count: int
    model_artifact: XGBoostModelArtifact = field(repr=False)
    calibrator_artifact: CalibratorArtifact = field(repr=False)
    orchestration_artifact: SyntheticModeOrchestrationArtifact = field(repr=False)
    run_artifact: SyntheticModeRunArtifact = field(repr=False)
    workflow: DeduplicationWorkflowResult = field(repr=False)
    evidence_audits: tuple[tuple[str, ProtectedModeEvidenceAudit], ...] = field(repr=False)
    calibration_binding: CombinedSurfaceCalibrationBinding = field(repr=False)
    qualification_digest: str
    operational_validation: Literal["not_established"] = "not_established"
    decision_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"

    def safe_summary(self) -> dict[str, object]:
        return {
            "linkage_mode": "link_and_dedupe",
            "assignment_constraint": "one_to_one",
            "run_id": self.run_id,
            "candidate_pair_count": self.candidate_pair_count,
            "decision_pair_count": self.decision_pair_count,
            "input_record_count": self.input_record_count,
            "orchestration": self.orchestration_artifact.safe_summary(),
            "run": self.run_artifact.safe_summary(),
            "workflow": self.workflow.safe_summary(),
            "protected_inference_boundaries": {
                name: audit.safe_summary() for name, audit in self.evidence_audits
            },
            "combined_calibration_binding_digest": self.calibration_binding.binding_digest,
            "qualification_digest": self.qualification_digest,
            "operational_validation": self.operational_validation,
            "decision_authority": self.decision_authority,
            "merge_authority": self.merge_authority,
        }


class SyntheticModeWorkflowRunner:
    """Run allow-listed synthetic linkage modes through fitted, reloaded evidence."""

    @staticmethod
    def run_link_only(
        plan: ExecutionPlan,
        *,
        generation: SyntheticGenerationConfig | None = None,
    ) -> SyntheticModeWorkflowResult:
        source_id, target_id, fixture_directory = _link_source_target_ids(plan)
        spec = _generation_spec(plan, generation)
        bundle = generate_synthetic_bundle(spec)
        bundle_digest = _bundle_digest(bundle)
        run_id = _canonical_digest(
            {
                "configuration_digest": plan.configuration_digest,
                "registry_digest": plan.registry_digest,
                "mode_dispatch_key": plan.mode_dispatch_key,
                "synthetic_bundle_digest": bundle_digest,
            }
        )[:24]
        write_synthetic_bundle(fixture_directory, bundle)
        artifact_base = f"artifacts/runs/{run_id}/synthetic_mode"

        with DuckDBStore() as store:
            catalog = ConfiguredDatasetPreparer(store).prepare_all(plan)
            left = catalog.require(source_id)
            right = catalog.require(target_id)
            if dict(left.variable_columns) != dict(right.variable_columns):
                raise PipelineError("ML-MODE-019", "Canonical variable contracts differ.")
            prepared_source_keys = _record_keys(store, left.table.table_name)
            if len(prepared_source_keys) != bundle.provenance.source_a_count:
                raise PipelineError("ML-MODE-018", "Synthetic source provenance is invalid.")
            candidate_result = DuckDBCandidateGenerator(store).generate(
                left=left.table,
                right=right.table,
                variable_columns=left.variable_columns,
                rules=runtime_blocking_rules(plan),
                maximum_candidate_pairs=plan.config.runtime.maximum_candidate_pairs,
            )
            snapshot = candidate_snapshot(store, candidate_result.table.table_name)
            if not snapshot.pairs:
                raise PipelineError("ML-MODE-020", "Synthetic mode candidate evidence is empty.")
            candidate_digest = _candidate_plan_digest(snapshot, shape="cross_table")
            feature_result = DuckDBComparisonFeatureBuilder(store).build(
                candidates=candidate_result.table,
                left=left,
                right=right,
                comparisons=plan.config.comparisons,
            )
            matrix_builder = DuckDBVerifiedMatrixBuilder(store)
            scoring_matrix = matrix_builder.build_scoring(features=feature_result)
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
            selected_model = next(
                (
                    model
                    for model in plan.config.models.all_boosted_trees()
                    if model.enabled
                    and plan.config.mode_orchestration is not None
                    and model.model_id == plan.config.mode_orchestration.pair_model_id
                ),
                None,
            )
            if selected_model is None:
                raise PipelineError("ML-MODE-021", "The configured pair model is unavailable.")
            training = matrix_builder.build_labelled(
                features=feature_result,
                labels=batches["training"],
                model=selected_model,
                random_seed=plan.random_seed,
                apply_training_selection=True,
            )
            validation = matrix_builder.build_labelled(
                features=feature_result,
                labels=batches["validation"],
                random_seed=plan.random_seed,
            )
            calibration = matrix_builder.build_labelled(
                features=feature_result,
                labels=batches["calibration"],
                random_seed=plan.random_seed,
            )
            locked_test = matrix_builder.build_labelled(
                features=feature_result,
                labels=batches["test"],
                random_seed=plan.random_seed,
            )
            decision = matrix_builder.build_labelled(
                features=feature_result,
                labels=batches["decision"],
                random_seed=plan.random_seed,
            )
            evidence_audit = _protected_evidence_audit(
                batches=batches,
                decision_matrix=decision,
            )
            classifier = XGBoostPairClassifier(store)
            fitted = classifier.fit(
                matrix=training,
                model=selected_model,
                random_seed=plan.random_seed,
                configuration_digest=plan.configuration_digest,
            )
            validation_scores = classifier._predict(matrix=validation, model=fitted)
            validation_report = evaluate_binary_scores(
                labels=validation.labels,
                scores=validation_scores,
                diagnostic_threshold=0.5,
                evaluation_scope="synthetic_mechanical_evaluation",
                partition_manifest_digest=disjointness.manifest_digest,
            )
            selection = _selection(
                artifact=fitted,
                validation_matrix=validation,
                report=validation_report,
                partition_manifest_digest=disjointness.manifest_digest,
                primary_metric=plan.config.model_selection.primary_metric,
            )
            calibration_scores = classifier._predict(matrix=calibration, model=fitted)
            calibration_batch = PairScoreBatch(
                pair_references=calibration.pair_references,
                pair_digests=calibration.pair_digests,
                scores=calibration_scores,
                labels=calibration.labels,
                partition="calibration",
                source_model_family="xgboost",
                source_model_id=fitted.model_id,
                source_model_version=fitted.model_version,
                source_evidence_digest=fitted.model_digest,
                feature_schema_digest=fitted.feature_schema_digest,
                label_authority_digest=calibration.label_authority_digest,
                partition_manifest_digest=disjointness.manifest_digest,
                champion_selection_digest=selection.selection_digest,
            )
            calibrator = ChampionCalibratorSelector.select(
                calibration_batch,
                selection,
                methods=(plan.config.calibration.method,),
            )
            locked_test_report = classifier.evaluate(
                matrix=locked_test,
                model=fitted,
                disjointness=disjointness,
            )

        persisted_model = _persist_reload_model(artifact=fitted, base=artifact_base, plan=plan)
        persisted_calibrator = _persist_reload_calibrator(
            artifact=calibrator, base=artifact_base, plan=plan
        )
        validation_evidence_digest = _canonical_digest(
            {
                "validation": validation_report.safe_summary(),
                "locked_test": locked_test_report.safe_summary(),
                "selection_digest": selection.selection_digest,
                "calibrator_digest": persisted_calibrator.calibrator_digest,
                "partition_manifest_digest": disjointness.manifest_digest,
                "truth_split_manifest_digest": split_manifest_digest,
                "decision_evidence_audit_digest": evidence_audit.audit_digest,
            }
        )
        constraint = cast(
            Literal["many_to_one", "one_to_many", "unconstrained"],
            plan.config.project.assignment_constraint,
        )
        recipe = PipelineRecipeArtifact(
            recipe_id=f"synthetic_mode_{constraint}",
            recipe_version="i1c-v1",
            linkage_mode="link_only",
            assignment_constraint=constraint,
            configuration_digest=plan.configuration_digest,
            candidate_plan_digest=candidate_digest,
            feature_schema_digest=scoring_matrix.feature_schema_digest,
            champion_model_id=persisted_model.model_id,
            champion_model_version=persisted_model.model_version,
            champion_artifact_digest=persisted_model.model_digest,
            calibrator_digest=persisted_calibrator.calibrator_digest,
            ranking_artifact_digest=None,
            decision_policy_digest=_canonical_digest(
                plan.config.decision_policy.model_dump(mode="json")
            ),
            validation_evidence_digest=validation_evidence_digest,
            approval_status=RecipeApprovalStatus.SYNTHETIC_VALIDATED,
            operational_validation=OperationalValidationStatus.NOT_ESTABLISHED,
        )
        persisted_recipe = _persist_reload_recipe(recipe=recipe, base=artifact_base, plan=plan)
        public_matrix = _public_matrix(
            matrix=_feature_view(decision),
            bundle=bundle,
            source_id=source_id,
            target_id=target_id,
        )
        public_source_keys = tuple(sorted({left for left, _ in public_matrix.pair_references}))
        attestation = attest_generated_synthetic_inference(
            bundle=bundle,
            source_record_keys=public_source_keys,
            pair_references=public_matrix.pair_references,
            feature_matrix=public_matrix,
            source_dataset_id="source_a",
            target_dataset_id="source_b",
        )
        inference = infer_with_approved_recipe(
            recipe=persisted_recipe,
            source_record_keys=public_source_keys,
            pair_references=public_matrix.pair_references,
            feature_matrix=public_matrix,
            champion_model_artifact=persisted_model,
            calibrator_artifact=persisted_calibrator,
            decision_policy=plan.config.decision_policy,
            execution_mode=RecipeExecutionMode.SYNTHETIC_INFERENCE,
            synthetic_attestation=attestation,
            synthetic_bundle=bundle,
            source_dataset_id="source_a",
            target_dataset_id="source_b",
        )
        workflow_digest = _canonical_digest(
            {
                "dispatch": plan.mode_dispatch_key,
                "bundle_digest": bundle_digest,
                "candidate_plan_digest": candidate_digest,
                "recipe_digest": persisted_recipe.recipe_digest,
                "model_digest": persisted_model.model_digest,
                "calibrator_digest": persisted_calibrator.calibrator_digest,
                "inference_digest": inference.inference_digest,
                "decision_evidence_audit_digest": evidence_audit.audit_digest,
            }
        )
        return SyntheticModeWorkflowResult(
            linkage_mode="link_only",
            assignment_constraint=constraint,
            run_id=run_id,
            candidate_pair_count=candidate_result.candidate_pair_count,
            protected_partition_count=len(batches),
            validation_pair_count=validation_report.pair_count,
            locked_test_pair_count=locked_test_report.pair_count,
            recipe=persisted_recipe,
            model_artifact=persisted_model,
            calibrator_artifact=persisted_calibrator,
            inference=inference,
            evidence_audit=evidence_audit,
            workflow_digest=workflow_digest,
        )

    @staticmethod
    def run_dedupe_only(
        plan: ExecutionPlan,
        *,
        generation: SyntheticGenerationConfig | None = None,
    ) -> SyntheticDedupeModeWorkflowResult:
        source_id, fixture_directory = _dedupe_source_id(plan)
        spec = _dedupe_generation_spec(plan, generation)
        bundle = generate_synthetic_bundle(spec)
        bundle_digest = _bundle_digest(bundle)
        run_id = _canonical_digest(
            {
                "configuration_digest": plan.configuration_digest,
                "registry_digest": plan.registry_digest,
                "mode_dispatch_key": plan.mode_dispatch_key,
                "synthetic_bundle_digest": bundle_digest,
            }
        )[:24]
        write_synthetic_bundle(fixture_directory, bundle)
        artifact_base = f"artifacts/runs/{run_id}/synthetic_mode"
        orchestration_config = plan.config.mode_orchestration
        if orchestration_config is None or orchestration_config.deduplication is None:
            raise PipelineError("ML-MODE-010", "The synthetic linkage-mode dispatch is invalid.")
        dedupe_config = orchestration_config.deduplication

        with DuckDBStore() as store:
            prepared = ConfiguredDatasetPreparer(store).prepare_all(plan).require(source_id)
            prepared_keys = _record_keys(store, prepared.table.table_name)
            if len(prepared_keys) != bundle.provenance.source_a_count:
                raise PipelineError("ML-MODE-018", "Synthetic source provenance is invalid.")
            candidate_result = DuckDBCandidateGenerator(store).generate_deduplication(
                dataset=prepared.table,
                variable_columns=prepared.variable_columns,
                rules=runtime_blocking_rules(plan),
                maximum_candidate_pairs=dedupe_config.maximum_candidate_edges,
            )
            snapshot = candidate_snapshot(store, candidate_result.table.table_name)
            if not snapshot.pairs:
                raise PipelineError("ML-MODE-020", "Synthetic mode candidate evidence is empty.")
            candidate_digest = _candidate_plan_digest(
                snapshot,
                shape="canonical_same_table",
            )
            feature_result = DuckDBComparisonFeatureBuilder(store).build(
                candidates=candidate_result.table,
                left=prepared,
                right=prepared,
                comparisons=plan.config.comparisons,
            )
            matrix_builder = DuckDBVerifiedMatrixBuilder(store)
            scoring_matrix = matrix_builder.build_scoring(features=feature_result)
            truth_records = _single_source_truth_records(
                bundle,
                source_dataset_id=source_id,
            )
            batches, disjointness, split_manifest_digest = protected_label_batches(
                candidate_pairs=snapshot.pairs,
                truth_records=truth_records,
                plan=plan,
            )
            selected_model = next(
                (
                    model
                    for model in plan.config.models.all_boosted_trees()
                    if model.enabled and model.model_id == orchestration_config.pair_model_id
                ),
                None,
            )
            if selected_model is None:
                raise PipelineError("ML-MODE-021", "The configured pair model is unavailable.")
            training = matrix_builder.build_labelled(
                features=feature_result,
                labels=batches["training"],
                model=selected_model,
                random_seed=plan.random_seed,
                apply_training_selection=True,
            )
            validation = matrix_builder.build_labelled(
                features=feature_result,
                labels=batches["validation"],
                random_seed=plan.random_seed,
            )
            calibration = matrix_builder.build_labelled(
                features=feature_result,
                labels=batches["calibration"],
                random_seed=plan.random_seed,
            )
            decision = matrix_builder.build_labelled(
                features=feature_result,
                labels=batches["decision"],
                random_seed=plan.random_seed,
            )
            locked_test = matrix_builder.build_labelled(
                features=feature_result,
                labels=batches["test"],
                random_seed=plan.random_seed,
            )
            evidence_audit = _protected_evidence_audit(
                batches=batches,
                decision_matrix=decision,
            )
            classifier = XGBoostPairClassifier(store)
            fitted = classifier.fit(
                matrix=training,
                model=selected_model,
                random_seed=plan.random_seed,
                configuration_digest=plan.configuration_digest,
            )
            validation_scores = classifier._predict(matrix=validation, model=fitted)
            validation_report = evaluate_binary_scores(
                labels=validation.labels,
                scores=validation_scores,
                diagnostic_threshold=0.5,
                evaluation_scope="synthetic_mechanical_evaluation",
                partition_manifest_digest=disjointness.manifest_digest,
            )
            selection = _selection(
                artifact=fitted,
                validation_matrix=validation,
                report=validation_report,
                partition_manifest_digest=disjointness.manifest_digest,
                primary_metric=plan.config.model_selection.primary_metric,
            )
            calibration_batch = PairScoreBatch(
                pair_references=calibration.pair_references,
                pair_digests=calibration.pair_digests,
                scores=classifier._predict(matrix=calibration, model=fitted),
                labels=calibration.labels,
                partition="calibration",
                source_model_family="xgboost",
                source_model_id=fitted.model_id,
                source_model_version=fitted.model_version,
                source_evidence_digest=fitted.model_digest,
                feature_schema_digest=fitted.feature_schema_digest,
                label_authority_digest=calibration.label_authority_digest,
                partition_manifest_digest=disjointness.manifest_digest,
                champion_selection_digest=selection.selection_digest,
            )
            calibrator = ChampionCalibratorSelector.select(
                calibration_batch,
                selection,
                methods=(plan.config.calibration.method,),
            )
            locked_test_report = classifier.evaluate(
                matrix=locked_test,
                model=fitted,
                disjointness=disjointness,
            )

        persisted_model = _persist_reload_model(
            artifact=fitted,
            base=artifact_base,
            plan=plan,
        )
        persisted_calibrator = _persist_reload_calibrator(
            artifact=calibrator,
            base=artifact_base,
            plan=plan,
        )
        decision_features = _feature_view(decision)
        decision_record_keys = tuple(
            sorted({key for pair in decision_features.pair_references for key in pair})
        )
        candidate_batch, calibrated_evidence_digest = _assignment_batch(
            matrix=decision_features,
            record_keys=decision_record_keys,
            model=persisted_model,
            calibrator=persisted_calibrator,
        )
        dedupe_plan = DeduplicationPlan(
            algorithm=dedupe_config.algorithm,
            threshold=dedupe_config.minimum_probability,
            no_match_utility=dedupe_config.no_match_utility,
            max_cluster_size=dedupe_config.maximum_cluster_size,
            maximum_candidate_edges=dedupe_config.maximum_candidate_edges,
            deterministic_tie_breaking=True,
        )
        workflow = DeduplicationWorkflowRunner.run_dedupe_only(
            record_keys=decision_record_keys,
            candidate_batch=candidate_batch,
            plan=dedupe_plan,
            dataset_id=source_id,
        )
        dedupe_result = workflow.deduplication_result
        if dedupe_result is None or workflow.manifest_path is not None:
            raise PipelineError("ML-MODE-023", "The dedupe result contract is invalid.")
        dedupe_plan_digest = _canonical_digest(asdict(dedupe_plan))
        assignment_plan_digest = _canonical_digest(
            {
                "linkage_mode": "dedupe_only",
                "assignment_constraint": "unconstrained",
                "assignment_authority": "none",
            }
        )
        orchestration_artifact = SyntheticModeOrchestrationArtifact(
            linkage_mode="dedupe_only",
            assignment_constraint="unconstrained",
            configuration_digest=plan.configuration_digest,
            registry_digest=plan.registry_digest,
            synthetic_bundle_digest=bundle_digest,
            generator_version=bundle.provenance.generator_version,
            random_seed=20260816,
            candidate_plan_digests=(candidate_digest,),
            calibrated_evidence_digests=(calibrated_evidence_digest,),
            feature_schema_digest=scoring_matrix.feature_schema_digest,
            champion_model_id=persisted_model.model_id,
            champion_model_version=persisted_model.model_version,
            champion_artifact_digest=persisted_model.model_digest,
            calibrator_digest=persisted_calibrator.calibrator_digest,
            partition_manifest_digest=disjointness.manifest_digest,
            deduplication_plan_digest=dedupe_plan_digest,
            assignment_plan_digest=assignment_plan_digest,
        )
        selected_edge_count = sum(cluster.edge_count for cluster in dedupe_result.clusters)
        run_artifact = SyntheticModeRunArtifact(
            linkage_mode="dedupe_only",
            orchestration_artifact_digest=orchestration_artifact.artifact_digest,
            configuration_digest=plan.configuration_digest,
            result_digest=workflow.workflow_digest,
            input_record_count=len(decision_record_keys),
            candidate_pair_count=candidate_batch.candidate_pair_count,
            cluster_count=dedupe_result.total_clusters,
            selected_edge_count=selected_edge_count,
        )
        persisted_orchestration, persisted_run = _persist_reload_mode_artifacts(
            orchestration=orchestration_artifact,
            run=run_artifact,
            base=artifact_base,
            plan=plan,
        )
        aggregate_digest = _canonical_digest(
            {
                "validation": validation_report.safe_summary(),
                "locked_test": locked_test_report.safe_summary(),
                "truth_split_manifest_digest": split_manifest_digest,
                "decision_evidence_audit_digest": evidence_audit.audit_digest,
                "orchestration_artifact_digest": persisted_orchestration.artifact_digest,
                "run_digest": persisted_run.run_digest,
            }
        )
        return SyntheticDedupeModeWorkflowResult(
            run_id=run_id,
            candidate_pair_count=candidate_result.candidate_pair_count,
            decision_pair_count=decision_features.pair_count,
            input_record_count=len(decision_record_keys),
            model_artifact=persisted_model,
            calibrator_artifact=persisted_calibrator,
            orchestration_artifact=persisted_orchestration,
            run_artifact=persisted_run,
            workflow=workflow,
            evidence_audit=evidence_audit,
            qualification_digest=aggregate_digest,
        )

    @staticmethod
    def run_link_and_dedupe(
        plan: ExecutionPlan,
        *,
        generation: SyntheticGenerationConfig | None = None,
    ) -> SyntheticLinkAndDedupeWorkflowResult:
        source_id, target_id, fixture_directory = _link_and_dedupe_ids(plan)
        spec = _generation_spec(plan, generation)
        bundle = generate_synthetic_bundle(spec)
        bundle_digest = _bundle_digest(bundle)
        run_id = _canonical_digest(
            {
                "configuration_digest": plan.configuration_digest,
                "registry_digest": plan.registry_digest,
                "mode_dispatch_key": plan.mode_dispatch_key,
                "synthetic_bundle_digest": bundle_digest,
            }
        )[:24]
        write_synthetic_bundle(fixture_directory, bundle)
        artifact_base = f"artifacts/runs/{run_id}/synthetic_mode"
        orchestration_config = plan.config.mode_orchestration
        if orchestration_config is None or orchestration_config.deduplication is None:
            raise PipelineError("ML-MODE-010", "The synthetic linkage-mode dispatch is invalid.")
        dedupe_config = orchestration_config.deduplication

        with DuckDBStore() as store:
            catalog = ConfiguredDatasetPreparer(store).prepare_all(plan)
            left = catalog.require(source_id)
            right = catalog.require(target_id)
            if dict(left.variable_columns) != dict(right.variable_columns):
                raise PipelineError("ML-MODE-019", "Canonical variable contracts differ.")
            generator = DuckDBCandidateGenerator(store)
            rules = runtime_blocking_rules(plan)
            aggregate_budget = plan.config.runtime.maximum_candidate_pairs
            candidate_results = {
                "cross": generator.generate(
                    left=left.table,
                    right=right.table,
                    variable_columns=left.variable_columns,
                    rules=rules,
                    maximum_candidate_pairs=aggregate_budget,
                )
            }
            remaining_budget = aggregate_budget - candidate_results["cross"].candidate_pair_count
            for surface, table in (("intra_a", left), ("intra_b", right)):
                if remaining_budget < 1:
                    raise PipelineError(
                        "ML-MODE-027",
                        "The combined linkage-mode candidate budget was exceeded.",
                    )
                surface_budget = min(
                    dedupe_config.maximum_candidate_edges,
                    remaining_budget,
                )
                try:
                    candidate_results[surface] = generator.generate_deduplication(
                        dataset=table.table,
                        variable_columns=table.variable_columns,
                        rules=rules,
                        maximum_candidate_pairs=surface_budget,
                    )
                except CandidateBudgetExceeded:
                    if remaining_budget <= dedupe_config.maximum_candidate_edges:
                        raise PipelineError(
                            "ML-MODE-027",
                            "The combined linkage-mode candidate budget was exceeded.",
                        ) from None
                    raise
                remaining_budget -= candidate_results[surface].candidate_pair_count
            aggregate_candidate_pair_count = sum(
                result.candidate_pair_count for result in candidate_results.values()
            )
            if aggregate_candidate_pair_count > aggregate_budget:
                raise PipelineError(
                    "ML-MODE-027",
                    "The combined linkage-mode candidate budget was exceeded.",
                )
            snapshots = {
                name: candidate_snapshot(store, result.table.table_name)
                for name, result in candidate_results.items()
            }
            if any(not snapshot.pairs for snapshot in snapshots.values()):
                raise PipelineError("ML-MODE-020", "Synthetic mode candidate evidence is empty.")
            candidate_digests = {
                "cross": _candidate_plan_digest(snapshots["cross"], shape="cross_table"),
                "intra_a": _candidate_plan_digest(
                    snapshots["intra_a"], shape="canonical_same_table_a"
                ),
                "intra_b": _candidate_plan_digest(
                    snapshots["intra_b"], shape="canonical_same_table_b"
                ),
            }
            feature_builder = DuckDBComparisonFeatureBuilder(store)
            feature_results = {
                "cross": feature_builder.build(
                    candidates=candidate_results["cross"].table,
                    left=left,
                    right=right,
                    comparisons=plan.config.comparisons,
                ),
                "intra_a": feature_builder.build(
                    candidates=candidate_results["intra_a"].table,
                    left=left,
                    right=left,
                    comparisons=plan.config.comparisons,
                ),
                "intra_b": feature_builder.build(
                    candidates=candidate_results["intra_b"].table,
                    left=right,
                    right=right,
                    comparisons=plan.config.comparisons,
                ),
            }
            matrix_builder = DuckDBVerifiedMatrixBuilder(store)
            scoring_matrices = {
                name: matrix_builder.build_scoring(features=result)
                for name, result in feature_results.items()
            }
            schema_digests = {matrix.feature_schema_digest for matrix in scoring_matrices.values()}
            feature_names = {matrix.feature_names for matrix in scoring_matrices.values()}
            if len(schema_digests) != 1 or len(feature_names) != 1:
                raise PipelineError(
                    "ML-MODE-025", "Cross and intra-source feature contracts are incompatible."
                )
            truth_records = synthetic_truth_records(
                bundle,
                source_dataset_id=source_id,
                target_dataset_id=target_id,
            )
            surface_batches: dict[str, dict[str, VerifiedLabelBatch]] = {}
            split_manifest_digests: set[str] = set()
            for surface in ("cross", "intra_a", "intra_b"):
                batches, _, split_digest = protected_label_batches(
                    candidate_pairs=snapshots[surface].pairs,
                    truth_records=truth_records,
                    plan=plan,
                )
                surface_batches[surface] = batches
                split_manifest_digests.add(split_digest)
            if len(split_manifest_digests) != 1:
                raise PipelineError("ML-MODE-024", "Protected surface splits are inconsistent.")
            combined_batches, combined_disjointness = _combined_label_batches(surface_batches)
            surface_matrices: dict[str, dict[str, BoostedLabelledMatrix]] = {
                surface: {
                    partition: matrix_builder.build_labelled(
                        features=feature_results[surface],
                        labels=surface_batches[surface][partition],
                        random_seed=plan.random_seed,
                    )
                    for partition in (
                        "training",
                        "validation",
                        "calibration",
                        "decision",
                        "test",
                    )
                }
                for surface in ("cross", "intra_a", "intra_b")
            }
            combined_matrices = {
                partition: _combined_matrix(
                    surface_matrices={
                        surface: surface_matrices[surface][partition]
                        for surface in ("cross", "intra_a", "intra_b")
                    },
                    combined_labels=combined_batches[partition],
                    random_seed=plan.random_seed,
                )
                for partition in ("training", "validation", "calibration")
            }
            evidence_audits = tuple(
                (
                    surface,
                    _protected_evidence_audit(
                        batches=surface_batches[surface],
                        decision_matrix=surface_matrices[surface]["decision"],
                    ),
                )
                for surface in ("cross", "intra_a", "intra_b")
            )
            selected_model = next(
                (
                    model
                    for model in plan.config.models.all_boosted_trees()
                    if model.enabled and model.model_id == orchestration_config.pair_model_id
                ),
                None,
            )
            if selected_model is None:
                raise PipelineError("ML-MODE-021", "The configured pair model is unavailable.")
            classifier = XGBoostPairClassifier(store)
            fitted = classifier.fit(
                matrix=combined_matrices["training"],
                model=selected_model,
                random_seed=plan.random_seed,
                configuration_digest=plan.configuration_digest,
            )
            validation = combined_matrices["validation"]
            validation_scores = classifier._predict(matrix=validation, model=fitted)
            validation_report = evaluate_binary_scores(
                labels=validation.labels,
                scores=validation_scores,
                diagnostic_threshold=0.5,
                evaluation_scope="synthetic_mechanical_evaluation",
                partition_manifest_digest=combined_disjointness.manifest_digest,
            )
            selection = _selection(
                artifact=fitted,
                validation_matrix=validation,
                report=validation_report,
                partition_manifest_digest=combined_disjointness.manifest_digest,
                primary_metric=plan.config.model_selection.primary_metric,
            )
            calibration = combined_matrices["calibration"]
            calibration_batch = PairScoreBatch(
                pair_references=calibration.pair_references,
                pair_digests=calibration.pair_digests,
                scores=classifier._predict(matrix=calibration, model=fitted),
                labels=calibration.labels,
                partition="calibration",
                source_model_family="xgboost",
                source_model_id=fitted.model_id,
                source_model_version=fitted.model_version,
                source_evidence_digest=fitted.model_digest,
                feature_schema_digest=fitted.feature_schema_digest,
                label_authority_digest=calibration.label_authority_digest,
                partition_manifest_digest=combined_disjointness.manifest_digest,
                champion_selection_digest=selection.selection_digest,
            )
            calibrator = ChampionCalibratorSelector.select(
                calibration_batch,
                selection,
                methods=(plan.config.calibration.method,),
            )
            locked_test_reports = {
                surface: evaluate_binary_scores(
                    labels=surface_matrices[surface]["test"].labels,
                    scores=classifier._predict(
                        matrix=surface_matrices[surface]["test"],
                        model=fitted,
                    ),
                    diagnostic_threshold=0.5,
                    evaluation_scope="synthetic_mechanical_evaluation",
                    partition_manifest_digest=combined_disjointness.manifest_digest,
                )
                for surface in ("cross", "intra_a", "intra_b")
            }

        persisted_model = _persist_reload_model(
            artifact=fitted,
            base=artifact_base,
            plan=plan,
        )
        persisted_calibrator = _persist_reload_calibrator(
            artifact=calibrator,
            base=artifact_base,
            plan=plan,
        )
        calibration_binding = CombinedSurfaceCalibrationBinding.from_protected_evidence(
            surface_batches=surface_batches,
            combined_batches=combined_batches,
            combined_matrices=combined_matrices,
            model=persisted_model,
            calibrator=persisted_calibrator,
            selection=selection,
            partition_manifest_digest=combined_disjointness.manifest_digest,
        )
        calibration_binding.assert_complete()
        decision_features = {
            surface: _feature_view(surface_matrices[surface]["decision"])
            for surface in ("cross", "intra_a", "intra_b")
        }
        source_a_keys = tuple(
            sorted(
                {
                    *(left for left, _ in decision_features["cross"].pair_references),
                    *(key for pair in decision_features["intra_a"].pair_references for key in pair),
                }
            )
        )
        source_b_keys = tuple(
            sorted(
                {
                    *(right for _, right in decision_features["cross"].pair_references),
                    *(key for pair in decision_features["intra_b"].pair_references for key in pair),
                }
            )
        )
        cross_batch, cross_evidence_digest = _assignment_batch(
            matrix=decision_features["cross"],
            record_keys=source_a_keys,
            model=persisted_model,
            calibrator=persisted_calibrator,
            surface="cross",
            calibration_binding=calibration_binding,
        )
        intra_a_batch, intra_a_evidence_digest = _assignment_batch(
            matrix=decision_features["intra_a"],
            record_keys=source_a_keys,
            model=persisted_model,
            calibrator=persisted_calibrator,
            surface="intra_a",
            calibration_binding=calibration_binding,
        )
        intra_b_batch, intra_b_evidence_digest = _assignment_batch(
            matrix=decision_features["intra_b"],
            record_keys=source_b_keys,
            model=persisted_model,
            calibrator=persisted_calibrator,
            surface="intra_b",
            calibration_binding=calibration_binding,
        )
        dedupe_plan = DeduplicationPlan(
            algorithm=dedupe_config.algorithm,
            threshold=dedupe_config.minimum_probability,
            no_match_utility=dedupe_config.no_match_utility,
            max_cluster_size=dedupe_config.maximum_cluster_size,
            maximum_candidate_edges=dedupe_config.maximum_candidate_edges,
            deterministic_tie_breaking=True,
        )
        cross_plan = AssignmentPlan(
            constraint="one_to_one",
            solver=plan.config.assignment.solver,
            no_match_utility=plan.config.assignment.no_match.utility,
            maximum_candidate_edges=plan.config.runtime.maximum_candidate_pairs,
            deterministic_tie_breaking=True,
        )
        workflow = DeduplicationWorkflowRunner.run_link_and_dedupe(
            source_a_keys=source_a_keys,
            source_b_keys=source_b_keys,
            cross_candidates=cross_batch,
            cross_plan=cross_plan,
            intra_a_candidates=intra_a_batch,
            intra_b_candidates=intra_b_batch,
            deduplication_plan=dedupe_plan,
            dataset_a_id=source_id,
            dataset_b_id=target_id,
        )
        combined_result = workflow.link_and_dedupe_result
        if combined_result is None or workflow.manifest_path is not None:
            raise PipelineError("ML-MODE-023", "The link-and-dedupe result is invalid.")
        combined_partition_digest = _canonical_digest(
            {
                "combined_partition_manifest_digest": combined_disjointness.manifest_digest,
                "split_manifest_digest": next(iter(split_manifest_digests)),
                "surface_audit_digests": {
                    name: audit.audit_digest for name, audit in evidence_audits
                },
                "calibration_binding_digest": calibration_binding.binding_digest,
            }
        )
        orchestration_artifact = SyntheticModeOrchestrationArtifact(
            linkage_mode="link_and_dedupe",
            assignment_constraint="one_to_one",
            configuration_digest=plan.configuration_digest,
            registry_digest=plan.registry_digest,
            synthetic_bundle_digest=bundle_digest,
            generator_version=bundle.provenance.generator_version,
            random_seed=20260816,
            candidate_plan_digests=tuple(
                candidate_digests[name] for name in ("cross", "intra_a", "intra_b")
            ),
            calibrated_evidence_digests=(
                cross_evidence_digest,
                intra_a_evidence_digest,
                intra_b_evidence_digest,
            ),
            feature_schema_digest=persisted_model.feature_schema_digest,
            champion_model_id=persisted_model.model_id,
            champion_model_version=persisted_model.model_version,
            champion_artifact_digest=persisted_model.model_digest,
            calibrator_digest=persisted_calibrator.calibrator_digest,
            partition_manifest_digest=combined_partition_digest,
            deduplication_plan_digest=_canonical_digest(asdict(dedupe_plan)),
            assignment_plan_digest=_canonical_digest(asdict(cross_plan)),
        )
        selected_edge_count = (
            sum(cluster.edge_count for cluster in combined_result.source_a_deduplication.clusters)
            + sum(cluster.edge_count for cluster in combined_result.source_b_deduplication.clusters)
            + combined_result.cross_assignment.real_assignment_count
        )
        run_artifact = SyntheticModeRunArtifact(
            linkage_mode="link_and_dedupe",
            orchestration_artifact_digest=orchestration_artifact.artifact_digest,
            configuration_digest=plan.configuration_digest,
            result_digest=workflow.workflow_digest,
            input_record_count=len(source_a_keys) + len(source_b_keys),
            candidate_pair_count=sum(
                batch.candidate_pair_count for batch in (cross_batch, intra_a_batch, intra_b_batch)
            ),
            cluster_count=(
                combined_result.source_a_cluster_count + combined_result.source_b_cluster_count
            ),
            selected_edge_count=selected_edge_count,
        )
        persisted_orchestration, persisted_run = _persist_reload_mode_artifacts(
            orchestration=orchestration_artifact,
            run=run_artifact,
            base=artifact_base,
            plan=plan,
        )
        qualification_digest = _canonical_digest(
            {
                "validation": validation_report.safe_summary(),
                "locked_tests": {
                    name: report.safe_summary() for name, report in locked_test_reports.items()
                },
                "calibration_binding_digest": calibration_binding.binding_digest,
                "orchestration_artifact_digest": persisted_orchestration.artifact_digest,
                "run_digest": persisted_run.run_digest,
            }
        )
        return SyntheticLinkAndDedupeWorkflowResult(
            run_id=run_id,
            candidate_pair_count=sum(
                result.candidate_pair_count for result in candidate_results.values()
            ),
            decision_pair_count=sum(matrix.pair_count for matrix in decision_features.values()),
            input_record_count=len(source_a_keys) + len(source_b_keys),
            model_artifact=persisted_model,
            calibrator_artifact=persisted_calibrator,
            orchestration_artifact=persisted_orchestration,
            run_artifact=persisted_run,
            workflow=workflow,
            evidence_audits=evidence_audits,
            calibration_binding=calibration_binding,
            qualification_digest=qualification_digest,
        )


__all__ = [
    "CombinedSurfaceCalibrationBinding",
    "ProtectedModeEvidenceAudit",
    "SyntheticDedupeModeWorkflowResult",
    "SyntheticLinkAndDedupeWorkflowResult",
    "SyntheticModeWorkflowResult",
    "SyntheticModeWorkflowRunner",
]
