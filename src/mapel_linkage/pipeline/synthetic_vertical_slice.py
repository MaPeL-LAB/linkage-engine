"""Complete two-source synthetic vertical slice with strict authority separation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
from numpy.typing import NDArray

from mapel_linkage import __version__
from mapel_linkage.adjudication import build_review_queue, write_review_queue
from mapel_linkage.anchors import DuckDBAnchorEvidenceEvaluator
from mapel_linkage.assignment import (
    AssignmentEdgeBatch,
    AssignmentPlan,
    OrToolsOneToOneAssignmentSolver,
    ScipyOneToOneAssignmentSolver,
    pair_digest,
)
from mapel_linkage.calibration import (
    BetaCalibrator,
    ChampionChallengerSelector,
    IsotonicCalibrator,
    ModelEvaluationCandidate,
    PairScoreBatch,
    SigmoidCalibrator,
    calibration_diagnostics,
    write_calibrator_artifact,
)
from mapel_linkage.calibration.contracts import CalibratedScoreBatch, CalibratorArtifact
from mapel_linkage.candidate_generation import (
    AllOf,
    AnyOf,
    BlockingRule,
    CandidatePredicate,
    DateWindow,
    DuckDBCandidateGenerator,
    Exact,
    PrefixEqual,
)
from mapel_linkage.comparisons import DuckDBComparisonFeatureBuilder
from mapel_linkage.configuration import ExecutionPlan
from mapel_linkage.configuration.models import (
    AllPredicate,
    AnyPredicate,
    BlockPredicate,
    DateWindowPredicate,
    ExactPredicate,
    PrefixEqualPredicate,
)
from mapel_linkage.decisions import DecisionEvidenceBuilder, RelationshipDecisionPolicy
from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.domain.sql_identifiers import quote_identifier
from mapel_linkage.governance.labels import (
    PartitionDisjointnessReport,
    VerifiedLabelBatch,
    assert_disjoint_label_partitions,
)
from mapel_linkage.io import DuckDBStore
from mapel_linkage.models.boosted import (
    BoostedFeatureMatrix,
    DuckDBVerifiedMatrixBuilder,
    XGBoostPairClassifier,
    write_xgboost_artifact,
)
from mapel_linkage.models.fellegi_sunter import (
    SplinkCandidateParityChecker,
    SplinkNativeDuckDBMatcher,
    SplinkSettingsPlanCompiler,
    deserialize_splink_native_model,
    serialize_splink_native_model,
)
from mapel_linkage.models.ranking import (
    XGBoostCandidateRanker,
    build_ranking_matrix,
    build_ranking_scoring_matrix,
    write_ranking_artifact,
)
from mapel_linkage.pipeline.contracts import StageSummary, SyntheticVerticalSliceResult
from mapel_linkage.pipeline.io import write_relationship_decisions, write_run_manifest
from mapel_linkage.preprocessing import ConfiguredDatasetPreparer, surrogate_record_key
from mapel_linkage.synthetic import (
    SyntheticBundle,
    SyntheticGenerationConfig,
    generate_synthetic_bundle,
    write_synthetic_bundle,
)
from mapel_linkage.validation import (
    EntityHouseholdRecord,
    PairValidationReport,
    evaluate_assignment,
    evaluate_binary_scores,
    evaluate_candidate_retrieval,
    evaluate_configured_decision_thresholds,
    evaluate_decisions,
    evaluate_ranking,
    evaluate_stratified_pair_performance,
    split_entity_household_components,
    write_aggregate_validation_report,
)
from mapel_linkage.validation.splitting import build_verified_candidate_label_batches


@dataclass(frozen=True, slots=True, repr=False)
class _CandidateSnapshot:
    pairs: tuple[tuple[str, str], ...]
    pair_digests: tuple[str, ...]
    rule_ids_by_pair: dict[tuple[str, str], tuple[str, ...]]
    rule_ids_by_digest: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _SyntheticProvenanceReport:
    entity_count: int
    source_record_count: int
    target_record_count: int
    truth_record_count: int
    duplicate_count: int
    competing_candidate_count: int
    generator_version: str
    seed: int
    evaluation_scope: str = "synthetic_mechanical_evaluation"
    real_data_validation_status: str = "not_established"

    def safe_summary(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _SelectionReport:
    selected_model_family: str
    selected_model_id: str
    selected_model_version: str
    primary_metric: str
    secondary_metric: str
    selection_digest: str
    test_partition_used: bool
    calibration_partition_used: bool

    def safe_summary(self) -> dict[str, str | bool]:
        return asdict(self)


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in (
        "duckdb",
        "networkx",
        "numpy",
        "ortools",
        "pydantic",
        "PyYAML",
        "scikit-learn",
        "scipy",
        "splink",
        "xgboost",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return versions


def _runtime_predicate(predicate: BlockPredicate) -> CandidatePredicate:
    if isinstance(predicate, ExactPredicate):
        return Exact(predicate.variable)
    if isinstance(predicate, PrefixEqualPredicate):
        return PrefixEqual(predicate.variable, predicate.length)
    if isinstance(predicate, AllPredicate):
        return AllOf(tuple(_runtime_predicate(term) for term in predicate.terms))
    if isinstance(predicate, AnyPredicate):
        return AnyOf(tuple(_runtime_predicate(term) for term in predicate.terms))
    if isinstance(predicate, DateWindowPredicate):
        return DateWindow(predicate.variable, predicate.maximum_days)
    raise PipelineError("ML-PIPE-004", "A configured blocking predicate is unsupported.")


def _blocking_rules(plan: ExecutionPlan) -> tuple[BlockingRule, ...]:
    return tuple(
        BlockingRule(rule.id, _runtime_predicate(rule.predicate))
        for rule in plan.config.blocking.rules
    )


def _source_target_ids(plan: ExecutionPlan) -> tuple[str, str]:
    sources = [dataset.id for dataset in plan.config.datasets if dataset.role == "source"]
    targets = [dataset.id for dataset in plan.config.datasets if dataset.role == "target"]
    if plan.config.project.linkage_mode != "link_only" or len(sources) != 1 or len(targets) != 1:
        raise PipelineError(
            "ML-PIPE-005",
            "The complete synthetic slice currently requires one source and one target dataset.",
        )
    if plan.config.project.assignment_constraint != "one_to_one":
        raise PipelineError(
            "ML-PIPE-006", "The complete synthetic slice currently requires one-to-one assignment."
        )
    return sources[0], targets[0]


def _synthetic_fixture_directory(
    plan: ExecutionPlan,
    *,
    source_id: str,
    target_id: str,
) -> Path:
    """Fail closed before a synthetic run can touch configured dataset inputs."""

    if plan.config.labels is None or plan.config.labels.source.kind != "synthetic_truth":
        raise PipelineError(
            "ML-PIPE-018",
            "Synthetic-demo execution requires synthetic-truth label authority.",
        )
    if plan.label_source_path is not None:
        raise PipelineError(
            "ML-PIPE-019",
            "Synthetic-demo execution cannot use a configured label-source path.",
        )
    fixture_directory = plan.path_policy.resolve_input("data/synthetic")
    expected_paths = {
        source_id: (fixture_directory / "source_a.jsonl").resolve(strict=False),
        target_id: (fixture_directory / "source_b.jsonl").resolve(strict=False),
    }
    if dict(plan.dataset_paths) != expected_paths:
        raise PipelineError(
            "ML-PIPE-020",
            "Synthetic-demo dataset paths must be the generated two-source fixtures.",
        )
    dataset_by_id = {dataset.id: dataset for dataset in plan.config.datasets}
    if any(
        dataset_by_id[dataset_id].format != "jsonl"
        or dataset_by_id[dataset_id].record_id_column != "record_key"
        for dataset_id in (source_id, target_id)
    ):
        raise PipelineError(
            "ML-PIPE-021",
            "Synthetic-demo datasets must use the generated JSONL record-key contract.",
        )
    return fixture_directory


def _record_keys(store: DuckDBStore, table_name: str) -> tuple[str, ...]:
    rows = store._fetch_model_rows(
        f"SELECT {quote_identifier('__ml_record_key')} FROM {quote_identifier(table_name)} "
        f"ORDER BY {quote_identifier('__ml_record_key')}"
    )
    return tuple(str(row[0]) for row in rows)


def _candidate_snapshot(store: DuckDBStore, table_name: str) -> _CandidateSnapshot:
    rows = store._fetch_model_rows(
        "SELECT left_record_key, right_record_key, retrieval_rule_ids "
        f"FROM {quote_identifier(table_name)} ORDER BY left_record_key, right_record_key"
    )
    pairs: list[tuple[str, str]] = []
    digests: list[str] = []
    by_pair: dict[tuple[str, str], tuple[str, ...]] = {}
    by_digest: dict[str, tuple[str, ...]] = {}
    for row in rows:
        left, right = str(row[0]), str(row[1])
        pair = (left, right)
        digest = pair_digest(left, right)
        rules = tuple(sorted(part for part in str(row[2]).split(",") if part))
        pairs.append(pair)
        digests.append(digest)
        by_pair[pair] = rules
        by_digest[digest] = rules
    return _CandidateSnapshot(tuple(pairs), tuple(digests), by_pair, by_digest)


def _truth_records(
    bundle: SyntheticBundle,
    *,
    source_dataset_id: str,
    target_dataset_id: str,
) -> tuple[EntityHouseholdRecord, ...]:
    dataset_map = {"source_a": source_dataset_id, "source_b": target_dataset_id}
    output: list[EntityHouseholdRecord] = []
    for record in bundle.truth:
        dataset_id = dataset_map[record.dataset_id]
        output.append(
            EntityHouseholdRecord(
                dataset_id=dataset_id,
                record_key=surrogate_record_key(dataset_id, record.record_key),
                entity_key=record.entity_key,
                household_key=record.household_key,
            )
        )
    return tuple(sorted(output, key=lambda item: (item.dataset_id, item.record_key)))


def _truth_relations(
    records: tuple[EntityHouseholdRecord, ...],
    *,
    source_dataset_id: str,
    target_dataset_id: str,
) -> tuple[frozenset[tuple[str, str]], dict[str, str | None]]:
    sources_by_entity: dict[str, list[str]] = defaultdict(list)
    targets_by_entity: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.dataset_id == source_dataset_id:
            sources_by_entity[record.entity_key].append(record.record_key)
        elif record.dataset_id == target_dataset_id:
            targets_by_entity[record.entity_key].append(record.record_key)
    true_pairs: set[tuple[str, str]] = set()
    target_by_source: dict[str, str | None] = {}
    for entity, sources in sources_by_entity.items():
        targets = sorted(targets_by_entity.get(entity, []))
        for source in sorted(sources):
            if len(targets) == 1:
                target_by_source[source] = targets[0]
                true_pairs.add((source, targets[0]))
            else:
                target_by_source[source] = None
    return frozenset(true_pairs), target_by_source


def _label_batches(
    *,
    candidate_pairs: tuple[tuple[str, str], ...],
    truth_records: tuple[EntityHouseholdRecord, ...],
    plan: ExecutionPlan,
) -> tuple[dict[str, VerifiedLabelBatch], PartitionDisjointnessReport, str]:
    split = plan.config.validation.split
    assignment = split_entity_household_components(
        truth_records,
        fractions=(
            split.training_fraction,
            split.validation_fraction,
            split.calibration_fraction,
            split.decision_fraction,
            split.test_fraction,
        ),
        random_seed=plan.random_seed,
    )
    truth_digest = _canonical_digest(
        [
            {
                "dataset_id": record.dataset_id,
                "record_digest": hashlib.sha256(record.record_key.encode("utf-8")).hexdigest(),
                "entity_digest": record.entity_digest,
                "household_digest": record.household_digest,
            }
            for record in truth_records
        ]
    )
    batches = build_verified_candidate_label_batches(
        candidate_pairs=candidate_pairs,
        truth_records=truth_records,
        assignment=assignment,
        verification_protocol="synthetic_v1",
        source_digest=truth_digest,
    )
    by_partition: dict[str, VerifiedLabelBatch] = {batch.partition: batch for batch in batches}
    required = {"training", "validation", "calibration", "decision", "test"}
    if set(by_partition) != required:
        raise PipelineError(
            "ML-PIPE-007",
            "The synthetic benchmark did not yield every protected label partition.",
        )
    if any(batch.positive_count <= 0 or batch.negative_count <= 0 for batch in batches):
        raise PipelineError(
            "ML-PIPE-008",
            "Every protected synthetic label partition requires matches and nonmatches.",
        )
    disjointness = assert_disjoint_label_partitions(batches)
    return by_partition, disjointness, assignment.manifest_digest


def _score_lookup(
    store: DuckDBStore,
    *,
    table_name: str,
    score_column: str,
) -> dict[tuple[str, str], float]:
    rows = store._fetch_model_rows(
        "SELECT left_record_key, right_record_key, "
        f"{quote_identifier(score_column)} FROM {quote_identifier(table_name)} "
        "ORDER BY left_record_key, right_record_key"
    )
    output: dict[tuple[str, str], float] = {}
    for row in rows:
        score = row[2]
        if not isinstance(score, (int, float)):
            raise PipelineError("ML-PIPE-017", "A model score is not numeric.")
        output[(str(row[0]), str(row[1]))] = float(score)
    return output


def _pair_score_batch(
    *,
    labels: VerifiedLabelBatch,
    score_lookup: dict[tuple[str, str], float],
    source_model_family: str,
    source_model_id: str,
    source_model_version: str,
    source_evidence_digest: str,
    feature_schema_digest: str,
    partition_manifest_digest: str,
    champion_selection_digest: str | None = None,
) -> PairScoreBatch:
    ordered = sorted(labels.labels, key=lambda item: item.pair_digest())
    try:
        scores = np.asarray(
            [score_lookup[(item.left_record_key, item.right_record_key)] for item in ordered],
            dtype=np.float64,
        )
    except KeyError:
        raise PipelineError(
            "ML-PIPE-009", "A protected label is unavailable in model-score evidence."
        ) from None
    return PairScoreBatch(
        pair_references=tuple((item.left_record_key, item.right_record_key) for item in ordered),
        pair_digests=tuple(item.pair_digest() for item in ordered),
        scores=scores,
        labels=np.asarray([item.label for item in ordered], dtype=np.int8),
        partition=labels.partition,
        source_model_family=source_model_family,
        source_model_id=source_model_id,
        source_model_version=source_model_version,
        source_evidence_digest=source_evidence_digest,
        feature_schema_digest=feature_schema_digest,
        label_authority_digest=labels.label_authority_digest,
        partition_manifest_digest=partition_manifest_digest,
        champion_selection_digest=champion_selection_digest,
    )


def _stratification_vectors(
    *,
    scoring_matrix: BoostedFeatureMatrix,
    batch: PairScoreBatch,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    index_by_pair = {pair: index for index, pair in enumerate(scoring_matrix.pair_references)}
    missing_indices = tuple(
        index
        for index, name in enumerate(scoring_matrix.feature_names)
        if name.endswith("_missing_any")
    )
    candidate_counts = Counter(left for left, _ in scoring_matrix.pair_references)
    patterns: list[str] = []
    sizes: list[int] = []
    try:
        for pair in batch.pair_references:
            row_index = index_by_pair[pair]
            if missing_indices:
                bits = "".join(
                    "1" if scoring_matrix.features[row_index, column] >= 0.5 else "0"
                    for column in missing_indices
                )
                patterns.append(f"pattern_{bits}")
            else:
                patterns.append("pattern_none")
            sizes.append(candidate_counts[pair[0]])
    except (KeyError, IndexError):
        raise PipelineError(
            "ML-PIPE-016",
            "Stratified evaluation could not align protected pair evidence.",
        ) from None
    return tuple(patterns), tuple(sizes)


def _candidate_model(
    *,
    family: str,
    model_id: str,
    model_version: str,
    evidence_digest: str,
    feature_schema_digest: str,
    validation_batch: PairScoreBatch,
    training_label_authority_digest: str | None,
) -> tuple[ModelEvaluationCandidate, PairValidationReport]:
    report = evaluate_binary_scores(
        labels=validation_batch.labels,
        scores=validation_batch.scores,
        diagnostic_threshold=0.5,
        evaluation_scope="synthetic_mechanical_evaluation",
        partition_manifest_digest=validation_batch.partition_manifest_digest,
    )
    candidate = ModelEvaluationCandidate(
        model_family=family,
        model_id=model_id,
        model_version=model_version,
        evidence_digest=evidence_digest,
        feature_schema_digest=feature_schema_digest,
        validation_label_authority_digest=validation_batch.label_authority_digest,
        partition_manifest_digest=validation_batch.partition_manifest_digest,
        average_precision=report.average_precision,
        brier_score=report.brier_score,
        pair_count=report.pair_count,
        training_label_authority_digest=training_label_authority_digest,
    )
    return candidate, report


def _calibrated_all_candidates(
    *,
    candidate_pairs: tuple[tuple[str, str], ...],
    score_lookup: dict[tuple[str, str], float],
    artifact: CalibratorArtifact,
) -> CalibratedScoreBatch:
    try:
        raw = np.asarray([score_lookup[pair] for pair in candidate_pairs], dtype=np.float64)
    except KeyError:
        raise PipelineError("ML-PIPE-010", "Candidate score coverage is incomplete.") from None
    if artifact.method == "sigmoid":
        probabilities = SigmoidCalibrator.apply(raw, artifact)
    elif artifact.method == "beta":
        probabilities = BetaCalibrator.apply(raw, artifact)
    else:
        probabilities = IsotonicCalibrator.apply(raw, artifact)
    return CalibratedScoreBatch(
        pair_references=candidate_pairs,
        pair_digests=tuple(pair_digest(*pair) for pair in candidate_pairs),
        probabilities=probabilities,
        source_model_family=artifact.source_model_family,
        source_model_id=artifact.source_model_id,
        source_model_version=artifact.source_model_version,
        source_evidence_digest=artifact.source_evidence_digest,
        feature_schema_digest=artifact.feature_schema_digest,
        calibrator_method=artifact.method,
        calibrator_version=artifact.calibrator_version,
        calibrator_digest=artifact.calibrator_digest,
        champion_selection_digest=artifact.champion_selection_digest,
    )


def _top_target_by_source(
    pairs: tuple[tuple[str, str], ...],
    digests: tuple[str, ...],
    scores: NDArray[np.float64],
) -> dict[str, str]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, (source, _) in enumerate(pairs):
        grouped[source].append(index)
    return {
        source: pairs[min(indices, key=lambda index: (-float(scores[index]), digests[index]))][1]
        for source, indices in grouped.items()
    }


def _anchor_diagnostics(
    store: DuckDBStore,
    table_name: str,
) -> tuple[frozenset[str], dict[str, tuple[str, ...]]]:
    rows = store._fetch_model_rows(
        "SELECT left_record_key, right_record_key, anchor_rule_id, uniqueness_pass "
        f"FROM {quote_identifier(table_name)} "
        "ORDER BY left_record_key, anchor_rule_id, right_record_key"
    )
    targets_by_source: dict[str, set[str]] = defaultdict(set)
    rules_by_source: dict[str, set[str]] = defaultdict(set)
    failed: set[str] = set()
    for row in rows:
        source, target, rule = str(row[0]), str(row[1]), str(row[2])
        targets_by_source[source].add(target)
        rules_by_source[source].add(rule)
        if not bool(row[3]):
            failed.add(source)
    conflicts = failed | {
        source for source, targets in targets_by_source.items() if len(targets) > 1
    }
    return frozenset(conflicts), {
        source: tuple(sorted(rules)) for source, rules in rules_by_source.items()
    }


def _relative_output(base: str, run_id: str, name: str) -> str:
    return str(PurePosixPath(base) / run_id / name)


class SyntheticVerticalSliceRunner:
    """Run the complete synthetic MVP without granting models merge authority."""

    @staticmethod
    def run(
        plan: ExecutionPlan,
        *,
        generation: SyntheticGenerationConfig | None = None,
        prefer_ortools: bool = True,
    ) -> SyntheticVerticalSliceResult:
        source_id, target_id = _source_target_ids(plan)
        fixture_directory = _synthetic_fixture_directory(
            plan,
            source_id=source_id,
            target_id=target_id,
        )
        spec = generation or SyntheticGenerationConfig(
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
        if spec.seed != plan.random_seed:
            raise PipelineError(
                "ML-PIPE-022",
                "Synthetic generation seed must match the compiled deterministic plan.",
            )
        bundle = generate_synthetic_bundle(spec)
        dependency_versions = _dependency_versions()
        run_id = _canonical_digest(
            {
                "configuration_digest": plan.configuration_digest,
                "generator": asdict(bundle.provenance),
                "engine_version": __version__,
                "python_version": platform.python_version(),
                "dependency_versions": dependency_versions,
            }
        )[:24]
        write_synthetic_bundle(fixture_directory, bundle)
        stages: list[StageSummary] = [
            StageSummary(
                "synthetic_generation",
                "completed",
                counts={
                    "source_record_count": bundle.provenance.source_a_count,
                    "target_record_count": bundle.provenance.source_b_count,
                    "truth_record_count": bundle.provenance.truth_record_count,
                },
            )
        ]

        with DuckDBStore() as store:
            catalog = ConfiguredDatasetPreparer(store).prepare_all(plan)
            left = catalog.require(source_id)
            right = catalog.require(target_id)
            source_keys = _record_keys(store, left.table.table_name)
            target_keys = _record_keys(store, right.table.table_name)
            stages.append(
                StageSummary(
                    "canonical_preprocessing",
                    "completed",
                    counts={
                        "source_record_count": len(source_keys),
                        "target_record_count": len(target_keys),
                    },
                    digests={
                        "source_schema_digest": left.table.schema_digest,
                        "target_schema_digest": right.table.schema_digest,
                    },
                )
            )

            if dict(left.variable_columns) != dict(right.variable_columns):
                raise PipelineError(
                    "ML-PIPE-011", "Source and target canonical variable contracts differ."
                )
            candidate_generator = DuckDBCandidateGenerator(store)
            candidates = candidate_generator.generate(
                left=left.table,
                right=right.table,
                variable_columns=left.variable_columns,
                rules=_blocking_rules(plan),
                maximum_candidate_pairs=plan.config.runtime.maximum_candidate_pairs,
            )
            candidate_diagnostics = candidate_generator.diagnostics(candidates)
            snapshot = _candidate_snapshot(store, candidates.table.table_name)
            splink_plan = SplinkSettingsPlanCompiler().compile(
                left=left,
                right=right,
                comparisons=plan.config.comparisons,
                blocking_rules=tuple(rule.predicate for rule in plan.config.blocking.rules),
                model=plan.config.models.fellegi_sunter,
            )
            splink_parity = SplinkCandidateParityChecker.check(
                store=store,
                left=left,
                right=right,
                settings_plan=splink_plan,
                expected_pairs=snapshot.pairs,
            )
            stages.append(
                StageSummary(
                    "candidate_generation",
                    "completed",
                    counts={
                        "candidate_pair_count": candidates.candidate_pair_count,
                        "multi_rule_pair_count": candidate_diagnostics.multi_rule_pair_count,
                        "splink_candidate_pair_count": splink_parity.observed_pair_count,
                    },
                    digests={
                        "candidate_schema_digest": candidates.table.schema_digest,
                        "splink_settings_digest": splink_parity.settings_digest,
                        "candidate_pair_set_digest": splink_parity.pair_set_digest,
                    },
                )
            )

            feature_builder = DuckDBComparisonFeatureBuilder(store)
            features = feature_builder.build(
                candidates=candidates.table,
                left=left,
                right=right,
                comparisons=plan.config.comparisons,
            )
            anchors = DuckDBAnchorEvidenceEvaluator(store).evaluate(
                left=left,
                right=right,
                anchors=plan.config.deterministic_anchors,
                maximum_anchor_pairs=plan.config.runtime.maximum_candidate_pairs,
            )
            anchor_conflicts, anchor_rules = _anchor_diagnostics(store, anchors.table.table_name)
            stages.append(
                StageSummary(
                    "comparison_and_anchor_evidence",
                    "completed",
                    counts={
                        "comparison_pair_count": features.candidate_pair_count,
                        "anchor_evidence_count": anchors.evidence_row_count,
                        "anchor_conflict_source_count": len(anchor_conflicts),
                    },
                    digests={"feature_schema_digest": features.table.schema_digest},
                )
            )

            truth_records = _truth_records(
                bundle,
                source_dataset_id=source_id,
                target_dataset_id=target_id,
            )
            true_pairs, true_target_by_source = _truth_relations(
                truth_records,
                source_dataset_id=source_id,
                target_dataset_id=target_id,
            )
            batches, disjointness, split_manifest_digest = _label_batches(
                candidate_pairs=snapshot.pairs,
                truth_records=truth_records,
                plan=plan,
            )
            stages.append(
                StageSummary(
                    "protected_label_partitions",
                    "completed",
                    counts={
                        "partition_count": len(batches),
                        "entity_component_count": disjointness.entity_component_count,
                        "household_component_count": disjointness.household_component_count,
                    },
                    digests={
                        "label_partition_manifest_digest": disjointness.manifest_digest,
                        "truth_split_manifest_digest": split_manifest_digest,
                    },
                )
            )

            fs_matcher = SplinkNativeDuckDBMatcher(store)
            trained_fs_model = fs_matcher.fit(
                left=left,
                right=right,
                settings_plan=splink_plan,
                model=plan.config.models.fellegi_sunter,
                configuration_digest=plan.configuration_digest,
                expected_pairs=snapshot.pairs,
                maximum_candidate_pairs=plan.config.runtime.maximum_candidate_pairs,
                random_seed=plan.random_seed,
            )
            fs_model = deserialize_splink_native_model(
                serialize_splink_native_model(trained_fs_model),
                settings_plan=splink_plan,
                model=plan.config.models.fellegi_sunter,
                configuration_digest=plan.configuration_digest,
                feature_schema_digest=trained_fs_model.feature_schema_digest,
                random_seed=plan.random_seed,
            )
            fs_scores = fs_matcher.score(
                left=left,
                right=right,
                settings_plan=splink_plan,
                artifact=fs_model,
                expected_pairs=snapshot.pairs,
                maximum_candidate_pairs=plan.config.runtime.maximum_candidate_pairs,
            )
            fs_lookup = _score_lookup(
                store,
                table_name=fs_scores.table.table_name,
                score_column="__ml_fs_model_probability",
            )

            boosted_config = plan.config.models.boosted_tree
            if boosted_config is None or not boosted_config.enabled:
                raise PipelineError(
                    "ML-PIPE-012", "The complete synthetic slice requires the XGBoost challenger."
                )
            matrix_builder = DuckDBVerifiedMatrixBuilder(store)
            training_matrix = matrix_builder.build_labelled(
                features=features,
                labels=batches["training"],
                model=boosted_config,
                random_seed=plan.random_seed,
                apply_training_selection=True,
            )
            scoring_matrix = matrix_builder.build_scoring(features=features)
            classifier = XGBoostPairClassifier(store)
            xgb_model = classifier.fit(
                matrix=training_matrix,
                model=boosted_config,
                random_seed=plan.random_seed,
                configuration_digest=plan.configuration_digest,
            )
            xgb_scores = classifier.score(matrix=scoring_matrix, model=xgb_model)
            xgb_lookup = _score_lookup(
                store,
                table_name=xgb_scores.table.table_name,
                score_column="__ml_bt_model_score",
            )
            stages.append(
                StageSummary(
                    "pair_model_training_and_scoring",
                    "completed",
                    counts={
                        "fs_native_training_candidate_pair_count": fs_model.candidate_pair_count,
                        "fs_native_scored_pair_count": fs_scores.pair_count,
                        "xgb_training_pair_count": xgb_model.training_pair_count,
                        "xgb_scored_pair_count": xgb_scores.pair_count,
                    },
                    digests={
                        "fs_native_artifact_digest": fs_model.artifact_digest,
                        "fs_native_model_digest": fs_model.model_digest,
                        "fs_native_parameter_digest": fs_model.parameter_digest,
                        "fs_native_score_digest": fs_scores.score_digest,
                        "fs_native_training_candidate_pair_set_digest": (
                            fs_model.training_candidate_pair_set_digest
                        ),
                        "fs_native_scoring_candidate_pair_set_digest": (
                            fs_scores.scoring_candidate_pair_set_digest
                        ),
                        "fs_native_decision_authority": fs_model.decision_authority,
                        "fs_native_relationship_authority": fs_model.relationship_authority,
                        "fs_native_assignment_authority": fs_model.assignment_authority,
                        "fs_native_merge_authority": fs_model.merge_authority,
                        "fs_native_operational_validation": fs_model.operational_validation,
                        "xgb_model_digest": xgb_model.model_digest,
                    },
                )
            )

            validation_fs = _pair_score_batch(
                labels=batches["validation"],
                score_lookup=fs_lookup,
                source_model_family="fellegi_sunter",
                source_model_id=fs_model.model_id,
                source_model_version=fs_model.model_version,
                source_evidence_digest=fs_model.artifact_digest,
                feature_schema_digest=fs_model.feature_schema_digest,
                partition_manifest_digest=disjointness.manifest_digest,
            )
            validation_xgb = _pair_score_batch(
                labels=batches["validation"],
                score_lookup=xgb_lookup,
                source_model_family="xgboost",
                source_model_id=xgb_model.model_id,
                source_model_version=xgb_model.model_version,
                source_evidence_digest=xgb_model.model_digest,
                feature_schema_digest=xgb_model.feature_schema_digest,
                partition_manifest_digest=disjointness.manifest_digest,
            )
            fs_candidate, fs_validation_report = _candidate_model(
                family="fellegi_sunter",
                model_id=fs_model.model_id,
                model_version=fs_model.model_version,
                evidence_digest=fs_model.artifact_digest,
                feature_schema_digest=fs_model.feature_schema_digest,
                validation_batch=validation_fs,
                training_label_authority_digest=None,
            )
            xgb_candidate, xgb_validation_report = _candidate_model(
                family="xgboost",
                model_id=xgb_model.model_id,
                model_version=xgb_model.model_version,
                evidence_digest=xgb_model.model_digest,
                feature_schema_digest=xgb_model.feature_schema_digest,
                validation_batch=validation_xgb,
                training_label_authority_digest=xgb_model.label_authority_digest,
            )
            selection = ChampionChallengerSelector.select(
                (fs_candidate, xgb_candidate),
                plan.config.model_selection,
            )
            configured_source_model = plan.config.calibration.source_model
            if configured_source_model not in {
                "selected_champion",
                selection.selected_model_id,
            }:
                raise PipelineError(
                    "ML-PIPE-015",
                    "The calibration source model does not match the selected champion.",
                )
            selected_lookup = (
                fs_lookup if selection.selected_model_family == "fellegi_sunter" else xgb_lookup
            )
            calibration_batch = _pair_score_batch(
                labels=batches["calibration"],
                score_lookup=selected_lookup,
                source_model_family=selection.selected_model_family,
                source_model_id=selection.selected_model_id,
                source_model_version=selection.selected_model_version,
                source_evidence_digest=selection.selected_evidence_digest,
                feature_schema_digest=selection.selected_feature_schema_digest,
                partition_manifest_digest=selection.partition_manifest_digest,
                champion_selection_digest=selection.selection_digest,
            )
            if plan.config.calibration.method == "sigmoid":
                calibrator = SigmoidCalibrator.fit(calibration_batch, selection)
            elif plan.config.calibration.method == "beta":
                calibrator = BetaCalibrator.fit(calibration_batch, selection)
            else:
                calibrator = IsotonicCalibrator.fit(calibration_batch, selection)
            decision_batch = _pair_score_batch(
                labels=batches["decision"],
                score_lookup=selected_lookup,
                source_model_family=selection.selected_model_family,
                source_model_id=selection.selected_model_id,
                source_model_version=selection.selected_model_version,
                source_evidence_digest=selection.selected_evidence_digest,
                feature_schema_digest=selection.selected_feature_schema_digest,
                partition_manifest_digest=selection.partition_manifest_digest,
                champion_selection_digest=selection.selection_digest,
            )
            if calibrator.method == "sigmoid":
                decision_probabilities = SigmoidCalibrator.apply(decision_batch.scores, calibrator)
            elif calibrator.method == "beta":
                decision_probabilities = BetaCalibrator.apply(decision_batch.scores, calibrator)
            else:
                decision_probabilities = IsotonicCalibrator.apply(decision_batch.scores, calibrator)
            decision_threshold_report = evaluate_configured_decision_thresholds(
                probabilities=decision_probabilities,
                labels=decision_batch.labels,
                confirmed_threshold=plan.config.decision_policy.confirmed.minimum_probability,
                review_threshold=plan.config.decision_policy.review_required.minimum_probability,
                no_match_threshold=plan.config.decision_policy.no_match.maximum_top_probability,
                partition_manifest_digest=decision_batch.partition_manifest_digest,
            )
            calibrated = _calibrated_all_candidates(
                candidate_pairs=snapshot.pairs,
                score_lookup=selected_lookup,
                artifact=calibrator,
            )
            stages.append(
                StageSummary(
                    "champion_selection_and_calibration",
                    "completed",
                    counts={
                        "validation_pair_count": validation_fs.pair_count,
                        "calibration_pair_count": calibration_batch.pair_count,
                        "decision_pair_count": decision_batch.pair_count,
                    },
                    digests={
                        "champion_selection_digest": selection.selection_digest,
                        "calibrator_digest": calibrator.calibrator_digest,
                    },
                )
            )

            ranking_config = plan.config.models.ranking
            if ranking_config is None or not ranking_config.enabled:
                raise PipelineError(
                    "ML-PIPE-013", "The complete synthetic slice requires a candidate ranker."
                )
            ranking_training_full = matrix_builder.build_labelled(
                features=features,
                labels=batches["training"],
            )
            ranking_training = build_ranking_matrix(
                ranking_training_full,
                query_side=ranking_config.query_side,
            )
            ranking_model = XGBoostCandidateRanker.fit(
                matrix=ranking_training,
                model=ranking_config,
                random_seed=plan.random_seed,
                configuration_digest=plan.configuration_digest,
            )
            ranking_scoring = build_ranking_scoring_matrix(
                scoring_matrix,
                query_side=ranking_config.query_side,
            )
            ranking_scores = XGBoostCandidateRanker.score(
                matrix=ranking_scoring,
                model=ranking_model,
            )
            probability_by_digest = {
                digest: float(probability)
                for digest, probability in zip(
                    calibrated.pair_digests,
                    calibrated.probabilities,
                    strict=True,
                )
            }
            probabilities_in_rank_order = np.asarray(
                [probability_by_digest[digest] for digest in ranking_scores.pair_digests],
                dtype=np.float64,
            )
            assignment_batch = AssignmentEdgeBatch(
                source_record_keys=source_keys,
                pair_references=ranking_scores.pair_references,
                pair_digests=ranking_scores.pair_digests,
                probabilities=probabilities_in_rank_order,
                candidate_ranks=ranking_scores.ranks,
                source_model_id=selection.selected_model_id,
                source_model_version=selection.selected_model_version,
                calibrator_digest=calibrator.calibrator_digest,
                ranking_model_digest=ranking_model.model_digest,
                candidate_search_complete=True,
                candidate_search_truncated=False,
            )
            assignment_plan = AssignmentPlan(
                solver=(
                    "ortools_min_cost_flow"
                    if prefer_ortools and plan.config.assignment.solver == "ortools_min_cost_flow"
                    else "scipy_linear_sum_assignment"
                ),
                no_match_utility=plan.config.assignment.no_match.utility,
                maximum_candidate_edges=plan.config.runtime.maximum_candidate_pairs,
            )
            if assignment_plan.solver == "ortools_min_cost_flow":
                assignment = OrToolsOneToOneAssignmentSolver.solve(
                    assignment_batch, assignment_plan
                )
            else:
                assignment = ScipyOneToOneAssignmentSolver.solve(assignment_batch, assignment_plan)
            stages.append(
                StageSummary(
                    "candidate_ranking_and_assignment",
                    "completed",
                    counts={
                        "ranking_pair_count": ranking_scores.pair_count,
                        "ranking_query_count": ranking_training.query_count,
                        "real_assignment_count": assignment.real_assignment_count,
                        "no_match_count": assignment.no_match_count,
                    },
                    digests={
                        "ranking_model_digest": ranking_model.model_digest,
                        "assignment_digest": assignment.assignment_digest,
                    },
                )
            )

            fs_all = np.asarray([fs_lookup[pair] for pair in scoring_matrix.pair_references])
            xgb_all = np.asarray([xgb_lookup[pair] for pair in scoring_matrix.pair_references])
            fs_top = _top_target_by_source(
                scoring_matrix.pair_references,
                scoring_matrix.pair_digests,
                fs_all,
            )
            xgb_top = _top_target_by_source(
                scoring_matrix.pair_references,
                scoring_matrix.pair_digests,
                xgb_all,
            )
            disagreement_sources = frozenset(
                source for source in set(fs_top) & set(xgb_top) if fs_top[source] != xgb_top[source]
            )
            evidence = DecisionEvidenceBuilder.build(
                candidates=assignment_batch,
                assignment=assignment,
                source_dataset_id=source_id,
                target_dataset_id=target_id,
                anchor_conflict_sources=anchor_conflicts,
                model_disagreement_sources=disagreement_sources,
                anchor_rules_by_source=anchor_rules,
                candidate_rules_by_pair_digest=snapshot.rule_ids_by_digest,
            )
            decisions = RelationshipDecisionPolicy.classify_all(
                evidence,
                plan.config.decision_policy,
                model_family=selection.selected_model_family,
                model_version=selection.selected_model_version,
                assignment_method=assignment.solver,
                assignment_constraint=assignment.constraint,
                run_id=run_id,
                configuration_digest=plan.configuration_digest,
                feature_schema_digest=selection.selected_feature_schema_digest,
                created_at=datetime(2000, 1, 1, tzinfo=UTC),
            )
            decision_report = evaluate_decisions(decisions)
            review_queue = build_review_queue(decisions)
            stages.append(
                StageSummary(
                    "relationship_decisions_and_review",
                    "completed",
                    counts={
                        "relationship_count": len(decisions),
                        "review_queue_count": review_queue.relationship_count,
                    },
                    digests={"review_queue_digest": review_queue.queue_digest},
                )
            )

            candidate_report = evaluate_candidate_retrieval(
                source_record_keys=source_keys,
                target_record_keys=target_keys,
                candidate_pairs=snapshot.pairs,
                true_pairs=true_pairs,
                rule_ids_by_pair=snapshot.rule_ids_by_pair,
            )
            ranking_report = evaluate_ranking(
                scores=ranking_scores,
                true_pair_digests=frozenset(pair_digest(*pair) for pair in true_pairs),
                eligible_query_keys=tuple(
                    source for source, target in true_target_by_source.items() if target is not None
                ),
                k_values=plan.config.validation.candidate_recall_k,
            )
            assignment_report = evaluate_assignment(
                assignment=assignment,
                true_target_by_source=true_target_by_source,
            )
            test_batch = _pair_score_batch(
                labels=batches["test"],
                score_lookup=selected_lookup,
                source_model_family=selection.selected_model_family,
                source_model_id=selection.selected_model_id,
                source_model_version=selection.selected_model_version,
                source_evidence_digest=selection.selected_evidence_digest,
                feature_schema_digest=selection.selected_feature_schema_digest,
                partition_manifest_digest=selection.partition_manifest_digest,
                champion_selection_digest=selection.selection_digest,
            )
            if calibrator.method == "sigmoid":
                test_probabilities = SigmoidCalibrator.apply(test_batch.scores, calibrator)
            elif calibrator.method == "beta":
                test_probabilities = BetaCalibrator.apply(test_batch.scores, calibrator)
            else:
                test_probabilities = IsotonicCalibrator.apply(test_batch.scores, calibrator)
            test_pair_report = replace(
                evaluate_binary_scores(
                    labels=test_batch.labels,
                    scores=test_probabilities,
                    diagnostic_threshold=0.5,
                    evaluation_scope="synthetic_mechanical_evaluation",
                    partition_manifest_digest=test_batch.partition_manifest_digest,
                ),
                calibration_status="calibrated_on_protected_partition",
            )
            test_calibration_report = calibration_diagnostics(
                test_probabilities,
                test_batch.labels,
            )
            missingness_patterns, candidate_set_sizes = _stratification_vectors(
                scoring_matrix=scoring_matrix,
                batch=test_batch,
            )
            stratified_report = evaluate_stratified_pair_performance(
                labels=test_batch.labels,
                probabilities=test_probabilities,
                missingness_patterns=missingness_patterns,
                candidate_set_sizes=candidate_set_sizes,
                diagnostic_threshold=0.5,
                partition_manifest_digest=test_batch.partition_manifest_digest,
            )

            restricted_base = plan.config.outputs.restricted_directory
            artifact_base = f"artifacts/runs/{run_id}"
            write_xgboost_artifact(
                artifact=xgb_model,
                model_path=f"{artifact_base}/models/xgboost.json",
                manifest_path=f"{artifact_base}/models/xgboost.manifest.json",
                policy=plan.path_policy,
            )
            write_calibrator_artifact(
                artifact=calibrator,
                payload_path=f"{artifact_base}/calibration/calibrator.json",
                manifest_path=f"{artifact_base}/calibration/calibrator.manifest.json",
                policy=plan.path_policy,
            )
            write_ranking_artifact(
                artifact=ranking_model,
                model_path=f"{artifact_base}/models/ranker.json",
                manifest_path=f"{artifact_base}/models/ranker.manifest.json",
                policy=plan.path_policy,
            )
            relationship_path = write_relationship_decisions(
                decisions=decisions,
                output=plan.config.outputs,
                path=_relative_output(restricted_base, run_id, "relationships.jsonl"),
                policy=plan.path_policy,
            )
            written_review = write_review_queue(
                queue=review_queue,
                output=plan.config.outputs,
                queue_path=_relative_output(restricted_base, run_id, "review_queue.jsonl"),
                manifest_path=f"{artifact_base}/review/review_queue.manifest.json",
                policy=plan.path_policy,
            )
            selection_report = _SelectionReport(
                selected_model_family=selection.selected_model_family,
                selected_model_id=selection.selected_model_id,
                selected_model_version=selection.selected_model_version,
                primary_metric=selection.primary_metric,
                secondary_metric=selection.secondary_metric,
                selection_digest=selection.selection_digest,
                test_partition_used=selection.test_partition_used,
                calibration_partition_used=selection.calibration_partition_used,
            )
            provenance_report = _SyntheticProvenanceReport(
                entity_count=bundle.provenance.entity_count,
                source_record_count=bundle.provenance.source_a_count,
                target_record_count=bundle.provenance.source_b_count,
                truth_record_count=bundle.provenance.truth_record_count,
                duplicate_count=bundle.provenance.duplicate_count,
                competing_candidate_count=bundle.provenance.competing_candidate_count,
                generator_version=bundle.provenance.generator_version,
                seed=bundle.provenance.seed,
            )
            aggregate_report_path = write_aggregate_validation_report(
                reports={
                    "synthetic_population": provenance_report,
                    "candidate_retrieval": candidate_report,
                    "fellegi_sunter_validation": fs_validation_report,
                    "xgboost_validation": xgb_validation_report,
                    "champion_selection": selection_report,
                    "calibration_fit": calibrator.diagnostics,
                    "configured_decision_thresholds": decision_threshold_report,
                    "calibrated_test_pairs": test_pair_report,
                    "calibrated_test_reliability": test_calibration_report,
                    "stratified_test_performance": stratified_report,
                    "candidate_ranking": ranking_report,
                    "assignment": assignment_report,
                    "relationship_decisions": decision_report,
                    "review_queue": review_queue,
                },
                path=f"{artifact_base}/evaluation/aggregate_report.json",
                policy=plan.path_policy,
            )
            stages.append(
                StageSummary(
                    "synthetic_evaluation",
                    "completed",
                    counts={
                        "test_pair_count": test_pair_report.pair_count,
                        "true_relationship_count": candidate_report.true_relationship_count,
                        "confirmed_count": decision_report.confirmed_count,
                        "review_required_count": decision_report.review_required_count,
                        "unresolved_count": decision_report.unresolved_count,
                        "no_match_count": decision_report.no_match_count,
                    },
                    digests={
                        "evaluation_report_digest": _canonical_digest(
                            {
                                "candidate": candidate_report.safe_summary(),
                                "ranking": ranking_report.safe_summary(),
                                "assignment": assignment_report.safe_summary(),
                                "decisions": decision_report.safe_summary(),
                            }
                        )
                    },
                )
            )

        manifest_payload: dict[str, object] = {
            "schema_version": "0.1",
            "run_id": run_id,
            "engine_version": __version__,
            "configuration_digest": plan.configuration_digest,
            "registry_digest": plan.registry_digest,
            "random_seed": plan.random_seed,
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "dependency_versions": dependency_versions,
            "stage_summaries": [stage.safe_summary() for stage in stages],
            "relationship_status_counts": {
                "confirmed": decision_report.confirmed_count,
                "review_required": decision_report.review_required_count,
                "unresolved": decision_report.unresolved_count,
                "no_match": decision_report.no_match_count,
            },
            "evaluation_scope": "synthetic_mechanical_evaluation",
            "real_data_validation_status": "not_established",
            "merge_authority": "none",
            "warning": (
                "Synthetic testing establishes software behaviour only; it does not validate "
                "linkage accuracy on real populations or systems."
            ),
        }
        run_manifest_path = write_run_manifest(
            payload=manifest_payload,
            path=f"artifacts/runs/{run_id}/run_manifest.json",
            policy=plan.path_policy,
        )
        status_counts = {
            "confirmed": decision_report.confirmed_count,
            "review_required": decision_report.review_required_count,
            "unresolved": decision_report.unresolved_count,
            "no_match": decision_report.no_match_count,
        }
        return SyntheticVerticalSliceResult(
            run_id=run_id,
            configuration_digest=plan.configuration_digest,
            package_version=__version__,
            stage_summaries=tuple(stages),
            relationship_status_counts=status_counts,
            selected_model_family=selection.selected_model_family,
            selected_model_id=selection.selected_model_id,
            calibrator_method=calibrator.method,
            calibrator_digest=calibrator.calibrator_digest,
            ranking_model_digest=ranking_model.model_digest,
            assignment_digest=assignment.assignment_digest,
            review_queue_count=review_queue.relationship_count,
            aggregate_report_path=aggregate_report_path,
            relationship_output_path=relationship_path,
            review_queue_path=written_review.queue_path,
            run_manifest_path=run_manifest_path,
        )
