"""Shared fail-closed contracts for generated-synthetic workflow runners."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from mapel_linkage.assignment import pair_digest
from mapel_linkage.candidate_generation import (
    AllOf,
    AnyOf,
    BlockingRule,
    CandidatePredicate,
    DateWindow,
    Exact,
    PrefixEqual,
)
from mapel_linkage.configuration import ExecutionPlan
from mapel_linkage.configuration.models import (
    AllPredicate,
    AnyPredicate,
    BlockPredicate,
    DateWindowPredicate,
    ExactPredicate,
    PrefixEqualPredicate,
)
from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.domain.sql_identifiers import quote_identifier
from mapel_linkage.governance.labels import (
    PartitionDisjointnessReport,
    VerifiedLabelBatch,
    assert_disjoint_label_partitions,
)
from mapel_linkage.io import DuckDBStore
from mapel_linkage.preprocessing import surrogate_record_key
from mapel_linkage.synthetic import SyntheticBundle
from mapel_linkage.validation import EntityHouseholdRecord, split_entity_household_components
from mapel_linkage.validation.splitting import build_verified_candidate_label_batches


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class SyntheticCandidateSnapshot:
    """Value-hidden ordered candidate metadata shared by synthetic runners."""

    pairs: tuple[tuple[str, str], ...]
    pair_digests: tuple[str, ...]
    rule_ids_by_pair: dict[tuple[str, str], tuple[str, ...]]
    rule_ids_by_digest: dict[str, tuple[str, ...]]


def runtime_blocking_rules(plan: ExecutionPlan) -> tuple[BlockingRule, ...]:
    """Compile declarative blocking predicates without granting scoring authority."""

    def convert(predicate: BlockPredicate) -> CandidatePredicate:
        if isinstance(predicate, ExactPredicate):
            return Exact(predicate.variable)
        if isinstance(predicate, PrefixEqualPredicate):
            return PrefixEqual(predicate.variable, predicate.length)
        if isinstance(predicate, AllPredicate):
            return AllOf(tuple(convert(term) for term in predicate.terms))
        if isinstance(predicate, AnyPredicate):
            return AnyOf(tuple(convert(term) for term in predicate.terms))
        if isinstance(predicate, DateWindowPredicate):
            return DateWindow(predicate.variable, predicate.maximum_days)
        raise PipelineError("ML-PIPE-004", "A configured blocking predicate is unsupported.")

    return tuple(
        BlockingRule(rule.id, convert(rule.predicate)) for rule in plan.config.blocking.rules
    )


def source_target_ids(plan: ExecutionPlan) -> tuple[str, str]:
    """Require the synthetic two-source link-only, one-to-one workflow shape."""

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


def synthetic_fixture_directory(
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


def candidate_snapshot(store: DuckDBStore, table_name: str) -> SyntheticCandidateSnapshot:
    """Read an exact ordered candidate set and its retrieval-rule provenance."""

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
    return SyntheticCandidateSnapshot(tuple(pairs), tuple(digests), by_pair, by_digest)


def synthetic_truth_records(
    bundle: SyntheticBundle,
    *,
    source_dataset_id: str,
    target_dataset_id: str,
) -> tuple[EntityHouseholdRecord, ...]:
    """Translate generated truth to prepared surrogate record keys."""

    dataset_map = {"source_a": source_dataset_id, "source_b": target_dataset_id}
    output = tuple(
        EntityHouseholdRecord(
            dataset_id=dataset_map[record.dataset_id],
            record_key=surrogate_record_key(dataset_map[record.dataset_id], record.record_key),
            entity_key=record.entity_key,
            household_key=record.household_key,
        )
        for record in bundle.truth
    )
    return tuple(sorted(output, key=lambda item: (item.dataset_id, item.record_key)))


def protected_label_batches(
    *,
    candidate_pairs: tuple[tuple[str, str], ...],
    truth_records: tuple[EntityHouseholdRecord, ...],
    plan: ExecutionPlan,
) -> tuple[dict[str, VerifiedLabelBatch], PartitionDisjointnessReport, str]:
    """Build and verify all five protected synthetic label partitions."""

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


__all__ = [
    "SyntheticCandidateSnapshot",
    "candidate_snapshot",
    "protected_label_batches",
    "runtime_blocking_rules",
    "source_target_ids",
    "synthetic_fixture_directory",
    "synthetic_truth_records",
]
