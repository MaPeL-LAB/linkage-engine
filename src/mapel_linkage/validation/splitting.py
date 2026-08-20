"""Leakage-resistant entity/household connected-component splitting."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import networkx as nx

from mapel_linkage.domain.errors import LabelProvenanceError, ValidationReportError
from mapel_linkage.governance.labels import (
    LabelPartition,
    PairLabel,
    VerifiedLabelBatch,
    VerifiedPairLabel,
)

_PARTITIONS: tuple[LabelPartition, ...] = (
    "training",
    "validation",
    "calibration",
    "decision",
    "test",
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class EntityHouseholdRecord:
    dataset_id: str
    record_key: str = field(repr=False)
    entity_key: str = field(repr=False)
    household_key: str | None = field(default=None, repr=False)

    @property
    def entity_digest(self) -> str:
        return _digest(f"entity\x00{self.entity_key}")

    @property
    def household_digest(self) -> str | None:
        if self.household_key is None:
            return None
        return _digest(f"household\x00{self.household_key}")


@dataclass(frozen=True, slots=True, repr=False)
class PartitionAssignment:
    record_partitions: Mapping[str, LabelPartition] = field(repr=False)
    entity_partitions: Mapping[str, LabelPartition] = field(repr=False)
    household_partitions: Mapping[str, LabelPartition] = field(repr=False)
    partition_record_counts: tuple[tuple[LabelPartition, int], ...]
    connected_component_count: int
    manifest_digest: str
    random_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "record_partitions", MappingProxyType(dict(self.record_partitions))
        )
        object.__setattr__(
            self, "entity_partitions", MappingProxyType(dict(self.entity_partitions))
        )
        object.__setattr__(
            self, "household_partitions", MappingProxyType(dict(self.household_partitions))
        )

    def partition_for_record(self, record_key: str) -> LabelPartition:
        try:
            return self.record_partitions[record_key]
        except KeyError:
            raise ValidationReportError(
                "ML-SPLIT-001", "A record is unavailable in the protected partition manifest."
            ) from None

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "record_count": len(self.record_partitions),
            "entity_component_count": len(self.entity_partitions),
            "household_component_count": len(self.household_partitions),
            "connected_component_count": self.connected_component_count,
            "manifest_digest": self.manifest_digest,
            "random_seed": self.random_seed,
        }


def split_entity_household_components(
    records: tuple[EntityHouseholdRecord, ...],
    *,
    fractions: tuple[float, float, float, float, float],
    random_seed: int,
) -> PartitionAssignment:
    if not records or len({record.record_key for record in records}) != len(records):
        raise ValidationReportError("ML-SPLIT-002", "Protected truth records are invalid.")
    if (
        len(fractions) != 5
        or any(not math.isfinite(value) or value <= 0.0 for value in fractions)
        or abs(sum(fractions) - 1.0) > 1e-9
        or random_seed < 0
    ):
        raise ValidationReportError("ML-SPLIT-003", "Protected split fractions are invalid.")
    graph = nx.Graph()
    records_by_entity: dict[str, list[str]] = defaultdict(list)
    entity_digest_by_key: dict[str, str] = {}
    household_digest_by_key: dict[str, str] = {}
    for record in records:
        entity_node = f"e:{record.entity_digest}"
        graph.add_node(entity_node)
        entity_digest_by_key[record.record_key] = record.entity_digest
        records_by_entity[record.entity_digest].append(record.record_key)
        household_digest = record.household_digest
        if household_digest is not None:
            household_node = f"h:{household_digest}"
            graph.add_edge(entity_node, household_node)
            household_digest_by_key[record.record_key] = household_digest

    components: list[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    for nodes in nx.connected_components(graph):
        entities = sorted(node[2:] for node in nodes if node.startswith("e:"))
        households = sorted(node[2:] for node in nodes if node.startswith("h:"))
        record_keys = sorted(
            record_key for entity in entities for record_key in records_by_entity[entity]
        )
        token = _canonical_digest({"nodes": sorted(nodes), "random_seed": random_seed})
        components.append((token, tuple(entities), tuple(households), tuple(record_keys)))
    components.sort(key=lambda item: item[0])

    total_records = len(records)
    targets = {
        partition: total_records * fraction
        for partition, fraction in zip(_PARTITIONS, fractions, strict=True)
    }
    current = {partition: 0 for partition in _PARTITIONS}
    assigned_components: list[
        tuple[LabelPartition, tuple[str, ...], tuple[str, ...], tuple[str, ...]]
    ] = []
    for _, component_entities, component_households, component_record_keys in sorted(
        components, key=lambda item: (-len(item[3]), item[0])
    ):
        partition = min(
            _PARTITIONS,
            key=lambda name: (
                (current[name] + len(component_record_keys)) / max(targets[name], 1e-9),
                current[name],
                name,
            ),
        )
        current[partition] += len(component_record_keys)
        assigned_components.append(
            (
                partition,
                component_entities,
                component_households,
                component_record_keys,
            )
        )

    record_partitions: dict[str, LabelPartition] = {}
    entity_partitions: dict[str, LabelPartition] = {}
    household_partitions: dict[str, LabelPartition] = {}
    for (
        assigned_partition,
        assigned_entities,
        assigned_households,
        assigned_keys,
    ) in assigned_components:
        for record_key in assigned_keys:
            record_partitions[record_key] = assigned_partition
        for entity in assigned_entities:
            if entity in entity_partitions and entity_partitions[entity] != assigned_partition:
                raise ValidationReportError(
                    "ML-SPLIT-004", "An entity crosses protected partitions."
                )
            entity_partitions[entity] = assigned_partition
        for household in assigned_households:
            if (
                household in household_partitions
                and household_partitions[household] != assigned_partition
            ):
                raise ValidationReportError(
                    "ML-SPLIT-005", "A household crosses protected partitions."
                )
            household_partitions[household] = assigned_partition

    if set(record_partitions) != {record.record_key for record in records}:
        raise ValidationReportError("ML-SPLIT-006", "Protected split coverage is incomplete.")
    manifest_payload = {
        "random_seed": random_seed,
        "components": [
            {
                "partition": partition,
                "entities": entities,
                "households": households,
                "record_key_digests": sorted(_digest(key) for key in record_keys),
            }
            for partition, entities, households, record_keys in sorted(
                assigned_components,
                key=lambda item: (item[0], item[1], item[2]),
            )
        ],
    }
    return PartitionAssignment(
        record_partitions=record_partitions,
        entity_partitions=entity_partitions,
        household_partitions=household_partitions,
        partition_record_counts=tuple((partition, current[partition]) for partition in _PARTITIONS),
        connected_component_count=len(components),
        manifest_digest=_canonical_digest(manifest_payload),
        random_seed=random_seed,
    )


def build_verified_candidate_label_batches(
    *,
    candidate_pairs: tuple[tuple[str, str], ...],
    truth_records: tuple[EntityHouseholdRecord, ...],
    assignment: PartitionAssignment,
    verification_protocol: str,
    source_digest: str,
) -> tuple[VerifiedLabelBatch, ...]:
    truth_by_record = {record.record_key: record for record in truth_records}
    labels_by_partition: dict[LabelPartition, list[VerifiedPairLabel]] = defaultdict(list)
    for left, right in candidate_pairs:
        try:
            left_truth = truth_by_record[left]
            right_truth = truth_by_record[right]
        except KeyError:
            raise LabelProvenanceError(
                "ML-LABEL-017", "A candidate pair is absent from the verified truth snapshot."
            ) from None
        left_partition = assignment.partition_for_record(left)
        right_partition = assignment.partition_for_record(right)
        if left_partition != right_partition:
            continue
        entity_digests = tuple(sorted({left_truth.entity_digest, right_truth.entity_digest}))
        household_digests = tuple(
            sorted(
                {
                    digest
                    for digest in (
                        left_truth.household_digest,
                        right_truth.household_digest,
                    )
                    if digest is not None
                }
            )
        )
        labels_by_partition[left_partition].append(
            VerifiedPairLabel(
                left_record_key=left,
                right_record_key=right,
                label=cast(PairLabel, int(left_truth.entity_key == right_truth.entity_key)),
                entity_component_digests=entity_digests,
                household_component_digests=household_digests,
            )
        )
    batches: list[VerifiedLabelBatch] = []
    for partition in _PARTITIONS:
        labels = labels_by_partition.get(partition, [])
        if not labels:
            continue
        batches.append(
            VerifiedLabelBatch(
                source_kind="synthetic_truth",
                verification_protocol=verification_protocol,
                source_digest=source_digest,
                partition=partition,
                labels=tuple(labels),
            )
        )
    return tuple(batches)
