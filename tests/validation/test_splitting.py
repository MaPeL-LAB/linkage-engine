from __future__ import annotations

import hashlib

from mapel_linkage.governance.labels import assert_disjoint_label_partitions
from mapel_linkage.validation import (
    EntityHouseholdRecord,
    build_verified_candidate_label_batches,
    split_entity_household_components,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def records() -> tuple[EntityHouseholdRecord, ...]:
    output = []
    for entity in range(30):
        household = f"h-{entity // 2}"
        output.append(EntityHouseholdRecord("source_a", f"a-{entity}", f"e-{entity}", household))
        output.append(EntityHouseholdRecord("source_b", f"b-{entity}", f"e-{entity}", household))
    return tuple(output)


def test_entity_household_components_are_deterministic_and_disjoint() -> None:
    first = split_entity_household_components(
        records(),
        fractions=(0.5, 0.15, 0.1, 0.1, 0.15),
        random_seed=20260817,
    )
    second = split_entity_household_components(
        records(),
        fractions=(0.5, 0.15, 0.1, 0.1, 0.15),
        random_seed=20260817,
    )
    assert first.manifest_digest == second.manifest_digest
    assert first.record_partitions == second.record_partitions
    for entity in range(30):
        assert first.partition_for_record(f"a-{entity}") == first.partition_for_record(
            f"b-{entity}"
        )
        if entity % 2 == 1:
            assert first.partition_for_record(f"a-{entity}") == first.partition_for_record(
                f"a-{entity - 1}"
            )


def test_verified_candidate_labels_skip_cross_partition_pairs() -> None:
    truth = records()
    split = split_entity_household_components(
        truth,
        fractions=(0.5, 0.15, 0.1, 0.1, 0.15),
        random_seed=5,
    )
    pairs = []
    for entity in range(30):
        pairs.append((f"a-{entity}", f"b-{entity}"))
        pairs.append((f"a-{entity}", f"b-{(entity + 2) % 30}"))
    batches = build_verified_candidate_label_batches(
        candidate_pairs=tuple(pairs),
        truth_records=truth,
        assignment=split,
        verification_protocol="synthetic_v1",
        source_digest=digest("truth"),
    )
    report = assert_disjoint_label_partitions(batches)
    assert report.partition_count >= 2
    assert all(batch.label_authority_digest for batch in batches)
    assert all(
        split.partition_for_record(item.left_record_key)
        == split.partition_for_record(item.right_record_key)
        for batch in batches
        for item in batch.labels
    )
