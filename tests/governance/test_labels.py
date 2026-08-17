from __future__ import annotations

import hashlib
from typing import cast

import pytest

from mapel_linkage.domain.errors import LabelProvenanceError
from mapel_linkage.governance.labels import (
    LabelPartition,
    LabelSourceKind,
    PairLabel,
    VerifiedLabelBatch,
    VerifiedPairLabel,
    assert_disjoint_label_partitions,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _label(
    left: str,
    right: str,
    label: int,
    entity: str,
    household: str | None = None,
) -> VerifiedPairLabel:
    return VerifiedPairLabel(
        left_record_key=left,
        right_record_key=right,
        label=cast(PairLabel, label),
        entity_component_digests=(_digest(entity),),
        household_component_digests=(() if household is None else (_digest(household),)),
    )


def _batch(partition: str, labels: tuple[VerifiedPairLabel, ...]) -> VerifiedLabelBatch:
    return VerifiedLabelBatch(
        source_kind="synthetic_truth",
        verification_protocol="synthetic_v1",
        source_digest=_digest("synthetic-source"),
        partition=cast(LabelPartition, partition),
        labels=labels,
    )


def test_verified_label_batch_is_deterministic_and_hides_pair_references() -> None:
    sentinel_left = "SYNTHETIC-PRIVATE-LEFT"
    sentinel_right = "SYNTHETIC-PRIVATE-RIGHT"
    first = _batch(
        "training",
        (
            _label(sentinel_left, sentinel_right, 1, "entity-1", "household-1"),
            _label("left-2", "right-2", 0, "entity-2", "household-2"),
        ),
    )
    second = _batch(
        "training",
        (
            _label(sentinel_left, sentinel_right, 1, "entity-1", "household-1"),
            _label("left-2", "right-2", 0, "entity-2", "household-2"),
        ),
    )

    assert first.label_authority_digest == second.label_authority_digest
    assert first.safe_summary()["positive_count"] == 1
    assert first.safe_summary()["negative_count"] == 1
    rendered = repr(first)
    assert sentinel_left not in rendered
    assert sentinel_right not in rendered
    assert _digest("entity-1") not in rendered


def test_unverified_reference_cannot_enter_verified_label_contract() -> None:
    with pytest.raises(LabelProvenanceError) as captured:
        VerifiedLabelBatch(
            source_kind=cast(LabelSourceKind, "unverified_reference"),
            verification_protocol="synthetic_v1",
            source_digest=_digest("source"),
            partition="training",
            labels=(_label("left", "right", 1, "entity"),),
        )

    assert str(captured.value) == (
        "ML-LABEL-005: The label source is not eligible for supervised use."
    )


def test_duplicate_and_conflicting_pair_labels_are_rejected_without_values() -> None:
    sentinel = "SYNTHETIC-CONFLICTING-PAIR"
    duplicate = _label(sentinel, "right", 1, "entity")
    conflict = _label(sentinel, "right", 0, "entity")
    with pytest.raises(LabelProvenanceError) as captured:
        _batch("training", (duplicate, conflict))

    assert captured.value.code == "ML-LABEL-010"
    assert sentinel not in str(captured.value)


def test_entity_and_household_components_must_be_partition_disjoint() -> None:
    training = _batch(
        "training",
        (
            _label("train-left", "train-right", 1, "entity-train", "household-train"),
            _label("train-left-2", "train-right-2", 0, "entity-train-2"),
        ),
    )
    validation = _batch(
        "validation",
        (
            _label("valid-left", "valid-right", 1, "entity-valid", "household-valid"),
            _label("valid-left-2", "valid-right-2", 0, "entity-valid-2"),
        ),
    )

    report = assert_disjoint_label_partitions((training, validation))
    assert report.partition_count == 2
    assert report.entity_component_count == 4
    assert report.household_component_count == 2
    assert report.covers("training", training.label_authority_digest)
    assert report.covers("validation", validation.label_authority_digest)


def test_cross_partition_entity_component_is_rejected_without_digest() -> None:
    shared = "shared-entity"
    shared_digest = _digest(shared)
    training = _batch("training", (_label("left-1", "right-1", 1, shared),))
    validation = _batch("validation", (_label("left-2", "right-2", 0, shared),))

    with pytest.raises(LabelProvenanceError) as captured:
        assert_disjoint_label_partitions((training, validation))

    assert captured.value.code == "ML-LABEL-013"
    assert shared_digest not in str(captured.value)


def test_duplicate_partition_batches_are_rejected() -> None:
    first = _batch("training", (_label("left-1", "right-1", 1, "entity-1"),))
    second = _batch("training", (_label("left-2", "right-2", 0, "entity-2"),))

    with pytest.raises(LabelProvenanceError) as captured:
        assert_disjoint_label_partitions((first, second))

    assert captured.value.code == "ML-LABEL-015"


def test_same_pair_cannot_cross_partitions_even_with_inconsistent_component_metadata() -> None:
    sentinel = "SYNTHETIC-CROSS-PARTITION-PAIR"
    training = _batch("training", (_label(sentinel, "right", 1, "entity-1"),))
    validation = _batch("validation", (_label(sentinel, "right", 1, "entity-2"),))

    with pytest.raises(LabelProvenanceError) as captured:
        assert_disjoint_label_partitions((training, validation))

    assert captured.value.code == "ML-LABEL-016"
    assert sentinel not in str(captured.value)


def test_pair_reference_with_nul_separator_is_rejected_without_value() -> None:
    sentinel = "SYNTHETIC\x00PRIVATE"
    with pytest.raises(LabelProvenanceError) as captured:
        _label(sentinel, "right", 1, "entity")

    assert captured.value.code == "ML-LABEL-001"
    assert "SYNTHETIC" not in str(captured.value)
