"""Verified label provenance and leakage-resistant partition contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from mapel_linkage.domain.errors import LabelProvenanceError

LabelSourceKind = Literal[
    "synthetic_truth",
    "verified_human_adjudication",
    "verified_gold_standard",
]
LabelPartition = Literal["training", "validation", "calibration", "decision", "test"]
PairLabel = Literal[0, 1]

_ALLOWED_SOURCE_KINDS: frozenset[str] = frozenset(
    {"synthetic_truth", "verified_human_adjudication", "verified_gold_standard"}
)
_ALLOWED_PARTITIONS: frozenset[str] = frozenset(
    {"training", "validation", "calibration", "decision", "test"}
)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class _CanonicalLabel(TypedDict):
    pair_digest: str
    label: PairLabel
    entity_components: list[str]
    household_components: list[str]


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_digest(value: str, *, code: str, message: str) -> None:
    if _DIGEST_PATTERN.fullmatch(value) is None:
        raise LabelProvenanceError(code, message)


def _require_component_digests(
    values: tuple[str, ...],
    *,
    allow_empty: bool,
    code: str,
    message: str,
) -> None:
    if (not allow_empty and not values) or len(values) > 4 or len(values) != len(set(values)):
        raise LabelProvenanceError(code, message)
    for value in values:
        if _DIGEST_PATTERN.fullmatch(value) is None:
            raise LabelProvenanceError(code, message)


@dataclass(frozen=True, slots=True)
class VerifiedPairLabel:
    """A verified binary label whose pair and grouping references stay private."""

    left_record_key: str = field(repr=False)
    right_record_key: str = field(repr=False)
    label: PairLabel
    entity_component_digests: tuple[str, ...] = field(repr=False)
    household_component_digests: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if (
            not self.left_record_key
            or not self.right_record_key
            or len(self.left_record_key) > 1024
            or len(self.right_record_key) > 1024
            or "\x00" in self.left_record_key
            or "\x00" in self.right_record_key
        ):
            raise LabelProvenanceError(
                "ML-LABEL-001", "A verified label has an invalid private pair reference."
            )
        if self.label not in (0, 1):
            raise LabelProvenanceError("ML-LABEL-002", "A verified label must be binary.")
        _require_component_digests(
            self.entity_component_digests,
            allow_empty=False,
            code="ML-LABEL-003",
            message="A verified label has invalid entity-component provenance.",
        )
        _require_component_digests(
            self.household_component_digests,
            allow_empty=True,
            code="ML-LABEL-004",
            message="A verified label has invalid household-component provenance.",
        )

    def pair_digest(self) -> str:
        """Return a deterministic pair digest without exposing either record key."""

        return hashlib.sha256(
            (self.left_record_key + "\x00" + self.right_record_key).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedLabelBatch:
    """A partition-specific snapshot of labels with explicit verification authority."""

    source_kind: LabelSourceKind
    verification_protocol: str
    source_digest: str
    partition: LabelPartition
    labels: tuple[VerifiedPairLabel, ...] = field(repr=False)
    label_authority_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.source_kind not in _ALLOWED_SOURCE_KINDS:
            raise LabelProvenanceError(
                "ML-LABEL-005", "The label source is not eligible for supervised use."
            )
        if self.partition not in _ALLOWED_PARTITIONS:
            raise LabelProvenanceError("ML-LABEL-006", "The label partition is not supported.")
        if _IDENTIFIER_PATTERN.fullmatch(self.verification_protocol) is None:
            raise LabelProvenanceError(
                "ML-LABEL-007", "The verification protocol identifier is invalid."
            )
        _require_digest(
            self.source_digest,
            code="ML-LABEL-008",
            message="The label source digest is invalid.",
        )
        if not self.labels:
            raise LabelProvenanceError(
                "ML-LABEL-009", "A verified label batch must contain at least one label."
            )

        by_pair: dict[str, int] = {}
        canonical_labels: list[_CanonicalLabel] = []
        for item in sorted(self.labels, key=lambda label: label.pair_digest()):
            pair_digest = item.pair_digest()
            prior = by_pair.get(pair_digest)
            if prior is not None:
                if prior != item.label:
                    raise LabelProvenanceError(
                        "ML-LABEL-010", "Conflicting verified labels were rejected."
                    )
                raise LabelProvenanceError(
                    "ML-LABEL-011", "Duplicate verified labels were rejected."
                )
            by_pair[pair_digest] = item.label
            canonical_labels.append(
                {
                    "pair_digest": pair_digest,
                    "label": item.label,
                    "entity_components": sorted(item.entity_component_digests),
                    "household_components": sorted(item.household_component_digests),
                }
            )

        authority = _canonical_digest(
            {
                "source_kind": self.source_kind,
                "verification_protocol": self.verification_protocol,
                "source_digest": self.source_digest,
                "partition": self.partition,
                "labels": canonical_labels,
            }
        )
        object.__setattr__(self, "label_authority_digest", authority)

    @property
    def positive_count(self) -> int:
        return sum(item.label for item in self.labels)

    @property
    def negative_count(self) -> int:
        return len(self.labels) - self.positive_count

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "source_kind": self.source_kind,
            "partition": self.partition,
            "label_count": len(self.labels),
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "entity_component_count": len(
                {value for item in self.labels for value in item.entity_component_digests}
            ),
            "household_component_count": len(
                {value for item in self.labels for value in item.household_component_digests}
            ),
            "label_authority_digest": self.label_authority_digest,
        }


@dataclass(frozen=True, slots=True)
class PartitionDisjointnessReport:
    """Aggregate proof that protected entity and household groups do not cross partitions."""

    partition_count: int
    entity_component_count: int
    household_component_count: int
    manifest_digest: str
    partition_authority_digests: tuple[tuple[LabelPartition, str], ...] = field(repr=False)

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "partition_count": self.partition_count,
            "entity_component_count": self.entity_component_count,
            "household_component_count": self.household_component_count,
            "manifest_digest": self.manifest_digest,
        }

    def covers(self, partition: LabelPartition, authority_digest: str) -> bool:
        """Return whether the proof includes one exact partition label snapshot."""

        return (partition, authority_digest) in self.partition_authority_digests


def assert_disjoint_label_partitions(
    batches: Iterable[VerifiedLabelBatch],
) -> PartitionDisjointnessReport:
    """Reject entity or household components that occur in different partitions."""

    materialised = tuple(batches)
    if len(materialised) < 2:
        raise LabelProvenanceError(
            "ML-LABEL-012", "At least two label partitions are required for a disjointness check."
        )

    partitions = [batch.partition for batch in materialised]
    if len(partitions) != len(set(partitions)):
        raise LabelProvenanceError(
            "ML-LABEL-015", "Duplicate protected label partitions were rejected."
        )

    pair_partition: dict[str, LabelPartition] = {}
    entity_partition: dict[str, LabelPartition] = {}
    household_partition: dict[str, LabelPartition] = {}
    partition_digests: list[tuple[LabelPartition, str]] = []
    for batch in materialised:
        partition_digests.append((batch.partition, batch.label_authority_digest))
        for item in batch.labels:
            pair_digest = item.pair_digest()
            prior_pair_partition = pair_partition.setdefault(pair_digest, batch.partition)
            if prior_pair_partition != batch.partition:
                raise LabelProvenanceError(
                    "ML-LABEL-016", "A verified pair crosses protected label partitions."
                )
            for digest in item.entity_component_digests:
                prior = entity_partition.setdefault(digest, batch.partition)
                if prior != batch.partition:
                    raise LabelProvenanceError(
                        "ML-LABEL-013",
                        "An entity component crosses protected label partitions.",
                    )
            for digest in item.household_component_digests:
                prior = household_partition.setdefault(digest, batch.partition)
                if prior != batch.partition:
                    raise LabelProvenanceError(
                        "ML-LABEL-014",
                        "A household component crosses protected label partitions.",
                    )

    manifest_digest = _canonical_digest(sorted(partition_digests))
    return PartitionDisjointnessReport(
        partition_count=len({batch.partition for batch in materialised}),
        entity_component_count=len(entity_partition),
        household_component_count=len(household_partition),
        manifest_digest=manifest_digest,
        partition_authority_digests=tuple(sorted(partition_digests)),
    )
