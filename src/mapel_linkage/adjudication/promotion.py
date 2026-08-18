"""Adjudication label promotion eligibility evaluation and verified batch creation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast

from mapel_linkage.adjudication.disagreement import ConsensusResult
from mapel_linkage.adjudication.review_import import AdjudicationRecord
from mapel_linkage.domain.errors import AdjudicationError
from mapel_linkage.governance.labels import (
    LabelPartition,
    PairLabel,
    VerifiedLabelBatch,
    VerifiedPairLabel,
)

_ALLOWED_PARTITIONS: frozenset[str] = frozenset(
    {"training", "validation", "calibration", "decision", "test"}
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    """Rules and thresholds governing label promotion to verified dataset partitions."""

    target_partition: LabelPartition = "training"
    min_confidence: float = 0.80
    require_consensus: bool = True
    require_double_review: bool = False
    minimum_reviewers: int = 1
    allowed_protocols: frozenset[str] = field(default_factory=frozenset)
    allow_audit_only: bool = True

    def __post_init__(self) -> None:
        if self.target_partition not in _ALLOWED_PARTITIONS:
            raise AdjudicationError(
                "ML-ADJ-018", f"Unsupported target partition for promotion: {self.target_partition}"
            )
        if self.min_confidence < 0.0 or self.min_confidence > 1.0:
            raise AdjudicationError(
                "ML-ADJ-011", "Promotion minimum confidence must be between 0.0 and 1.0."
            )
        if self.minimum_reviewers < 1:
            raise AdjudicationError("ML-ADJ-016", "Promotion minimum reviewers must be at least 1.")


@dataclass(frozen=True, slots=True, repr=False)
class PromotionEvaluation:
    """Eligibility determination for a single adjudicated candidate pair."""

    pair_digest: str
    is_eligible: bool
    promoted_label: PairLabel | None
    target_partition: LabelPartition | None
    rejection_reasons: tuple[str, ...]
    is_audit_only: bool
    left_record_key: str = field(repr=False)
    right_record_key: str = field(repr=False)
    confidence: float
    reviewer_count: int
    protocol_version: str
    entity_component_digests: tuple[str, ...] = field(default=(), repr=False)
    household_component_digests: tuple[str, ...] = field(default=(), repr=False)

    def safe_summary(self) -> dict[str, object]:
        return {
            "pair_digest": self.pair_digest,
            "is_eligible": self.is_eligible,
            "promoted_label": self.promoted_label,
            "target_partition": self.target_partition,
            "is_audit_only": self.is_audit_only,
            "rejection_reasons": list(self.rejection_reasons),
            "confidence": round(self.confidence, 6),
            "reviewer_count": self.reviewer_count,
            "protocol_version": self.protocol_version,
        }


@dataclass(frozen=True, slots=True)
class PromotionSummary:
    """Aggregate counts for a batch label promotion evaluation."""

    total_evaluated: int
    eligible_count: int
    promoted_positive_count: int
    promoted_negative_count: int
    audit_only_count: int
    rejected_count: int
    target_partition: LabelPartition

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "total_evaluated": self.total_evaluated,
            "eligible_count": self.eligible_count,
            "promoted_positive_count": self.promoted_positive_count,
            "promoted_negative_count": self.promoted_negative_count,
            "audit_only_count": self.audit_only_count,
            "rejected_count": self.rejected_count,
            "target_partition": self.target_partition,
        }


def evaluate_promotion_eligibility(
    item: ConsensusResult | AdjudicationRecord,
    config: PromotionConfig,
    *,
    locked_test_pairs: frozenset[str] = frozenset(),
) -> PromotionEvaluation:
    """Evaluate whether an adjudicated pair qualifies for promotion to a verified label batch."""
    rejection_reasons: list[str] = []
    is_audit_only = False

    pair_digest = item.pair_digest if isinstance(item, ConsensusResult) else item.pair_digest()
    left_key = item.left_record_key
    right_key = item.right_record_key
    entity_digests = item.entity_component_digests
    household_digests = item.household_component_digests
    protocol_version = item.protocol_version

    if isinstance(item, ConsensusResult):
        outcome = item.consensus_outcome
        confidence = item.consensus_confidence
        reviewer_count = item.reviewer_count
        is_resolved = item.is_resolved
    else:
        outcome = item.decision
        confidence = item.confidence
        reviewer_count = 1
        is_resolved = True

    # Check 1: Resolution status
    if not is_resolved or outcome is None:
        rejection_reasons.append("unresolved_dispute")
        is_audit_only = True

    # Check 2: Binary outcome validity
    promoted_label: PairLabel | None = None
    if outcome == "match":
        promoted_label = 1
    elif outcome == "nonmatch":
        promoted_label = 0
    else:
        rejection_reasons.append("non_binary_outcome_audit_only")
        is_audit_only = True

    # Check 3: Review confidence threshold
    if confidence < config.min_confidence:
        rejection_reasons.append("insufficient_confidence")

    # Check 4: Reviewer count / double-review requirement
    if reviewer_count < config.minimum_reviewers:
        rejection_reasons.append("insufficient_reviewers")
    if config.require_double_review and reviewer_count < 2:
        rejection_reasons.append("requires_double_review")

    # Check 5: Protocol approval
    if config.allowed_protocols and protocol_version not in config.allowed_protocols:
        rejection_reasons.append("unapproved_protocol_version")

    # Check 6: Locked test partition protection
    if pair_digest in locked_test_pairs and config.target_partition != "test":
        rejection_reasons.append("locked_test_partition_violation")
        is_audit_only = True

    is_eligible = len(rejection_reasons) == 0 and promoted_label is not None

    return PromotionEvaluation(
        pair_digest=pair_digest,
        is_eligible=is_eligible,
        promoted_label=promoted_label if is_eligible else None,
        target_partition=config.target_partition if is_eligible else None,
        rejection_reasons=tuple(rejection_reasons),
        is_audit_only=is_audit_only or (not is_eligible and config.allow_audit_only),
        left_record_key=left_key,
        right_record_key=right_key,
        confidence=confidence,
        reviewer_count=reviewer_count,
        protocol_version=protocol_version,
        entity_component_digests=entity_digests,
        household_component_digests=household_digests,
    )


def evaluate_promotion_batch(
    items: Iterable[ConsensusResult | AdjudicationRecord],
    config: PromotionConfig,
    *,
    locked_test_pairs: frozenset[str] = frozenset(),
) -> tuple[tuple[PromotionEvaluation, ...], PromotionSummary]:
    """Evaluate a batch of adjudicated pairs against label promotion criteria."""
    evaluations: list[PromotionEvaluation] = []
    for item in items:
        evaluations.append(
            evaluate_promotion_eligibility(item, config, locked_test_pairs=locked_test_pairs)
        )

    eligible_count = sum(1 for e in evaluations if e.is_eligible)
    positive_count = sum(1 for e in evaluations if e.is_eligible and e.promoted_label == 1)
    negative_count = sum(1 for e in evaluations if e.is_eligible and e.promoted_label == 0)
    audit_only_count = sum(1 for e in evaluations if e.is_audit_only and not e.is_eligible)
    rejected_count = len(evaluations) - eligible_count

    summary = PromotionSummary(
        total_evaluated=len(evaluations),
        eligible_count=eligible_count,
        promoted_positive_count=positive_count,
        promoted_negative_count=negative_count,
        audit_only_count=audit_only_count,
        rejected_count=rejected_count,
        target_partition=config.target_partition,
    )

    return tuple(evaluations), summary


def promote_to_verified_batch(
    items: Iterable[ConsensusResult | AdjudicationRecord],
    config: PromotionConfig,
    *,
    verification_protocol: str,
    source_digest: str,
    locked_test_pairs: frozenset[str] = frozenset(),
) -> tuple[VerifiedLabelBatch, PromotionSummary]:
    """Evaluate and promote eligible adjudicated pairs into a VerifiedLabelBatch."""
    evaluations, summary = evaluate_promotion_batch(
        items, config, locked_test_pairs=locked_test_pairs
    )

    eligible_evals = [e for e in evaluations if e.is_eligible and e.promoted_label is not None]
    if not eligible_evals:
        raise AdjudicationError("ML-ADJ-017", "No adjudicated pairs qualified for label promotion.")

    verified_labels: list[VerifiedPairLabel] = []
    for eval_item in eligible_evals:
        # Guarantee non-empty entity component digests
        entity_digests = eval_item.entity_component_digests
        if not entity_digests:
            entity_digests = (_digest(f"adjudicated-entity:{eval_item.pair_digest}"),)

        label_obj = VerifiedPairLabel(
            left_record_key=eval_item.left_record_key,
            right_record_key=eval_item.right_record_key,
            label=cast(PairLabel, eval_item.promoted_label),
            entity_component_digests=entity_digests,
            household_component_digests=eval_item.household_component_digests,
        )
        verified_labels.append(label_obj)

    batch = VerifiedLabelBatch(
        source_kind="verified_human_adjudication",
        verification_protocol=verification_protocol,
        source_digest=source_digest,
        partition=config.target_partition,
        labels=tuple(verified_labels),
    )

    return batch, summary
