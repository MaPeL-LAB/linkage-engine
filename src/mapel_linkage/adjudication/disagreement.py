"""Double-review consensus, disagreement resolution, and dispute tracking."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from mapel_linkage.adjudication.review_import import AdjudicationOutcome, AdjudicationRecord
from mapel_linkage.domain.errors import AdjudicationError

ConsensusMethod = Literal[
    "unanimous",
    "majority",
    "senior_override",
    "single_reviewer",
    "unresolved",
]

ConsensusPolicy = Literal[
    "unanimous_only",
    "majority_vote",
    "senior_reviewer_override",
    "strict_double_review",
]

_ALLOWED_POLICIES: frozenset[str] = frozenset(
    {"unanimous_only", "majority_vote", "senior_reviewer_override", "strict_double_review"}
)


@dataclass(frozen=True, slots=True, repr=False)
class ReviewConflict:
    """A conflict between competing reviewer decisions for a single candidate pair."""

    pair_digest: str
    competing_outcomes: tuple[AdjudicationOutcome, ...]
    reviewers: tuple[str, ...]
    is_disputed: bool
    dispute_reason: str

    def safe_summary(self) -> dict[str, object]:
        return {
            "pair_digest": self.pair_digest,
            "competing_outcomes": list(self.competing_outcomes),
            "reviewer_count": len(self.reviewers),
            "is_disputed": self.is_disputed,
            "dispute_reason": self.dispute_reason,
        }


@dataclass(frozen=True, slots=True, repr=False)
class ConsensusResult:
    """Consensus outcome and provenance for an adjudicated candidate pair."""

    pair_digest: str
    left_record_key: str = field(repr=False)
    right_record_key: str = field(repr=False)
    consensus_outcome: AdjudicationOutcome | None
    consensus_confidence: float
    resolution_method: ConsensusMethod
    is_resolved: bool
    has_conflict: bool
    reviewer_count: int
    reviewing_event_ids: tuple[str, ...]
    senior_reviewer_id: str | None = None
    dispute_reason: str | None = None
    entity_component_digests: tuple[str, ...] = field(default=(), repr=False)
    household_component_digests: tuple[str, ...] = field(default=(), repr=False)
    protocol_version: str = ""

    def safe_summary(self) -> dict[str, object]:
        return {
            "pair_digest": self.pair_digest,
            "consensus_outcome": self.consensus_outcome,
            "consensus_confidence": round(self.consensus_confidence, 6),
            "resolution_method": self.resolution_method,
            "is_resolved": self.is_resolved,
            "has_conflict": self.has_conflict,
            "reviewer_count": self.reviewer_count,
            "senior_reviewer_id": self.senior_reviewer_id,
            "dispute_reason": self.dispute_reason,
            "protocol_version": self.protocol_version,
        }


@dataclass(frozen=True, slots=True)
class DisagreementReport:
    """Aggregate statistics over batch disagreement and consensus evaluation."""

    total_pairs: int
    resolved_pairs: int
    unresolved_pairs: int
    conflict_count: int
    unanimous_count: int
    majority_count: int
    senior_override_count: int
    single_reviewer_count: int

    def safe_summary(self) -> dict[str, int]:
        return {
            "total_pairs": self.total_pairs,
            "resolved_pairs": self.resolved_pairs,
            "unresolved_pairs": self.unresolved_pairs,
            "conflict_count": self.conflict_count,
            "unanimous_count": self.unanimous_count,
            "majority_count": self.majority_count,
            "senior_override_count": self.senior_override_count,
            "single_reviewer_count": self.single_reviewer_count,
        }


def resolve_pair_consensus(
    records: tuple[AdjudicationRecord, ...],
    *,
    policy: ConsensusPolicy = "majority_vote",
    senior_reviewers: frozenset[str] | set[str] | Sequence[str] = frozenset(),
    min_reviewers: int = 1,
) -> ConsensusResult:
    """Resolve adjudication consensus for multiple reviews on a single pair."""
    if not records:
        raise AdjudicationError(
            "ML-ADJ-015", "Cannot resolve consensus on an empty set of adjudication reviews."
        )

    if policy not in _ALLOWED_POLICIES:
        raise AdjudicationError("ML-ADJ-016", f"Unsupported consensus policy: {policy}")

    first = records[0]
    pair_digest = first.pair_digest()
    for rec in records[1:]:
        if rec.pair_digest() != pair_digest:
            raise AdjudicationError(
                "ML-ADJ-015", "Adjudication reviews in a single resolution must share pair digest."
            )

    senior_set = frozenset(senior_reviewers)
    event_ids = tuple(rec.event_id for rec in records)

    # Collect entity and household component digests across reviews
    entity_digests: set[str] = set()
    household_digests: set[str] = set()
    for rec in records:
        entity_digests.update(rec.entity_component_digests)
        household_digests.update(rec.household_component_digests)

    protocol_versions = {rec.protocol_version for rec in records}
    protocol_version = next(iter(protocol_versions)) if len(protocol_versions) == 1 else "mixed"

    unique_outcomes = {rec.decision for rec in records}
    has_conflict = len(unique_outcomes) > 1

    # Check minimum reviewer requirement
    effective_min_reviewers = max(min_reviewers, 2 if policy == "strict_double_review" else 1)
    if len(records) < effective_min_reviewers:
        reason = (
            "insufficient_reviewers_with_conflict" if has_conflict else "insufficient_reviewers"
        )
        return ConsensusResult(
            pair_digest=pair_digest,
            left_record_key=first.left_record_key,
            right_record_key=first.right_record_key,
            consensus_outcome=None,
            consensus_confidence=0.0,
            resolution_method="unresolved",
            is_resolved=False,
            has_conflict=has_conflict,
            reviewer_count=len(records),
            reviewing_event_ids=event_ids,
            dispute_reason=reason,
            entity_component_digests=tuple(sorted(entity_digests)),
            household_component_digests=tuple(sorted(household_digests)),
            protocol_version=protocol_version,
        )

    # Case 1: Unanimous agreement across all reviewers
    if not has_conflict:
        avg_confidence = sum(rec.confidence for rec in records) / len(records)
        method: ConsensusMethod = "unanimous" if len(records) > 1 else "single_reviewer"
        return ConsensusResult(
            pair_digest=pair_digest,
            left_record_key=first.left_record_key,
            right_record_key=first.right_record_key,
            consensus_outcome=first.decision,
            consensus_confidence=avg_confidence,
            resolution_method=method,
            is_resolved=True,
            has_conflict=False,
            reviewer_count=len(records),
            reviewing_event_ids=event_ids,
            entity_component_digests=tuple(sorted(entity_digests)),
            household_component_digests=tuple(sorted(household_digests)),
            protocol_version=protocol_version,
        )

    # Case 2: Conflict exists - check Senior Reviewer Override
    senior_records = [rec for rec in records if rec.reviewer_id in senior_set]
    if senior_records and policy in (
        "senior_reviewer_override",
        "majority_vote",
        "strict_double_review",
    ):
        senior_outcomes = {rec.decision for rec in senior_records}
        if len(senior_outcomes) == 1:
            senior_outcome = senior_records[0].decision
            senior_conf = sum(rec.confidence for rec in senior_records) / len(senior_records)
            return ConsensusResult(
                pair_digest=pair_digest,
                left_record_key=first.left_record_key,
                right_record_key=first.right_record_key,
                consensus_outcome=senior_outcome,
                consensus_confidence=senior_conf,
                resolution_method="senior_override",
                is_resolved=True,
                has_conflict=True,
                reviewer_count=len(records),
                reviewing_event_ids=event_ids,
                senior_reviewer_id=senior_records[0].reviewer_id,
                entity_component_digests=tuple(sorted(entity_digests)),
                household_component_digests=tuple(sorted(household_digests)),
                protocol_version=protocol_version,
            )
        # Conflicting senior reviewers
        return ConsensusResult(
            pair_digest=pair_digest,
            left_record_key=first.left_record_key,
            right_record_key=first.right_record_key,
            consensus_outcome=None,
            consensus_confidence=0.0,
            resolution_method="unresolved",
            is_resolved=False,
            has_conflict=True,
            reviewer_count=len(records),
            reviewing_event_ids=event_ids,
            dispute_reason="conflicting_senior_reviewers",
            entity_component_digests=tuple(sorted(entity_digests)),
            household_component_digests=tuple(sorted(household_digests)),
            protocol_version=protocol_version,
        )

    # Case 3: Majority Vote policy without senior override
    if policy == "majority_vote":
        counts = Counter(rec.decision for rec in records)
        top_outcome, top_count = counts.most_common(1)[0]
        total_votes = len(records)
        # Check strict majority (> 50%)
        if top_count > total_votes / 2:
            majority_recs = [rec for rec in records if rec.decision == top_outcome]
            avg_confidence = sum(rec.confidence for rec in majority_recs) / len(majority_recs)
            return ConsensusResult(
                pair_digest=pair_digest,
                left_record_key=first.left_record_key,
                right_record_key=first.right_record_key,
                consensus_outcome=top_outcome,
                consensus_confidence=avg_confidence,
                resolution_method="majority",
                is_resolved=True,
                has_conflict=True,
                reviewer_count=len(records),
                reviewing_event_ids=event_ids,
                entity_component_digests=tuple(sorted(entity_digests)),
                household_component_digests=tuple(sorted(household_digests)),
                protocol_version=protocol_version,
            )
        # Tied or split vote without majority
        return ConsensusResult(
            pair_digest=pair_digest,
            left_record_key=first.left_record_key,
            right_record_key=first.right_record_key,
            consensus_outcome=None,
            consensus_confidence=0.0,
            resolution_method="unresolved",
            is_resolved=False,
            has_conflict=True,
            reviewer_count=len(records),
            reviewing_event_ids=event_ids,
            dispute_reason="tied_vote",
            entity_component_digests=tuple(sorted(entity_digests)),
            household_component_digests=tuple(sorted(household_digests)),
            protocol_version=protocol_version,
        )

    # Case 4: Unanimous Only or Strict Double Review with conflict
    dispute_reason = (
        "unanimous_policy_conflict" if policy == "unanimous_only" else "double_review_conflict"
    )
    return ConsensusResult(
        pair_digest=pair_digest,
        left_record_key=first.left_record_key,
        right_record_key=first.right_record_key,
        consensus_outcome=None,
        consensus_confidence=0.0,
        resolution_method="unresolved",
        is_resolved=False,
        has_conflict=True,
        reviewer_count=len(records),
        reviewing_event_ids=event_ids,
        dispute_reason=dispute_reason,
        entity_component_digests=tuple(sorted(entity_digests)),
        household_component_digests=tuple(sorted(household_digests)),
        protocol_version=protocol_version,
    )


def evaluate_disagreements(
    records: Iterable[AdjudicationRecord],
    *,
    policy: ConsensusPolicy = "majority_vote",
    senior_reviewers: frozenset[str] | set[str] | Sequence[str] = frozenset(),
    min_reviewers: int = 1,
) -> tuple[tuple[ConsensusResult, ...], DisagreementReport]:
    """Evaluate and resolve adjudication consensus across a batch of adjudication records."""
    # Group records by pair_digest
    pairs_map: dict[str, list[AdjudicationRecord]] = {}
    for rec in records:
        pairs_map.setdefault(rec.pair_digest(), []).append(rec)

    results: list[ConsensusResult] = []
    senior_set = frozenset(senior_reviewers)

    for pair_digest in sorted(pairs_map.keys()):
        pair_records = tuple(pairs_map[pair_digest])
        consensus = resolve_pair_consensus(
            pair_records,
            policy=policy,
            senior_reviewers=senior_set,
            min_reviewers=min_reviewers,
        )
        results.append(consensus)

    resolved_count = sum(1 for r in results if r.is_resolved)
    unresolved_count = len(results) - resolved_count
    conflict_count = sum(1 for r in results if r.has_conflict)
    unanimous_count = sum(1 for r in results if r.resolution_method == "unanimous")
    majority_count = sum(1 for r in results if r.resolution_method == "majority")
    senior_override_count = sum(1 for r in results if r.resolution_method == "senior_override")
    single_reviewer_count = sum(1 for r in results if r.resolution_method == "single_reviewer")

    report = DisagreementReport(
        total_pairs=len(results),
        resolved_pairs=resolved_count,
        unresolved_pairs=unresolved_count,
        conflict_count=conflict_count,
        unanimous_count=unanimous_count,
        majority_count=majority_count,
        senior_override_count=senior_override_count,
        single_reviewer_count=single_reviewer_count,
    )

    return tuple(results), report
