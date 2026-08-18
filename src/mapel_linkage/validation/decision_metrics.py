"""Aggregate four-status relationship-decision diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from mapel_linkage.decisions import RelationshipDecision
from mapel_linkage.domain.errors import ValidationReportError


@dataclass(frozen=True, slots=True)
class DecisionValidationReport:
    relationship_count: int
    confirmed_count: int
    review_required_count: int
    unresolved_count: int
    no_match_count: int
    status_counts: tuple[tuple[str, int], ...] = field(repr=False)
    evaluation_scope: str = "synthetic_mechanical_evaluation"
    merge_authority: str = "none"
    real_data_validation_status: str = "not_established"

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "relationship_count": self.relationship_count,
            "confirmed_count": self.confirmed_count,
            "review_required_count": self.review_required_count,
            "unresolved_count": self.unresolved_count,
            "no_match_count": self.no_match_count,
            "evaluation_scope": self.evaluation_scope,
            "merge_authority": self.merge_authority,
            "real_data_validation_status": self.real_data_validation_status,
        }


def evaluate_decisions(decisions: tuple[RelationshipDecision, ...]) -> DecisionValidationReport:
    if not decisions or len({decision.source_record_ref for decision in decisions}) != len(
        decisions
    ):
        raise ValidationReportError("ML-VALID-007", "Decision evaluation coverage is invalid.")
    counts = Counter(decision.relationship_status for decision in decisions)
    expected = {"confirmed", "review_required", "unresolved", "no_match"}
    if set(counts) - expected:
        raise ValidationReportError(
            "ML-VALID-008", "An unsupported relationship status was rejected."
        )
    return DecisionValidationReport(
        relationship_count=len(decisions),
        confirmed_count=counts["confirmed"],
        review_required_count=counts["review_required"],
        unresolved_count=counts["unresolved"],
        no_match_count=counts["no_match"],
        status_counts=tuple(sorted(counts.items())),
    )
