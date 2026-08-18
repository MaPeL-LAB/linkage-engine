"""Aggregate one-to-one assignment and no-match diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from mapel_linkage.assignment import AssignmentResult
from mapel_linkage.domain.errors import ValidationReportError


@dataclass(frozen=True, slots=True)
class AssignmentValidationReport:
    source_record_count: int
    true_match_count: int
    true_no_match_count: int
    correct_assignment_count: int
    false_assignment_count: int
    missed_match_count: int
    correct_no_match_count: int
    assignment_accuracy: float
    false_assignment_rate: float
    missed_match_rate: float
    no_match_accuracy: float
    changed_from_independent_top1_count: int
    constraint_violation_count: int
    evaluation_scope: str = "synthetic_mechanical_evaluation"
    real_data_validation_status: str = "not_established"

    def safe_summary(self) -> dict[str, int | float | str]:
        return {
            "source_record_count": self.source_record_count,
            "true_match_count": self.true_match_count,
            "true_no_match_count": self.true_no_match_count,
            "correct_assignment_count": self.correct_assignment_count,
            "false_assignment_count": self.false_assignment_count,
            "missed_match_count": self.missed_match_count,
            "correct_no_match_count": self.correct_no_match_count,
            "assignment_accuracy": self.assignment_accuracy,
            "false_assignment_rate": self.false_assignment_rate,
            "missed_match_rate": self.missed_match_rate,
            "no_match_accuracy": self.no_match_accuracy,
            "changed_from_independent_top1_count": self.changed_from_independent_top1_count,
            "constraint_violation_count": self.constraint_violation_count,
            "evaluation_scope": self.evaluation_scope,
            "real_data_validation_status": self.real_data_validation_status,
        }


def evaluate_assignment(
    *,
    assignment: AssignmentResult,
    true_target_by_source: dict[str, str | None],
) -> AssignmentValidationReport:
    assigned_by_source = {
        item.source_record_key: item.target_record_key for item in assignment.assignments
    }
    if set(assigned_by_source) != set(true_target_by_source):
        raise ValidationReportError("ML-VALID-006", "Assignment truth coverage is inconsistent.")
    true_match_count = sum(target is not None for target in true_target_by_source.values())
    true_no_match_count = len(true_target_by_source) - true_match_count
    correct_real = 0
    false_real = 0
    missed = 0
    correct_no_match = 0
    for source, truth in true_target_by_source.items():
        predicted = assigned_by_source[source]
        if truth is None:
            if predicted is None:
                correct_no_match += 1
            else:
                false_real += 1
        elif predicted == truth:
            correct_real += 1
        else:
            missed += 1
            if predicted is not None:
                false_real += 1
    total = len(true_target_by_source)
    return AssignmentValidationReport(
        source_record_count=total,
        true_match_count=true_match_count,
        true_no_match_count=true_no_match_count,
        correct_assignment_count=correct_real,
        false_assignment_count=false_real,
        missed_match_count=missed,
        correct_no_match_count=correct_no_match,
        assignment_accuracy=(correct_real + correct_no_match) / total if total else 0.0,
        false_assignment_rate=false_real / max(1, assignment.real_assignment_count),
        missed_match_rate=missed / max(1, true_match_count),
        no_match_accuracy=correct_no_match / max(1, true_no_match_count),
        changed_from_independent_top1_count=assignment.changed_from_independent_top1_count,
        constraint_violation_count=assignment.constraint_violation_count,
    )
