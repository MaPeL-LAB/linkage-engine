"""Constrained assignment contracts with explicit private no-match alternatives."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mapel_linkage.domain.errors import AssignmentError

_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True, slots=True, repr=False)
class AssignmentEdgeBatch:
    """Calibrated real candidate edges plus the complete source-record universe."""

    source_record_keys: tuple[str, ...] = field(repr=False)
    pair_references: tuple[tuple[str, str], ...] = field(repr=False)
    pair_digests: tuple[str, ...] = field(repr=False)
    probabilities: NDArray[np.float64] = field(repr=False)
    candidate_ranks: NDArray[np.int64] = field(repr=False)
    source_model_id: str
    source_model_version: str
    calibrator_digest: str
    ranking_model_digest: str | None
    candidate_search_complete: bool
    candidate_search_truncated: bool

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=np.float64).copy()
        ranks = np.asarray(self.candidate_ranks, dtype=np.int64).copy()
        probabilities.setflags(write=False)
        ranks.setflags(write=False)
        count = len(self.pair_references)
        if not self.source_record_keys or len(set(self.source_record_keys)) != len(
            self.source_record_keys
        ):
            raise AssignmentError("ML-ASSIGN-001", "The assignment source universe is invalid.")
        if len(self.pair_digests) != count or len(probabilities) != count or len(ranks) != count:
            raise AssignmentError("ML-ASSIGN-002", "Assignment candidate coverage is invalid.")
        if len(set(self.pair_references)) != count or len(set(self.pair_digests)) != count:
            raise AssignmentError(
                "ML-ASSIGN-003", "Duplicate assignment candidate edges were rejected."
            )
        if any(
            digest != pair_digest(left, right)
            for (left, right), digest in zip(
                self.pair_references,
                self.pair_digests,
                strict=True,
            )
        ):
            raise AssignmentError(
                "ML-ASSIGN-023", "An assignment pair digest does not match its edge."
            )
        source_set = set(self.source_record_keys)
        if any(left not in source_set for left, _ in self.pair_references):
            raise AssignmentError(
                "ML-ASSIGN-004", "An assignment edge has an unknown source record."
            )
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
        ):
            raise AssignmentError("ML-ASSIGN-005", "Assignment probabilities are invalid.")
        if np.any(ranks < 1):
            raise AssignmentError("ML-ASSIGN-006", "Assignment candidate ranks are invalid.")
        ranks_by_source: dict[str, list[int]] = {source: [] for source in self.source_record_keys}
        for (source, _), rank in zip(self.pair_references, ranks, strict=True):
            ranks_by_source[source].append(int(rank))
        if any(
            sorted(source_ranks) != list(range(1, len(source_ranks) + 1))
            for source_ranks in ranks_by_source.values()
        ):
            raise AssignmentError(
                "ML-ASSIGN-024", "Assignment candidate ranks are not complete per source."
            )
        for digest in (self.calibrator_digest, *self.pair_digests):
            if _DIGEST_PATTERN.fullmatch(digest) is None:
                raise AssignmentError("ML-ASSIGN-007", "An assignment digest is invalid.")
        if (
            self.ranking_model_digest is not None
            and _DIGEST_PATTERN.fullmatch(self.ranking_model_digest) is None
        ):
            raise AssignmentError("ML-ASSIGN-008", "A ranking-model digest is invalid.")
        if not self.source_model_id or not self.source_model_version:
            raise AssignmentError("ML-ASSIGN-025", "Assignment model provenance is incomplete.")
        if self.candidate_search_complete and self.candidate_search_truncated:
            raise AssignmentError(
                "ML-ASSIGN-026", "A truncated candidate search cannot be marked complete."
            )
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "candidate_ranks", ranks)

    @property
    def candidate_pair_count(self) -> int:
        return len(self.pair_references)

    def safe_summary(self) -> dict[str, int | str | bool]:
        return {
            "source_record_count": len(self.source_record_keys),
            "candidate_pair_count": self.candidate_pair_count,
            "source_model_id": self.source_model_id,
            "source_model_version": self.source_model_version,
            "calibrator_digest": self.calibrator_digest,
            "candidate_search_complete": self.candidate_search_complete,
            "candidate_search_truncated": self.candidate_search_truncated,
        }


@dataclass(frozen=True, slots=True)
class AssignmentPlan:
    constraint: Literal["one_to_one"] = "one_to_one"
    solver: Literal["ortools_min_cost_flow", "scipy_linear_sum_assignment", "unconstrained"] = (
        "ortools_min_cost_flow"
    )
    no_match_utility: float = 0.0
    utility_scale: int = 1_000_000
    maximum_candidate_edges: int = 10_000_000
    deterministic_tie_breaking: Literal[True] = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.no_match_utility):
            raise AssignmentError("ML-ASSIGN-009", "The no-match utility is invalid.")
        if self.utility_scale <= 0 or self.utility_scale > 1_000_000_000:
            raise AssignmentError("ML-ASSIGN-010", "The assignment utility scale is invalid.")
        if self.maximum_candidate_edges <= 0:
            raise AssignmentError("ML-ASSIGN-011", "The assignment edge budget is invalid.")


@dataclass(frozen=True, slots=True, repr=False)
class AssignedEdge:
    source_record_key: str = field(repr=False)
    target_record_key: str | None = field(repr=False)
    pair_digest: str | None = field(repr=False)
    calibrated_probability: float | None
    candidate_rank: int | None
    selected_no_match: bool
    assignment_utility: float
    changed_from_independent_top1: bool = False

    def __post_init__(self) -> None:
        if self.selected_no_match:
            if (
                self.target_record_key is not None
                or self.pair_digest is not None
                or self.calibrated_probability is not None
                or self.candidate_rank is not None
            ):
                raise AssignmentError(
                    "ML-ASSIGN-012", "A no-match assignment contains a real candidate."
                )
        elif (
            self.target_record_key is None
            or self.pair_digest is None
            or self.calibrated_probability is None
            or self.candidate_rank is None
        ):
            raise AssignmentError("ML-ASSIGN-013", "A real assignment is incomplete.")
        elif (
            self.pair_digest != pair_digest(self.source_record_key, self.target_record_key)
            or not math.isfinite(self.calibrated_probability)
            or not 0.0 <= self.calibrated_probability <= 1.0
            or self.candidate_rank < 1
        ):
            raise AssignmentError("ML-ASSIGN-027", "A real assignment is inconsistent.")
        if not math.isfinite(self.assignment_utility):
            raise AssignmentError("ML-ASSIGN-028", "Assignment utility is invalid.")


@dataclass(frozen=True, slots=True, repr=False)
class AssignmentResult:
    assignments: tuple[AssignedEdge, ...] = field(repr=False)
    solver: str
    constraint: Literal["one_to_one"]
    source_record_count: int
    candidate_pair_count: int
    real_assignment_count: int
    no_match_count: int
    target_record_count: int
    changed_from_independent_top1_count: int
    constraint_violation_count: int
    assignment_digest: str
    assignment_authority: Literal["global_selection_only"] = "global_selection_only"
    decision_authority: Literal["none"] = "none"

    def __post_init__(self) -> None:
        if len(self.assignments) != self.source_record_count:
            raise AssignmentError("ML-ASSIGN-014", "Assignment coverage is incomplete.")
        if self.real_assignment_count + self.no_match_count != self.source_record_count:
            raise AssignmentError("ML-ASSIGN-015", "Assignment aggregate counts are inconsistent.")
        if self.constraint_violation_count != 0:
            raise AssignmentError(
                "ML-ASSIGN-016", "A successful assignment cannot retain violations."
            )
        if _DIGEST_PATTERN.fullmatch(self.assignment_digest) is None:
            raise AssignmentError("ML-ASSIGN-017", "The assignment digest is invalid.")
        if len({item.source_record_key for item in self.assignments}) != len(self.assignments):
            raise AssignmentError("ML-ASSIGN-018", "A source record was assigned more than once.")
        real = tuple(item for item in self.assignments if not item.selected_no_match)
        if len({item.target_record_key for item in real}) != len(real):
            raise AssignmentError("ML-ASSIGN-016", "One-to-one assignment contains a conflict.")
        if (
            sum(not item.selected_no_match for item in self.assignments)
            != self.real_assignment_count
            or sum(item.selected_no_match for item in self.assignments) != self.no_match_count
            or len({item.target_record_key for item in real}) != self.target_record_count
            or sum(item.changed_from_independent_top1 for item in self.assignments)
            != self.changed_from_independent_top1_count
        ):
            raise AssignmentError("ML-ASSIGN-029", "Assignment aggregate counts are inconsistent.")

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "solver": self.solver,
            "constraint": self.constraint,
            "source_record_count": self.source_record_count,
            "candidate_pair_count": self.candidate_pair_count,
            "target_record_count": self.target_record_count,
            "real_assignment_count": self.real_assignment_count,
            "no_match_count": self.no_match_count,
            "changed_from_independent_top1_count": self.changed_from_independent_top1_count,
            "constraint_violation_count": self.constraint_violation_count,
            "assignment_digest": self.assignment_digest,
            "assignment_authority": self.assignment_authority,
            "decision_authority": self.decision_authority,
        }


def pair_digest(left: str, right: str) -> str:
    return hashlib.sha256(f"{left}\x00{right}".encode()).hexdigest()
