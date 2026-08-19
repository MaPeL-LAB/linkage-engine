"""Structural Pareto analysis without empirical performance claims."""

from __future__ import annotations

from collections.abc import Iterable

from mapel_linkage.recommendation.contracts import StructuralPipelineCandidate


def _dominates(
    left: StructuralPipelineCandidate,
    right: StructuralPipelineCandidate,
) -> bool:
    """Return whether left structurally dominates right.

    Costs are minimized and capability/interpretability/portability attributes are maximized.
    These are static design attributes, not estimates of sensitivity, PPV, calibration, or
    operational accuracy.
    """

    left_costs = (
        int(left.requires_verified_labels),
        len(left.required_runtimes),
        left.structural_complexity,
    )
    right_costs = (
        int(right.requires_verified_labels),
        len(right.required_runtimes),
        right.structural_complexity,
    )
    left_benefits = (
        left.interaction_capacity,
        left.interpretability_score,
        left.artifact_portability_score,
    )
    right_benefits = (
        right.interaction_capacity,
        right.interpretability_score,
        right.artifact_portability_score,
    )
    no_worse = all(a <= b for a, b in zip(left_costs, right_costs, strict=True)) and all(
        a >= b for a, b in zip(left_benefits, right_benefits, strict=True)
    )
    strictly_better = any(a < b for a, b in zip(left_costs, right_costs, strict=True)) or any(
        a > b for a, b in zip(left_benefits, right_benefits, strict=True)
    )
    return no_worse and strictly_better


def structural_pareto_frontier(
    candidates: Iterable[StructuralPipelineCandidate],
) -> tuple[StructuralPipelineCandidate, ...]:
    """Return deterministic non-dominated structural recipe templates."""

    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    return tuple(
        candidate
        for candidate in ordered
        if not any(
            other.candidate_id != candidate.candidate_id and _dominates(other, candidate)
            for other in ordered
        )
    )


def build_diverse_shortlist(
    candidates: Iterable[StructuralPipelineCandidate],
    *,
    mandatory_baseline_id: str,
    maximum_challengers: int,
) -> tuple[StructuralPipelineCandidate, ...]:
    """Retain the baseline and structurally diverse challenger families deterministically."""

    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    baseline = next(
        (item for item in ordered if item.pair_model_id == mandatory_baseline_id),
        None,
    )
    if baseline is None:
        return ()

    frontier = structural_pareto_frontier(ordered)
    limit = maximum_challengers + 1
    selected: list[StructuralPipelineCandidate] = [baseline]
    seen_families = {baseline.pair_model_family}

    for candidate in frontier:
        if candidate.candidate_id == baseline.candidate_id:
            continue
        if candidate.pair_model_family in seen_families:
            continue
        selected.append(candidate)
        seen_families.add(candidate.pair_model_family)
        if len(selected) >= limit:
            return tuple(selected)

    for candidate in frontier:
        if candidate not in selected:
            selected.append(candidate)
            if len(selected) >= limit:
                return tuple(selected)

    for candidate in ordered:
        if candidate not in selected:
            selected.append(candidate)
            if len(selected) >= limit:
                break
    return tuple(selected)


__all__ = ["build_diverse_shortlist", "structural_pareto_frontier"]
