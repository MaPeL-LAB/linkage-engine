"""Sparse and dense assignment solvers supporting multiple constraint modes."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from mapel_linkage.assignment.contracts import (
    AssignedEdge,
    AssignmentEdgeBatch,
    AssignmentPlan,
    AssignmentResult,
)
from mapel_linkage.domain.errors import AssignmentError

try:
    from ortools.graph.python import min_cost_flow
except ImportError:  # pragma: no cover - optional dependency boundary
    min_cost_flow = None

_min_cost_flow: Any = min_cost_flow

_TIE_SPAN = 1_000_003
_MAX_ABS_LOGIT = 30.0


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utility(probability: float) -> float:
    clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
    return max(min(math.log(clipped / (1.0 - clipped)), _MAX_ABS_LOGIT), -_MAX_ABS_LOGIT)


def _tie_rank(pair_digest: str) -> int:
    return int(pair_digest[:12], 16) % _TIE_SPAN


def _cost(utility: float, pair_digest: str, plan: AssignmentPlan) -> int:
    base = -round(utility * plan.utility_scale)
    return base * _TIE_SPAN + _tie_rank(pair_digest)


def _no_match_digest(source_key: str) -> str:
    return hashlib.sha256(f"no-match\x00{source_key}".encode()).hexdigest()


def _candidate_index(batch: AssignmentEdgeBatch) -> dict[tuple[str, str], int]:
    return {pair: index for index, pair in enumerate(batch.pair_references)}


def _top1(batch: AssignmentEdgeBatch, plan: AssignmentPlan) -> dict[str, str | None]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, (left, _) in enumerate(batch.pair_references):
        grouped[left].append(index)
    output: dict[str, str | None] = {}
    for source in batch.source_record_keys:
        indices = grouped.get(source, [])
        if not indices:
            output[source] = None
            continue
        best = min(
            indices,
            key=lambda index: (
                -_utility(float(batch.probabilities[index])),
                batch.pair_digests[index],
            ),
        )
        if _utility(float(batch.probabilities[best])) <= plan.no_match_utility:
            output[source] = None
        else:
            output[source] = batch.pair_references[best][1]
    return output


def _result(
    batch: AssignmentEdgeBatch,
    plan: AssignmentPlan,
    assignments: Iterable[AssignedEdge],
    *,
    solver: str,
    constraint: (
        Literal["one_to_one", "many_to_one", "one_to_many", "unconstrained"] | None
    ) = None,
) -> AssignmentResult:
    constraint_mode = constraint or plan.constraint
    raw_ordered = tuple(
        sorted(
            assignments,
            key=lambda item: (
                item.source_record_key,
                "" if item.selected_no_match else (item.target_record_key or ""),
            ),
        )
    )
    independent = _top1(batch, plan)
    ordered = tuple(
        replace(
            item,
            changed_from_independent_top1=(
                item.target_record_key != independent[item.source_record_key]
            ),
        )
        for item in raw_ordered
    )
    sources_in_assignments = {item.source_record_key for item in ordered}
    if len(sources_in_assignments) != len(batch.source_record_keys):
        raise AssignmentError("ML-ASSIGN-014", "Assignment coverage is incomplete.")
    if constraint_mode in ("one_to_one", "many_to_one") and len(ordered) != len(
        sources_in_assignments
    ):
        raise AssignmentError("ML-ASSIGN-018", "A source record was assigned more than once.")

    real_targets = [item.target_record_key for item in ordered if not item.selected_no_match]
    violations = 0
    if constraint_mode in ("one_to_one", "one_to_many"):
        violations = len(real_targets) - len(set(real_targets))

    changed = sum(item.changed_from_independent_top1 for item in ordered)
    payload = [
        {
            "source_digest": hashlib.sha256(item.source_record_key.encode("utf-8")).hexdigest(),
            "pair_digest": item.pair_digest,
            "selected_no_match": item.selected_no_match,
            "candidate_rank": item.candidate_rank,
        }
        for item in ordered
    ]
    digest = _canonical_digest(
        {
            "solver": solver,
            "constraint": constraint_mode,
            "source_model_id": batch.source_model_id,
            "source_model_version": batch.source_model_version,
            "calibrator_digest": batch.calibrator_digest,
            "assignments": payload,
        }
    )
    return AssignmentResult(
        assignments=ordered,
        solver=solver,
        constraint=constraint_mode,
        source_record_count=len(batch.source_record_keys),
        candidate_pair_count=batch.candidate_pair_count,
        real_assignment_count=sum(not item.selected_no_match for item in ordered),
        no_match_count=sum(item.selected_no_match for item in ordered),
        target_record_count=len(set(real_targets)),
        changed_from_independent_top1_count=changed,
        constraint_violation_count=violations,
        assignment_digest=digest,
    )


class ScipyOneToOneAssignmentSolver:
    """Dense reference solver for small deterministic one-to-one problems."""

    @staticmethod
    def solve(batch: AssignmentEdgeBatch, plan: AssignmentPlan) -> AssignmentResult:
        if batch.candidate_pair_count > plan.maximum_candidate_edges:
            raise AssignmentError("ML-ASSIGN-019", "The assignment candidate budget was exceeded.")
        sources = tuple(sorted(batch.source_record_keys))
        targets = tuple(sorted({right for _, right in batch.pair_references}))
        target_index = {target: index for index, target in enumerate(targets)}
        source_index = {source: index for index, source in enumerate(sources)}
        columns = len(targets) + len(sources)
        unavailable = 10**18
        matrix = np.full((len(sources), columns), unavailable, dtype=np.int64)
        candidate_lookup = _candidate_index(batch)
        for index, (left, right) in enumerate(batch.pair_references):
            matrix[source_index[left], target_index[right]] = _cost(
                _utility(float(batch.probabilities[index])), batch.pair_digests[index], plan
            )
        for row, source in enumerate(sources):
            matrix[row, len(targets) + row] = _cost(
                plan.no_match_utility, _no_match_digest(source), plan
            )
        row_indices, column_indices = linear_sum_assignment(matrix)
        assignments: list[AssignedEdge] = []
        for row, column in zip(row_indices, column_indices, strict=True):
            source = sources[int(row)]
            if column >= len(targets):
                assignments.append(
                    AssignedEdge(
                        source_record_key=source,
                        target_record_key=None,
                        pair_digest=None,
                        calibrated_probability=None,
                        candidate_rank=None,
                        selected_no_match=True,
                        assignment_utility=plan.no_match_utility,
                    )
                )
                continue
            target = targets[int(column)]
            index = candidate_lookup[(source, target)]
            assignments.append(
                AssignedEdge(
                    source_record_key=source,
                    target_record_key=target,
                    pair_digest=batch.pair_digests[index],
                    calibrated_probability=float(batch.probabilities[index]),
                    candidate_rank=int(batch.candidate_ranks[index]),
                    selected_no_match=False,
                    assignment_utility=_utility(float(batch.probabilities[index])),
                )
            )
        return _result(
            batch,
            plan,
            assignments,
            solver="scipy_linear_sum_assignment",
            constraint="one_to_one",
        )


class OrToolsOneToOneAssignmentSolver:
    """Sparse production assignment using OR-Tools minimum-cost flow."""

    @staticmethod
    def solve(batch: AssignmentEdgeBatch, plan: AssignmentPlan) -> AssignmentResult:
        if _min_cost_flow is None:
            raise AssignmentError(
                "ML-ASSIGN-020", "The OR-Tools assignment dependency is unavailable."
            )
        if batch.candidate_pair_count > plan.maximum_candidate_edges:
            raise AssignmentError("ML-ASSIGN-019", "The assignment candidate budget was exceeded.")
        sources = tuple(sorted(batch.source_record_keys))
        targets = tuple(sorted({right for _, right in batch.pair_references}))
        source_nodes = {source: 1 + index for index, source in enumerate(sources)}
        target_offset = 1 + len(sources)
        target_nodes = {target: target_offset + index for index, target in enumerate(targets)}
        dummy_offset = target_offset + len(targets)
        dummy_nodes = {source: dummy_offset + index for index, source in enumerate(sources)}
        sink = dummy_offset + len(sources)
        super_source = 0
        flow = _min_cost_flow.SimpleMinCostFlow()
        arc_metadata: dict[int, tuple[str, str | None, int | None]] = {}
        for source in sources:
            flow.add_arc_with_capacity_and_unit_cost(super_source, source_nodes[source], 1, 0)
        for target in targets:
            flow.add_arc_with_capacity_and_unit_cost(target_nodes[target], sink, 1, 0)
        for source in sources:
            flow.add_arc_with_capacity_and_unit_cost(dummy_nodes[source], sink, 1, 0)
        for index, (left, right) in enumerate(batch.pair_references):
            arc = flow.add_arc_with_capacity_and_unit_cost(
                source_nodes[left],
                target_nodes[right],
                1,
                _cost(_utility(float(batch.probabilities[index])), batch.pair_digests[index], plan),
            )
            arc_metadata[int(arc)] = (left, right, index)
        for source in sources:
            arc = flow.add_arc_with_capacity_and_unit_cost(
                source_nodes[source],
                dummy_nodes[source],
                1,
                _cost(plan.no_match_utility, _no_match_digest(source), plan),
            )
            arc_metadata[int(arc)] = (source, None, None)
        flow.set_node_supply(super_source, len(sources))
        flow.set_node_supply(sink, -len(sources))
        status = flow.solve()
        if status != flow.OPTIMAL:
            raise AssignmentError(
                "ML-ASSIGN-021", "The one-to-one assignment solver did not converge."
            )
        assignments: list[AssignedEdge] = []
        for arc, (source, arc_target, candidate_index) in arc_metadata.items():
            if flow.flow(arc) != 1:
                continue
            if arc_target is None or candidate_index is None:
                assignments.append(
                    AssignedEdge(
                        source_record_key=source,
                        target_record_key=None,
                        pair_digest=None,
                        calibrated_probability=None,
                        candidate_rank=None,
                        selected_no_match=True,
                        assignment_utility=plan.no_match_utility,
                    )
                )
            else:
                assignments.append(
                    AssignedEdge(
                        source_record_key=source,
                        target_record_key=arc_target,
                        pair_digest=batch.pair_digests[candidate_index],
                        calibrated_probability=float(batch.probabilities[candidate_index]),
                        candidate_rank=int(batch.candidate_ranks[candidate_index]),
                        selected_no_match=False,
                        assignment_utility=_utility(float(batch.probabilities[candidate_index])),
                    )
                )
        return _result(
            batch,
            plan,
            assignments,
            solver="ortools_min_cost_flow",
            constraint="one_to_one",
        )


class ManyToOneAssignmentSolver:
    """Many-to-one assignment solver allowing multiple sources to match a single target."""

    @staticmethod
    def solve(batch: AssignmentEdgeBatch, plan: AssignmentPlan) -> AssignmentResult:
        if batch.candidate_pair_count > plan.maximum_candidate_edges:
            raise AssignmentError("ML-ASSIGN-019", "The assignment candidate budget was exceeded.")
        sources = tuple(sorted(batch.source_record_keys))
        targets = tuple(sorted({right for _, right in batch.pair_references}))
        target_capacity = plan.max_matches_per_target

        if target_capacity >= len(sources):
            grouped: dict[str, list[int]] = defaultdict(list)
            for index, (left, _) in enumerate(batch.pair_references):
                grouped[left].append(index)

            assignments: list[AssignedEdge] = []
            for source in sources:
                indices = grouped.get(source, [])
                if not indices:
                    assignments.append(
                        AssignedEdge(
                            source_record_key=source,
                            target_record_key=None,
                            pair_digest=None,
                            calibrated_probability=None,
                            candidate_rank=None,
                            selected_no_match=True,
                            assignment_utility=plan.no_match_utility,
                        )
                    )
                    continue
                best = min(
                    indices,
                    key=lambda idx: _cost(
                        _utility(float(batch.probabilities[idx])),
                        batch.pair_digests[idx],
                        plan,
                    ),
                )
                best_prob = float(batch.probabilities[best])
                best_util = _utility(best_prob)
                if best_util <= plan.no_match_utility:
                    assignments.append(
                        AssignedEdge(
                            source_record_key=source,
                            target_record_key=None,
                            pair_digest=None,
                            calibrated_probability=None,
                            candidate_rank=None,
                            selected_no_match=True,
                            assignment_utility=plan.no_match_utility,
                        )
                    )
                else:
                    target = batch.pair_references[best][1]
                    assignments.append(
                        AssignedEdge(
                            source_record_key=source,
                            target_record_key=target,
                            pair_digest=batch.pair_digests[best],
                            calibrated_probability=best_prob,
                            candidate_rank=int(batch.candidate_ranks[best]),
                            selected_no_match=False,
                            assignment_utility=best_util,
                        )
                    )
            return _result(
                batch,
                plan,
                assignments,
                solver="many_to_one",
                constraint="many_to_one",
            )

        if _min_cost_flow is not None:
            source_nodes = {source: 1 + index for index, source in enumerate(sources)}
            target_offset = 1 + len(sources)
            target_nodes = {target: target_offset + index for index, target in enumerate(targets)}
            dummy_offset = target_offset + len(targets)
            dummy_nodes = {source: dummy_offset + index for index, source in enumerate(sources)}
            sink = dummy_offset + len(sources)
            super_source = 0

            flow = _min_cost_flow.SimpleMinCostFlow()
            arc_metadata: dict[int, tuple[str, str | None, int | None]] = {}
            for source in sources:
                flow.add_arc_with_capacity_and_unit_cost(super_source, source_nodes[source], 1, 0)
            for target in targets:
                flow.add_arc_with_capacity_and_unit_cost(
                    target_nodes[target], sink, target_capacity, 0
                )
            for source in sources:
                flow.add_arc_with_capacity_and_unit_cost(dummy_nodes[source], sink, 1, 0)
            for index, (left, right) in enumerate(batch.pair_references):
                arc = flow.add_arc_with_capacity_and_unit_cost(
                    source_nodes[left],
                    target_nodes[right],
                    1,
                    _cost(
                        _utility(float(batch.probabilities[index])),
                        batch.pair_digests[index],
                        plan,
                    ),
                )
                arc_metadata[int(arc)] = (left, right, index)
            for source in sources:
                arc = flow.add_arc_with_capacity_and_unit_cost(
                    source_nodes[source],
                    dummy_nodes[source],
                    1,
                    _cost(plan.no_match_utility, _no_match_digest(source), plan),
                )
                arc_metadata[int(arc)] = (source, None, None)

            flow.set_node_supply(super_source, len(sources))
            flow.set_node_supply(sink, -len(sources))
            status = flow.solve()
            if status != flow.OPTIMAL:
                raise AssignmentError(
                    "ML-ASSIGN-021", "The many-to-one assignment solver did not converge."
                )

            assignments = []
            for arc, (source, arc_target, candidate_index) in arc_metadata.items():
                if flow.flow(arc) != 1:
                    continue
                if arc_target is None or candidate_index is None:
                    assignments.append(
                        AssignedEdge(
                            source_record_key=source,
                            target_record_key=None,
                            pair_digest=None,
                            calibrated_probability=None,
                            candidate_rank=None,
                            selected_no_match=True,
                            assignment_utility=plan.no_match_utility,
                        )
                    )
                else:
                    assignments.append(
                        AssignedEdge(
                            source_record_key=source,
                            target_record_key=arc_target,
                            pair_digest=batch.pair_digests[candidate_index],
                            calibrated_probability=float(batch.probabilities[candidate_index]),
                            candidate_rank=int(batch.candidate_ranks[candidate_index]),
                            selected_no_match=False,
                            assignment_utility=_utility(
                                float(batch.probabilities[candidate_index])
                            ),
                        )
                    )
            return _result(
                batch,
                plan,
                assignments,
                solver="many_to_one",
                constraint="many_to_one",
            )

        candidate_indices = sorted(
            range(len(batch.pair_references)),
            key=lambda idx: _cost(
                _utility(float(batch.probabilities[idx])), batch.pair_digests[idx], plan
            ),
        )
        assigned_sources: set[str] = set()
        target_counts: dict[str, int] = defaultdict(int)
        greedy_assignments: list[AssignedEdge] = []
        for idx in candidate_indices:
            left, right = batch.pair_references[idx]
            if left in assigned_sources:
                continue
            prob = float(batch.probabilities[idx])
            util = _utility(prob)
            if util <= plan.no_match_utility:
                continue
            if target_counts[right] < target_capacity:
                target_counts[right] += 1
                assigned_sources.add(left)
                greedy_assignments.append(
                    AssignedEdge(
                        source_record_key=left,
                        target_record_key=right,
                        pair_digest=batch.pair_digests[idx],
                        calibrated_probability=prob,
                        candidate_rank=int(batch.candidate_ranks[idx]),
                        selected_no_match=False,
                        assignment_utility=util,
                    )
                )
        for source in sources:
            if source not in assigned_sources:
                greedy_assignments.append(
                    AssignedEdge(
                        source_record_key=source,
                        target_record_key=None,
                        pair_digest=None,
                        calibrated_probability=None,
                        candidate_rank=None,
                        selected_no_match=True,
                        assignment_utility=plan.no_match_utility,
                    )
                )
        return _result(
            batch,
            plan,
            greedy_assignments,
            solver="many_to_one",
            constraint="many_to_one",
        )


class OneToManyAssignmentSolver:
    """One-to-many assignment solver allowing a single source to match multiple targets."""

    @staticmethod
    def solve(batch: AssignmentEdgeBatch, plan: AssignmentPlan) -> AssignmentResult:
        if batch.candidate_pair_count > plan.maximum_candidate_edges:
            raise AssignmentError("ML-ASSIGN-019", "The assignment candidate budget was exceeded.")

        sources = tuple(sorted(batch.source_record_keys))
        targets = tuple(sorted({right for _, right in batch.pair_references}))
        source_capacity = plan.max_matches_per_source

        if source_capacity >= len(targets):
            grouped_by_target: dict[str, list[int]] = defaultdict(list)
            for index, (_, right) in enumerate(batch.pair_references):
                grouped_by_target[right].append(index)

            assigned_targets_by_source: dict[str, list[int]] = defaultdict(list)
            for target in targets:
                indices = grouped_by_target[target]
                best = min(
                    indices,
                    key=lambda idx: _cost(
                        _utility(float(batch.probabilities[idx])),
                        batch.pair_digests[idx],
                        plan,
                    ),
                )
                best_prob = float(batch.probabilities[best])
                best_util = _utility(best_prob)
                if best_util > plan.no_match_utility:
                    source = batch.pair_references[best][0]
                    assigned_targets_by_source[source].append(best)

            assignments: list[AssignedEdge] = []
            for source in sources:
                matched_indices = assigned_targets_by_source.get(source, [])
                if not matched_indices:
                    assignments.append(
                        AssignedEdge(
                            source_record_key=source,
                            target_record_key=None,
                            pair_digest=None,
                            calibrated_probability=None,
                            candidate_rank=None,
                            selected_no_match=True,
                            assignment_utility=plan.no_match_utility,
                        )
                    )
                else:
                    for idx in matched_indices:
                        target = batch.pair_references[idx][1]
                        prob = float(batch.probabilities[idx])
                        assignments.append(
                            AssignedEdge(
                                source_record_key=source,
                                target_record_key=target,
                                pair_digest=batch.pair_digests[idx],
                                calibrated_probability=prob,
                                candidate_rank=int(batch.candidate_ranks[idx]),
                                selected_no_match=False,
                                assignment_utility=_utility(prob),
                            )
                        )
            return _result(
                batch,
                plan,
                assignments,
                solver="one_to_many",
                constraint="one_to_many",
            )

        if _min_cost_flow is not None:
            source_nodes = {source: 1 + index for index, source in enumerate(sources)}
            target_offset = 1 + len(sources)
            target_nodes = {target: target_offset + index for index, target in enumerate(targets)}
            dummy_target_offset = target_offset + len(targets)
            dummy_target_nodes = {
                target: dummy_target_offset + index for index, target in enumerate(targets)
            }
            sink = dummy_target_offset + len(targets)
            super_source = 0

            flow = _min_cost_flow.SimpleMinCostFlow()
            arc_metadata: dict[int, tuple[str, str, int]] = {}

            for source in sources:
                flow.add_arc_with_capacity_and_unit_cost(
                    super_source, source_nodes[source], source_capacity, 0
                )
            for target in targets:
                flow.add_arc_with_capacity_and_unit_cost(
                    super_source,
                    dummy_target_nodes[target],
                    1,
                    _cost(plan.no_match_utility, _no_match_digest(target), plan),
                )
                flow.add_arc_with_capacity_and_unit_cost(target_nodes[target], sink, 1, 0)
                flow.add_arc_with_capacity_and_unit_cost(dummy_target_nodes[target], sink, 1, 0)

            for index, (left, right) in enumerate(batch.pair_references):
                arc = flow.add_arc_with_capacity_and_unit_cost(
                    source_nodes[left],
                    target_nodes[right],
                    1,
                    _cost(
                        _utility(float(batch.probabilities[index])),
                        batch.pair_digests[index],
                        plan,
                    ),
                )
                arc_metadata[int(arc)] = (left, right, index)

            flow.set_node_supply(super_source, len(targets))
            flow.set_node_supply(sink, -len(targets))
            status = flow.solve()
            if status != flow.OPTIMAL:
                raise AssignmentError(
                    "ML-ASSIGN-021", "The one-to-many assignment solver did not converge."
                )

            assigned_targets_by_source = defaultdict(list)
            for arc, (source, _arc_target, candidate_index) in arc_metadata.items():
                if flow.flow(arc) == 1:
                    assigned_targets_by_source[source].append(candidate_index)

            assignments = []
            for source in sources:
                matched_indices = assigned_targets_by_source.get(source, [])
                if not matched_indices:
                    assignments.append(
                        AssignedEdge(
                            source_record_key=source,
                            target_record_key=None,
                            pair_digest=None,
                            calibrated_probability=None,
                            candidate_rank=None,
                            selected_no_match=True,
                            assignment_utility=plan.no_match_utility,
                        )
                    )
                else:
                    for idx in matched_indices:
                        target = batch.pair_references[idx][1]
                        prob = float(batch.probabilities[idx])
                        assignments.append(
                            AssignedEdge(
                                source_record_key=source,
                                target_record_key=target,
                                pair_digest=batch.pair_digests[idx],
                                calibrated_probability=prob,
                                candidate_rank=int(batch.candidate_ranks[idx]),
                                selected_no_match=False,
                                assignment_utility=_utility(prob),
                            )
                        )
            return _result(
                batch,
                plan,
                assignments,
                solver="one_to_many",
                constraint="one_to_many",
            )

        candidate_indices = sorted(
            range(len(batch.pair_references)),
            key=lambda idx: _cost(
                _utility(float(batch.probabilities[idx])), batch.pair_digests[idx], plan
            ),
        )
        assigned_targets: set[str] = set()
        source_counts: dict[str, int] = defaultdict(int)
        greedy_by_source: dict[str, list[int]] = defaultdict(list)
        for idx in candidate_indices:
            left, right = batch.pair_references[idx]
            if right in assigned_targets:
                continue
            prob = float(batch.probabilities[idx])
            util = _utility(prob)
            if util <= plan.no_match_utility:
                continue
            if source_counts[left] < source_capacity:
                source_counts[left] += 1
                assigned_targets.add(right)
                greedy_by_source[left].append(idx)

        greedy_assignments_otm: list[AssignedEdge] = []
        for source in sources:
            matched_indices = greedy_by_source.get(source, [])
            if not matched_indices:
                greedy_assignments_otm.append(
                    AssignedEdge(
                        source_record_key=source,
                        target_record_key=None,
                        pair_digest=None,
                        calibrated_probability=None,
                        candidate_rank=None,
                        selected_no_match=True,
                        assignment_utility=plan.no_match_utility,
                    )
                )
            else:
                for idx in matched_indices:
                    target = batch.pair_references[idx][1]
                    prob = float(batch.probabilities[idx])
                    greedy_assignments_otm.append(
                        AssignedEdge(
                            source_record_key=source,
                            target_record_key=target,
                            pair_digest=batch.pair_digests[idx],
                            calibrated_probability=prob,
                            candidate_rank=int(batch.candidate_ranks[idx]),
                            selected_no_match=False,
                            assignment_utility=_utility(prob),
                        )
                    )
        return _result(
            batch,
            plan,
            greedy_assignments_otm,
            solver="one_to_many",
            constraint="one_to_many",
        )


class UnconstrainedAssignmentSolver:
    """Unconstrained bipartite threshold selection with capacity limits."""

    @staticmethod
    def solve(batch: AssignmentEdgeBatch, plan: AssignmentPlan) -> AssignmentResult:
        if batch.candidate_pair_count > plan.maximum_candidate_edges:
            raise AssignmentError("ML-ASSIGN-019", "The assignment candidate budget was exceeded.")

        sources = tuple(sorted(batch.source_record_keys))
        source_capacity = plan.max_matches_per_source
        target_capacity = plan.max_matches_per_target

        candidate_indices = sorted(
            range(len(batch.pair_references)),
            key=lambda idx: (
                batch.pair_references[idx][0],
                _cost(_utility(float(batch.probabilities[idx])), batch.pair_digests[idx], plan),
            ),
        )

        source_counts: dict[str, int] = defaultdict(int)
        target_counts: dict[str, int] = defaultdict(int)
        selected_by_source: dict[str, list[int]] = defaultdict(list)

        for idx in candidate_indices:
            left, right = batch.pair_references[idx]
            prob = float(batch.probabilities[idx])
            util = _utility(prob)
            if util <= plan.no_match_utility:
                continue
            if source_counts[left] < source_capacity and target_counts[right] < target_capacity:
                source_counts[left] += 1
                target_counts[right] += 1
                selected_by_source[left].append(idx)

        assignments: list[AssignedEdge] = []
        for source in sources:
            matched_indices = selected_by_source.get(source, [])
            if not matched_indices:
                assignments.append(
                    AssignedEdge(
                        source_record_key=source,
                        target_record_key=None,
                        pair_digest=None,
                        calibrated_probability=None,
                        candidate_rank=None,
                        selected_no_match=True,
                        assignment_utility=plan.no_match_utility,
                    )
                )
            else:
                for idx in matched_indices:
                    target = batch.pair_references[idx][1]
                    prob = float(batch.probabilities[idx])
                    assignments.append(
                        AssignedEdge(
                            source_record_key=source,
                            target_record_key=target,
                            pair_digest=batch.pair_digests[idx],
                            calibrated_probability=prob,
                            candidate_rank=int(batch.candidate_ranks[idx]),
                            selected_no_match=False,
                            assignment_utility=_utility(prob),
                        )
                    )
        return _result(
            batch,
            plan,
            assignments,
            solver="unconstrained",
            constraint="unconstrained",
        )
