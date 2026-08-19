"""Configuration-only preflight profiling without reading row-level inputs."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from mapel_linkage.configuration.compiler import ExecutionPlan
from mapel_linkage.profiling.contracts import (
    CountBand,
    LabelEvidenceClass,
    PreflightTaskProfile,
    ProfileScope,
    VariableTypeCount,
)


def _candidate_budget_band(value: int) -> CountBand:
    if value <= 100_000:
        return CountBand.SMALL
    if value <= 1_000_000:
        return CountBand.MEDIUM
    if value <= 10_000_000:
        return CountBand.LARGE
    return CountBand.VERY_LARGE


def _label_evidence(plan: ExecutionPlan) -> LabelEvidenceClass:
    labels = plan.config.labels
    if labels is None:
        return LabelEvidenceClass.NONE
    return LabelEvidenceClass(labels.source.kind)


def build_preflight_task_profile(
    plan: ExecutionPlan,
    *,
    profile_scope: ProfileScope = ProfileScope.LOCAL_RESTRICTED,
) -> PreflightTaskProfile:
    """Build an aggregate profile from validated configuration only.

    Record counts, missingness rates, uniqueness, candidate graph properties, and model-score
    evidence are deliberately not inferred at this stage. They require later restricted stages.
    """

    role_counts = Counter(dataset.role for dataset in plan.config.datasets)
    type_counts = Counter(variable.data_type for variable in plan.config.variables)
    variable_types: tuple[
        Literal["string", "date", "integer", "float", "boolean", "categorical"], ...
    ] = ("boolean", "categorical", "date", "float", "integer", "string")
    evidence_class = _label_evidence(plan)
    verified = evidence_class in {
        LabelEvidenceClass.SYNTHETIC_TRUTH,
        LabelEvidenceClass.VERIFIED_HUMAN_ADJUDICATION,
        LabelEvidenceClass.VERIFIED_GOLD_STANDARD,
    }
    return PreflightTaskProfile(
        profile_scope=profile_scope,
        linkage_mode=plan.config.project.linkage_mode,
        assignment_constraint=plan.config.project.assignment_constraint,
        dataset_count=len(plan.config.datasets),
        source_count=role_counts["source"],
        target_count=role_counts["target"],
        reference_count=role_counts["reference"],
        auxiliary_count=role_counts["auxiliary"],
        variable_count=len(plan.config.variables),
        variable_type_counts=tuple(
            VariableTypeCount(data_type=data_type, count=type_counts[data_type])
            for data_type in variable_types
            if type_counts[data_type]
        ),
        restricted_variable_count=sum(
            variable.restricted_output for variable in plan.config.variables
        ),
        transformation_count=sum(len(variable.normalisation) for variable in plan.config.variables),
        blocking_rule_count=len(plan.config.blocking.rules),
        comparison_count=len(plan.config.comparisons),
        record_count_band=CountBand.NOT_OBSERVED,
        candidate_pair_budget_band=_candidate_budget_band(
            plan.config.runtime.maximum_candidate_pairs
        ),
        label_evidence_class=evidence_class,
        verified_labels_available=verified,
    )


__all__ = ["build_preflight_task_profile"]
