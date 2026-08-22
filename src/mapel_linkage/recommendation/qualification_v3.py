"""Outcome-free prospective qualification policy for advisor-v3."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from mapel_linkage.benchmarking.advisor_v3_features import (
    advisor_v3_feature_source_policy_digest,
)
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.recommendation.distance_v3 import (
    MechanismAwareMetaFeatureDistanceComputer,
    MechanismAwareTaskMetaFeatureVector,
    advisor_v3_feature_model_schema_digest,
    advisor_v3_ood_distance_rule_digest,
)
from mapel_linkage.recommendation.utility import (
    ADVISOR_UTILITY_POLICY_DIGEST,
    REQUIRED_ADVISOR_RECIPE_TOKENS,
    AdvisorRecipeToken,
)

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def advisor_v3_evaluation_algorithm_digest() -> str:
    """Bind every evaluator choice before v3 run outcomes exist."""

    return _digest(
        {
            "algorithm_id": "advisor_v3_family_qualification_algorithm_v1",
            "statistical_unit": "family",
            "family_recipe_aggregation": {
                "cell_utility": "advisor_utility_policy_digest_bounded_zero_one",
                "formula": "sum_20_cell_utilities_divided_by_20",
                "instance_weight": "one_quarter",
                "replicate_weight_within_instance": "one_fifth",
                "recipe_weight_in_model_fit": "one_third_within_each_family",
                "required_cells": "4_instances_x_5_replicates_for_each_recipe",
                "missing_or_non_success_required_cell": "fail_closed",
            },
            "family_regret": {
                "oracle": "maximum_family_recipe_utility_over_3_recipes",
                "oracle_tie_membership": "absolute_difference_lte_1e-12",
                "selected_recipe": "maximum_predicted_utility_then_recipe_token_ascending",
                "formula": "oracle_utility_minus_selected_recipe_actual_utility",
                "mean": "sum_family_regret_divided_by_12_locked_families",
                "median": "arithmetic_middle_two_when_even",
                "top1": "selected_recipe_in_oracle_tie_set",
                "top2": "nonempty_intersection_first_2_ranked_and_oracle_tie_set",
            },
            "stage2": {
                "neighbours": 3,
                "distance": "preregistered_weighted_gower",
                "ordering": "distance_then_family_id",
                "prediction": "unweighted_arithmetic_mean_neighbour_family_utility_by_recipe",
                "recipe_ordering": "predicted_utility_desc_then_recipe_token",
            },
            "stage3": {
                "model": "ridge_linear_regression",
                "ridge_alpha": 1.0,
                "intercept_penalized": False,
                "row_unit": "one_family_recipe_with_equal_weight",
                "row_count": 144,
                "target": "family_recipe_utility",
                "continuous_columns": "feature_model_schema_learned_continuous_order_raw_zero_one",
                "recipe_columns": "feature_model_schema_learned_recipe_one_hot_order",
                "categorical_columns": "excluded_from_learned_model",
                "fit_formula": (
                    "center_X_and_y; solve((X_centered.T@X_centered+alpha*I),"
                    "X_centered.T@y_centered); intercept=y_mean-X_mean@coef"
                ),
                "singular_fallback": "numpy_pinv_of_penalized_normal_matrix",
                "prediction_clip": "zero_one",
                "training_roles": ("meta_training",),
                "conformal_roles": ("conformal",),
            },
            "conformal": {
                "statistical_unit": "family",
                "calibration_family_count": 12,
                "score_per_family": "maximum_absolute_residual_over_3_recipes",
                "coverage": 0.90,
                "rank": "ceil((n+1)*coverage)_capped_at_n",
                "rank_with_n_12": 12,
                "tie_rule": "stable_numeric_order_statistic",
                "interval": (
                    "prediction_plus_or_minus_shared_family_score_quantile_clipped_zero_one"
                ),
                "locked_empirical_coverage": "covered_family_recipe_cells_divided_by_36",
                "locked_mean_width": "sum_36_interval_widths_divided_by_36",
            },
            "locked_metrics": (
                "mean_regret",
                "median_regret",
                "top1_oracle_coverage",
                "top2_oracle_coverage",
                "leave_one_training_family_out_selection_stability",
                "family_recipe_interval_coverage",
                "mean_interval_width",
                "mean_absolute_error",
            ),
            "ood": {
                "threshold": "conformal_nearest_training_geometry_rule_digest",
                "abstain_comparison": "nearest_training_distance_strictly_greater_than_threshold",
                "equality_behavior": "not_abstained",
                "ood_detection_rate": "detected_ood_families_divided_by_12",
                "locked_false_abstention_rate": "abstained_locked_families_divided_by_12",
                "balanced_abstention_accuracy": (
                    "(ood_detection_rate+(1-locked_false_abstention_rate))/2"
                ),
                "metrics": (
                    "ood_detection_rate",
                    "locked_false_abstention_rate",
                    "balanced_abstention_accuracy",
                ),
            },
            "fixed_baselines": {
                "recipes": REQUIRED_ADVISOR_RECIPE_TOKENS,
                "rule": "always_select_same_recipe_for_all_12_locked_families",
                "best_fixed": "minimum_mean_family_regret_then_recipe_token_ascending",
                "stage2_improvement": "best_fixed_mean_regret_minus_stage2_mean_regret",
                "stage3_improvement": "stage2_mean_regret_minus_stage3_mean_regret",
            },
            "leave_one_training_family_out_stability": {
                "full_selection_tie_rule": "predicted_utility_desc_then_recipe_token",
                "perturbations": 48,
                "exclusion": "omit_exactly_one_meta_training_family_from_fit_or_neighbour_pool",
                "conformal_families": "all_12_unchanged",
                "locked_families": "all_12_prediction_targets_unchanged",
                "numerator": "agreement_count_with_full_fit_selection_over_48_x_12_comparisons",
                "denominator": 576,
            },
            "learning_curve": {
                "family_counts": (12, 24, 36, 48),
                "order": "sha256(policy_digest_colon_family_id)_then_family_id",
                "subsets": "nested_prefixes_of_the_fixed_48_family_order",
                "refit": "stage2_and_stage3_at_each_count_with_all_12_conformal_families",
                "evaluation": "same_12_locked_families_at_each_count",
                "tail_rule": "abs(regret_at_48_minus_regret_at_36)_lte_0.02",
                "nondegradation_rule": "regret_at_48_lte_regret_at_12",
            },
            "gate_conjunctions": {
                "ood_qualified": "ood_detection_gte_0.75_AND_locked_false_abstention_lte_0.125",
                "conformal_qualified": "coverage_gte_0.80_AND_mean_width_lte_0.50",
                "learning_curve_qualified": "tail_rule_AND_nondegradation_rule",
                "stage2_qualified": (
                    "improvement_gte_0.005_AND_top2_gte_0.875_AND_stability_gte_0.80_"
                    "AND_ood_qualified"
                ),
                "stage3_qualified": (
                    "stage2_qualified_AND_improvement_gte_0.01_AND_top2_gte_0.875_"
                    "AND_stability_gte_0.80_AND_conformal_qualified_AND_"
                    "learning_curve_qualified"
                ),
                "overall_qualified": "stage2_qualified_AND_stage3_qualified",
                "failure_behavior": "not_qualified_with_all_failed_gate_codes_no_promotion",
            },
            "numeric_canonicalization": (
                "binary64_computation_then_python_round_12; canonical_json_sort_keys_"
                "compact_separators_allow_nan_false"
            ),
            "qualification_approval_schema_id": "advisor_v3_locked_qualification_approval_v1",
            "approval_binding": {
                "required_digests": (
                    "design_digest",
                    "catalogue_manifest_digest",
                    "readiness_digest",
                    "registry_snapshot_digest",
                    "policy_digest",
                    "preregistration_digest",
                    "evaluation_algorithm_digest",
                ),
                "required_literals": (
                    "human_approved_true",
                    "locked_evaluation_access_authorized_true",
                    "ood_evaluation_access_authorized_true",
                ),
            },
            "aggregate_report_schema_id": "advisor_v3_qualification_aggregate_report_v1",
            "output": {
                "encoding": "canonical_tamper_evident_json",
                "content": "aggregate_metrics_gate_results_status_and_bound_digests",
                "prohibited": "family_ids_instance_ids_local_paths_record_values_candidate_pairs",
                "automatic_release_or_promotion": False,
            },
            "locked_or_ood_used_for_fit": False,
            "qualification_execution_availability": "not_implemented_until_separate_reviewed_slice",
            "automatic_promotion": False,
        }
    )


def advisor_v31_evaluation_algorithm_digest() -> str:
    """Bind the post-corpus, pre-qualification role-specific evidence amendment."""

    return _digest(
        {
            "algorithm_id": "advisor_v31_family_qualification_algorithm_amendment_v1",
            "base_evaluation_algorithm_digest": advisor_v3_evaluation_algorithm_digest(),
            "amendment_scope": "qualification_input_evidence_contract_only",
            "family_is_statistical_unit": True,
            "recipe_utility_required_roles": (
                "meta_training",
                "conformal",
                "locked_evaluation",
            ),
            "recipe_utility_required_cells": (
                "72_families_x_4_instances_x_5_replicates_x_3_recipes"
            ),
            "missing_or_non_success_required_role_cell": "fail_closed",
            "ood_holdout_evidence": {
                "required": (
                    "12_complete_observable_mechanism_profiles",
                    "preregistered_training_conformal_distance_geometry",
                ),
                "recipe_utility_requirement": "none",
                "recipe_metric_use_for_fit_threshold_or_qualification": "prohibited",
                "adapter_status_use": "aggregate_diagnostic_integrity_only",
            },
            "unchanged": (
                "family_roles",
                "catalogue",
                "seeds",
                "replicates",
                "utility_policy",
                "performance_thresholds",
                "distance_threshold",
                "locked_and_ood_human_approval_gate",
                "no_automatic_promotion",
            ),
            "source_registry": "immutable_advisor_v3_execution_v1",
            "remediation_registry": "governance_only_digest_bound_reference",
            "amendment_trigger_metadata": ("adapter_status_and_failure_code_metadata"),
            "adapter_status_metadata_accessed_to_select_amendment": True,
            "failure_code_metadata_accessed_to_select_amendment": True,
            "performance_metric_values_accessed_to_select_amendment": False,
            "qualification_execution_availability": "not_implemented_until_separate_review",
            "automatic_promotion": False,
        }
    )


class AdvisorV3QualificationPolicy(BaseModel):
    """Fixed v3 thresholds declared before any v3 locked or OOD result exists."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    policy_schema_version: Literal["3"] = "3"
    policy_id: Literal["advisor_family_qualification_v3"] = "advisor_family_qualification_v3"
    feature_source_policy_digest: Digest = advisor_v3_feature_source_policy_digest()
    feature_model_schema_digest: Digest = advisor_v3_feature_model_schema_digest()
    evaluation_algorithm_digest: Digest = advisor_v3_evaluation_algorithm_digest()
    utility_policy_digest: Digest = ADVISOR_UTILITY_POLICY_DIGEST
    nearest_family_count: StrictInt = 3
    oracle_coverage_k: StrictInt = 2
    ridge_alpha: StrictFloat = 1.0
    conformal_coverage_level: StrictFloat = 0.90
    ood_distance_rule_digest: Digest = advisor_v3_ood_distance_rule_digest()
    ood_distance_quantile: StrictFloat = 0.90
    minimum_stage2_regret_improvement: StrictFloat = 0.005
    minimum_stage3_regret_improvement: StrictFloat = 0.01
    minimum_top2_oracle_coverage: StrictFloat = 0.875
    minimum_locked_interval_coverage: StrictFloat = 0.80
    maximum_mean_interval_width: StrictFloat = 0.50
    minimum_selection_stability: StrictFloat = 0.80
    minimum_ood_detection_rate: StrictFloat = 0.75
    maximum_locked_false_abstention_rate: StrictFloat = 0.125
    maximum_learning_curve_tail_regret_range: StrictFloat = 0.02
    learning_curve_family_counts: tuple[StrictInt, ...] = (12, 24, 36, 48)
    meta_training_family_count: StrictInt = 48
    conformal_family_count: StrictInt = 12
    locked_evaluation_family_count: StrictInt = 12
    ood_holdout_family_count: StrictInt = 12
    required_recipe_tokens: tuple[AdvisorRecipeToken, ...] = REQUIRED_ADVISOR_RECIPE_TOKENS
    family_is_statistical_unit: Literal[True] = True
    performance_thresholds_relaxed_from_v2: Literal[False] = False
    ood_distance_policy_status: Literal["training_conformal_geometry_only"] = (
        "training_conformal_geometry_only"
    )
    locked_evaluation_requires_later_approval: Literal[True] = True
    ood_evaluation_requires_later_approval: Literal[True] = True
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    release_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_fixed_policy(self) -> Self:
        fixed: dict[str, object] = {
            "feature_source_policy_digest": advisor_v3_feature_source_policy_digest(),
            "feature_model_schema_digest": advisor_v3_feature_model_schema_digest(),
            "evaluation_algorithm_digest": advisor_v3_evaluation_algorithm_digest(),
            "utility_policy_digest": ADVISOR_UTILITY_POLICY_DIGEST,
            "nearest_family_count": 3,
            "oracle_coverage_k": 2,
            "ridge_alpha": 1.0,
            "conformal_coverage_level": 0.90,
            "ood_distance_rule_digest": advisor_v3_ood_distance_rule_digest(),
            "ood_distance_quantile": 0.90,
            "minimum_stage2_regret_improvement": 0.005,
            "minimum_stage3_regret_improvement": 0.01,
            "minimum_top2_oracle_coverage": 0.875,
            "minimum_locked_interval_coverage": 0.80,
            "maximum_mean_interval_width": 0.50,
            "minimum_selection_stability": 0.80,
            "minimum_ood_detection_rate": 0.75,
            "maximum_locked_false_abstention_rate": 0.125,
            "maximum_learning_curve_tail_regret_range": 0.02,
            "learning_curve_family_counts": (12, 24, 36, 48),
            "meta_training_family_count": 48,
            "conformal_family_count": 12,
            "locked_evaluation_family_count": 12,
            "ood_holdout_family_count": 12,
            "required_recipe_tokens": REQUIRED_ADVISOR_RECIPE_TOKENS,
        }
        if {name: getattr(self, name) for name in fixed} != fixed:
            raise ValueError("Advisor-v3 qualification thresholds are prospectively fixed.")
        return self

    @property
    def policy_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {**self.model_dump(mode="json"), "policy_digest": self.policy_digest}


class AdvisorV31QualificationApproval(BaseModel):
    """Separate human authorization for one v3.1 locked/OOD evaluation.

    This is deliberately distinct from the remediation approval, whose literal
    access permissions remain false.  It binds the frozen source evidence and
    the current evaluator analysis provenance without granting any operational
    promotion authority.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    approval_schema_version: Literal["3.1"] = "3.1"
    approval_reference: Annotated[
        str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$", repr=False)
    ]
    human_approved: Literal[True]
    locked_evaluation_access_authorized: Literal[True]
    ood_evaluation_access_authorized: Literal[True]
    amendment_digest: Digest
    source_execution_approval_digest: Digest
    source_execution_provenance_digest: Digest
    source_v3_readiness_digest: Digest
    source_registry_snapshot_digest: Digest
    analysis_provenance_digest: Digest
    remediation_approval_digest: Digest
    remediation_readiness_digest: Digest
    policy_digest: Digest
    evaluation_algorithm_digest: Digest
    synthetic_only: Literal[True] = True
    ood_recipe_metric_use_for_fit_threshold_or_qualification: Literal["prohibited"] = "prohibited"
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    release_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_current_policy(self) -> Self:
        policy = AdvisorV3QualificationPolicy()
        if (
            self.policy_digest != policy.policy_digest
            or self.evaluation_algorithm_digest != advisor_v31_evaluation_algorithm_digest()
        ):
            raise ValueError("Advisor-v3.1 qualification approval is stale or conflicting.")
        return self

    @property
    def approval_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {
            **self.model_dump(mode="json", exclude={"approval_reference"}),
            "approval_digest": self.approval_digest,
        }


class AdvisorV31QualificationReadinessArtifact(BaseModel):
    """Governance-only re-audit binding; it neither reads nor evaluates outcomes."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    readiness_schema_version: Literal["3.1"] = "3.1"
    artifact_id: Literal["advisor_v31_qualification_readiness_v1"] = (
        "advisor_v31_qualification_readiness_v1"
    )
    amendment_digest: Digest
    source_execution_approval_digest: Digest
    source_execution_provenance_digest: Digest
    source_v3_readiness_digest: Digest
    source_registry_snapshot_digest: Digest
    analysis_provenance_digest: Digest
    remediation_approval_digest: Digest
    remediation_readiness_digest: Digest
    policy_digest: Digest
    evaluation_algorithm_digest: Digest
    advisor_evidence_ready: Literal[True]
    qualification_evaluation_accessed: Literal[False] = False
    locked_evaluation_access_authorized: Literal[False] = False
    ood_evaluation_access_authorized: Literal[False] = False
    ood_recipe_metric_use_for_fit_threshold_or_qualification: Literal["prohibited"] = "prohibited"
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_current_policy(self) -> Self:
        policy = AdvisorV3QualificationPolicy()
        if (
            self.policy_digest != policy.policy_digest
            or self.evaluation_algorithm_digest != advisor_v31_evaluation_algorithm_digest()
        ):
            raise ValueError("Advisor-v3.1 qualification readiness is stale or conflicting.")
        return self

    @property
    def readiness_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {**self.model_dump(mode="json"), "readiness_digest": self.readiness_digest}


def build_advisor_v31_qualification_readiness(
    *,
    amendment_digest: str,
    source_execution_approval_digest: str,
    source_execution_provenance_digest: str,
    source_v3_readiness_digest: str,
    source_registry_snapshot_digest: str,
    analysis_provenance_digest: str,
    remediation_approval_digest: str,
    remediation_readiness_digest: str,
    advisor_evidence_ready: bool,
) -> AdvisorV31QualificationReadinessArtifact:
    """Create a no-outcome-access readiness artifact from reviewed aggregate bindings."""

    if not advisor_evidence_ready:
        raise ValueError("Advisor-v3.1 qualification readiness must fail closed.")
    return AdvisorV31QualificationReadinessArtifact(
        amendment_digest=amendment_digest,
        source_execution_approval_digest=source_execution_approval_digest,
        source_execution_provenance_digest=source_execution_provenance_digest,
        source_v3_readiness_digest=source_v3_readiness_digest,
        source_registry_snapshot_digest=source_registry_snapshot_digest,
        analysis_provenance_digest=analysis_provenance_digest,
        remediation_approval_digest=remediation_approval_digest,
        remediation_readiness_digest=remediation_readiness_digest,
        policy_digest=AdvisorV3QualificationPolicy().policy_digest,
        evaluation_algorithm_digest=advisor_v31_evaluation_algorithm_digest(),
        advisor_evidence_ready=True,
    )


@dataclass(frozen=True, slots=True, repr=False)
class AdvisorV31FamilyUtilityEvidence:
    """One aggregate family/recipe utility input; never serialized into reports."""

    family_id: str
    family_role: Literal["meta_training", "conformal", "locked_evaluation", "ood_holdout"]
    recipe_token: AdvisorRecipeToken
    mean_utility: float
    run_count: int
    evidence_digest: str


class AdvisorV31MethodQualificationMetrics(BaseModel):
    """Complete locked-family metrics for one advisory method."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    method: Literal["similarity", "meta_ranker"]
    locked_family_count: Literal[12] = 12
    mean_regret: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    median_regret: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    top1_oracle_coverage: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    top2_oracle_coverage: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    selection_stability: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]


class AdvisorV31FixedRecipeBaselineMetrics(BaseModel):
    """Locked-family regret for one prespecified fixed recipe."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    recipe_token: AdvisorRecipeToken
    locked_family_count: Literal[12] = 12
    mean_regret: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    median_regret: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    oracle_coverage: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]


class AdvisorV31ConformalQualificationMetrics(BaseModel):
    """Locked family-by-recipe coverage for the fixed split-conformal interval."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    nominal_coverage: StrictFloat = 0.9
    locked_family_count: Literal[12] = 12
    locked_family_recipe_count: Literal[36] = 36
    empirical_coverage: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    mean_interval_width: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    mean_absolute_error: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def validate_nominal_coverage(self) -> Self:
        if self.nominal_coverage != 0.9:
            raise ValueError("Advisor-v3.1 nominal conformal coverage is fixed.")
        return self


class AdvisorV31OODQualificationMetrics(BaseModel):
    """Distance-only OOD and locked false-abstention metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    ood_family_count: Literal[12] = 12
    locked_family_count: Literal[12] = 12
    detected_ood_family_count: Annotated[StrictInt, Field(ge=0, le=12)]
    falsely_abstained_locked_family_count: Annotated[StrictInt, Field(ge=0, le=12)]
    ood_detection_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    locked_false_abstention_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    balanced_abstention_accuracy: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def validate_derived_rates(self) -> Self:
        expected_ood = _metric(self.detected_ood_family_count / 12)
        expected_false = _metric(self.falsely_abstained_locked_family_count / 12)
        expected_balanced = _metric((expected_ood + (1.0 - expected_false)) / 2.0)
        if (
            self.ood_detection_rate != expected_ood
            or self.locked_false_abstention_rate != expected_false
            or self.balanced_abstention_accuracy != expected_balanced
        ):
            raise ValueError("Advisor-v3.1 OOD aggregate rates are inconsistent.")
        return self


class AdvisorV31LearningCurvePoint(BaseModel):
    """One fixed nested-family learning-curve point."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    training_family_count: StrictInt
    similarity_mean_regret: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    meta_ranker_mean_regret: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def validate_family_count(self) -> Self:
        if self.training_family_count not in {12, 24, 36, 48}:
            raise ValueError("Advisor-v3.1 learning-curve family counts are fixed.")
        return self


class AdvisorV31QualificationReport(BaseModel):
    """Aggregate-only v3.1 qualification result, with no promotion authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    report_schema_version: Literal["3.1"] = "3.1"
    qualification_status: Literal["qualified", "not_qualified"]
    amendment_digest: Digest
    source_execution_approval_digest: Digest
    source_execution_provenance_digest: Digest
    source_v3_readiness_digest: Digest
    source_registry_snapshot_digest: Digest
    analysis_provenance_digest: Digest
    remediation_approval_digest: Digest
    remediation_readiness_digest: Digest
    qualification_readiness_digest: Digest
    approval_digest: Digest
    policy_digest: Digest
    evaluation_algorithm_digest: Digest
    evidence_digest: Digest
    family_split_digest: Digest
    meta_model_digest: Digest
    expected_run_count: Literal[11760] = 11_760
    qualification_required_adapter_run_count: Literal[4320] = 4_320
    retained_ineligible_run_count: Literal[6720] = 6_720
    ood_diagnostic_adapter_run_count: Literal[720] = 720
    meta_training_family_count: Literal[48] = 48
    conformal_family_count: Literal[12] = 12
    locked_evaluation_family_count: Literal[12] = 12
    ood_holdout_family_count: Literal[12] = 12
    fixed_baselines: Annotated[
        tuple[AdvisorV31FixedRecipeBaselineMetrics, ...], Field(min_length=3, max_length=3)
    ]
    stage2_similarity: AdvisorV31MethodQualificationMetrics
    stage3_meta_ranker: AdvisorV31MethodQualificationMetrics
    conformal: AdvisorV31ConformalQualificationMetrics
    ood: AdvisorV31OODQualificationMetrics
    learning_curve: Annotated[
        tuple[AdvisorV31LearningCurvePoint, ...], Field(min_length=4, max_length=4)
    ]
    stage2_regret_improvement_over_best_fixed: StrictFloat
    stage3_regret_improvement_over_stage2: StrictFloat
    learning_curve_tail_regret_range: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    stage2_qualified: StrictBool
    stage3_qualified: StrictBool
    ood_qualified: StrictBool
    conformal_qualified: StrictBool
    learning_curve_qualified: StrictBool
    fallback_to_similarity_required: StrictBool
    failed_gate_codes: tuple[StrictStr, ...]
    hard_constraint_violation_count: Literal[0] = 0
    ood_recipe_metric_payloads_parsed_for_digest_integrity_only: Literal[True] = True
    ood_recipe_metric_values_used_for_fit_threshold_or_qualification: Literal[False] = False
    locked_evaluation_accessed: Literal[True] = True
    ood_evaluation_accessed: Literal[True] = True
    locked_evaluation_used_for_fit: Literal[False] = False
    ood_evidence_used_for_fit: Literal[False] = False
    ood_evidence_used_for_interval_calibration: Literal[False] = False
    contains_record_values: Literal[False] = False
    contains_identifiers: Literal[False] = False
    contains_candidate_pairs: Literal[False] = False
    contains_local_paths: Literal[False] = False
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    release_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        policy = AdvisorV3QualificationPolicy()
        if (
            self.policy_digest != policy.policy_digest
            or self.evaluation_algorithm_digest != advisor_v31_evaluation_algorithm_digest()
        ):
            raise ValueError("Advisor-v3.1 qualification report policy binding is stale.")
        if tuple(item.recipe_token for item in self.fixed_baselines) != (
            REQUIRED_ADVISOR_RECIPE_TOKENS
        ):
            raise ValueError("Advisor-v3.1 fixed baselines are not in canonical order.")
        if tuple(item.training_family_count for item in self.learning_curve) != (
            policy.learning_curve_family_counts
        ):
            raise ValueError("Advisor-v3.1 learning-curve points are not canonical.")
        if (
            self.stage2_similarity.method != "similarity"
            or self.stage3_meta_ranker.method != "meta_ranker"
        ):
            raise ValueError("Advisor-v3.1 method metrics are misbound.")
        if len(self.failed_gate_codes) != len(set(self.failed_gate_codes)):
            raise ValueError("Advisor-v3.1 failed gate codes must be unique.")
        stage2_improvement = (
            min(item.mean_regret for item in self.fixed_baselines)
            - self.stage2_similarity.mean_regret
        )
        stage3_improvement = (
            self.stage2_similarity.mean_regret - self.stage3_meta_ranker.mean_regret
        )
        tail_range = abs(
            self.learning_curve[-1].meta_ranker_mean_regret
            - self.learning_curve[-2].meta_ranker_mean_regret
        )
        if (
            self.stage2_regret_improvement_over_best_fixed != _metric(stage2_improvement)
            or self.stage3_regret_improvement_over_stage2 != _metric(stage3_improvement)
            or self.learning_curve_tail_regret_range != _metric(tail_range)
        ):
            raise ValueError("Advisor-v3.1 derived regret metrics are inconsistent.")
        expected_ood = (
            self.ood.ood_detection_rate >= policy.minimum_ood_detection_rate
            and self.ood.locked_false_abstention_rate <= policy.maximum_locked_false_abstention_rate
        )
        expected_conformal = (
            self.conformal.empirical_coverage >= policy.minimum_locked_interval_coverage
            and self.conformal.mean_interval_width <= policy.maximum_mean_interval_width
        )
        expected_curve = (
            tail_range <= policy.maximum_learning_curve_tail_regret_range
            and self.learning_curve[-1].meta_ranker_mean_regret
            <= self.learning_curve[0].meta_ranker_mean_regret
        )
        expected_stage2 = (
            stage2_improvement >= policy.minimum_stage2_regret_improvement
            and self.stage2_similarity.top2_oracle_coverage >= policy.minimum_top2_oracle_coverage
            and self.stage2_similarity.selection_stability >= policy.minimum_selection_stability
            and expected_ood
        )
        expected_stage3 = (
            expected_stage2
            and stage3_improvement >= policy.minimum_stage3_regret_improvement
            and self.stage3_meta_ranker.top2_oracle_coverage >= policy.minimum_top2_oracle_coverage
            and self.stage3_meta_ranker.selection_stability >= policy.minimum_selection_stability
            and expected_conformal
            and expected_curve
        )
        if (
            self.ood_qualified != expected_ood
            or self.conformal_qualified != expected_conformal
            or self.learning_curve_qualified != expected_curve
            or self.stage2_qualified != expected_stage2
            or self.stage3_qualified != expected_stage3
        ):
            raise ValueError("Advisor-v3.1 qualification gate outcomes are inconsistent.")
        expected_failed = tuple(
            code
            for code, passed in (
                (
                    "stage2_regret_improvement",
                    stage2_improvement >= policy.minimum_stage2_regret_improvement,
                ),
                (
                    "stage2_top2_oracle_coverage",
                    self.stage2_similarity.top2_oracle_coverage
                    >= policy.minimum_top2_oracle_coverage,
                ),
                (
                    "stage2_selection_stability",
                    self.stage2_similarity.selection_stability
                    >= policy.minimum_selection_stability,
                ),
                (
                    "stage3_regret_improvement",
                    stage3_improvement >= policy.minimum_stage3_regret_improvement,
                ),
                (
                    "stage3_top2_oracle_coverage",
                    self.stage3_meta_ranker.top2_oracle_coverage
                    >= policy.minimum_top2_oracle_coverage,
                ),
                (
                    "stage3_selection_stability",
                    self.stage3_meta_ranker.selection_stability
                    >= policy.minimum_selection_stability,
                ),
                (
                    "locked_interval_coverage",
                    self.conformal.empirical_coverage >= policy.minimum_locked_interval_coverage,
                ),
                (
                    "mean_interval_width",
                    self.conformal.mean_interval_width <= policy.maximum_mean_interval_width,
                ),
                (
                    "ood_detection_rate",
                    self.ood.ood_detection_rate >= policy.minimum_ood_detection_rate,
                ),
                (
                    "locked_false_abstention_rate",
                    self.ood.locked_false_abstention_rate
                    <= policy.maximum_locked_false_abstention_rate,
                ),
                ("learning_curve_stability", expected_curve),
            )
            if not passed
        )
        if self.failed_gate_codes != expected_failed:
            raise ValueError("Advisor-v3.1 failed gate evidence is inconsistent.")
        expected_status = "qualified" if expected_stage2 and expected_stage3 else "not_qualified"
        if self.qualification_status != expected_status:
            raise ValueError("Advisor-v3.1 qualification status is inconsistent.")
        if self.fallback_to_similarity_required != (not expected_stage3):
            raise ValueError("Advisor-v3.1 fallback status is inconsistent.")
        return self

    @property
    def report_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {**self.model_dump(mode="json"), "report_digest": self.report_digest}


class AdvisorV31QualificationArtifact(BaseModel):
    """Tamper-evident aggregate envelope for one explicitly approved evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    artifact_schema_version: Literal["3.1"] = "3.1"
    report: AdvisorV31QualificationReport
    report_digest: Digest
    contains_record_values: Literal[False] = False
    contains_identifiers: Literal[False] = False
    contains_candidate_pairs: Literal[False] = False
    contains_local_paths: Literal[False] = False
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_integrity(self) -> Self:
        if self.report_digest != self.report.report_digest:
            raise ValueError("Advisor-v3.1 qualification artifact integrity failed.")
        return self

    @property
    def artifact_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "artifact_digest": self.artifact_digest,
            "report": self.report.safe_summary(),
            "contains_record_values": False,
            "contains_identifiers": False,
            "contains_candidate_pairs": False,
            "contains_local_paths": False,
            "operational_validity": self.operational_validity,
        }


def serialize_advisor_v31_qualification_artifact(
    artifact: AdvisorV31QualificationArtifact,
) -> str:
    return json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def deserialize_advisor_v31_qualification_artifact(text: str) -> AdvisorV31QualificationArtifact:
    if len(text.encode("utf-8")) > 4 * 1024 * 1024:
        raise ValueError("The advisor-v3.1 qualification artifact exceeds its aggregate bound.")
    artifact = AdvisorV31QualificationArtifact.model_validate_json(text)
    if text != serialize_advisor_v31_qualification_artifact(artifact):
        raise ValueError("The advisor-v3.1 qualification artifact is not canonical JSON.")
    return artifact


def _reject_symlink_components(path: Path) -> None:
    lexical = path.absolute()
    if any(component.is_symlink() for component in (*reversed(lexical.parents), lexical)):
        raise ValueError("Advisor-v3.1 qualification paths cannot use symbolic links.")


def write_advisor_v31_qualification_artifact(
    path: Path, artifact: AdvisorV31QualificationArtifact
) -> None:
    """Write once or accept only an exact canonical replay."""

    try:
        _reject_symlink_components(path)
        text = serialize_advisor_v31_qualification_artifact(artifact)
        if path.exists():
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                raise FileExistsError("A conflicting advisor-v3.1 qualification artifact exists.")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, text)
    except (FileExistsError, ValueError):
        raise
    except OSError:
        raise ValueError(
            "Advisor-v3.1 qualification artifact could not be written safely."
        ) from None


def load_advisor_v31_qualification_artifact(path: Path) -> AdvisorV31QualificationArtifact:
    try:
        _reject_symlink_components(path)
        if not path.is_file():
            raise FileNotFoundError
        return deserialize_advisor_v31_qualification_artifact(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError("Advisor-v3.1 qualification artifact is unavailable.") from None
    except OSError:
        raise ValueError("Advisor-v3.1 qualification artifact could not be read safely.") from None


def _metric(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Advisor-v3.1 qualification produced a non-finite aggregate metric.")
    return round(float(value), 12)


def _utility_map(
    evidence: tuple[AdvisorV31FamilyUtilityEvidence, ...],
) -> dict[tuple[str, AdvisorRecipeToken], float]:
    utilities: dict[tuple[str, AdvisorRecipeToken], float] = {}
    for item in evidence:
        key = (item.family_id, item.recipe_token)
        if (
            key in utilities
            or not math.isfinite(item.mean_utility)
            or not 0.0 <= item.mean_utility <= 1.0
            or item.run_count != 20
            or len(item.evidence_digest) != 64
            or any(character not in "0123456789abcdef" for character in item.evidence_digest)
        ):
            raise ValueError(
                "Advisor-v3.1 family utility evidence must be complete, unique, and finite."
            )
        utilities[key] = float(item.mean_utility)
    return utilities


def _select_recipe(predictions: Mapping[AdvisorRecipeToken, float]) -> AdvisorRecipeToken:
    return min(REQUIRED_ADVISOR_RECIPE_TOKENS, key=lambda token: (-predictions[token], token))


def _fit_ridge_predict(
    *,
    training_family_ids: tuple[str, ...],
    conformal_family_ids: tuple[str, ...],
    target_family_ids: tuple[str, ...],
    vectors: Mapping[str, MechanismAwareTaskMetaFeatureVector],
    utilities: Mapping[tuple[str, AdvisorRecipeToken], float],
    policy: AdvisorV3QualificationPolicy,
) -> tuple[dict[str, dict[AdvisorRecipeToken, tuple[float, float, float]]], str]:
    """Implement the digest-bound centered ridge and family-wise conformal interval."""

    def row(family_id: str, token: AdvisorRecipeToken) -> list[float]:
        vector = vectors[family_id]
        return [
            *(float(getattr(vector, name)) for name in vector.CONTINUOUS_FEATURES),
            *(1.0 if token == candidate else 0.0 for candidate in REQUIRED_ADVISOR_RECIPE_TOKENS),
        ]

    train_rows = tuple(
        (family_id, token)
        for family_id in training_family_ids
        for token in REQUIRED_ADVISOR_RECIPE_TOKENS
    )
    X_train = np.asarray([row(*item) for item in train_rows], dtype=np.float64)
    y_train = np.asarray([utilities[item] for item in train_rows], dtype=np.float64)
    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
        raise ValueError("Advisor-v3.1 learned features must be finite.")
    x_mean = X_train.mean(axis=0)
    y_mean = float(y_train.mean())
    centered_x = X_train - x_mean
    normal_matrix = centered_x.T @ centered_x + policy.ridge_alpha * np.eye(X_train.shape[1])
    right_hand_side = centered_x.T @ (y_train - y_mean)
    try:
        weights = np.linalg.solve(normal_matrix, right_hand_side)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(normal_matrix) @ right_hand_side
    intercept = y_mean - float(x_mean @ weights)
    scores: list[float] = []
    for family_id in conformal_family_ids:
        residuals = [
            abs(
                float(np.clip(np.asarray(row(family_id, token)) @ weights + intercept, 0.0, 1.0))
                - utilities[(family_id, token)]
            )
            for token in REQUIRED_ADVISOR_RECIPE_TOKENS
        ]
        scores.append(max(residuals))
    if len(scores) != policy.conformal_family_count:
        raise ValueError("Advisor-v3.1 conformal evidence is incomplete.")
    rank = min(len(scores), math.ceil((len(scores) + 1) * policy.conformal_coverage_level))
    radius = sorted(scores)[rank - 1]
    predictions: dict[str, dict[AdvisorRecipeToken, tuple[float, float, float]]] = {}
    for family_id in target_family_ids:
        predictions[family_id] = {}
        for token in REQUIRED_ADVISOR_RECIPE_TOKENS:
            point = float(
                np.clip(np.asarray(row(family_id, token)) @ weights + intercept, 0.0, 1.0)
            )
            predictions[family_id][token] = (
                point,
                max(0.0, point - radius),
                min(1.0, point + radius),
            )
    model_digest = _digest(
        {
            "model_schema_id": "advisor_v31_centered_ridge_conformal_v1",
            "weights": [float(item) for item in weights],
            "intercept": intercept,
            "conformal_radius": radius,
            "training_family_count": len(training_family_ids),
            "conformal_family_count": len(conformal_family_ids),
            "family_split_digest": _digest(
                {
                    "training": training_family_ids,
                    "conformal": conformal_family_ids,
                }
            ),
            "policy_digest": policy.policy_digest,
            "feature_model_schema_digest": policy.feature_model_schema_digest,
        }
    )
    return predictions, model_digest


def evaluate_advisor_v31_aggregate_evidence(
    *,
    readiness: AdvisorV31QualificationReadinessArtifact,
    approval: AdvisorV31QualificationApproval,
    evidence: tuple[AdvisorV31FamilyUtilityEvidence, ...],
    role_by_family: Mapping[str, str],
    family_vectors: Mapping[str, MechanismAwareTaskMetaFeatureVector],
) -> AdvisorV31QualificationReport:
    """Evaluate only reviewed aggregate evidence after explicit locked/OOD authorization.

    OOD contributes only observable feature geometry.  No OOD utility evidence
    is accepted, which makes metric use for fitting, threshold selection, and
    qualification structurally impossible at this interface.
    """

    binding_names = (
        "amendment_digest",
        "source_execution_approval_digest",
        "source_execution_provenance_digest",
        "source_v3_readiness_digest",
        "source_registry_snapshot_digest",
        "analysis_provenance_digest",
        "remediation_approval_digest",
        "remediation_readiness_digest",
        "policy_digest",
        "evaluation_algorithm_digest",
    )
    if any(getattr(readiness, name) != getattr(approval, name) for name in binding_names):
        raise ValueError("Advisor-v3.1 approval does not bind the exact readiness evidence.")
    policy = AdvisorV3QualificationPolicy()
    roles = {family_id: role for family_id, role in role_by_family.items()}
    role_counts = {
        role: sum(value == role for value in roles.values()) for role in set(roles.values())
    }
    expected_counts = {
        "meta_training": policy.meta_training_family_count,
        "conformal": policy.conformal_family_count,
        "locked_evaluation": policy.locked_evaluation_family_count,
        "ood_holdout": policy.ood_holdout_family_count,
    }
    if role_counts != expected_counts or set(roles) != set(family_vectors):
        raise ValueError("Advisor-v3.1 roles or observable vectors do not cover the fixed design.")
    if any(roles.get(item.family_id) != item.family_role for item in evidence):
        raise ValueError("Advisor-v3.1 utility evidence conflicts with the fixed family roles.")
    if any(item.family_role == "ood_holdout" for item in evidence):
        raise ValueError("Advisor-v3.1 OOD recipe metrics are prohibited qualification inputs.")
    utilities = _utility_map(evidence)
    training = tuple(sorted(key for key, value in roles.items() if value == "meta_training"))
    conformal = tuple(sorted(key for key, value in roles.items() if value == "conformal"))
    locked = tuple(sorted(key for key, value in roles.items() if value == "locked_evaluation"))
    ood = tuple(sorted(key for key, value in roles.items() if value == "ood_holdout"))
    expected_keys = {
        (family_id, token)
        for family_id in (*training, *conformal, *locked)
        for token in REQUIRED_ADVISOR_RECIPE_TOKENS
    }
    if set(utilities) != expected_keys:
        raise ValueError("Advisor-v3.1 evidence omits or crosses a required protected role.")

    distance = MechanismAwareMetaFeatureDistanceComputer()
    train_vectors = {family_id: family_vectors[family_id] for family_id in training}

    def similarity(
        targets: tuple[str, ...], subset: tuple[str, ...]
    ) -> dict[str, dict[AdvisorRecipeToken, float]]:
        subset_vectors = {family_id: family_vectors[family_id] for family_id in subset}
        return {
            family_id: {
                token: float(
                    np.mean(
                        [
                            utilities[(near, token)]
                            for near, _ in distance.find_nearest_families(
                                family_vectors[family_id],
                                subset_vectors,
                                k=policy.nearest_family_count,
                            )
                        ]
                    )
                )
                for token in REQUIRED_ADVISOR_RECIPE_TOKENS
            }
            for family_id in targets
        }

    def method_metrics(
        method: Literal["similarity", "meta_ranker"],
        predicted: Mapping[str, Mapping[AdvisorRecipeToken, float]],
        *,
        selection_stability: float,
    ) -> tuple[AdvisorV31MethodQualificationMetrics, dict[str, AdvisorRecipeToken]]:
        regrets: list[float] = []
        top1 = 0
        top2 = 0
        choices: dict[str, AdvisorRecipeToken] = {}
        for family_id in locked:
            ranked = sorted(
                REQUIRED_ADVISOR_RECIPE_TOKENS,
                key=lambda token: (-predicted[family_id][token], token),
            )
            actual = {
                token: utilities[(family_id, token)] for token in REQUIRED_ADVISOR_RECIPE_TOKENS
            }
            oracle = max(actual.values())
            oracle_tokens = {
                token for token, value in actual.items() if abs(value - oracle) <= 1e-12
            }
            choices[family_id] = ranked[0]
            regrets.append(oracle - actual[ranked[0]])
            top1 += ranked[0] in oracle_tokens
            top2 += bool(set(ranked[:2]) & oracle_tokens)
        return (
            AdvisorV31MethodQualificationMetrics(
                method=method,
                mean_regret=_metric(float(np.mean(regrets))),
                median_regret=_metric(float(statistics.median(regrets))),
                top1_oracle_coverage=_metric(top1 / len(locked)),
                top2_oracle_coverage=_metric(top2 / len(locked)),
                selection_stability=_metric(selection_stability),
            ),
            choices,
        )

    similarity_pred = similarity(locked, training)
    stage2_preliminary, stage2_choices = method_metrics(
        "similarity", similarity_pred, selection_stability=1.0
    )
    meta_intervals, meta_model_digest = _fit_ridge_predict(
        training_family_ids=training,
        conformal_family_ids=conformal,
        target_family_ids=locked,
        vectors=family_vectors,
        utilities=utilities,
        policy=policy,
    )
    meta_pred = {
        family_id: {token: item[0] for token, item in values.items()}
        for family_id, values in meta_intervals.items()
    }
    stage3_preliminary, stage3_choices = method_metrics(
        "meta_ranker", meta_pred, selection_stability=1.0
    )

    similarity_alternatives: list[Mapping[str, AdvisorRecipeToken]] = []
    meta_alternatives: list[Mapping[str, AdvisorRecipeToken]] = []
    for omitted in training:
        subset = tuple(item for item in training if item != omitted)
        _, selected = method_metrics(
            "similarity", similarity(locked, subset), selection_stability=1.0
        )
        similarity_alternatives.append(selected)
        alternate_intervals, _ = _fit_ridge_predict(
            training_family_ids=subset,
            conformal_family_ids=conformal,
            target_family_ids=locked,
            vectors=family_vectors,
            utilities=utilities,
            policy=policy,
        )
        _, selected = method_metrics(
            "meta_ranker",
            {
                family_id: {token: value[0] for token, value in values.items()}
                for family_id, values in alternate_intervals.items()
            },
            selection_stability=1.0,
        )
        meta_alternatives.append(selected)

    def stability(
        baseline: Mapping[str, AdvisorRecipeToken],
        alternatives: list[Mapping[str, AdvisorRecipeToken]],
    ) -> float:
        return _metric(
            sum(
                other[family] == choice
                for other in alternatives
                for family, choice in baseline.items()
            )
            / (len(baseline) * len(alternatives))
        )

    stage2 = stage2_preliminary.model_copy(
        update={"selection_stability": stability(stage2_choices, similarity_alternatives)}
    )
    stage3 = stage3_preliminary.model_copy(
        update={"selection_stability": stability(stage3_choices, meta_alternatives)}
    )
    widths = [
        upper - lower for values in meta_intervals.values() for _, lower, upper in values.values()
    ]
    coverages = [
        lower <= utilities[(family, token)] <= upper
        for family, values in meta_intervals.items()
        for token, (_, lower, upper) in values.items()
    ]
    errors = [
        abs(point - utilities[(family, token)])
        for family, values in meta_intervals.items()
        for token, (point, _, _) in values.items()
    ]
    conformal_metrics = AdvisorV31ConformalQualificationMetrics(
        empirical_coverage=_metric(sum(coverages) / len(coverages)),
        mean_interval_width=_metric(float(np.mean(widths))),
        mean_absolute_error=_metric(float(np.mean(errors))),
    )

    threshold_vectors = {family_id: family_vectors[family_id] for family_id in conformal}
    nearest_conformal = sorted(
        min(distance.compute_distance(vector, train) for train in train_vectors.values())
        for vector in threshold_vectors.values()
    )
    threshold = nearest_conformal[
        min(
            len(nearest_conformal),
            math.ceil((len(nearest_conformal) + 1) * policy.ood_distance_quantile),
        )
        - 1
    ]
    locked_false = sum(
        min(
            distance.compute_distance(family_vectors[family], train)
            for train in train_vectors.values()
        )
        > threshold
        for family in locked
    )
    detected = sum(
        min(
            distance.compute_distance(family_vectors[family], train)
            for train in train_vectors.values()
        )
        > threshold
        for family in ood
    )
    ood_rate = _metric(detected / len(ood))
    false_rate = _metric(locked_false / len(locked))
    ood_metrics = AdvisorV31OODQualificationMetrics(
        detected_ood_family_count=detected,
        falsely_abstained_locked_family_count=locked_false,
        ood_detection_rate=ood_rate,
        locked_false_abstention_rate=false_rate,
        balanced_abstention_accuracy=_metric((ood_rate + (1.0 - false_rate)) / 2.0),
    )

    order = tuple(
        sorted(
            training,
            key=lambda family: hashlib.sha256(
                f"{policy.policy_digest}:{family}".encode()
            ).hexdigest(),
        )
    )
    curve: list[AdvisorV31LearningCurvePoint] = []
    for count in policy.learning_curve_family_counts:
        curve_similarity, _ = method_metrics(
            "similarity",
            similarity(locked, order[:count]),
            selection_stability=1.0,
        )
        intervals, _ = _fit_ridge_predict(
            training_family_ids=order[:count],
            conformal_family_ids=conformal,
            target_family_ids=locked,
            vectors=family_vectors,
            utilities=utilities,
            policy=policy,
        )
        curve_meta, _ = method_metrics(
            "meta_ranker",
            {
                family: {token: value[0] for token, value in values.items()}
                for family, values in intervals.items()
            },
            selection_stability=1.0,
        )
        curve.append(
            AdvisorV31LearningCurvePoint(
                training_family_count=count,
                similarity_mean_regret=curve_similarity.mean_regret,
                meta_ranker_mean_regret=curve_meta.mean_regret,
            )
        )
    fixed_baselines: list[AdvisorV31FixedRecipeBaselineMetrics] = []
    for token in REQUIRED_ADVISOR_RECIPE_TOKENS:
        regrets = [
            max(utilities[(family, candidate)] for candidate in REQUIRED_ADVISOR_RECIPE_TOKENS)
            - utilities[(family, token)]
            for family in locked
        ]
        fixed_baselines.append(
            AdvisorV31FixedRecipeBaselineMetrics(
                recipe_token=token,
                mean_regret=_metric(float(np.mean(regrets))),
                median_regret=_metric(float(statistics.median(regrets))),
                oracle_coverage=_metric(sum(regret <= 1e-12 for regret in regrets) / len(locked)),
            )
        )
    best_fixed = min(item.mean_regret for item in fixed_baselines)
    stage2_improvement = best_fixed - stage2.mean_regret
    stage3_improvement = stage2.mean_regret - stage3.mean_regret
    curve_tail = abs(curve[-1].meta_ranker_mean_regret - curve[-2].meta_ranker_mean_regret)
    ood_qualified = (
        ood_metrics.ood_detection_rate >= policy.minimum_ood_detection_rate
        and ood_metrics.locked_false_abstention_rate <= policy.maximum_locked_false_abstention_rate
    )
    conformal_qualified = (
        conformal_metrics.empirical_coverage >= policy.minimum_locked_interval_coverage
        and conformal_metrics.mean_interval_width <= policy.maximum_mean_interval_width
    )
    learning_curve_qualified = (
        curve_tail <= policy.maximum_learning_curve_tail_regret_range
        and curve[-1].meta_ranker_mean_regret <= curve[0].meta_ranker_mean_regret
    )
    stage2_qualified = (
        stage2_improvement >= policy.minimum_stage2_regret_improvement
        and stage2.top2_oracle_coverage >= policy.minimum_top2_oracle_coverage
        and stage2.selection_stability >= policy.minimum_selection_stability
        and ood_qualified
    )
    stage3_qualified = (
        stage2_qualified
        and stage3_improvement >= policy.minimum_stage3_regret_improvement
        and stage3.top2_oracle_coverage >= policy.minimum_top2_oracle_coverage
        and stage3.selection_stability >= policy.minimum_selection_stability
        and conformal_qualified
        and learning_curve_qualified
    )
    gates = {
        "stage2_regret_improvement": stage2_improvement >= policy.minimum_stage2_regret_improvement,
        "stage2_top2_oracle_coverage": stage2.top2_oracle_coverage
        >= policy.minimum_top2_oracle_coverage,
        "stage2_selection_stability": stage2.selection_stability
        >= policy.minimum_selection_stability,
        "stage3_regret_improvement": stage3_improvement >= policy.minimum_stage3_regret_improvement,
        "stage3_top2_oracle_coverage": stage3.top2_oracle_coverage
        >= policy.minimum_top2_oracle_coverage,
        "stage3_selection_stability": stage3.selection_stability
        >= policy.minimum_selection_stability,
        "locked_interval_coverage": conformal_metrics.empirical_coverage
        >= policy.minimum_locked_interval_coverage,
        "mean_interval_width": conformal_metrics.mean_interval_width
        <= policy.maximum_mean_interval_width,
        "ood_detection_rate": ood_metrics.ood_detection_rate >= policy.minimum_ood_detection_rate,
        "locked_false_abstention_rate": ood_metrics.locked_false_abstention_rate
        <= policy.maximum_locked_false_abstention_rate,
        "learning_curve_stability": learning_curve_qualified,
    }
    failed = tuple(code for code, passed in gates.items() if not passed)
    evidence_digest = _digest(
        sorted((item.evidence_digest, item.run_count, item.recipe_token) for item in evidence)
    )
    family_split_digest = _digest(
        {
            "role_assignments": tuple(sorted(roles.items())),
            "policy_digest": policy.policy_digest,
        }
    )
    return AdvisorV31QualificationReport(
        qualification_status=(
            "qualified" if stage2_qualified and stage3_qualified else "not_qualified"
        ),
        amendment_digest=readiness.amendment_digest,
        source_execution_approval_digest=readiness.source_execution_approval_digest,
        source_execution_provenance_digest=readiness.source_execution_provenance_digest,
        source_v3_readiness_digest=readiness.source_v3_readiness_digest,
        source_registry_snapshot_digest=readiness.source_registry_snapshot_digest,
        analysis_provenance_digest=readiness.analysis_provenance_digest,
        remediation_approval_digest=readiness.remediation_approval_digest,
        remediation_readiness_digest=readiness.remediation_readiness_digest,
        qualification_readiness_digest=readiness.readiness_digest,
        approval_digest=approval.approval_digest,
        policy_digest=policy.policy_digest,
        evaluation_algorithm_digest=advisor_v31_evaluation_algorithm_digest(),
        evidence_digest=evidence_digest,
        family_split_digest=family_split_digest,
        meta_model_digest=meta_model_digest,
        fixed_baselines=tuple(fixed_baselines),
        stage2_similarity=stage2,
        stage3_meta_ranker=stage3,
        conformal=conformal_metrics,
        ood=ood_metrics,
        learning_curve=tuple(curve),
        stage2_regret_improvement_over_best_fixed=_metric(stage2_improvement),
        stage3_regret_improvement_over_stage2=_metric(stage3_improvement),
        learning_curve_tail_regret_range=_metric(curve_tail),
        stage2_qualified=stage2_qualified,
        stage3_qualified=stage3_qualified,
        ood_qualified=ood_qualified,
        conformal_qualified=conformal_qualified,
        learning_curve_qualified=learning_curve_qualified,
        fallback_to_similarity_required=not stage3_qualified,
        failed_gate_codes=failed,
    )


__all__ = [
    "AdvisorV3QualificationPolicy",
    "AdvisorV31ConformalQualificationMetrics",
    "AdvisorV31FamilyUtilityEvidence",
    "AdvisorV31FixedRecipeBaselineMetrics",
    "AdvisorV31LearningCurvePoint",
    "AdvisorV31MethodQualificationMetrics",
    "AdvisorV31OODQualificationMetrics",
    "AdvisorV31QualificationApproval",
    "AdvisorV31QualificationArtifact",
    "AdvisorV31QualificationReadinessArtifact",
    "AdvisorV31QualificationReport",
    "advisor_v3_evaluation_algorithm_digest",
    "advisor_v31_evaluation_algorithm_digest",
    "build_advisor_v31_qualification_readiness",
    "deserialize_advisor_v31_qualification_artifact",
    "load_advisor_v31_qualification_artifact",
    "serialize_advisor_v31_qualification_artifact",
    "write_advisor_v31_qualification_artifact",
]
