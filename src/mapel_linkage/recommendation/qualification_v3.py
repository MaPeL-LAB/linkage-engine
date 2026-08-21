"""Outcome-free prospective qualification policy for advisor-v3."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from mapel_linkage.benchmarking.advisor_v3_features import (
    advisor_v3_feature_source_policy_digest,
)
from mapel_linkage.recommendation.distance_v3 import (
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


__all__ = [
    "AdvisorV3QualificationPolicy",
    "advisor_v3_evaluation_algorithm_digest",
    "advisor_v31_evaluation_algorithm_digest",
]
