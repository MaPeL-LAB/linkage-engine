"""Observable, aggregate-only mechanism profile for the prospective advisor-v3 design."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from mapel_linkage.profiling.contracts import PreflightTaskProfile
from mapel_linkage.synthetic.generator import SyntheticRecord


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_FEATURE_NAMES = (
    "script_variation_rate",
    "punctuation_variation_rate",
    "tokenization_variation_rate",
    "missingness_mean",
    "missingness_asymmetry",
    "frequency_concentration",
    "candidate_ambiguity_scale",
    "duplicate_signature_rate",
    "planned_training_label_budget_scale",
)


class AdvisorV3MechanismProfile(BaseModel):
    """Versioned non-truth aggregate inputs shared by fitting and target recommendation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    profile_schema_version: Literal["3"] = "3"
    feature_source: Literal["generated_synthetic_aggregate", "unavailable"]
    feature_source_policy: Literal["observable_non_truth_aggregate_only"] = (
        "observable_non_truth_aggregate_only"
    )
    observed_record_count: Annotated[StrictInt, Field(ge=0)]
    script_variation_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] | None
    punctuation_variation_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] | None
    tokenization_variation_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] | None
    missingness_mean: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] | None
    missingness_asymmetry: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] | None
    frequency_concentration: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] | None
    candidate_ambiguity_scale: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] | None
    duplicate_signature_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] | None
    planned_training_label_budget_scale: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] | None
    complete: bool
    contains_truth_values: Literal[False] = False
    contains_outcomes: Literal[False] = False
    contains_record_values: Literal[False] = False
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        values = tuple(getattr(self, name) for name in _FEATURE_NAMES)
        expected_complete = all(value is not None for value in values)
        if self.complete != expected_complete:
            raise ValueError("Mechanism-profile completeness must match explicit availability.")
        if self.feature_source == "unavailable" and (
            self.complete
            or self.observed_record_count != 0
            or any(value is not None for value in values)
        ):
            raise ValueError("Unavailable mechanism evidence cannot silently encode zero features.")
        if self.feature_source != "unavailable" and not self.complete:
            raise ValueError(
                "Observed mechanism evidence must provide the complete feature schema."
            )
        return self

    @property
    def profile_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    @classmethod
    def unavailable(cls) -> AdvisorV3MechanismProfile:
        """Represent missing target-side mechanism evidence without zero imputation."""

        return cls(
            feature_source="unavailable",
            observed_record_count=0,
            script_variation_rate=None,
            punctuation_variation_rate=None,
            tokenization_variation_rate=None,
            missingness_mean=None,
            missingness_asymmetry=None,
            frequency_concentration=None,
            candidate_ambiguity_scale=None,
            duplicate_signature_rate=None,
            planned_training_label_budget_scale=None,
            complete=False,
        )


def advisor_v3_feature_source_policy_digest() -> str:
    """Bind the exact observable feature schema and missing-evidence behavior."""

    return _digest(
        {
            "profile_schema_version": "3",
            "feature_source_policy": "observable_non_truth_aggregate_only",
            "feature_names": _FEATURE_NAMES,
            "dataset_and_record_order": {
                "datasets": "mapping_keys_lexicographic_ascending",
                "records": "dataset_order_then_input_tuple_order",
                "observed_record_count": "sum_lengths_of_all_ordered_datasets",
            },
            "shared_ratio": "float(numerator/denominator)_if_denominator_gt_0_else_0.0",
            "script_variation_rate": {
                "eligible_values": "str(label_value)_where_label_value_is_not_None",
                "numerator": "count_values_with_any_unicode_codepoint_ord_gt_127",
                "denominator": "count_eligible_values",
                "cap": "ratio_in_zero_one_no_additional_cap",
            },
            "punctuation_variation_rate": {
                "eligible_values": "str(label_value)_where_label_value_is_not_None",
                "regex": "[^\\w\\s-]",
                "regex_api": "re.search_flags_re.UNICODE_boolean",
                "numerator": "count_values_with_regex_match",
                "denominator": "count_eligible_values",
                "cap": "ratio_in_zero_one_no_additional_cap",
            },
            "tokenization_variation_rate": {
                "eligible_values": "str(label_value)_where_label_value_is_not_None",
                "split_regex": "[-\\s']+",
                "token_filter": "filter_None_equivalent_remove_empty_strings",
                "numerator": "count_values_with_token_count_not_equal_2",
                "denominator": "count_eligible_values",
                "cap": "ratio_in_zero_one_no_additional_cap",
            },
            "missingness": {
                "per_dataset": "count_label_value_is_None_divided_by_dataset_length",
                "eligible_datasets": "nonempty_datasets_only",
                "mean": "unweighted_sum_dataset_rates_divided_by_nonempty_dataset_count",
                "asymmetry": "max_dataset_rate_minus_min_dataset_rate",
                "cap": "both_intrinsically_zero_one_no_additional_cap",
            },
            "frequency_concentration": {
                "eligible_values": "group_value_where_not_None_preserving_python_equality",
                "numerator": "maximum_Counter_count_default_0",
                "denominator": "max(1,count_eligible_group_values)",
                "cap": "intrinsically_zero_one_no_additional_cap",
            },
            "candidate_ambiguity_scale": {
                "two_dataset_condition": "exactly_2_lexicographically_ordered_datasets",
                "blocking_union": (
                    "left.group_value_is_not_None_AND_right.group_value_is_not_None_AND_"
                    "str(left.group_value)==str(right.group_value)",
                    "OR_left.label_value_is_not_None_AND_right.label_value_is_not_None_AND_"
                    "len(str(left.label_value))>=3_AND_len(str(right.label_value))>=3_AND_"
                    "str(left.label_value)[:3]==str(right.label_value)[:3]",
                ),
                "union_identity": "deduplicated_positional_(left_index,right_index)",
                "candidate_count": "cardinality_of_blocking_union",
                "denominator": "max(1,len(left)*len(right))",
                "normalization": "sqrt(candidate_count/denominator)",
                "cap": "min(1.0,normalization)",
                "non_two_dataset_value": 1.0,
            },
            "duplicate_signature_rate": {
                "scope": "within_each_dataset_never_cross_dataset",
                "signature": "(label_value,date_value,group_value)_including_None",
                "repeated_record_numerator": (
                    "sum_full_signature_count_for_each_within_dataset_signature_count_gt_1"
                ),
                "denominator": "all_records_across_all_datasets",
                "cap": "min(1.0,numerator/denominator)",
            },
            "planned_training_label_budget_scale": {
                "formula": "min(1.0,log1p(planned_training_label_budget)/log1p(1000))",
                "input_failure": "planned_training_label_budget_lt_2_fails",
            },
            "task_profile_scope_check": (
                "task_profile.source_count+task_profile.target_count_must_be_gt_0"
            ),
            "empty_and_missing_behavior": {
                "no_records": "fail",
                "no_nonmissing_labels": (
                    "script_punctuation_tokenization_rates_are_explicit_observed_0.0"
                ),
                "unavailable_target_profile": "all_features_None_count_0_complete_false",
                "missing_complete_feature": "abstain_or_v2_fallback_no_zero_imputation",
                "observed_profile_missing_any_feature": "fail",
            },
            "runtime_aggregate_producer": "not_implemented",
            "synthetic_family_profile_replicates": 5,
            "synthetic_family_profile_seed_policy": "benchmark_replicate_seed_v1",
            "contains_truth_values": False,
            "contains_outcomes": False,
        }
    )


def _bounded_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _candidate_pair_count(
    left: tuple[SyntheticRecord, ...], right: tuple[SyntheticRecord, ...]
) -> int:
    """Count the exact package blocking union without returning or persisting pair keys."""

    by_group_right: dict[str, list[int]] = {}
    by_prefix_right: dict[str, list[int]] = {}
    for index, item in enumerate(right):
        if item.group_value is not None:
            by_group_right.setdefault(str(item.group_value), []).append(index)
        if item.label_value is not None and len(str(item.label_value)) >= 3:
            by_prefix_right.setdefault(str(item.label_value)[:3], []).append(index)
    pairs: set[tuple[int, int]] = set()
    for left_index, item in enumerate(left):
        if item.group_value is not None:
            pairs.update(
                (left_index, right_index)
                for right_index in by_group_right.get(str(item.group_value), ())
            )
        if item.label_value is not None and len(str(item.label_value)) >= 3:
            pairs.update(
                (left_index, right_index)
                for right_index in by_prefix_right.get(str(item.label_value)[:3], ())
            )
    return len(pairs)


def build_advisor_v3_mechanism_profile(
    *,
    datasets: Mapping[str, tuple[SyntheticRecord, ...]],
    task_profile: PreflightTaskProfile,
    planned_training_label_budget: int,
) -> AdvisorV3MechanismProfile:
    """Extract synthetic v3 features without establishing a runtime-data producer."""

    if planned_training_label_budget < 2:
        raise ValueError("Planned v3 training-label budgets must retain both classes.")
    ordered = tuple(datasets[name] for name in sorted(datasets))
    records = tuple(item for dataset in ordered for item in dataset)
    if not records:
        raise ValueError("Mechanism profiling requires at least one observable record.")

    label_values = tuple(str(item.label_value) for item in records if item.label_value is not None)
    script_rate = _bounded_ratio(
        sum(any(ord(character) > 127 for character in value) for value in label_values),
        len(label_values),
    )
    punctuation_rate = _bounded_ratio(
        sum(bool(re.search(r"[^\w\s-]", value, flags=re.UNICODE)) for value in label_values),
        len(label_values),
    )
    tokenization_rate = _bounded_ratio(
        sum(len(tuple(filter(None, re.split(r"[-\s']+", value)))) != 2 for value in label_values),
        len(label_values),
    )

    missing_rates = tuple(
        _bounded_ratio(sum(item.label_value is None for item in dataset), len(dataset))
        for dataset in ordered
        if dataset
    )
    missing_mean = sum(missing_rates) / len(missing_rates)
    missing_asymmetry = max(missing_rates) - min(missing_rates)

    groups = tuple(item.group_value for item in records if item.group_value is not None)
    group_counts = Counter(groups)
    concentration = max(group_counts.values(), default=0) / max(1, len(groups))

    if len(ordered) == 2:
        candidate_count = _candidate_pair_count(ordered[0], ordered[1])
        candidate_scale = math.sqrt(candidate_count / max(1, len(ordered[0]) * len(ordered[1])))
    else:
        candidate_scale = 1.0

    repeated = 0
    for dataset in ordered:
        signatures = Counter(
            (item.label_value, item.date_value, item.group_value) for item in dataset
        )
        repeated += sum(count for count in signatures.values() if count > 1)
    duplicate_rate = repeated / len(records)
    label_scale = min(1.0, math.log1p(planned_training_label_budget) / math.log1p(1_000))

    # The task profile is used as an explicit provenance check. No truth or model outcome is read.
    if task_profile.source_count + task_profile.target_count <= 0:
        raise ValueError("Mechanism profiling requires an observed task source scope.")
    return AdvisorV3MechanismProfile(
        feature_source="generated_synthetic_aggregate",
        observed_record_count=len(records),
        script_variation_rate=float(script_rate),
        punctuation_variation_rate=float(punctuation_rate),
        tokenization_variation_rate=float(tokenization_rate),
        missingness_mean=float(missing_mean),
        missingness_asymmetry=float(missing_asymmetry),
        frequency_concentration=float(concentration),
        candidate_ambiguity_scale=float(min(1.0, candidate_scale)),
        duplicate_signature_rate=float(min(1.0, duplicate_rate)),
        planned_training_label_budget_scale=float(label_scale),
        complete=True,
    )


__all__ = [
    "AdvisorV3MechanismProfile",
    "advisor_v3_feature_source_policy_digest",
    "build_advisor_v3_mechanism_profile",
]
