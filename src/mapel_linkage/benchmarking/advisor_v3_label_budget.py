"""V3-only deterministic training-label budget with protected-partition preservation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from mapel_linkage.governance.labels import VerifiedLabelBatch, VerifiedPairLabel


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def advisor_v3_label_budget_policy_digest() -> str:
    """Bind the exact v3-only supervised label-selection algorithm."""

    return _digest(
        {
            "policy_id": "advisor_v3_training_label_budget_v1",
            "required_partitions_exactly_once": (
                "training",
                "validation",
                "calibration",
                "decision",
                "test",
            ),
            "eligible_partition": "training_only",
            "input_failures": {
                "planned_training_label_budget_lt_2": "fail",
                "missing_or_duplicate_required_partition": "fail",
                "training_has_no_positive_or_no_negative": "fail",
            },
            "effective_budget": "min(planned_training_label_budget,len(training.labels))",
            "initial_positive_target": (
                "python_builtin_round_half_to_even(effective_budget*positive_available/"
                "total_available)"
            ),
            "allocation_adjustments_in_order": (
                "positive_target=min(positive_available,max(1,initial_positive_target))",
                "negative_target=effective_budget-positive_target",
                "if_negative_target_lt_1_set_negative_1_and_positive_effective_minus_1",
                "if_negative_target_gt_negative_available_set_negative_available_and_"
                "positive_effective_minus_negative",
                "fail_if_positive_target_lt_1_or_gt_positive_available",
            ),
            "selection_hash": {
                "input_string": "str(random_seed)+U+0000+pair_digest_lower_hex",
                "encoding": "utf-8_strict",
                "hash": "sha256_hexdigest_lower_hex",
                "class_order": "selection_hash_ascending_then_pair_digest_ascending",
                "retained": "positive_prefix_then_negative_prefix_at_allocated_counts",
                "final_label_order": "pair_digest_ascending",
            },
            "seed_scope": (
                "random_seed_argument_is_exact_benchmark_instance_replicate_seed_and_is_"
                "shared_across_recipes_for_that_instance_replicate"
            ),
            "supervised_adapters_bounded": ("xgboost_classifier", "xgboost_ranker"),
            "fellegi_sunter_training_feature_surface": "full_protected_training_partition",
            "retained_batch": {
                "source_kind": "original_training.source_kind",
                "verification_protocol": "synthetic_benchmark_v3_label_budget",
                "partition": "training",
                "labels": "final_pair_digest_order",
            },
            "retained_source_digest": {
                "hash": "sha256_canonical_json_sort_keys_compact_utf8",
                "inputs": (
                    "original_source_digest",
                    "original_training_authority_digest",
                    "planned_training_label_budget",
                    "retained_pair_digests_sorted",
                    "random_seed",
                    "policy_digest",
                ),
            },
            "retained_authority_digest": {
                "derivation": "VerifiedLabelBatch.label_authority_digest_v1",
                "inputs": (
                    "source_kind",
                    "verification_protocol",
                    "retained_source_digest",
                    "partition",
                    "labels_sorted_by_pair_digest_with_binary_label_sorted_entity_components_"
                    "and_sorted_household_components",
                ),
            },
            "output_batch_order": "preserve_input_batch_order_replace_training_in_place",
            "non_training_partitions_mutated": False,
            "calibration_or_locked_access": False,
        }
    )


@dataclass(frozen=True, slots=True)
class AdvisorV3LabelBudgetReport:
    """Aggregate label-budget provenance with no pair or record references."""

    planned_training_label_budget: int
    available_training_label_count: int
    retained_training_label_count: int
    retained_positive_count: int
    retained_negative_count: int
    original_training_authority_digest: str
    retained_training_authority_digest: str
    policy_digest: str
    non_training_authority_digests_unchanged: bool
    contains_record_values: bool = False
    contains_pair_references: bool = False
    operational_validity: str = "not_established"

    @property
    def report_digest(self) -> str:
        return _digest(
            {
                "planned_training_label_budget": self.planned_training_label_budget,
                "available_training_label_count": self.available_training_label_count,
                "retained_training_label_count": self.retained_training_label_count,
                "retained_positive_count": self.retained_positive_count,
                "retained_negative_count": self.retained_negative_count,
                "original_training_authority_digest": self.original_training_authority_digest,
                "retained_training_authority_digest": self.retained_training_authority_digest,
                "policy_digest": self.policy_digest,
                "non_training_authority_digests_unchanged": (
                    self.non_training_authority_digests_unchanged
                ),
                "contains_record_values": self.contains_record_values,
                "contains_pair_references": self.contains_pair_references,
                "operational_validity": self.operational_validity,
            }
        )


def _selection_key(item: VerifiedPairLabel, *, random_seed: int) -> str:
    return hashlib.sha256(f"{random_seed}\x00{item.pair_digest()}".encode()).hexdigest()


def apply_advisor_v3_training_label_budget(
    batches: tuple[VerifiedLabelBatch, ...],
    *,
    planned_training_label_budget: int,
    random_seed: int,
) -> tuple[tuple[VerifiedLabelBatch, ...], AdvisorV3LabelBudgetReport]:
    """Bound only training labels; validation/calibration/decision/test stay byte-semantic equal."""

    if planned_training_label_budget < 2:
        raise ValueError("Advisor-v3 training-label budgets must retain both classes.")
    by_partition = {batch.partition: batch for batch in batches}
    if len(batches) != 5 or set(by_partition) != {
        "training",
        "validation",
        "calibration",
        "decision",
        "test",
    }:
        raise ValueError("Advisor-v3 label budgeting requires every protected partition.")
    training = by_partition["training"]
    positives = sorted(
        (item for item in training.labels if item.label == 1),
        key=lambda item: (_selection_key(item, random_seed=random_seed), item.pair_digest()),
    )
    negatives = sorted(
        (item for item in training.labels if item.label == 0),
        key=lambda item: (_selection_key(item, random_seed=random_seed), item.pair_digest()),
    )
    if not positives or not negatives:
        raise ValueError("Advisor-v3 training-label budgeting requires both classes.")
    retained_count = min(planned_training_label_budget, len(training.labels))
    positive_target = round(retained_count * len(positives) / len(training.labels))
    positive_target = min(len(positives), max(1, positive_target))
    negative_target = retained_count - positive_target
    if negative_target < 1:
        negative_target = 1
        positive_target = retained_count - 1
    if negative_target > len(negatives):
        negative_target = len(negatives)
        positive_target = retained_count - negative_target
    if positive_target < 1 or positive_target > len(positives):
        raise ValueError("Advisor-v3 label budget cannot retain both classes at this size.")
    retained = tuple(
        sorted(
            (*positives[:positive_target], *negatives[:negative_target]),
            key=lambda item: item.pair_digest(),
        )
    )
    retained_batch = VerifiedLabelBatch(
        source_kind=training.source_kind,
        verification_protocol="synthetic_benchmark_v3_label_budget",
        source_digest=_digest(
            {
                "original_source_digest": training.source_digest,
                "original_training_authority_digest": training.label_authority_digest,
                "planned_training_label_budget": planned_training_label_budget,
                "retained_pair_digests": sorted(item.pair_digest() for item in retained),
                "random_seed": random_seed,
                "policy_digest": advisor_v3_label_budget_policy_digest(),
            }
        ),
        partition="training",
        labels=retained,
    )
    output = tuple(retained_batch if batch.partition == "training" else batch for batch in batches)
    report = AdvisorV3LabelBudgetReport(
        planned_training_label_budget=planned_training_label_budget,
        available_training_label_count=len(training.labels),
        retained_training_label_count=len(retained),
        retained_positive_count=retained_batch.positive_count,
        retained_negative_count=retained_batch.negative_count,
        original_training_authority_digest=training.label_authority_digest,
        retained_training_authority_digest=retained_batch.label_authority_digest,
        policy_digest=advisor_v3_label_budget_policy_digest(),
        non_training_authority_digests_unchanged=all(
            original.label_authority_digest == updated.label_authority_digest
            for original, updated in zip(batches, output, strict=True)
            if original.partition != "training"
        ),
    )
    return output, report


__all__ = [
    "AdvisorV3LabelBudgetReport",
    "advisor_v3_label_budget_policy_digest",
    "apply_advisor_v3_training_label_budget",
]
