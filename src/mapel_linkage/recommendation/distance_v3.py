"""Versioned mechanism-aware distances for the prospective advisor-v3 design."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, NoReturn

from mapel_linkage.benchmarking.advisor_v3_features import (
    AdvisorV3MechanismProfile,
    advisor_v3_feature_source_policy_digest,
)
from mapel_linkage.benchmarking.advisor_v3_label_budget import (
    advisor_v3_label_budget_policy_digest,
)
from mapel_linkage.benchmarking.generator import BenchmarkScenarioGenerator, ScenarioLatentSpec
from mapel_linkage.benchmarking.runner import benchmark_replicate_seed
from mapel_linkage.profiling.contracts import LabelEvidenceClass, PreflightTaskProfile
from mapel_linkage.recommendation.utility import REQUIRED_ADVISOR_RECIPE_TOKENS


class AdvisorV3FeatureUnavailableError(ValueError):
    """Raised when target-side aggregate evidence cannot populate the v3 feature schema."""


ADVISOR_V3_META_FEATURE_WEIGHTS: dict[str, float] = {
    "linkage_mode": 2.0,
    "assignment_constraint": 1.5,
    "label_volume_class": 1.0,
    "script_variation_rate": 1.5,
    "punctuation_variation_rate": 1.0,
    "tokenization_variation_rate": 1.0,
    "missingness_mean": 1.0,
    "missingness_asymmetry": 1.2,
    "frequency_concentration": 1.0,
    "candidate_ambiguity_scale": 1.2,
    "duplicate_signature_rate": 1.0,
    "planned_training_label_budget_scale": 1.0,
}


@dataclass(frozen=True, slots=True)
class MechanismAwareTaskMetaFeatureVector:
    """Advisor-v3 features with fitting/target source parity and no latent truth inputs."""

    linkage_mode: str
    assignment_constraint: str
    label_volume_class: str
    script_variation_rate: float
    punctuation_variation_rate: float
    tokenization_variation_rate: float
    missingness_mean: float
    missingness_asymmetry: float
    frequency_concentration: float
    candidate_ambiguity_scale: float
    duplicate_signature_rate: float
    planned_training_label_budget_scale: float

    CATEGORICAL_FEATURES: ClassVar[tuple[str, ...]] = (
        "linkage_mode",
        "assignment_constraint",
        "label_volume_class",
    )
    CONTINUOUS_FEATURES: ClassVar[tuple[str, ...]] = (
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

    @classmethod
    def from_profiles(
        cls,
        task_profile: PreflightTaskProfile | Any,
        mechanism_profile: AdvisorV3MechanismProfile,
    ) -> MechanismAwareTaskMetaFeatureVector:
        """Build the vector only when the complete observable feature schema is available."""

        if not mechanism_profile.complete or mechanism_profile.feature_source == "unavailable":
            raise AdvisorV3FeatureUnavailableError(
                "Advisor-v3 mechanism features are unavailable; abstain or use the v2 fallback."
            )
        values = {name: getattr(mechanism_profile, name) for name in cls.CONTINUOUS_FEATURES}
        if any(value is None for value in values.values()):
            raise AdvisorV3FeatureUnavailableError(
                "Advisor-v3 mechanism features are incomplete; zero imputation is prohibited."
            )
        evidence = getattr(task_profile, "label_evidence_class", LabelEvidenceClass.NONE)
        if evidence is LabelEvidenceClass.NONE:
            label_class = "none"
        elif evidence in (
            LabelEvidenceClass.UNVERIFIED_REFERENCE,
            LabelEvidenceClass.SYNTHETIC_TRUTH,
        ):
            label_class = "sparse"
        else:
            label_class = "dense"
        return cls(
            linkage_mode=str(getattr(task_profile, "linkage_mode", "link_only")),
            assignment_constraint=str(getattr(task_profile, "assignment_constraint", "one_to_one")),
            label_volume_class=label_class,
            **{name: float(value) for name, value in values.items() if value is not None},
        )

    @classmethod
    def from_latent_spec(cls, _spec: ScenarioLatentSpec) -> NoReturn:
        """Reject the v2 latent shortcut because runtime targets cannot supply latent rates."""

        raise AdvisorV3FeatureUnavailableError(
            "Latent simulator parameters are prohibited advisor-v3 meta-features."
        )

    def to_dict(self) -> dict[str, str | float]:
        return {
            name: getattr(self, name)
            for name in (*self.CATEGORICAL_FEATURES, *self.CONTINUOUS_FEATURES)
        }


class MechanismAwareMetaFeatureDistanceComputer:
    """Weighted Gower distance over the complete v3 feature schema."""

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights = (
            dict(weights) if weights is not None else dict(ADVISOR_V3_META_FEATURE_WEIGHTS)
        )

    def compute_distance(
        self,
        left: MechanismAwareTaskMetaFeatureVector,
        right: MechanismAwareTaskMetaFeatureVector,
    ) -> float:
        left_values = left.to_dict()
        right_values = right.to_dict()
        numerator = 0.0
        denominator = 0.0
        for name in MechanismAwareTaskMetaFeatureVector.CATEGORICAL_FEATURES:
            weight = self.weights.get(name, 1.0)
            denominator += weight
            numerator += weight * (0.0 if left_values[name] == right_values[name] else 1.0)
        for name in MechanismAwareTaskMetaFeatureVector.CONTINUOUS_FEATURES:
            weight = self.weights.get(name, 1.0)
            denominator += weight
            numerator += weight * abs(float(left_values[name]) - float(right_values[name]))
        return min(1.0, max(0.0, numerator / denominator)) if denominator > 0 else 0.0

    def find_nearest_families(
        self,
        target: MechanismAwareTaskMetaFeatureVector,
        family_vectors: Mapping[str, MechanismAwareTaskMetaFeatureVector],
        *,
        k: int = 3,
    ) -> tuple[tuple[str, float], ...]:
        if k <= 0:
            return ()
        scored = tuple(
            (family_id, self.compute_distance(target, vector))
            for family_id, vector in family_vectors.items()
        )
        return tuple(sorted(scored, key=lambda item: (item[1], item[0]))[:k])


def advisor_v3_ood_distance_rule_digest() -> str:
    """Bind the outcome-free conformal-distance threshold selection algorithm."""

    return hashlib.sha256(
        json.dumps(
            {
                "rule_id": "advisor_v3_conformal_nearest_training_distance_v1",
                "fit_roles": ("meta_training", "conformal"),
                "prohibited_roles": ("locked_evaluation", "ood_holdout"),
                "distance_metric": "weighted_gower",
                "quantile": 0.90,
                "finite_sample_rank": "ceil((n+1)*quantile)_capped_at_n",
                "tie_rule": "stable_numeric_order_statistic",
                "outcomes_used": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def select_advisor_v3_ood_distance_threshold(
    *,
    training_vectors: Mapping[str, MechanismAwareTaskMetaFeatureVector],
    conformal_vectors: Mapping[str, MechanismAwareTaskMetaFeatureVector],
) -> float:
    """Select a threshold from conformal-to-training geometry without outcomes or holdouts."""

    if not training_vectors or not conformal_vectors:
        raise ValueError("Advisor-v3 OOD threshold selection requires both fitting roles.")
    if set(training_vectors) & set(conformal_vectors):
        raise ValueError("Advisor-v3 OOD threshold roles must be family-disjoint.")
    computer = MechanismAwareMetaFeatureDistanceComputer()
    nearest = sorted(
        min(computer.compute_distance(vector, train) for train in training_vectors.values())
        for vector in conformal_vectors.values()
    )
    rank = min(len(nearest), math.ceil((len(nearest) + 1) * 0.90))
    return float(nearest[rank - 1])


def advisor_v3_feature_model_schema_digest() -> str:
    """Bind v3 feature order, geometry, profiling seeds, and learned design columns."""

    payload = {
        "schema_id": "advisor_v3_mechanism_feature_model_v1",
        "feature_source_policy_digest": advisor_v3_feature_source_policy_digest(),
        "label_budget_policy_digest": advisor_v3_label_budget_policy_digest(),
        "categorical_feature_order": MechanismAwareTaskMetaFeatureVector.CATEGORICAL_FEATURES,
        "continuous_feature_order": MechanismAwareTaskMetaFeatureVector.CONTINUOUS_FEATURES,
        "distance_metric": "weighted_gower",
        "distance_normalization": "weighted_sum_divided_by_total_weight_clipped_0_1",
        "distance_weights": ADVISOR_V3_META_FEATURE_WEIGHTS,
        "ood_distance_rule_digest": advisor_v3_ood_distance_rule_digest(),
        "family_profile_aggregation": "arithmetic_mean_across_instances_and_replicates",
        "family_profile_base_seed": 20260816,
        "family_profile_replicates": 5,
        "family_profile_seed_policy": "benchmark_replicate_seed_v1",
        "learned_continuous_feature_order": (
            MechanismAwareTaskMetaFeatureVector.CONTINUOUS_FEATURES
        ),
        "learned_recipe_feature_order": REQUIRED_ADVISOR_RECIPE_TOKENS,
        "missing_feature_behavior": "abstain_or_v2_fallback_no_zero_imputation",
        "runtime_feature_producer": "not_implemented",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def extract_advisor_v3_family_meta_features(
    generator: BenchmarkScenarioGenerator,
    *,
    family_ids: frozenset[str],
    base_seed: int = 20260816,
    profiling_replicates: int = 5,
) -> dict[str, MechanismAwareTaskMetaFeatureVector]:
    """Average observable generated profiles by family without reading truth or outcomes."""

    if profiling_replicates != 5:
        raise ValueError("Advisor-v3 observable family profiling is fixed at five replicates.")
    result: dict[str, MechanismAwareTaskMetaFeatureVector] = {}
    for family_id in sorted(family_ids):
        instances = generator.list_instances(family_id=family_id)
        if not instances:
            raise ValueError("Advisor-v3 feature extraction found an empty family.")
        vectors = []
        for instance in instances:
            for replicate_number in range(profiling_replicates):
                seed = benchmark_replicate_seed(
                    instance_id=instance.instance_id,
                    replicate_number=replicate_number,
                    base_seed=base_seed,
                )
                bundle = generator.generate(instance.instance_id, seed=seed)
                mechanism_profile = generator.build_advisor_v3_mechanism_profile(
                    instance.instance_id,
                    seed=seed,
                    generated_bundle=bundle,
                )
                vectors.append(
                    MechanismAwareTaskMetaFeatureVector.from_profiles(
                        bundle.task_profile,
                        mechanism_profile,
                    )
                )
        representative = vectors[0]
        if any(
            tuple(getattr(item, name) for name in item.CATEGORICAL_FEATURES)
            != tuple(getattr(representative, name) for name in representative.CATEGORICAL_FEATURES)
            for item in vectors[1:]
        ):
            raise ValueError("Advisor-v3 family categorical features are not coherent.")
        result[family_id] = MechanismAwareTaskMetaFeatureVector(
            linkage_mode=representative.linkage_mode,
            assignment_constraint=representative.assignment_constraint,
            label_volume_class=representative.label_volume_class,
            **{
                name: math.fsum(getattr(item, name) for item in vectors) / len(vectors)
                for name in representative.CONTINUOUS_FEATURES
            },
        )
    return result


__all__ = [
    "ADVISOR_V3_META_FEATURE_WEIGHTS",
    "AdvisorV3FeatureUnavailableError",
    "MechanismAwareMetaFeatureDistanceComputer",
    "MechanismAwareTaskMetaFeatureVector",
    "advisor_v3_feature_model_schema_digest",
    "advisor_v3_ood_distance_rule_digest",
    "extract_advisor_v3_family_meta_features",
    "select_advisor_v3_ood_distance_threshold",
]
