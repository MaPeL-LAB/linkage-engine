"""Meta-feature extraction and distance computation for Stage-2 Linkage Strategy Advisor."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from mapel_linkage.benchmarking.generator import (
    BenchmarkScenarioGenerator,
    ScenarioLatentSpec,
)
from mapel_linkage.profiling.contracts import (
    CountBand,
    LabelEvidenceClass,
    PreflightTaskProfile,
)

_COUNT_BAND_SCALES: dict[CountBand, float] = {
    CountBand.NOT_OBSERVED: 0.0,
    CountBand.VERY_SMALL: 0.2,
    CountBand.SMALL: 0.4,
    CountBand.MEDIUM: 0.6,
    CountBand.LARGE: 0.8,
    CountBand.VERY_LARGE: 1.0,
}

_LABEL_EVIDENCE_SCALES: dict[LabelEvidenceClass, float] = {
    LabelEvidenceClass.NONE: 0.0,
    LabelEvidenceClass.UNVERIFIED_REFERENCE: 0.25,
    LabelEvidenceClass.SYNTHETIC_TRUTH: 0.5,
    LabelEvidenceClass.VERIFIED_HUMAN_ADJUDICATION: 0.8,
    LabelEvidenceClass.VERIFIED_GOLD_STANDARD: 1.0,
}

DEFAULT_META_FEATURE_WEIGHTS: dict[str, float] = {
    "linkage_mode": 2.5,
    "assignment_constraint": 1.5,
    "label_volume_class": 1.5,
    "candidate_edge_budget_scale": 1.0,
    "missingness_mean": 1.0,
    "missingness_max": 0.8,
    "entropy_estimate": 0.8,
    "error_estimate_approx": 1.2,
    "label_volume_scale": 1.0,
    "variable_count_scale": 0.5,
    "comparison_count_scale": 0.5,
    "blocking_rule_count_scale": 0.5,
    "record_count_log_ratio": 0.5,
}


@dataclass(frozen=True, slots=True)
class TaskMetaFeatureVector:
    """Normalized categorical and continuous meta-features extracted from task profile."""

    # Categorical meta-features
    linkage_mode: str
    assignment_constraint: str
    label_volume_class: str

    # Normalized continuous meta-features [0.0, 1.0]
    record_count_log_ratio: float
    missingness_mean: float
    missingness_max: float
    entropy_estimate: float
    candidate_edge_budget_scale: float
    error_estimate_approx: float
    label_volume_scale: float
    variable_count_scale: float
    comparison_count_scale: float
    blocking_rule_count_scale: float

    CATEGORICAL_FEATURES: ClassVar[tuple[str, ...]] = (
        "linkage_mode",
        "assignment_constraint",
        "label_volume_class",
    )

    CONTINUOUS_FEATURES: ClassVar[tuple[str, ...]] = (
        "record_count_log_ratio",
        "missingness_mean",
        "missingness_max",
        "entropy_estimate",
        "candidate_edge_budget_scale",
        "error_estimate_approx",
        "label_volume_scale",
        "variable_count_scale",
        "comparison_count_scale",
        "blocking_rule_count_scale",
    )

    @classmethod
    def from_profile(
        cls,
        profile: PreflightTaskProfile | Any,
        *,
        missingness_mean: float = 0.0,
        missingness_max: float = 0.0,
        error_estimate_approx: float = 0.0,
    ) -> TaskMetaFeatureVector:
        """Extract normalized meta-feature vector from PreflightTaskProfile or similar profile."""
        linkage_mode = getattr(profile, "linkage_mode", "link_only")
        assignment_constraint = getattr(profile, "assignment_constraint", "one_to_one")
        label_evidence_class = getattr(profile, "label_evidence_class", LabelEvidenceClass.NONE)

        if label_evidence_class is LabelEvidenceClass.NONE:
            label_volume_class = "none"
        elif label_evidence_class in (
            LabelEvidenceClass.UNVERIFIED_REFERENCE,
            LabelEvidenceClass.SYNTHETIC_TRUTH,
        ):
            label_volume_class = "sparse"
        else:
            label_volume_class = "dense"

        src_cnt = getattr(profile, "source_count", 1)
        tgt_cnt = getattr(profile, "target_count", 1)
        if src_cnt > 0 and tgt_cnt > 0:
            ratio = float(tgt_cnt) / float(src_cnt)
            log_r = math.log(ratio)
            record_count_log_ratio = min(1.0, max(0.0, (math.tanh(log_r) + 1.0) / 2.0))
        else:
            record_count_log_ratio = 0.5

        # Variable type entropy
        var_type_counts = getattr(profile, "variable_type_counts", ())
        total_vars = getattr(profile, "variable_count", sum(item.count for item in var_type_counts))
        if total_vars > 0 and var_type_counts:
            probs = [item.count / total_vars for item in var_type_counts if item.count > 0]
            denom = math.log2(6.0)
            entropy = -sum(p * math.log2(p) for p in probs) / denom if denom > 0 else 0.0
            entropy_estimate = min(1.0, max(0.0, entropy))
        else:
            entropy_estimate = 0.0

        budget_band = getattr(profile, "candidate_pair_budget_band", CountBand.NOT_OBSERVED)
        budget_scale = _COUNT_BAND_SCALES.get(budget_band, 0.0)

        label_scale = _LABEL_EVIDENCE_SCALES.get(label_evidence_class, 0.0)
        var_count = getattr(profile, "variable_count", 0)
        comp_count = getattr(profile, "comparison_count", 0)
        block_count = getattr(profile, "blocking_rule_count", 0)

        return cls(
            linkage_mode=linkage_mode,
            assignment_constraint=assignment_constraint,
            label_volume_class=label_volume_class,
            record_count_log_ratio=record_count_log_ratio,
            missingness_mean=min(1.0, max(0.0, missingness_mean)),
            missingness_max=min(1.0, max(0.0, missingness_max)),
            entropy_estimate=entropy_estimate,
            candidate_edge_budget_scale=budget_scale,
            error_estimate_approx=min(1.0, max(0.0, error_estimate_approx)),
            label_volume_scale=label_scale,
            variable_count_scale=min(1.0, max(0.0, var_count / 20.0)),
            comparison_count_scale=min(1.0, max(0.0, comp_count / 20.0)),
            blocking_rule_count_scale=min(1.0, max(0.0, block_count / 10.0)),
        )

    @classmethod
    def from_latent_spec(cls, spec: ScenarioLatentSpec) -> TaskMetaFeatureVector:
        """Extract normalized meta-feature vector from synthetic generator ScenarioLatentSpec."""
        if spec.label_volume <= 0:
            label_volume_class = "none"
            label_scale = 0.0
        elif spec.label_volume < 200:
            label_volume_class = "sparse"
            label_scale = min(1.0, spec.label_volume / 200.0 * 0.5)
        else:
            label_volume_class = "dense"
            label_scale = min(1.0, 0.5 + (spec.label_volume - 200) / 800.0 * 0.5)

        assignment_constraint = (
            "one_to_one" if spec.linkage_mode == "link_only" else "unconstrained"
        )
        total_error = (
            spec.typo_rate
            + spec.token_transposition_rate
            + spec.date_shift_rate
            + spec.date_ambiguity_rate
        )
        error_estimate = min(1.0, total_error / 1.5)

        missing_mean = spec.missingness_rate
        missing_max = (
            min(1.0, missing_mean * 2.0 + 0.2) if spec.informative_missingness else missing_mean
        )

        return cls(
            linkage_mode=spec.linkage_mode,
            assignment_constraint=assignment_constraint,
            label_volume_class=label_volume_class,
            record_count_log_ratio=0.5,
            missingness_mean=missing_mean,
            missingness_max=missing_max,
            entropy_estimate=0.61,
            candidate_edge_budget_scale=0.4,
            error_estimate_approx=error_estimate,
            label_volume_scale=label_scale,
            variable_count_scale=3 / 20.0,
            comparison_count_scale=3 / 20.0,
            blocking_rule_count_scale=1 / 10.0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert meta-features to serializable dictionary."""
        return {
            "linkage_mode": self.linkage_mode,
            "assignment_constraint": self.assignment_constraint,
            "label_volume_class": self.label_volume_class,
            "record_count_log_ratio": self.record_count_log_ratio,
            "missingness_mean": self.missingness_mean,
            "missingness_max": self.missingness_max,
            "entropy_estimate": self.entropy_estimate,
            "candidate_edge_budget_scale": self.candidate_edge_budget_scale,
            "error_estimate_approx": self.error_estimate_approx,
            "label_volume_scale": self.label_volume_scale,
            "variable_count_scale": self.variable_count_scale,
            "comparison_count_scale": self.comparison_count_scale,
            "blocking_rule_count_scale": self.blocking_rule_count_scale,
        }


class MetaFeatureDistanceComputer:
    """Computes weighted Gower or Euclidean distances between task and family meta-features."""

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights = dict(weights) if weights is not None else dict(DEFAULT_META_FEATURE_WEIGHTS)

    def compute_distance(
        self,
        v1: TaskMetaFeatureVector,
        v2: TaskMetaFeatureVector,
        metric: Literal["gower", "weighted_euclidean"] = "gower",
    ) -> float:
        """Compute normalized distance between two meta-feature vectors."""
        total_weight = 0.0
        dist_accum = 0.0
        d1 = v1.to_dict()
        d2 = v2.to_dict()

        for cat_feat in TaskMetaFeatureVector.CATEGORICAL_FEATURES:
            w = self.weights.get(cat_feat, 1.0)
            total_weight += w
            diff = 0.0 if d1[cat_feat] == d2[cat_feat] else 1.0
            if metric == "gower":
                dist_accum += w * diff
            else:
                dist_accum += w * (diff**2)

        for cont_feat in TaskMetaFeatureVector.CONTINUOUS_FEATURES:
            w = self.weights.get(cont_feat, 1.0)
            total_weight += w
            diff = abs(float(d1[cont_feat]) - float(d2[cont_feat]))
            if metric == "gower":
                dist_accum += w * diff
            else:
                dist_accum += w * (diff**2)

        if total_weight <= 0.0:
            return 0.0

        if metric == "gower":
            return min(1.0, max(0.0, dist_accum / total_weight))
        else:
            return min(1.0, max(0.0, math.sqrt(dist_accum / total_weight)))

    def find_nearest_families(
        self,
        target_vector: TaskMetaFeatureVector,
        family_vectors: Mapping[str, TaskMetaFeatureVector],
        k: int = 3,
        metric: Literal["gower", "weighted_euclidean"] = "gower",
    ) -> tuple[tuple[str, float], ...]:
        """Rank and return top-k nearest scenario families with their distances."""
        if not family_vectors or k <= 0:
            return ()

        scored: list[tuple[str, float]] = []
        for fam_id, fam_vec in family_vectors.items():
            dist = self.compute_distance(target_vector, fam_vec, metric=metric)
            scored.append((fam_id, dist))

        scored.sort(key=lambda item: (item[1], item[0]))
        return tuple(scored[:k])


def extract_family_meta_features(
    generator: BenchmarkScenarioGenerator | None = None,
) -> dict[str, TaskMetaFeatureVector]:
    """Extract canonical average meta-feature vectors for all families in the generator."""
    gen = generator or BenchmarkScenarioGenerator()
    family_vectors: dict[str, TaskMetaFeatureVector] = {}

    for fam in gen.list_families():
        instances = gen.list_instances(family_id=fam.family_id)
        if not instances:
            continue

        instance_vectors: list[TaskMetaFeatureVector] = []
        for inst in instances:
            spec = getattr(gen, "_instances", {}).get(inst.instance_id, (None, None))[1]
            if spec is not None:
                instance_vectors.append(TaskMetaFeatureVector.from_latent_spec(spec))
            else:
                profile = gen.build_task_profile(inst.instance_id)
                instance_vectors.append(TaskMetaFeatureVector.from_profile(profile))

        if not instance_vectors:
            continue

        n = len(instance_vectors)
        rep = instance_vectors[0]
        avg_log_ratio = sum(v.record_count_log_ratio for v in instance_vectors) / n
        avg_miss_mean = sum(v.missingness_mean for v in instance_vectors) / n
        avg_miss_max = sum(v.missingness_max for v in instance_vectors) / n
        avg_entropy = sum(v.entropy_estimate for v in instance_vectors) / n
        avg_budget = sum(v.candidate_edge_budget_scale for v in instance_vectors) / n
        avg_error = sum(v.error_estimate_approx for v in instance_vectors) / n
        avg_label_scale = sum(v.label_volume_scale for v in instance_vectors) / n
        avg_var_scale = sum(v.variable_count_scale for v in instance_vectors) / n
        avg_comp_scale = sum(v.comparison_count_scale for v in instance_vectors) / n
        avg_block_scale = sum(v.blocking_rule_count_scale for v in instance_vectors) / n

        family_vectors[fam.family_id] = TaskMetaFeatureVector(
            linkage_mode=rep.linkage_mode,
            assignment_constraint=rep.assignment_constraint,
            label_volume_class=rep.label_volume_class,
            record_count_log_ratio=avg_log_ratio,
            missingness_mean=avg_miss_mean,
            missingness_max=avg_miss_max,
            entropy_estimate=avg_entropy,
            candidate_edge_budget_scale=avg_budget,
            error_estimate_approx=avg_error,
            label_volume_scale=avg_label_scale,
            variable_count_scale=avg_var_scale,
            comparison_count_scale=avg_comp_scale,
            blocking_rule_count_scale=avg_block_scale,
        )

    return family_vectors


__all__ = [
    "DEFAULT_META_FEATURE_WEIGHTS",
    "MetaFeatureDistanceComputer",
    "TaskMetaFeatureVector",
    "extract_family_meta_features",
]
