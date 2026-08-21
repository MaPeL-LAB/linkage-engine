"""Prospectively fixed advisor-v3 catalogue, roles, shards, and preregistration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from mapel_linkage.benchmarking.advisor_catalogue import AdvisorFamilyRole
from mapel_linkage.benchmarking.advisor_v3_features import (
    advisor_v3_feature_source_policy_digest,
)
from mapel_linkage.benchmarking.advisor_v3_label_budget import (
    advisor_v3_label_budget_policy_digest,
)
from mapel_linkage.benchmarking.generator import (
    BenchmarkScenarioGenerator,
    ScenarioLatentSpec,
    ScenarioMechanicExtensionV3,
)
from mapel_linkage.recommendation.distance_v3 import (
    MechanismAwareMetaFeatureDistanceComputer,
    advisor_v3_feature_model_schema_digest,
    extract_advisor_v3_family_meta_features,
    select_advisor_v3_ood_distance_threshold,
)
from mapel_linkage.recommendation.qualification_v3 import (
    AdvisorV3QualificationPolicy,
    advisor_v3_evaluation_algorithm_digest,
)
from mapel_linkage.recommendation.utility import ADVISOR_UTILITY_POLICY_DIGEST

_CATALOGUE_ID: Literal["advisor_v3"] = "advisor_v3"
_REQUIRED_ADAPTERS = (
    "recipe.fellegi_sunter_reference",
    "recipe.xgboost_classifier",
    "recipe.xgboost_ranker",
)
_PORTFOLIO_RECIPE_COUNT: Literal[7] = 7
_REPLICATES: Literal[5] = 5
_BASE_SEED: Literal[20260816] = 20260816
_SHARD_COUNT: Literal[42] = 42


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class AdvisorV3CorpusDesignManifest(BaseModel):
    """Outcome-free, family-disjoint prospective advisor-v3 experiment design."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    design_schema_version: Literal["3"] = "3"
    catalogue_id: Literal["advisor_v3"] = _CATALOGUE_ID
    family_count: Literal[84] = 84
    instance_count: Literal[336] = 336
    instances_per_family: Literal[4] = 4
    meta_training_family_count: Literal[48] = 48
    conformal_family_count: Literal[12] = 12
    locked_evaluation_family_count: Literal[12] = 12
    ood_holdout_family_count: Literal[12] = 12
    family_roles: tuple[tuple[StrictStr, AdvisorFamilyRole], ...]
    catalogue_manifest_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    required_success_adapters: tuple[StrictStr, ...] = _REQUIRED_ADAPTERS
    feature_source_policy_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    feature_model_schema_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    label_budget_policy_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    utility_policy_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    family_is_statistical_unit: Literal[True] = True
    role_assignment_fixed_before_outcomes: Literal[True] = True
    v2_locked_or_ood_units_reused: Literal[False] = False
    synthetic_only: Literal[True] = True
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    release_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_design(self) -> Self:
        family_ids = tuple(family_id for family_id, _ in self.family_roles)
        if family_ids != tuple(sorted(family_ids)) or len(set(family_ids)) != self.family_count:
            raise ValueError("Advisor-v3 family roles must be unique and canonically ordered.")
        counts = {
            role: sum(item_role == role for _, item_role in self.family_roles)
            for role in ("meta_training", "conformal", "locked_evaluation", "ood_holdout")
        }
        if counts != {
            "meta_training": self.meta_training_family_count,
            "conformal": self.conformal_family_count,
            "locked_evaluation": self.locked_evaluation_family_count,
            "ood_holdout": self.ood_holdout_family_count,
        }:
            raise ValueError("Advisor-v3 family-role counts do not match the fixed design.")
        if any(not family_id.startswith("family.advisor_v3.") for family_id in family_ids):
            raise ValueError("Advisor-v3 cannot reuse a prior catalogue family identifier.")
        if self.feature_source_policy_digest != advisor_v3_feature_source_policy_digest():
            raise ValueError("Advisor-v3 feature-source policy binding is stale.")
        if self.feature_model_schema_digest != advisor_v3_feature_model_schema_digest():
            raise ValueError("Advisor-v3 feature/model schema binding is stale.")
        if self.label_budget_policy_digest != advisor_v3_label_budget_policy_digest():
            raise ValueError("Advisor-v3 training-label budget binding is stale.")
        if self.utility_policy_digest != ADVISOR_UTILITY_POLICY_DIGEST:
            raise ValueError("Advisor-v3 utility policy binding is stale.")
        if self.catalogue_manifest_digest != _advisor_v3_catalogue_manifest_digest():
            raise ValueError("Advisor-v3 family or instance catalogue binding is stale.")
        return self

    @property
    def role_manifest_digest(self) -> str:
        return _digest(self.family_roles)

    @property
    def design_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {
            "design_schema_version": self.design_schema_version,
            "catalogue_id": self.catalogue_id,
            "design_digest": self.design_digest,
            "role_manifest_digest": self.role_manifest_digest,
            "catalogue_manifest_digest": self.catalogue_manifest_digest,
            "family_count": self.family_count,
            "instance_count": self.instance_count,
            "meta_training_family_count": self.meta_training_family_count,
            "conformal_family_count": self.conformal_family_count,
            "locked_evaluation_family_count": self.locked_evaluation_family_count,
            "ood_holdout_family_count": self.ood_holdout_family_count,
            "synthetic_only": self.synthetic_only,
            "recommendation_authority": self.recommendation_authority,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "automatic_promotion": self.automatic_promotion,
            "operational_validity": self.operational_validity,
        }


class AdvisorV3ShardFamily(BaseModel):
    """One whole-family unit assigned to exactly one execution shard."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    family_id: StrictStr
    instance_ids: Annotated[tuple[StrictStr, ...], Field(min_length=4, max_length=4)]


class AdvisorV3Shard(BaseModel):
    """A deterministic collection of disjoint whole-family execution units."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    shard_index: Annotated[StrictInt, Field(ge=0)]
    families: Annotated[tuple[AdvisorV3ShardFamily, ...], Field(min_length=1)]

    @property
    def instance_ids(self) -> tuple[str, ...]:
        return tuple(instance_id for family in self.families for instance_id in family.instance_ids)


class AdvisorV3ShardPlan(BaseModel):
    """Fixed 42-way whole-family plan safe for disjoint concurrent workers."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    shard_schema_version: Literal["3"] = "3"
    design_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    shard_count: Literal[42] = _SHARD_COUNT
    shards: tuple[AdvisorV3Shard, ...]
    family_count: Literal[84] = 84
    instance_count: Literal[336] = 336
    whole_family_shards: Literal[True] = True
    serial_governance_preparation_required: Literal[True] = True
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_shards(self) -> Self:
        if tuple(shard.shard_index for shard in self.shards) != tuple(range(self.shard_count)):
            raise ValueError("Advisor-v3 shard indices must be contiguous and ordered.")
        families = tuple(family for shard in self.shards for family in shard.families)
        family_ids = tuple(item.family_id for item in families)
        instance_ids = tuple(instance for item in families for instance in item.instance_ids)
        if len(families) != self.family_count or len(set(family_ids)) != self.family_count:
            raise ValueError("Advisor-v3 shards must cover every family exactly once.")
        if (
            len(instance_ids) != self.instance_count
            or len(set(instance_ids)) != self.instance_count
        ):
            raise ValueError("Advisor-v3 shards must cover every instance exactly once.")
        if any(
            len(item.instance_ids) != 4 or len(set(item.instance_ids)) != 4 for item in families
        ):
            raise ValueError("Advisor-v3 shards cannot split or duplicate family instances.")
        return self

    @property
    def plan_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class AdvisorV3PreregistrationManifest(BaseModel):
    """Canonical outcome-free binding that must exist before heavy v3 execution."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    preregistration_schema_version: Literal["3"] = "3"
    preregistration_id: Literal["advisor_v3_prospective_20260821"] = (
        "advisor_v3_prospective_20260821"
    )
    design_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    catalogue_manifest_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    role_manifest_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    feature_source_policy_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    feature_model_schema_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    label_budget_policy_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    qualification_policy_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    evaluation_algorithm_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    geometry_coherence_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    utility_policy_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    shard_plan_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    required_success_adapters: tuple[StrictStr, ...] = _REQUIRED_ADAPTERS
    portfolio_recipe_count: Literal[7] = _PORTFOLIO_RECIPE_COUNT
    replicates_per_instance: Literal[5] = _REPLICATES
    base_seed: Literal[20260816] = _BASE_SEED
    expected_run_count: Literal[11760] = 11_760
    expected_required_success_run_count: Literal[5040] = 5_040
    expected_ineligible_run_count: Literal[6720] = 6_720
    outcome_fields_present: Literal[False] = False
    held_out_metric_values_used_for_design_fit_or_threshold: Literal[False] = False
    held_out_digest_integrity_checked: StrictBool = False
    qualification_evaluation_accessed: Literal[False] = False
    locked_and_ood_access_requires_later_human_approval: Literal[True] = True
    runtime_feature_producer_status: Literal["not_implemented"] = "not_implemented"
    missing_runtime_feature_behavior: Literal["abstain_or_v2_fallback"] = "abstain_or_v2_fallback"
    synthetic_only: Literal[True] = True
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    release_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    @property
    def preregistration_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class AdvisorV3CorpusReadinessManifest(BaseModel):
    """Fail-closed completion evidence for the exact preregistered v3 grid."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    readiness_schema_version: Literal["3"] = "3"
    execution_protocol_id: Literal["advisor_corpus_execution_v3"] = "advisor_corpus_execution_v3"
    preregistration_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    adapter_statuses: tuple[tuple[StrictStr, Literal["success_capable", "ineligible"]], ...]
    required_adapter_count: Literal[3] = 3
    success_capable_required_adapter_count: Annotated[StrictInt, Field(ge=0, le=3)]
    execution_ready: StrictBool
    execution_status: Literal["not_started", "partial", "complete"] = "not_started"
    expected_run_count: Literal[11760] = 11_760
    completed_run_count: Annotated[StrictInt, Field(ge=0, le=11760)] = 0
    required_evidence_cell_count: Literal[1680] = 1_680
    successful_evidence_cell_count: Annotated[StrictInt, Field(ge=0, le=1680)] = 0
    expected_required_adapter_run_count: Literal[5040] = 5_040
    successful_required_adapter_run_count: Annotated[StrictInt, Field(ge=0, le=5040)] = 0
    failed_required_adapter_run_count: Annotated[StrictInt, Field(ge=0, le=5040)] = 0
    missing_required_adapter_run_count: Annotated[StrictInt, Field(ge=0, le=5040)] = 5_040
    successful_overlap_family_count: Annotated[StrictInt, Field(ge=0, le=84)] = 0
    advisor_evidence_ready: StrictBool = False
    held_out_metric_values_used_for_design_fit_or_threshold: Literal[False] = False
    held_out_digest_integrity_checked: StrictBool = False
    qualification_evaluation_accessed: Literal[False] = False
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_readiness(self) -> Self:
        expected_ready = self.success_capable_required_adapter_count == self.required_adapter_count
        if self.execution_ready != expected_ready:
            raise ValueError("Advisor-v3 execution readiness must fail closed on adapter gaps.")
        if (
            self.successful_required_adapter_run_count
            + self.failed_required_adapter_run_count
            + self.missing_required_adapter_run_count
            != self.expected_required_adapter_run_count
        ):
            raise ValueError("Advisor-v3 required-adapter counts are inconsistent.")
        status = (
            "not_started"
            if self.completed_run_count == 0
            else "complete"
            if self.completed_run_count == self.expected_run_count
            else "partial"
        )
        if self.execution_status != status:
            raise ValueError("Advisor-v3 execution status does not match retained evidence.")
        expected_evidence_ready = (
            status == "complete"
            and self.successful_evidence_cell_count == self.required_evidence_cell_count
            and self.successful_required_adapter_run_count
            == self.expected_required_adapter_run_count
            and self.failed_required_adapter_run_count == 0
            and self.missing_required_adapter_run_count == 0
            and self.successful_overlap_family_count == 84
        )
        if self.advisor_evidence_ready != expected_evidence_ready:
            raise ValueError("Advisor-v3 evidence readiness must fail closed on any grid gap.")
        return self

    @property
    def readiness_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class AdvisorV3GeometryCoherenceManifest(BaseModel):
    """Outcome-free proof that preregistered role geometry can satisfy OOD gates."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    geometry_schema_version: Literal["3"] = "3"
    profiling_base_seed: Literal[20260816] = 20260816
    profiling_replicates: Literal[5] = 5
    conformal_family_count: Literal[12] = 12
    locked_family_count: Literal[12] = 12
    ood_family_count: Literal[12] = 12
    selected_distance_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    locked_above_threshold_count: Literal[0] = 0
    ood_above_threshold_count: Literal[12] = 12
    maximum_locked_nearest_training_distance: Annotated[float, Field(ge=0.0, le=1.0)]
    minimum_ood_nearest_training_distance: Annotated[float, Field(ge=0.0, le=1.0)]
    outcome_values_accessed: Literal[False] = False
    family_role_geometry_coherent: Literal[True] = True
    operational_validity: Literal["not_established"] = "not_established"

    @property
    def geometry_coherence_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class _V3FamilyBlueprint:
    slug: str
    role: AdvisorFamilyRole
    axes: tuple[str, ...]
    category: str
    variant: int = 0


_TRAINING_ARCHETYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("script", ("script", "typo")),
    ("punctuation", ("punctuation", "token")),
    ("tokenization", ("token", "punctuation")),
    ("missing_asymmetry", ("missing_asymmetry", "date_shift")),
    ("frequency_skew", ("skew", "scale")),
    ("candidate_ambiguity", ("collision", "skew")),
    ("duplication", ("duplicate", "exact_duplicate")),
    ("label_regime", ("labels",)),
    ("script_missing", ("script", "missing_asymmetry")),
    ("punctuation_ambiguity", ("punctuation", "collision")),
    ("skew_duplication", ("skew", "duplicate")),
    ("token_label", ("token", "labels")),
)

_CONFORMAL_AXES: tuple[tuple[str, ...], ...] = (
    ("script", "punctuation", "typo"),
    ("token", "missing_asymmetry"),
    ("skew", "collision", "scale"),
    ("duplicate", "exact_duplicate", "labels"),
    ("date_shift", "date_ambiguity", "missing_asymmetry"),
    ("script", "collision"),
    ("punctuation", "skew"),
    ("token", "duplicate"),
    ("labels", "missing_asymmetry"),
    ("typo", "collision", "duplicate"),
    ("script", "labels", "scale"),
    ("punctuation", "date_shift", "skew"),
)

_LOCKED_AXES: tuple[tuple[str, ...], ...] = tuple(axes for _, axes in _TRAINING_ARCHETYPES)

_OOD_AXES: tuple[tuple[str, ...], ...] = (
    ("script", "punctuation", "missing_asymmetry", "collision"),
    ("script", "skew", "duplicate", "exact_duplicate"),
    ("punctuation", "missing_asymmetry", "skew", "collision"),
    ("script", "punctuation", "duplicate", "exact_duplicate"),
    ("missing_asymmetry", "skew", "collision", "scale"),
    ("script", "labels", "collision", "missing_asymmetry"),
    ("punctuation", "skew", "duplicate", "exact_duplicate"),
    ("script", "punctuation", "skew", "scale"),
    ("missing_asymmetry", "collision", "duplicate", "exact_duplicate"),
    ("script", "skew", "labels", "collision", "duplicate"),
    ("punctuation", "missing_asymmetry", "labels", "duplicate", "exact_duplicate"),
    ("script", "punctuation", "missing_asymmetry", "skew", "collision", "duplicate"),
)


def _blueprints() -> tuple[_V3FamilyBlueprint, ...]:
    training = tuple(
        _V3FamilyBlueprint(
            slug=f"train_{name}_v{variant + 1:02d}",
            role="meta_training",
            axes=axes,
            category=name,
            variant=variant,
        )
        for name, axes in _TRAINING_ARCHETYPES
        for variant in range(4)
    )
    conformal = tuple(
        _V3FamilyBlueprint(
            slug=f"conformal_{index + 1:02d}",
            role="conformal",
            axes=axes,
            category="interpolation_boundary",
        )
        for index, axes in enumerate(_CONFORMAL_AXES)
    )
    locked = tuple(
        _V3FamilyBlueprint(
            slug=f"locked_{index + 1:02d}",
            role="locked_evaluation",
            axes=axes,
            category="prospective_locked_combination",
        )
        for index, axes in enumerate(_LOCKED_AXES)
    )
    ood = tuple(
        _V3FamilyBlueprint(
            slug=f"ood_{index + 1:02d}",
            role="ood_holdout",
            axes=axes,
            category="prospective_mechanism_ood",
        )
        for index, axes in enumerate(_OOD_AXES)
    )
    result = training + conformal + locked + ood
    if len(result) != 84:
        raise RuntimeError("The advisor-v3 design must contain exactly 84 families.")
    return result


def _bounded(value: float) -> float:
    return min(1.0, max(0.0, value))


def _effective_intensity(blueprint: _V3FamilyBlueprint, level: float) -> float:
    if blueprint.role == "meta_training":
        return _bounded(0.10 + 0.58 * level + 0.03 * blueprint.variant)
    if blueprint.role == "conformal":
        return _bounded(0.16 + 0.66 * level)
    if blueprint.role == "locked_evaluation":
        return _bounded(0.10 + 0.52 * level)
    return _bounded(0.72 + 0.28 * level)


def _settings(
    blueprint: _V3FamilyBlueprint, intensity: float
) -> tuple[dict[str, object], ScenarioMechanicExtensionV3]:
    effective = _effective_intensity(blueprint, intensity)
    spec: dict[str, object] = {
        "entity_count": 140,
        "typo_rate": 0.03,
        "token_transposition_rate": 0.02,
        "date_shift_rate": 0.02,
        "date_ambiguity_rate": 0.01,
        "missingness_rate": 0.0,
        "zipf_skew_parameter": 0.2,
        "duplicate_density": 0.02,
        "label_volume": 200,
    }
    extension: dict[str, float | str] = {
        "mechanic_schema_version": "3",
        "unicode_transliteration_rate": 0.0,
        "punctuation_change_rate": 0.0,
        "source_a_missingness_rate": 0.02,
        "source_b_missingness_rate": 0.02,
        "label_collision_rate": 0.0,
        "exact_duplicate_fraction": 0.0,
    }
    for axis in blueprint.axes:
        if axis == "typo":
            spec["typo_rate"] = round(0.34 * effective, 6)
        elif axis == "token":
            spec["token_transposition_rate"] = round(0.42 * effective, 6)
        elif axis == "date_shift":
            spec["date_shift_rate"] = round(0.36 * effective, 6)
        elif axis == "date_ambiguity":
            spec["date_ambiguity_rate"] = round(0.28 * effective, 6)
        elif axis == "skew":
            spec["zipf_skew_parameter"] = round(3.2 * effective, 6)
        elif axis == "duplicate":
            spec["duplicate_density"] = round(0.36 * effective, 6)
        elif axis == "labels":
            spec["label_volume"] = round(500 - 475 * effective)
        elif axis == "scale":
            spec["entity_count"] = round(100 + 180 * effective)
        elif axis == "script":
            extension["unicode_transliteration_rate"] = round(0.88 * effective, 6)
        elif axis == "punctuation":
            extension["punctuation_change_rate"] = round(0.78 * effective, 6)
        elif axis == "missing_asymmetry":
            extension["source_a_missingness_rate"] = round(0.04 * (1.0 - effective), 6)
            extension["source_b_missingness_rate"] = round(0.10 + 0.62 * effective, 6)
        elif axis == "collision":
            extension["label_collision_rate"] = round(0.74 * effective, 6)
        elif axis == "exact_duplicate":
            extension["exact_duplicate_fraction"] = round(0.20 + 0.80 * effective, 6)
        else:
            raise RuntimeError("An advisor-v3 mechanism axis is not package-owned.")
    return spec, ScenarioMechanicExtensionV3.model_validate(extension)


def build_advisor_v3_generator() -> BenchmarkScenarioGenerator:
    """Return seed-v1 plus only the new, prospectively fixed advisor-v3 catalogue."""

    generator = BenchmarkScenarioGenerator()
    for ordinal, blueprint in enumerate(_blueprints(), start=1):
        family_id = f"family.advisor_v3.{blueprint.slug}"
        instances: list[ScenarioLatentSpec] = []
        extensions: dict[str, ScenarioMechanicExtensionV3] = {}
        for level_index, level in enumerate((0.2, 0.4, 0.6, 0.8), start=1):
            instance_id = f"instance.advisor_v3.f{ordinal:03d}.p{level_index:02d}"
            spec_values, extension = _settings(blueprint, level)
            instances.append(
                ScenarioLatentSpec.model_validate(
                    {
                        "family_id": family_id,
                        "instance_id": instance_id,
                        "planned_replicates": _REPLICATES,
                        **spec_values,
                    }
                )
            )
            extensions[instance_id] = extension
        generator.register_family(
            family_id=family_id,
            mechanism_tags=tuple(
                dict.fromkeys(
                    (
                        "advisor_v3",
                        blueprint.role,
                        blueprint.category,
                        *blueprint.axes,
                    )
                )
            ),
            prospectively_held_out=blueprint.role == "ood_holdout",
            instances=tuple(instances),
            instance_extensions=extensions,
        )
    return generator


def advisor_v3_family_roles() -> tuple[tuple[str, AdvisorFamilyRole], ...]:
    return tuple(
        sorted(
            (f"family.advisor_v3.{blueprint.slug}", blueprint.role) for blueprint in _blueprints()
        )
    )


def _advisor_v3_catalogue_manifest_digest(
    generator: BenchmarkScenarioGenerator | None = None,
) -> str:
    catalogue = generator or build_advisor_v3_generator()
    return _digest(
        {
            "families": sorted(
                (item.family_id, item.family_digest)
                for item in catalogue.list_families()
                if item.family_id.startswith("family.advisor_v3.")
            ),
            "instances": sorted(
                (item.instance_id, item.instance_digest)
                for item in catalogue.list_instances()
                if item.instance_id.startswith("instance.advisor_v3.")
            ),
        }
    )


def build_advisor_v3_corpus_design() -> AdvisorV3CorpusDesignManifest:
    generator = build_advisor_v3_generator()
    instances = tuple(
        item
        for item in generator.list_instances()
        if item.instance_id.startswith("instance.advisor_v3.")
    )
    if len(instances) != 336:
        raise RuntimeError("The advisor-v3 catalogue must contain exactly 336 instances.")
    return AdvisorV3CorpusDesignManifest(
        family_roles=advisor_v3_family_roles(),
        catalogue_manifest_digest=_advisor_v3_catalogue_manifest_digest(generator),
        feature_source_policy_digest=advisor_v3_feature_source_policy_digest(),
        feature_model_schema_digest=advisor_v3_feature_model_schema_digest(),
        label_budget_policy_digest=advisor_v3_label_budget_policy_digest(),
        utility_policy_digest=ADVISOR_UTILITY_POLICY_DIGEST,
    )


def build_advisor_v3_shard_plan() -> AdvisorV3ShardPlan:
    design = build_advisor_v3_corpus_design()
    generator = build_advisor_v3_generator()
    family_ids = sorted(
        (family_id for family_id, _ in advisor_v3_family_roles()),
        key=lambda value: (_digest(value), value),
    )
    buckets: list[list[str]] = [[] for _ in range(_SHARD_COUNT)]
    for index, family_id in enumerate(family_ids):
        buckets[index % _SHARD_COUNT].append(family_id)
    shards = tuple(
        AdvisorV3Shard(
            shard_index=shard_index,
            families=tuple(
                AdvisorV3ShardFamily(
                    family_id=family_id,
                    instance_ids=tuple(
                        sorted(
                            item.instance_id
                            for item in generator.list_instances(family_id=family_id)
                        )
                    ),
                )
                for family_id in sorted(bucket)
            ),
        )
        for shard_index, bucket in enumerate(buckets)
    )
    return AdvisorV3ShardPlan(design_digest=design.design_digest, shards=shards)


@lru_cache(maxsize=1)
def build_advisor_v3_geometry_coherence() -> AdvisorV3GeometryCoherenceManifest:
    """Return the outcome-free geometry values fixed in the preregistration."""

    return AdvisorV3GeometryCoherenceManifest(
        selected_distance_threshold=0.01517688761542686,
        locked_above_threshold_count=0,
        ood_above_threshold_count=12,
        maximum_locked_nearest_training_distance=0.0049136087911625485,
        minimum_ood_nearest_training_distance=0.047076718405300845,
    )


def validate_advisor_v3_geometry_coherence() -> AdvisorV3GeometryCoherenceManifest:
    """Recompute the fixed geometry from all five profiling seeds without outcomes."""

    roles = dict(advisor_v3_family_roles())
    vectors = extract_advisor_v3_family_meta_features(
        build_advisor_v3_generator(),
        family_ids=frozenset(roles),
        base_seed=20260816,
        profiling_replicates=5,
    )
    training = {
        family_id: vector
        for family_id, vector in vectors.items()
        if roles[family_id] == "meta_training"
    }
    conformal = {
        family_id: vector
        for family_id, vector in vectors.items()
        if roles[family_id] == "conformal"
    }
    locked = {
        family_id: vector
        for family_id, vector in vectors.items()
        if roles[family_id] == "locked_evaluation"
    }
    ood = {
        family_id: vector
        for family_id, vector in vectors.items()
        if roles[family_id] == "ood_holdout"
    }
    threshold = select_advisor_v3_ood_distance_threshold(
        training_vectors=training,
        conformal_vectors=conformal,
    )
    computer = MechanismAwareMetaFeatureDistanceComputer()
    locked_distances = tuple(
        min(computer.compute_distance(vector, item) for item in training.values())
        for vector in locked.values()
    )
    ood_distances = tuple(
        min(computer.compute_distance(vector, item) for item in training.values())
        for vector in ood.values()
    )
    locked_above = sum(value > threshold for value in locked_distances)
    ood_above = sum(value > threshold for value in ood_distances)
    if locked_above != 0 or ood_above != 12:
        raise RuntimeError("Advisor-v3 prospective family-role geometry is incoherent.")
    observed = AdvisorV3GeometryCoherenceManifest(
        selected_distance_threshold=threshold,
        locked_above_threshold_count=cast(Literal[0], locked_above),
        ood_above_threshold_count=cast(Literal[12], ood_above),
        maximum_locked_nearest_training_distance=max(locked_distances),
        minimum_ood_nearest_training_distance=min(ood_distances),
    )
    if observed != build_advisor_v3_geometry_coherence():
        raise RuntimeError("Advisor-v3 geometry no longer matches its preregistration.")
    return observed


def build_advisor_v3_preregistration() -> AdvisorV3PreregistrationManifest:
    design = build_advisor_v3_corpus_design()
    plan = build_advisor_v3_shard_plan()
    policy = AdvisorV3QualificationPolicy()
    geometry = build_advisor_v3_geometry_coherence()
    return AdvisorV3PreregistrationManifest(
        design_digest=design.design_digest,
        catalogue_manifest_digest=design.catalogue_manifest_digest,
        role_manifest_digest=design.role_manifest_digest,
        feature_source_policy_digest=design.feature_source_policy_digest,
        feature_model_schema_digest=design.feature_model_schema_digest,
        label_budget_policy_digest=design.label_budget_policy_digest,
        qualification_policy_digest=policy.policy_digest,
        evaluation_algorithm_digest=advisor_v3_evaluation_algorithm_digest(),
        geometry_coherence_digest=geometry.geometry_coherence_digest,
        utility_policy_digest=design.utility_policy_digest,
        shard_plan_digest=plan.plan_digest,
    )


def build_advisor_v3_corpus_readiness(
    *,
    adapter_statuses: Mapping[str, Literal["success_capable", "ineligible"]],
) -> AdvisorV3CorpusReadinessManifest:
    preregistration = build_advisor_v3_preregistration()
    successful = sum(adapter_statuses.get(name) == "success_capable" for name in _REQUIRED_ADAPTERS)
    return AdvisorV3CorpusReadinessManifest(
        preregistration_digest=preregistration.preregistration_digest,
        adapter_statuses=tuple(sorted(adapter_statuses.items())),
        success_capable_required_adapter_count=successful,
        execution_ready=successful == len(_REQUIRED_ADAPTERS),
    )


__all__ = [
    "AdvisorV3CorpusDesignManifest",
    "AdvisorV3CorpusReadinessManifest",
    "AdvisorV3GeometryCoherenceManifest",
    "AdvisorV3PreregistrationManifest",
    "AdvisorV3Shard",
    "AdvisorV3ShardFamily",
    "AdvisorV3ShardPlan",
    "advisor_v3_family_roles",
    "build_advisor_v3_corpus_design",
    "build_advisor_v3_corpus_readiness",
    "build_advisor_v3_generator",
    "build_advisor_v3_geometry_coherence",
    "build_advisor_v3_preregistration",
    "build_advisor_v3_shard_plan",
    "validate_advisor_v3_geometry_coherence",
]
