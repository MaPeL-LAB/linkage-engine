"""Versioned advisor-scale synthetic benchmark design and deterministic shard planning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from mapel_linkage.benchmarking.generator import (
    BenchmarkScenarioGenerator,
    ScenarioLatentSpec,
    ScenarioMechanicExtension,
)

AdvisorFamilyRole = Literal["meta_training", "conformal", "locked_evaluation", "ood_holdout"]

_CATALOGUE_ID: Literal["advisor_v2"] = "advisor_v2"
_REQUIRED_ADAPTERS = (
    "recipe.fellegi_sunter_reference",
    "recipe.xgboost_classifier",
    "recipe.xgboost_ranker",
)
_PORTFOLIO_RECIPE_COUNT: Literal[7] = 7
_MINIMUM_ADVISOR_REPLICATES: Literal[5] = 5


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class AdvisorCorpusDesignManifest(BaseModel):
    """Aggregate, versioned prospective design; it contains no generated row values."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    design_schema_version: Literal["2"] = "2"
    catalogue_id: Literal["advisor_v2"] = _CATALOGUE_ID
    seed_v1_family_count: Literal[10] = 10
    seed_v1_instance_count: Literal[19] = 19
    seed_v1_binding_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    family_count: Literal[64] = 64
    instance_count: Literal[280] = 280
    meta_training_family_count: Literal[40] = 40
    conformal_family_count: Literal[8] = 8
    locked_evaluation_family_count: Literal[8] = 8
    ood_holdout_family_count: Literal[8] = 8
    family_roles: tuple[tuple[StrictStr, AdvisorFamilyRole], ...]
    required_success_adapters: tuple[StrictStr, ...] = _REQUIRED_ADAPTERS
    design_components: tuple[StrictStr, ...] = (
        "main_effects",
        "selected_interactions",
        "composite_regimes",
        "stress_regimes",
        "space_filling_coverage",
        "prospective_mechanism_holdout",
    )
    synthetic_only: Literal[True] = True
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_roles(self) -> AdvisorCorpusDesignManifest:
        if len(self.family_roles) != self.family_count:
            raise ValueError("The advisor design must bind every family exactly once.")
        family_ids = [family_id for family_id, _ in self.family_roles]
        if len(family_ids) != len(set(family_ids)) or family_ids != sorted(family_ids):
            raise ValueError("Advisor family roles must be unique and canonically ordered.")
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
            raise ValueError("Advisor family-role counts do not match the prospective design.")
        if len(self.required_success_adapters) < 3:
            raise ValueError("Advisor strategy evidence requires at least three real adapters.")
        return self

    @property
    def design_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {
            "design_schema_version": self.design_schema_version,
            "catalogue_id": self.catalogue_id,
            "design_digest": self.design_digest,
            "seed_v1_family_count": self.seed_v1_family_count,
            "seed_v1_instance_count": self.seed_v1_instance_count,
            "family_count": self.family_count,
            "instance_count": self.instance_count,
            "meta_training_family_count": self.meta_training_family_count,
            "conformal_family_count": self.conformal_family_count,
            "locked_evaluation_family_count": self.locked_evaluation_family_count,
            "ood_holdout_family_count": self.ood_holdout_family_count,
            "required_success_adapter_count": len(self.required_success_adapters),
            "synthetic_only": self.synthetic_only,
            "recommendation_authority": self.recommendation_authority,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "automatic_promotion": self.automatic_promotion,
            "operational_validity": self.operational_validity,
        }


class AdvisorCorpusReadinessManifest(BaseModel):
    """Fail-closed readiness evidence for an approved heavy corpus execution."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    readiness_schema_version: Literal["2"] = "2"
    execution_protocol_id: Literal["advisor_corpus_execution_v2"] = "advisor_corpus_execution_v2"
    design_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    adapter_statuses: tuple[tuple[StrictStr, Literal["success_capable", "ineligible"]], ...]
    required_adapter_count: Annotated[StrictInt, Field(ge=3)]
    success_capable_required_adapter_count: Annotated[StrictInt, Field(ge=0)]
    design_valid: StrictBool
    execution_ready: StrictBool
    execution_status: Literal["not_started", "partial", "complete"] = "not_started"
    expected_run_count: Annotated[StrictInt, Field(ge=0)] = 0
    completed_run_count: Annotated[StrictInt, Field(ge=0)] = 0
    portfolio_recipe_count: Literal[7] = _PORTFOLIO_RECIPE_COUNT
    minimum_replicates_per_instance: Literal[5] = _MINIMUM_ADVISOR_REPLICATES
    planned_replicates_per_instance: Annotated[StrictInt, Field(ge=1, le=100)] = 5
    required_evidence_cell_count: Annotated[StrictInt, Field(ge=1)] = 1_400
    successful_evidence_cell_count: Annotated[StrictInt, Field(ge=0)] = 0
    expected_required_adapter_run_count: Annotated[StrictInt, Field(ge=1)] = 4_200
    successful_required_adapter_run_count: Annotated[StrictInt, Field(ge=0)] = 0
    failed_required_adapter_run_count: Annotated[StrictInt, Field(ge=0)] = 0
    missing_required_adapter_run_count: Annotated[StrictInt, Field(ge=0)] = 4_200
    required_overlap_family_count: Literal[64] = 64
    successful_overlap_family_count: Annotated[StrictInt, Field(ge=0, le=64)] = 0
    advisor_evidence_ready: StrictBool = False
    proxy_ood_family_excluded: Literal[True] = True
    incomplete_mode_adapters: tuple[Literal["dedupe_only", "multi_source"], ...] = (
        "dedupe_only",
        "multi_source",
    )
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_readiness(self) -> AdvisorCorpusReadinessManifest:
        if len(self.adapter_statuses) != len({name for name, _ in self.adapter_statuses}):
            raise ValueError("Adapter readiness entries must be unique.")
        expected_ready = (
            self.design_valid
            and self.success_capable_required_adapter_count == self.required_adapter_count
        )
        if self.execution_ready != expected_ready:
            raise ValueError("Corpus execution readiness must fail closed on missing adapters.")
        if self.completed_run_count > self.expected_run_count:
            raise ValueError("Completed advisor runs cannot exceed the expected run count.")
        expected_cells = 280 * self.planned_replicates_per_instance
        if self.required_evidence_cell_count != expected_cells:
            raise ValueError("Required evidence cells do not match the approved replicate grid.")
        if self.expected_run_count != expected_cells * self.portfolio_recipe_count:
            raise ValueError("Expected portfolio runs do not match the approved replicate grid.")
        expected_required_runs = expected_cells * self.required_adapter_count
        if self.expected_required_adapter_run_count != expected_required_runs:
            raise ValueError("Required adapter runs do not match the approved evidence grid.")
        if (
            self.successful_required_adapter_run_count
            + self.failed_required_adapter_run_count
            + self.missing_required_adapter_run_count
            != expected_required_runs
        ):
            raise ValueError("Required adapter completion counts are inconsistent.")
        if self.successful_evidence_cell_count > self.required_evidence_cell_count:
            raise ValueError("Successful evidence cells cannot exceed the approved grid.")
        if (
            self.successful_required_adapter_run_count
            < self.successful_evidence_cell_count * self.required_adapter_count
            or self.failed_required_adapter_run_count + self.missing_required_adapter_run_count
            < self.required_evidence_cell_count - self.successful_evidence_cell_count
        ):
            raise ValueError("Required adapter and evidence-cell counts are inconsistent.")
        expected_status = (
            "not_started"
            if self.completed_run_count == 0
            else "complete"
            if self.completed_run_count == self.expected_run_count
            else "partial"
        )
        if self.execution_status != expected_status:
            raise ValueError("Corpus execution status does not match retained run evidence.")
        expected_evidence_ready = (
            self.execution_status == "complete"
            and self.planned_replicates_per_instance >= self.minimum_replicates_per_instance
            and self.successful_evidence_cell_count == self.required_evidence_cell_count
            and self.successful_required_adapter_run_count
            == self.expected_required_adapter_run_count
            and self.failed_required_adapter_run_count == 0
            and self.missing_required_adapter_run_count == 0
            and self.successful_overlap_family_count == self.required_overlap_family_count
        )
        if self.advisor_evidence_ready != expected_evidence_ready:
            raise ValueError("Advisor evidence readiness must fail closed on recipe gaps.")
        return self

    @property
    def readiness_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {
            "readiness_schema_version": self.readiness_schema_version,
            "execution_protocol_id": self.execution_protocol_id,
            "design_digest": self.design_digest,
            "readiness_digest": self.readiness_digest,
            "required_adapter_count": self.required_adapter_count,
            "success_capable_required_adapter_count": self.success_capable_required_adapter_count,
            "design_valid": self.design_valid,
            "execution_ready": self.execution_ready,
            "execution_status": self.execution_status,
            "expected_run_count": self.expected_run_count,
            "completed_run_count": self.completed_run_count,
            "portfolio_recipe_count": self.portfolio_recipe_count,
            "minimum_replicates_per_instance": self.minimum_replicates_per_instance,
            "planned_replicates_per_instance": self.planned_replicates_per_instance,
            "required_evidence_cell_count": self.required_evidence_cell_count,
            "successful_evidence_cell_count": self.successful_evidence_cell_count,
            "expected_required_adapter_run_count": self.expected_required_adapter_run_count,
            "successful_required_adapter_run_count": (self.successful_required_adapter_run_count),
            "failed_required_adapter_run_count": self.failed_required_adapter_run_count,
            "missing_required_adapter_run_count": self.missing_required_adapter_run_count,
            "required_overlap_family_count": self.required_overlap_family_count,
            "successful_overlap_family_count": self.successful_overlap_family_count,
            "advisor_evidence_ready": self.advisor_evidence_ready,
            "proxy_ood_family_excluded": self.proxy_ood_family_excluded,
            "incomplete_mode_adapter_count": len(self.incomplete_mode_adapters),
            "recommendation_authority": self.recommendation_authority,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "automatic_promotion": self.automatic_promotion,
            "operational_validity": self.operational_validity,
        }


class BenchmarkShard(BaseModel):
    """One deterministic shard of public synthetic instance identifiers."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    shard_index: Annotated[StrictInt, Field(ge=0)]
    instance_ids: tuple[StrictStr, ...]

    @property
    def shard_digest(self) -> str:
        return _digest({"shard_index": self.shard_index, "instance_ids": self.instance_ids})


class BenchmarkShardPlan(BaseModel):
    """Digest-bound deterministic execution decomposition for the v2 corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    shard_schema_version: Literal["1"] = "1"
    design_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    shard_count: Annotated[StrictInt, Field(ge=1, le=256)]
    shards: tuple[BenchmarkShard, ...]
    instance_count: Literal[280] = 280
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_shards(self) -> BenchmarkShardPlan:
        if len(self.shards) != self.shard_count:
            raise ValueError("Shard-plan count does not match its shard entries.")
        if tuple(shard.shard_index for shard in self.shards) != tuple(range(self.shard_count)):
            raise ValueError("Shard indices must be contiguous and ordered.")
        instance_ids = [item for shard in self.shards for item in shard.instance_ids]
        if len(instance_ids) != self.instance_count or len(instance_ids) != len(set(instance_ids)):
            raise ValueError("Shard plans must cover every advisor instance exactly once.")
        sizes = [len(shard.instance_ids) for shard in self.shards]
        if max(sizes) - min(sizes) > 1:
            raise ValueError("Deterministic shard plans must remain balanced.")
        return self

    @property
    def plan_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {
            "shard_schema_version": self.shard_schema_version,
            "design_digest": self.design_digest,
            "plan_digest": self.plan_digest,
            "shard_count": self.shard_count,
            "instance_count": self.instance_count,
            "minimum_shard_size": min(len(shard.instance_ids) for shard in self.shards),
            "maximum_shard_size": max(len(shard.instance_ids) for shard in self.shards),
            "recommendation_authority": self.recommendation_authority,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "automatic_promotion": self.automatic_promotion,
            "operational_validity": self.operational_validity,
        }


@dataclass(frozen=True, slots=True)
class _FamilyBlueprint:
    slug: str
    role: AdvisorFamilyRole
    mechanism_tags: tuple[str, ...]
    varying_axes: tuple[str, ...]
    base: Mapping[str, float | int | bool] = field(default_factory=dict)
    true_transliteration: bool = False


_AXIS_RANGES: Mapping[str, tuple[float | int, float | int]] = {
    "typo_rate": (0.0, 0.40),
    "token_transposition_rate": (0.0, 0.40),
    "date_shift_rate": (0.0, 0.40),
    "date_ambiguity_rate": (0.0, 0.30),
    "missingness_rate": (0.0, 0.40),
    "zipf_skew_parameter": (0.0, 2.20),
    "duplicate_density": (0.0, 0.30),
    "label_volume": (25, 500),
    "entity_count": (80, 240),
}


def _main_blueprints() -> list[_FamilyBlueprint]:
    axes = (
        ("character_error", "typo_rate", ("character_substitution",)),
        ("token_order", "token_transposition_rate", ("token_transposition",)),
        ("date_shift", "date_shift_rate", ("date_shift",)),
        ("date_ambiguity", "date_ambiguity_rate", ("date_ambiguity",)),
        ("mcar_missingness", "missingness_rate", ("source_specific_missingness",)),
        ("informative_missingness", "missingness_rate", ("informative_missingness",)),
        ("frequency_skew", "zipf_skew_parameter", ("frequency_skew",)),
        ("duplicate_density", "duplicate_density", ("within_source_duplication",)),
        ("label_volume", "label_volume", ("label_scarcity",)),
        ("population_scale", "entity_count", ("population_scale",)),
    )
    output = []
    for slug, axis, tags in axes:
        base: dict[str, float | int | bool] = {}
        if slug == "informative_missingness":
            base["informative_missingness"] = True
        output.append(_FamilyBlueprint(slug, "meta_training", tags, (axis,), base))
    return output


def _interaction_blueprints() -> list[_FamilyBlueprint]:
    definitions = (
        ("typo_by_missing", "typo_rate", "missingness_rate"),
        ("typo_by_skew", "typo_rate", "zipf_skew_parameter"),
        ("typo_by_date", "typo_rate", "date_shift_rate"),
        ("token_by_missing", "token_transposition_rate", "missingness_rate"),
        ("token_by_skew", "token_transposition_rate", "zipf_skew_parameter"),
        ("date_by_missing", "date_shift_rate", "missingness_rate"),
        ("date_ambiguity_by_missing", "date_ambiguity_rate", "missingness_rate"),
        ("skew_by_missing", "zipf_skew_parameter", "missingness_rate"),
        ("duplicate_by_typo", "duplicate_density", "typo_rate"),
        ("duplicate_by_missing", "duplicate_density", "missingness_rate"),
        ("labels_by_typo", "label_volume", "typo_rate"),
        ("labels_by_missing", "label_volume", "missingness_rate"),
        ("scale_by_skew", "entity_count", "zipf_skew_parameter"),
        ("scale_by_missing", "entity_count", "missingness_rate"),
    )
    return [
        _FamilyBlueprint(
            slug,
            "meta_training",
            ("selected_interaction", left, right),
            (left, right),
        )
        for slug, left, right in definitions
    ]


def _scaled_regime_blueprints(
    *, role: AdvisorFamilyRole, category: str, count: int, offset: int = 0
) -> list[_FamilyBlueprint]:
    axis_sets = (
        ("typo_rate", "missingness_rate", "date_shift_rate"),
        ("typo_rate", "zipf_skew_parameter", "duplicate_density"),
        ("token_transposition_rate", "missingness_rate", "label_volume"),
        ("date_ambiguity_rate", "zipf_skew_parameter", "entity_count"),
        ("missingness_rate", "duplicate_density", "label_volume"),
        ("typo_rate", "date_shift_rate", "zipf_skew_parameter", "label_volume"),
        ("token_transposition_rate", "date_ambiguity_rate", "duplicate_density"),
        ("typo_rate", "missingness_rate", "zipf_skew_parameter", "entity_count"),
    )
    output: list[_FamilyBlueprint] = []
    for index in range(count):
        axes = axis_sets[(index + offset) % len(axis_sets)]
        output.append(
            _FamilyBlueprint(
                f"{category}_{index + 1:02d}",
                role,
                (category, *axes),
                axes,
                {"informative_missingness": bool(index % 2)},
            )
        )
    return output


def _blueprints() -> tuple[_FamilyBlueprint, ...]:
    training = (
        _main_blueprints()
        + _interaction_blueprints()
        + _scaled_regime_blueprints(role="meta_training", category="composite", count=8)
        + _scaled_regime_blueprints(
            role="meta_training", category="upper_tail_stress", count=8, offset=3
        )
    )
    conformal = _scaled_regime_blueprints(
        role="conformal", category="conformal_boundary", count=8, offset=1
    )
    locked = _scaled_regime_blueprints(
        role="locked_evaluation", category="locked_space_filling", count=8, offset=2
    )
    ood_axes = (
        (),
        ("missingness_rate",),
        ("zipf_skew_parameter",),
        ("date_shift_rate",),
        ("token_transposition_rate",),
        ("duplicate_density",),
        ("label_volume",),
        ("missingness_rate", "zipf_skew_parameter", "date_shift_rate"),
    )
    ood = [
        _FamilyBlueprint(
            f"ood_transliteration_{index + 1:02d}",
            "ood_holdout",
            ("unicode_variation", "true_script_transliteration", "prospective_ood", *axes),
            axes,
            true_transliteration=True,
        )
        for index, axes in enumerate(ood_axes)
    ]
    result = tuple(training + conformal + locked + ood)
    if len(result) != 64:
        raise RuntimeError("The package-owned advisor design must contain exactly 64 families.")
    return result


def _level_value(axis: str, intensity: float, *, reverse: bool = False) -> float | int:
    low, high = _AXIS_RANGES[axis]
    fraction = 1.0 - intensity if reverse else intensity
    value = float(low) + (float(high) - float(low)) * fraction
    if axis in {"entity_count", "label_volume"}:
        return round(value)
    return round(value, 6)


def _register_blueprint(
    generator: BenchmarkScenarioGenerator,
    blueprint: _FamilyBlueprint,
    ordinal: int,
) -> None:
    family_id = f"family.advisor_v2.{blueprint.slug}"
    level_count = {
        "meta_training": 4,
        "conformal": 4,
        "locked_evaluation": 5,
        "ood_holdout": 6,
    }[blueprint.role]
    intensities = tuple((index + 1) / (level_count + 1) for index in range(level_count))
    specs: list[ScenarioLatentSpec] = []
    extensions: dict[str, ScenarioMechanicExtension] = {}
    for index, intensity in enumerate(intensities, start=1):
        values: dict[str, object] = dict(blueprint.base)
        for axis_index, axis in enumerate(blueprint.varying_axes):
            values[axis] = _level_value(
                axis,
                intensity,
                reverse=axis == "label_volume" or (axis_index % 2 == 1 and ordinal % 2 == 1),
            )
        instance_id = f"instance.advisor_v2.f{ordinal:02d}.p{index:02d}"
        specs.append(
            ScenarioLatentSpec.model_validate(
                {
                    "family_id": family_id,
                    "instance_id": instance_id,
                    "planned_replicates": 5,
                    **values,
                }
            )
        )
        if blueprint.true_transliteration:
            extensions[instance_id] = ScenarioMechanicExtension(
                unicode_transliteration_rate=round(intensity, 6),
                punctuation_change_rate=round(intensity / 2.0, 6),
            )
    generator.register_family(
        family_id=family_id,
        mechanism_tags=(
            "advisor_v2",
            blueprint.role,
            *blueprint.mechanism_tags,
        ),
        prospectively_held_out=blueprint.role == "ood_holdout",
        instances=tuple(specs),
        instance_extensions=extensions,
    )


def build_advisor_v2_generator() -> BenchmarkScenarioGenerator:
    """Return seed-v1 plus the prospectively fixed advisor-v2 package catalogue."""

    generator = BenchmarkScenarioGenerator()
    for ordinal, blueprint in enumerate(_blueprints(), start=1):
        _register_blueprint(generator, blueprint, ordinal)
    return generator


def advisor_v2_family_roles() -> tuple[tuple[str, AdvisorFamilyRole], ...]:
    return tuple(
        sorted(
            (f"family.advisor_v2.{blueprint.slug}", blueprint.role) for blueprint in _blueprints()
        )
    )


def _seed_v1_binding_digest() -> str:
    generator = BenchmarkScenarioGenerator()
    return _digest(
        {
            "families": [
                (manifest.family_id, manifest.family_digest)
                for manifest in generator.list_families()
            ],
            "instances": [
                (manifest.instance_id, manifest.instance_digest)
                for manifest in generator.list_instances()
            ],
        }
    )


def build_advisor_corpus_design() -> AdvisorCorpusDesignManifest:
    generator = build_advisor_v2_generator()
    advisor_instances = tuple(
        item
        for item in generator.list_instances()
        if item.instance_id.startswith("instance.advisor_v2.")
    )
    if len(advisor_instances) != 280:
        raise RuntimeError("The advisor-v2 package catalogue must contain exactly 280 instances.")
    return AdvisorCorpusDesignManifest(
        seed_v1_binding_digest=_seed_v1_binding_digest(),
        family_roles=advisor_v2_family_roles(),
    )


def build_advisor_corpus_readiness(
    *,
    adapter_statuses: Mapping[str, Literal["success_capable", "ineligible"]],
    planned_replicates_per_instance: int = 5,
) -> AdvisorCorpusReadinessManifest:
    if not 1 <= planned_replicates_per_instance <= 100:
        raise ValueError("Advisor corpus replicate planning must remain between 1 and 100.")
    design = build_advisor_corpus_design()
    ordered = tuple(sorted(adapter_statuses.items()))
    successful = sum(adapter_statuses.get(name) == "success_capable" for name in _REQUIRED_ADAPTERS)
    required_cells = design.instance_count * planned_replicates_per_instance
    required_runs = required_cells * len(_REQUIRED_ADAPTERS)
    return AdvisorCorpusReadinessManifest(
        design_digest=design.design_digest,
        adapter_statuses=ordered,
        required_adapter_count=len(_REQUIRED_ADAPTERS),
        success_capable_required_adapter_count=successful,
        design_valid=True,
        execution_ready=successful == len(_REQUIRED_ADAPTERS),
        expected_run_count=required_cells * _PORTFOLIO_RECIPE_COUNT,
        planned_replicates_per_instance=planned_replicates_per_instance,
        required_evidence_cell_count=required_cells,
        expected_required_adapter_run_count=required_runs,
        missing_required_adapter_run_count=required_runs,
    )


def build_benchmark_shard_plan(*, shard_count: int) -> BenchmarkShardPlan:
    design = build_advisor_corpus_design()
    generator = build_advisor_v2_generator()
    instance_ids = sorted(
        (
            item.instance_id
            for item in generator.list_instances()
            if item.instance_id.startswith("instance.advisor_v2.")
        ),
        key=lambda value: (_digest(value), value),
    )
    buckets: list[list[str]] = [[] for _ in range(shard_count)]
    for index, instance_id in enumerate(instance_ids):
        buckets[index % shard_count].append(instance_id)
    shards = tuple(
        BenchmarkShard(shard_index=index, instance_ids=tuple(sorted(bucket)))
        for index, bucket in enumerate(buckets)
    )
    return BenchmarkShardPlan(
        design_digest=design.design_digest,
        shard_count=shard_count,
        shards=shards,
    )


__all__ = [
    "AdvisorCorpusDesignManifest",
    "AdvisorCorpusReadinessManifest",
    "AdvisorFamilyRole",
    "BenchmarkShard",
    "BenchmarkShardPlan",
    "advisor_v2_family_roles",
    "build_advisor_corpus_design",
    "build_advisor_corpus_readiness",
    "build_advisor_v2_generator",
    "build_benchmark_shard_plan",
]
