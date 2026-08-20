"""Stage-4 active synthetic benchmark planning with no linkage authority."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from mapel_linkage.benchmarking.contracts import BenchmarkRunStatus
from mapel_linkage.benchmarking.generator import (
    BenchmarkScenarioGenerator,
    ScenarioLatentSpec,
)
from mapel_linkage.benchmarking.registry import BenchmarkRegistry, build_registry_snapshot
from mapel_linkage.benchmarking.runner import (
    BenchmarkPortfolioRunner,
    BenchmarkRecipe,
    BenchmarkRunResult,
)
from mapel_linkage.configuration.compiler import ExecutionPlan
from mapel_linkage.domain.errors import AdvisorError
from mapel_linkage.profiling.contracts import PreflightTaskProfile
from mapel_linkage.recommendation.contracts import (
    MetaRankingAdvisoryReport,
    SimilarityAdvisoryReport,
)
from mapel_linkage.recommendation.distance import (
    MetaFeatureDistanceComputer,
    TaskMetaFeatureVector,
)
from mapel_linkage.recommendation.eligibility import AdvisorContext
from mapel_linkage.recommendation.meta_ranker import MetaRankingLinkageAdvisor

Identifier = Annotated[
    StrictStr,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$"),
]
Digest = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]

_DEFAULT_BASE_SEED = 20260816
_UINT32_MODULUS = 4_294_967_296


def _digest_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_uint32(payload: object) -> int:
    return int(_digest_payload(payload)[:8], 16)


class CoverageDimension(StrEnum):
    """Package-owned latent axes used only for simulator experiment design."""

    ERROR_RATE = "error_rate"
    MISSINGNESS = "missingness"
    FREQUENCY_SKEW = "frequency_skew"
    LABEL_VOLUME = "label_volume"


class ExperimentPlanningTrigger(StrEnum):
    """Stable reasons that may open an advisory experiment plan."""

    COVERAGE_GAP = "coverage_gap"
    SIMILARITY_OUT_OF_DISTRIBUTION = "similarity_out_of_distribution"
    META_RANKER_WIDE_INTERVAL = "meta_ranker_wide_interval"
    META_RANKER_FALLBACK = "meta_ranker_fallback"
    NOT_TRIGGERED = "not_triggered"


class ExperimentPlanningStatus(StrEnum):
    """Whether a plan is ready for separate human approval."""

    READY_FOR_HUMAN_APPROVAL = "ready_for_human_approval"
    ABSTAINED = "abstained"
    NOT_TRIGGERED = "not_triggered"


class MetaModelRefitStatus(StrEnum):
    """Truthful outcome of the post-execution advisory model refit."""

    FITTED = "fitted"
    ABSTAINED_INSUFFICIENT_EVIDENCE = "abstained_insufficient_evidence"
    FAILED = "failed"


class PlannerNode(BaseModel):
    """Strict immutable planning contract with value-safe validation errors."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        hide_input_in_errors=True,
    )


class CoverageDensityCell(PlannerNode):
    """Aggregate retained-evidence density for one simulator axis band."""

    dimension: CoverageDimension
    band: Identifier
    catalogue_instance_count: Annotated[StrictInt, Field(ge=1, le=100_000)]
    covered_instance_count: Annotated[StrictInt, Field(ge=0, le=100_000)]
    retained_run_count: Annotated[StrictInt, Field(ge=0, le=10_000_000)]
    successful_run_count: Annotated[StrictInt, Field(ge=0, le=10_000_000)]
    density: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    missing_instance_ids: Annotated[tuple[Identifier, ...], Field(max_length=100_000)] = ()
    operational_validity: Literal["not_established"] = "not_established"


class BenchmarkCoverageAnalysis(PlannerNode):
    """Digest-bound coverage analysis over the package-owned synthetic catalogue."""

    analysis_schema_version: Literal["1"] = "1"
    analysis_id: Identifier
    registry_snapshot_digest: Digest
    cells: Annotated[tuple[CoverageDensityCell, ...], Field(min_length=1, max_length=64)]
    catalogue_family_count: Annotated[StrictInt, Field(ge=1, le=100_000)]
    catalogue_instance_count: Annotated[StrictInt, Field(ge=1, le=100_000)]
    retained_run_count: Annotated[StrictInt, Field(ge=0, le=10_000_000)]
    prospectively_held_out_family_count: Annotated[StrictInt, Field(ge=0, le=100_000)]
    contains_record_values: Literal[False] = False
    contains_record_identifiers: Literal[False] = False
    contains_candidate_pairs: Literal[False] = False
    latent_values_persisted: Literal[False] = False
    operational_validity: Literal["not_established"] = "not_established"

    @property
    def analysis_digest(self) -> str:
        return _digest_payload(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {
            "analysis_schema_version": self.analysis_schema_version,
            "analysis_id": self.analysis_id,
            "analysis_digest": self.analysis_digest,
            "registry_snapshot_digest": self.registry_snapshot_digest,
            "catalogue_family_count": self.catalogue_family_count,
            "catalogue_instance_count": self.catalogue_instance_count,
            "retained_run_count": self.retained_run_count,
            "prospectively_held_out_family_count": self.prospectively_held_out_family_count,
            "cells": [cell.model_dump(mode="json") for cell in self.cells],
            "contains_record_values": self.contains_record_values,
            "contains_record_identifiers": self.contains_record_identifiers,
            "contains_candidate_pairs": self.contains_candidate_pairs,
            "latent_values_persisted": self.latent_values_persisted,
            "operational_validity": self.operational_validity,
        }


class PlannedBenchmarkExperiment(PlannerNode):
    """One bounded synthetic instance/recipe/replicate experiment request."""

    family_id: Identifier
    instance_id: Identifier
    recipe_ids: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=32)]
    gap_cell_ids: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=16)]
    replicates: Annotated[StrictInt, Field(ge=1, le=100)] = 1
    replicate_start: Annotated[StrictInt, Field(ge=0, le=9_999_999)]
    base_seed: Annotated[StrictInt, Field(ge=0, le=4_294_967_295)]
    evidence_scope: Literal["global_synthetic"] = "global_synthetic"
    contains_record_values: Literal[False] = False
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        if len(self.recipe_ids) != len(set(self.recipe_ids)):
            raise ValueError("Planned recipe IDs must be unique.")
        if len(self.gap_cell_ids) != len(set(self.gap_cell_ids)):
            raise ValueError("Planned gap-cell IDs must be unique.")
        return self


class ExperimentPlan(PlannerNode):
    """Advisory-only experiment plan that cannot execute without separate approval."""

    plan_schema_version: Literal["1"] = "1"
    plan_id: Identifier
    planning_status: ExperimentPlanningStatus
    trigger: ExperimentPlanningTrigger
    trigger_report_digest: Digest | None = None
    registry_snapshot_digest: Digest
    target_profile_digest: Digest | None = None
    coverage_analysis_digest: Digest
    experiments: Annotated[tuple[PlannedBenchmarkExperiment, ...], Field(max_length=16)] = ()
    reason_codes: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=16)]
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    execution_authority: Literal["explicit_human_approval_required"] = (
        "explicit_human_approval_required"
    )
    contains_record_values: Literal[False] = False
    contains_record_identifiers: Literal[False] = False
    contains_candidate_pairs: Literal[False] = False

    @model_validator(mode="after")
    def validate_status_and_experiments(self) -> Self:
        instance_ids = [item.instance_id for item in self.experiments]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("Planned experiment instance IDs must be unique.")
        ready = self.planning_status is ExperimentPlanningStatus.READY_FOR_HUMAN_APPROVAL
        if ready != bool(self.experiments):
            raise ValueError("Only a ready experiment plan may contain experiments.")
        return self

    @property
    def plan_digest(self) -> str:
        return _digest_payload(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {
            "plan_schema_version": self.plan_schema_version,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "planning_status": self.planning_status.value,
            "trigger": self.trigger.value,
            "trigger_report_digest": self.trigger_report_digest,
            "registry_snapshot_digest": self.registry_snapshot_digest,
            "target_profile_digest": self.target_profile_digest,
            "coverage_analysis_digest": self.coverage_analysis_digest,
            "experiments": [item.model_dump(mode="json") for item in self.experiments],
            "reason_codes": list(self.reason_codes),
            "recommendation_authority": self.recommendation_authority,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "operational_validity": self.operational_validity,
            "automatic_promotion": self.automatic_promotion,
            "execution_authority": self.execution_authority,
            "contains_record_values": self.contains_record_values,
            "contains_record_identifiers": self.contains_record_identifiers,
            "contains_candidate_pairs": self.contains_candidate_pairs,
        }


class ExperimentExecutionApproval(PlannerNode):
    """Plan-bound human approval without a personal identifier or free-form content."""

    approval_schema_version: Literal["1"] = "1"
    approval_id: Identifier
    plan_digest: Digest
    approved_by_human: Literal[True]
    approved_scope: Literal["synthetic_benchmark_execution"] = "synthetic_benchmark_execution"
    automatic_approval: Literal[False] = False


class ExperimentExecutionReport(PlannerNode):
    """Aggregate execution and substantive advisory refit evidence."""

    report_schema_version: Literal["1"] = "1"
    report_id: Identifier
    plan_digest: Digest
    approval_id: Identifier
    registry_snapshot_digest_before: Digest
    registry_snapshot_digest_after: Digest
    appended_run_count: Annotated[StrictInt, Field(ge=0, le=10_000_000)]
    successful_run_count: Annotated[StrictInt, Field(ge=0, le=10_000_000)]
    unsuccessful_run_count: Annotated[StrictInt, Field(ge=0, le=10_000_000)]
    meta_model_refit_status: MetaModelRefitStatus
    refitted_meta_model_digest: Digest | None = None
    meta_ranking_report_digest: Digest | None = None
    meta_model_trained_run_count: Annotated[StrictInt, Field(ge=0, le=10_000_000)] = 0
    refit_failure_code: Identifier | None = None
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    contains_record_values: Literal[False] = False
    contains_record_identifiers: Literal[False] = False
    contains_candidate_pairs: Literal[False] = False

    def safe_summary(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def _axis_bands(spec: ScenarioLatentSpec) -> Mapping[CoverageDimension, str]:
    total_error = (
        spec.typo_rate
        + spec.token_transposition_rate
        + spec.date_shift_rate
        + spec.date_ambiguity_rate
    )
    if total_error == 0.0:
        error_band = "none"
    elif total_error <= 0.20:
        error_band = "low"
    elif total_error <= 0.60:
        error_band = "moderate"
    else:
        error_band = "high"

    if spec.missingness_rate == 0.0:
        missingness_band = "none"
    elif spec.missingness_rate <= 0.10:
        missingness_band = "low"
    elif spec.missingness_rate <= 0.25:
        missingness_band = "moderate"
    else:
        missingness_band = "high"

    if spec.zipf_skew_parameter == 0.0:
        skew_band = "none"
    elif spec.zipf_skew_parameter <= 1.2:
        skew_band = "moderate"
    else:
        skew_band = "high"

    if spec.label_volume == 0:
        label_band = "none"
    elif spec.label_volume < 200:
        label_band = "sparse"
    else:
        label_band = "dense"

    return {
        CoverageDimension.ERROR_RATE: error_band,
        CoverageDimension.MISSINGNESS: missingness_band,
        CoverageDimension.FREQUENCY_SKEW: skew_band,
        CoverageDimension.LABEL_VOLUME: label_band,
    }


class BenchmarkGapAnalyzer:
    """Analyze retained coverage against an injected package-owned synthetic catalogue."""

    def __init__(
        self,
        registry: BenchmarkRegistry,
        *,
        generator: BenchmarkScenarioGenerator | None = None,
    ) -> None:
        self.registry = registry
        self.generator = generator or BenchmarkScenarioGenerator()

    def _validate_registry_provenance(self) -> None:
        catalogue_families = {item.family_id: item for item in self.generator.list_families()}
        catalogue_instances = {item.instance_id: item for item in self.generator.list_instances()}
        persisted_families = {item.family_id: item for item in self.registry.list_families()}
        persisted_instances = {item.instance_id: item for item in self.registry.list_instances()}

        for family_id, family_manifest in persisted_families.items():
            expected_family = catalogue_families.get(family_id)
            if (
                expected_family is None
                or expected_family.family_digest != family_manifest.family_digest
            ):
                raise AdvisorError(
                    "ML-ADVISOR-040",
                    "Benchmark family provenance does not match the planning catalogue.",
                )
        for instance_id, instance_manifest in persisted_instances.items():
            expected_instance = catalogue_instances.get(instance_id)
            if (
                expected_instance is None
                or expected_instance.instance_digest != instance_manifest.instance_digest
            ):
                raise AdvisorError(
                    "ML-ADVISOR-041",
                    "Benchmark instance provenance does not match the planning catalogue.",
                )

        for record in self.registry.list_run_records():
            instance = persisted_instances.get(record.instance_id)
            family = persisted_families.get(record.family_id)
            if instance is None or family is None or instance.family_id != record.family_id:
                raise AdvisorError(
                    "ML-ADVISOR-042",
                    "Benchmark run evidence is missing its matching persisted manifests.",
                )

    def analyze(self) -> BenchmarkCoverageAnalysis:
        """Compute aggregate density; unsuccessful runs remain retained coverage evidence."""
        self._validate_registry_provenance()
        records = self.registry.list_run_records()
        snapshot = build_registry_snapshot(
            snapshot_id="snapshot.active_planner",
            records=records,
        )
        catalogue_instances = self.generator.list_instances()
        catalogue_families = self.generator.list_families()

        instances_by_cell: dict[tuple[CoverageDimension, str], set[str]] = defaultdict(set)
        for instance in catalogue_instances:
            spec = self.generator.get_latent_spec(instance.instance_id)
            for dimension, band in _axis_bands(spec).items():
                instances_by_cell[(dimension, band)].add(instance.instance_id)

        records_by_instance: dict[str, list[object]] = defaultdict(list)
        for record in records:
            records_by_instance[record.instance_id].append(record)

        cells: list[CoverageDensityCell] = []
        for (dimension, band), instance_ids in sorted(
            instances_by_cell.items(), key=lambda item: (item[0][0].value, item[0][1])
        ):
            covered = {item for item in instance_ids if records_by_instance.get(item)}
            retained_records = [
                record for item in instance_ids for record in records_by_instance.get(item, [])
            ]
            successful = sum(
                getattr(record, "status", None) is BenchmarkRunStatus.SUCCESS
                for record in retained_records
            )
            cells.append(
                CoverageDensityCell(
                    dimension=dimension,
                    band=band,
                    catalogue_instance_count=len(instance_ids),
                    covered_instance_count=len(covered),
                    retained_run_count=len(retained_records),
                    successful_run_count=successful,
                    density=len(covered) / len(instance_ids),
                    missing_instance_ids=tuple(sorted(instance_ids - covered)),
                )
            )

        analysis_seed = {
            "snapshot": snapshot.registry_digest,
            "cells": [cell.model_dump(mode="json") for cell in cells],
        }
        return BenchmarkCoverageAnalysis(
            analysis_id=f"coverage.active.{_digest_payload(analysis_seed)[:24]}",
            registry_snapshot_digest=snapshot.registry_digest,
            cells=tuple(cells),
            catalogue_family_count=len(catalogue_families),
            catalogue_instance_count=len(catalogue_instances),
            retained_run_count=len(records),
            prospectively_held_out_family_count=sum(
                family.prospectively_held_out for family in catalogue_families
            ),
        )


def _next_replicate_start(instance_id: str, registry: BenchmarkRegistry) -> int:
    observed: list[int] = []
    for record in registry.list_run_records(instance_id=instance_id):
        prefix = "replicate."
        if record.replicate_id.startswith(prefix):
            suffix = record.replicate_id[len(prefix) :]
            if suffix.isdigit():
                observed.append(int(suffix))
    return max(observed, default=-1) + 1


class ActiveBenchmarkPlanner:
    """Generate bounded experiment plans when aggregate epistemic uncertainty is high."""

    def __init__(
        self,
        registry: BenchmarkRegistry,
        *,
        generator: BenchmarkScenarioGenerator | None = None,
        runner: BenchmarkPortfolioRunner | None = None,
        minimum_coverage_density: float = 1.0,
        maximum_interval_width: float = 0.35,
        maximum_ood_distance: float = 0.45,
        maximum_experiments: int = 4,
    ) -> None:
        if not 0.0 <= minimum_coverage_density <= 1.0:
            raise ValueError("Minimum coverage density must be between zero and one.")
        if not 0.0 <= maximum_interval_width <= 1.0:
            raise ValueError("Maximum interval width must be between zero and one.")
        if not 0.0 <= maximum_ood_distance <= 1.0:
            raise ValueError("Maximum OOD distance must be between zero and one.")
        if not 1 <= maximum_experiments <= 16:
            raise ValueError("Maximum planned experiments must be between one and sixteen.")
        self.registry = registry
        self.generator = generator or BenchmarkScenarioGenerator()
        self.runner = runner or BenchmarkPortfolioRunner()
        self.minimum_coverage_density = minimum_coverage_density
        self.maximum_interval_width = maximum_interval_width
        self.maximum_ood_distance = maximum_ood_distance
        self.maximum_experiments = maximum_experiments
        self.gap_analyzer = BenchmarkGapAnalyzer(registry, generator=self.generator)

    def _trigger_from_report(
        self,
        report: SimilarityAdvisoryReport | MetaRankingAdvisoryReport,
    ) -> tuple[ExperimentPlanningTrigger, tuple[str, ...], str]:
        if isinstance(report, SimilarityAdvisoryReport):
            if (
                report.out_of_distribution
                or report.out_of_distribution_score > self.maximum_ood_distance
            ):
                return (
                    ExperimentPlanningTrigger.SIMILARITY_OUT_OF_DISTRIBUTION,
                    ("planner.high_ood_distance",),
                    report.report_digest,
                )
            return (
                ExperimentPlanningTrigger.NOT_TRIGGERED,
                ("planner.uncertainty_below_trigger",),
                report.report_digest,
            )

        if report.fallback_to_similarity:
            return (
                ExperimentPlanningTrigger.META_RANKER_FALLBACK,
                ("planner.meta_ranker_fallback",),
                report.report_digest,
            )
        widths = [
            item.uncertainty_upper_bound - item.uncertainty_lower_bound
            for item in report.predicted_candidate_utilities.values()
        ]
        if widths and max(widths) > self.maximum_interval_width:
            return (
                ExperimentPlanningTrigger.META_RANKER_WIDE_INTERVAL,
                ("planner.wide_conformal_interval",),
                report.report_digest,
            )
        return (
            ExperimentPlanningTrigger.NOT_TRIGGERED,
            ("planner.uncertainty_below_trigger",),
            report.report_digest,
        )

    def _compatible_recipes(self, instance_id: str) -> tuple[BenchmarkRecipe, ...]:
        profile = self.generator.build_task_profile(instance_id)
        return tuple(
            recipe
            for recipe in self.runner.list_recipes()
            if recipe.linkage_mode == profile.linkage_mode
            and (not recipe.requires_verified_labels or profile.verified_labels_available)
        )

    def plan(
        self,
        report: SimilarityAdvisoryReport | MetaRankingAdvisoryReport | None = None,
        *,
        target_profile: PreflightTaskProfile | None = None,
    ) -> ExperimentPlan:
        """Build a deterministic snapshot-bound plan; this method never executes it."""
        analysis = self.gap_analyzer.analyze()
        gap_cells = tuple(
            cell
            for cell in analysis.cells
            if cell.density < self.minimum_coverage_density and cell.missing_instance_ids
        )

        if report is None:
            trigger = (
                ExperimentPlanningTrigger.COVERAGE_GAP
                if gap_cells
                else ExperimentPlanningTrigger.NOT_TRIGGERED
            )
            reason_codes: tuple[str, ...] = (
                ("planner.coverage_gap",) if gap_cells else ("planner.coverage_sufficient",)
            )
            trigger_report_digest = None
        else:
            trigger, reason_codes, trigger_report_digest = self._trigger_from_report(report)
            report_profile_digest = (
                report.target_task_profile_digest
                if isinstance(report, SimilarityAdvisoryReport)
                else report.recommendation.task_profile_digest
            )
            if (
                target_profile is not None
                and target_profile.profile_digest != report_profile_digest
            ):
                raise AdvisorError(
                    "ML-ADVISOR-043",
                    "The supplied target profile does not match the uncertainty report.",
                )

        profile_digest = (
            target_profile.profile_digest
            if target_profile is not None
            else (
                report.target_task_profile_digest
                if isinstance(report, SimilarityAdvisoryReport)
                else (
                    report.recommendation.task_profile_digest
                    if isinstance(report, MetaRankingAdvisoryReport)
                    else None
                )
            )
        )

        experiments: tuple[PlannedBenchmarkExperiment, ...] = ()
        status = ExperimentPlanningStatus.NOT_TRIGGERED
        if trigger is not ExperimentPlanningTrigger.NOT_TRIGGERED:
            held_out_families = {
                family.family_id
                for family in self.generator.list_families()
                if family.prospectively_held_out
            }
            cell_ids_by_instance: dict[str, set[str]] = defaultdict(set)
            for cell in gap_cells:
                cell_id = f"{cell.dimension.value}.{cell.band}"
                for instance_id in cell.missing_instance_ids:
                    cell_ids_by_instance[instance_id].add(cell_id)

            target_vector = (
                TaskMetaFeatureVector.from_profile(target_profile)
                if target_profile is not None
                else None
            )
            distance_computer = MetaFeatureDistanceComputer()
            candidates: list[tuple[str, str, int, float]] = []
            for instance_id, cell_ids in cell_ids_by_instance.items():
                manifest = self.generator.get_instance(instance_id)
                if manifest.family_id in held_out_families:
                    continue
                distance = 0.0
                if target_vector is not None:
                    spec_vector = TaskMetaFeatureVector.from_latent_spec(
                        self.generator.get_latent_spec(instance_id)
                    )
                    distance = distance_computer.compute_distance(target_vector, spec_vector)
                candidates.append((instance_id, manifest.family_id, len(cell_ids), distance))

            candidates.sort(key=lambda item: (-item[2], item[3], item[1], item[0]))
            diverse = []
            seen_families: set[str] = set()
            for item in candidates:
                if item[1] not in seen_families:
                    diverse.append(item)
                    seen_families.add(item[1])
                if len(diverse) >= self.maximum_experiments:
                    break
            if len(diverse) < self.maximum_experiments:
                for item in candidates:
                    if item not in diverse:
                        diverse.append(item)
                    if len(diverse) >= self.maximum_experiments:
                        break

            basis_digest = _digest_payload(
                {
                    "analysis": analysis.analysis_digest,
                    "trigger": trigger.value,
                    "report": trigger_report_digest,
                    "profile": profile_digest,
                    "maximum_experiments": self.maximum_experiments,
                }
            )
            planned: list[PlannedBenchmarkExperiment] = []
            for instance_id, family_id, _gap_count, _distance in diverse:
                recipes = self._compatible_recipes(instance_id)
                if not recipes:
                    continue
                seed = (
                    _DEFAULT_BASE_SEED
                    + _stable_uint32(
                        {
                            "basis": basis_digest,
                            "family": family_id,
                            "instance": instance_id,
                        }
                    )
                ) % _UINT32_MODULUS
                planned.append(
                    PlannedBenchmarkExperiment(
                        family_id=family_id,
                        instance_id=instance_id,
                        recipe_ids=tuple(recipe.recipe_id for recipe in recipes),
                        gap_cell_ids=tuple(sorted(cell_ids_by_instance[instance_id])),
                        replicates=1,
                        replicate_start=_next_replicate_start(instance_id, self.registry),
                        base_seed=seed,
                    )
                )
            experiments = tuple(planned)
            if experiments:
                status = ExperimentPlanningStatus.READY_FOR_HUMAN_APPROVAL
            else:
                status = ExperimentPlanningStatus.ABSTAINED
                reason_codes = tuple(dict.fromkeys((*reason_codes, "planner.catalogue_exhausted")))

        plan_seed = {
            "analysis": analysis.analysis_digest,
            "status": status.value,
            "trigger": trigger.value,
            "report": trigger_report_digest,
            "profile": profile_digest,
            "experiments": [item.model_dump(mode="json") for item in experiments],
            "reasons": reason_codes,
        }
        return ExperimentPlan(
            plan_id=f"plan.active.{_digest_payload(plan_seed)[:24]}",
            planning_status=status,
            trigger=trigger,
            trigger_report_digest=trigger_report_digest,
            registry_snapshot_digest=analysis.registry_snapshot_digest,
            target_profile_digest=profile_digest,
            coverage_analysis_digest=analysis.analysis_digest,
            experiments=experiments,
            reason_codes=reason_codes,
        )


def execute_planned_experiments(
    plan: ExperimentPlan,
    *,
    approval: ExperimentExecutionApproval,
    registry: BenchmarkRegistry,
    linkage_plan: ExecutionPlan,
    advisor_context: AdvisorContext,
    target_profile: PreflightTaskProfile,
    generator: BenchmarkScenarioGenerator | None = None,
    runner: BenchmarkPortfolioRunner | None = None,
) -> ExperimentExecutionReport:
    """Execute an approved, current plan, append evidence, and refit the advisory model."""
    if plan.planning_status is not ExperimentPlanningStatus.READY_FOR_HUMAN_APPROVAL:
        raise AdvisorError("ML-ADVISOR-044", "Only a ready experiment plan may be executed.")
    if approval.plan_digest != plan.plan_digest:
        raise AdvisorError(
            "ML-ADVISOR-045",
            "The human approval is not bound to this experiment plan.",
        )
    if plan.target_profile_digest != target_profile.profile_digest:
        raise AdvisorError(
            "ML-ADVISOR-046",
            "The execution target profile does not match the approved plan.",
        )

    gen = generator or BenchmarkScenarioGenerator()
    run_engine = runner or BenchmarkPortfolioRunner()
    analyzer = BenchmarkGapAnalyzer(registry, generator=gen)
    current = analyzer.analyze()
    if current.registry_snapshot_digest != plan.registry_snapshot_digest:
        raise AdvisorError(
            "ML-ADVISOR-047",
            "The benchmark registry changed after planning; a new plan and approval are required.",
        )

    recipe_by_id = {recipe.recipe_id: recipe for recipe in run_engine.list_recipes()}
    held_out = {family.family_id for family in gen.list_families() if family.prospectively_held_out}
    results: list[BenchmarkRunResult] = []
    for experiment in plan.experiments:
        manifest = gen.get_instance(experiment.instance_id)
        if manifest.family_id != experiment.family_id or experiment.family_id in held_out:
            raise AdvisorError(
                "ML-ADVISOR-048",
                "A planned experiment no longer satisfies catalogue or holdout constraints.",
            )
        try:
            recipes = tuple(recipe_by_id[item] for item in experiment.recipe_ids)
        except KeyError as error:
            raise AdvisorError(
                "ML-ADVISOR-049",
                "A planned benchmark recipe is unavailable from the package-owned runner.",
            ) from error
        results.extend(
            run_engine.run_portfolio(
                gen,
                families=(experiment.family_id,),
                instances=(experiment.instance_id,),
                recipes=recipes,
                replicates=experiment.replicates,
                base_seed=experiment.base_seed,
                replicate_start=experiment.replicate_start,
            )
        )

    result_run_ids = [result.record.run_id for result in results]
    existing_run_ids = {record.run_id for record in registry.list_run_records()}
    if len(result_run_ids) != len(set(result_run_ids)) or existing_run_ids.intersection(
        result_run_ids
    ):
        raise AdvisorError(
            "ML-ADVISOR-050",
            "Planned benchmark run IDs would collide with retained evidence.",
        )

    for experiment in plan.experiments:
        registry.save_family(gen.get_family(experiment.family_id))
        registry.save_instance(gen.get_instance(experiment.instance_id))
    for result in results:
        registry.save_run_record(
            result.record,
            metrics=result.metrics,
            failure=result.failure,
        )

    after = build_registry_snapshot(
        snapshot_id="snapshot.active_planner",
        records=registry.list_run_records(),
    )
    meta_report = None
    fitted_model_digest = None
    fitted_model_training_count = 0
    refit_failure_code = None
    try:
        meta_advisor = MetaRankingLinkageAdvisor(registry=registry, generator=gen)
        meta_report = meta_advisor.advise(
            linkage_plan,
            context=advisor_context,
            profile=target_profile,
        )
        if meta_advisor.last_fitted_model is None:
            refit_status = MetaModelRefitStatus.ABSTAINED_INSUFFICIENT_EVIDENCE
        else:
            refit_status = MetaModelRefitStatus.FITTED
            fitted_model_digest = meta_advisor.last_fitted_model.model_digest
            fitted_model_training_count = meta_advisor.last_fitted_model.trained_run_count
    except AdvisorError as error:
        refit_status = MetaModelRefitStatus.FAILED
        refit_failure_code = error.code

    success_count = sum(result.record.status is BenchmarkRunStatus.SUCCESS for result in results)
    report_seed = {
        "plan": plan.plan_digest,
        "approval": approval.approval_id,
        "after": after.registry_digest,
        "refit": refit_status.value,
        "model": fitted_model_digest,
    }
    return ExperimentExecutionReport(
        report_id=f"execution.active.{_digest_payload(report_seed)[:24]}",
        plan_digest=plan.plan_digest,
        approval_id=approval.approval_id,
        registry_snapshot_digest_before=current.registry_snapshot_digest,
        registry_snapshot_digest_after=after.registry_digest,
        appended_run_count=len(results),
        successful_run_count=success_count,
        unsuccessful_run_count=len(results) - success_count,
        meta_model_refit_status=refit_status,
        refitted_meta_model_digest=fitted_model_digest,
        meta_ranking_report_digest=(meta_report.report_digest if meta_report is not None else None),
        meta_model_trained_run_count=fitted_model_training_count,
        refit_failure_code=refit_failure_code,
    )


__all__ = [
    "ActiveBenchmarkPlanner",
    "BenchmarkCoverageAnalysis",
    "BenchmarkGapAnalyzer",
    "CoverageDensityCell",
    "CoverageDimension",
    "ExperimentExecutionApproval",
    "ExperimentExecutionReport",
    "ExperimentPlan",
    "ExperimentPlanningStatus",
    "ExperimentPlanningTrigger",
    "MetaModelRefitStatus",
    "PlannedBenchmarkExperiment",
    "execute_planned_experiments",
]
