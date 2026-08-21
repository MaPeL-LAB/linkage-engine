"""Approved, append-only execution and resume controls for advisor-scale benchmarks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from mapel_linkage.benchmarking.advisor_catalogue import (
    AdvisorCorpusDesignManifest,
    AdvisorCorpusReadinessManifest,
    BenchmarkShardPlan,
    build_advisor_corpus_design,
    build_advisor_corpus_readiness,
    build_advisor_v2_generator,
)
from mapel_linkage.benchmarking.contracts import BenchmarkRunRecord, BenchmarkRunStatus
from mapel_linkage.benchmarking.registry import BenchmarkRegistry
from mapel_linkage.benchmarking.runner import (
    BenchmarkPortfolioRunner,
    benchmark_replicate_seed,
    benchmark_run_id,
)
from mapel_linkage.governance.atomic import atomic_write_text


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class CorpusExecutionApproval(BaseModel):
    """Non-identifying human authorization for one exact synthetic shard plan."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    approval_schema_version: Literal["1"] = "1"
    approval_reference: Annotated[StrictStr, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")]
    human_approved: Literal[True]
    design_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    shard_plan_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    replicates: Annotated[StrictInt, Field(ge=1, le=100)]
    base_seed: Annotated[StrictInt, Field(ge=0, le=4_294_967_295)] = 20260816
    synthetic_only: Literal[True] = True
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    release_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    @property
    def approval_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {
            "approval_schema_version": self.approval_schema_version,
            "approval_digest": self.approval_digest,
            "human_approved": self.human_approved,
            "design_digest": self.design_digest,
            "shard_plan_digest": self.shard_plan_digest,
            "replicates": self.replicates,
            "base_seed": self.base_seed,
            "synthetic_only": self.synthetic_only,
            "recommendation_authority": self.recommendation_authority,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "automatic_promotion": self.automatic_promotion,
            "release_authority": self.release_authority,
            "operational_validity": self.operational_validity,
        }


class CorpusShardExecutionReport(BaseModel):
    """Aggregate result for an idempotent execution or resume attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    report_schema_version: Literal["1"] = "1"
    design_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    shard_plan_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    approval_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    shard_index: Annotated[StrictInt, Field(ge=0)]
    shard_instance_count: Annotated[StrictInt, Field(ge=1)]
    expected_run_count: Annotated[StrictInt, Field(ge=1)]
    completed_run_count: Annotated[StrictInt, Field(ge=0)]
    newly_persisted_run_count: Annotated[StrictInt, Field(ge=0)]
    resumed_run_count: Annotated[StrictInt, Field(ge=0)]
    successful_run_count: Annotated[StrictInt, Field(ge=0)]
    retained_non_success_run_count: Annotated[StrictInt, Field(ge=0)]
    shard_complete: StrictBool
    append_only: Literal[True] = True
    idempotent_resume: Literal[True] = True
    contains_record_values: Literal[False] = False
    contains_identifiers: Literal[False] = False
    contains_candidate_pairs: Literal[False] = False
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_counts(self) -> CorpusShardExecutionReport:
        if (
            self.completed_run_count
            != self.successful_run_count + self.retained_non_success_run_count
        ):
            raise ValueError("Shard completion counts are inconsistent.")
        if self.newly_persisted_run_count + self.resumed_run_count != self.completed_run_count:
            raise ValueError("Shard resume counts are inconsistent.")
        if self.shard_complete != (self.completed_run_count == self.expected_run_count):
            raise ValueError("Shard completeness must fail closed on missing run evidence.")
        return self

    @property
    def report_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {
            "report_schema_version": self.report_schema_version,
            "report_digest": self.report_digest,
            "design_digest": self.design_digest,
            "shard_plan_digest": self.shard_plan_digest,
            "approval_digest": self.approval_digest,
            "shard_index": self.shard_index,
            "shard_instance_count": self.shard_instance_count,
            "expected_run_count": self.expected_run_count,
            "completed_run_count": self.completed_run_count,
            "newly_persisted_run_count": self.newly_persisted_run_count,
            "resumed_run_count": self.resumed_run_count,
            "successful_run_count": self.successful_run_count,
            "retained_non_success_run_count": self.retained_non_success_run_count,
            "shard_complete": self.shard_complete,
            "append_only": self.append_only,
            "idempotent_resume": self.idempotent_resume,
            "contains_record_values": self.contains_record_values,
            "contains_identifiers": self.contains_identifiers,
            "contains_candidate_pairs": self.contains_candidate_pairs,
            "recommendation_authority": self.recommendation_authority,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "automatic_promotion": self.automatic_promotion,
            "operational_validity": self.operational_validity,
        }


def _canonical_text(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def _persist_exact(path: Path, model: BaseModel) -> None:
    """Persist one immutable canonical manifest or accept its exact idempotent replay."""

    text = _canonical_text(model)
    try:
        if path.is_symlink():
            raise ValueError("Benchmark governance artifacts cannot be symbolic links.")
        if path.exists():
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                raise FileExistsError("A conflicting benchmark governance artifact exists.")
            return
        if path.parent.is_symlink():
            raise ValueError("Benchmark governance directories cannot be symbolic links.")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, text)
    except (FileExistsError, ValueError):
        raise
    except OSError:
        raise ValueError("A benchmark governance artifact could not be persisted safely.") from None


def _validate_existing_record(
    record: BenchmarkRunRecord,
    *,
    family_id: str,
    instance_id: str,
    recipe_digest: str,
    replicate_id: str,
    seed: int,
    expected_provenance: dict[str, str],
) -> None:
    if (
        record.family_id != family_id
        or record.instance_id != instance_id
        or record.pipeline_recipe_digest != recipe_digest
        or record.replicate_id != replicate_id
        or record.random_seed != seed
    ):
        raise FileExistsError("Existing benchmark run provenance conflicts with the shard plan.")
    if (
        record.engine_commit != expected_provenance["engine_commit"]
        or record.dependency_lock_digest != expected_provenance["dependency_lock_digest"]
        or record.environment_digest != expected_provenance["environment_digest"]
    ):
        raise FileExistsError("Existing benchmark evidence is stale for the current environment.")


def _governance_preflight(
    *,
    registry: BenchmarkRegistry,
    design: AdvisorCorpusDesignManifest,
    readiness: AdvisorCorpusReadinessManifest,
    shard_plan: BenchmarkShardPlan,
    approval: CorpusExecutionApproval,
    replicates: int,
    base_seed: int,
) -> None:
    if (
        not readiness.execution_ready
        or shard_plan.design_digest != design.design_digest
        or approval.design_digest != design.design_digest
        or approval.shard_plan_digest != shard_plan.plan_digest
        or approval.replicates != replicates
        or approval.base_seed != base_seed
    ):
        raise ValueError("Advisor corpus execution is not approved and ready for this exact plan.")
    governance = registry.root_directory / "governance"
    _persist_exact(governance / f"design.{design.design_digest}.json", design)
    _persist_exact(governance / f"shards.{shard_plan.plan_digest}.json", shard_plan)
    _persist_exact(governance / f"readiness.{readiness.readiness_digest}.json", readiness)
    _persist_exact(governance / f"approval.{approval.approval_digest}.json", approval)


def execute_advisor_corpus_shard(
    *,
    registry: BenchmarkRegistry,
    shard_plan: BenchmarkShardPlan,
    shard_index: int,
    approval: CorpusExecutionApproval,
    replicates: int = 5,
    base_seed: int = 20260816,
    runner: BenchmarkPortfolioRunner | None = None,
) -> CorpusShardExecutionReport:
    """Execute or resume one exact approved shard without overwriting evidence."""

    if not 1 <= replicates <= 100 or not 0 <= shard_index < shard_plan.shard_count:
        raise ValueError("Advisor corpus shard execution inputs are outside safe bounds.")
    run_engine = runner or BenchmarkPortfolioRunner()
    design = build_advisor_corpus_design()
    readiness = build_advisor_corpus_readiness(
        adapter_statuses=run_engine.adapter_statuses(),
        planned_replicates_per_instance=replicates,
    )
    _governance_preflight(
        registry=registry,
        design=design,
        readiness=readiness,
        shard_plan=shard_plan,
        approval=approval,
        replicates=replicates,
        base_seed=base_seed,
    )
    generator = build_advisor_v2_generator()
    shard = shard_plan.shards[shard_index]
    recipes = run_engine.list_recipes()
    expected_provenance = run_engine.provenance_summary()

    family_ids = {
        generator.get_instance(instance_id).family_id for instance_id in shard.instance_ids
    }
    for family_id in sorted(family_ids):
        registry.save_family(generator.get_family(family_id))
    for instance_id in shard.instance_ids:
        registry.save_instance(generator.get_instance(instance_id))

    newly_persisted = 0
    resumed = 0
    final_records: dict[str, BenchmarkRunRecord] = {}
    for instance_id in shard.instance_ids:
        family_id = generator.get_instance(instance_id).family_id
        for replicate_number in range(replicates):
            replicate_id = f"replicate.{replicate_number:07d}"
            seed = benchmark_replicate_seed(
                instance_id=instance_id,
                replicate_number=replicate_number,
                base_seed=base_seed,
            )
            existing: dict[str, BenchmarkRunRecord] = {}
            for recipe in recipes:
                expected_id = benchmark_run_id(
                    instance_id=instance_id,
                    recipe_id=recipe.recipe_id,
                    replicate_id=replicate_id,
                )
                try:
                    record = registry.load_run_record(expected_id)
                except FileNotFoundError:
                    continue
                _validate_existing_record(
                    record,
                    family_id=family_id,
                    instance_id=instance_id,
                    recipe_digest=recipe.recipe_digest,
                    replicate_id=replicate_id,
                    seed=seed,
                    expected_provenance=expected_provenance,
                )
                existing[expected_id] = record
                final_records[expected_id] = record
            if len(existing) == len(recipes):
                resumed += len(existing)
                continue

            results = run_engine.run_portfolio(
                generator,
                instances=(instance_id,),
                recipes=recipes,
                replicates=1,
                base_seed=base_seed,
                replicate_start=replicate_number,
            )
            if len(results) != len(recipes):
                raise ValueError("Benchmark portfolio output does not cover the approved recipes.")
            for result in results:
                prior = existing.get(result.record.run_id)
                if prior is not None:
                    resumed += 1
                    continue
                registry.save_run_record(
                    result.record,
                    metrics=result.metrics,
                    failure=result.failure,
                )
                newly_persisted += 1
                final_records[result.record.run_id] = result.record

    expected_run_count = len(shard.instance_ids) * replicates * len(recipes)
    if len(final_records) != expected_run_count:
        raise ValueError("Approved shard execution did not produce complete retained evidence.")
    successful = sum(
        record.status == BenchmarkRunStatus.SUCCESS for record in final_records.values()
    )
    report = CorpusShardExecutionReport(
        design_digest=design.design_digest,
        shard_plan_digest=shard_plan.plan_digest,
        approval_digest=approval.approval_digest,
        shard_index=shard_index,
        shard_instance_count=len(shard.instance_ids),
        expected_run_count=expected_run_count,
        completed_run_count=len(final_records),
        newly_persisted_run_count=newly_persisted,
        resumed_run_count=resumed,
        successful_run_count=successful,
        retained_non_success_run_count=len(final_records) - successful,
        shard_complete=True,
    )
    completion = report.model_copy(
        update={
            "newly_persisted_run_count": 0,
            "resumed_run_count": report.completed_run_count,
        }
    )
    _persist_exact(
        registry.root_directory
        / "governance"
        / f"completion.{shard_plan.plan_digest}.{shard_index:03d}.json",
        completion,
    )
    return report


def audit_advisor_corpus(
    *,
    registry: BenchmarkRegistry,
    shard_plan: BenchmarkShardPlan,
    replicates: int = 5,
    base_seed: int = 20260816,
    runner: BenchmarkPortfolioRunner | None = None,
) -> AdvisorCorpusReadinessManifest:
    """Audit aggregate completion and the complete required-adapter cell grid."""

    if not 1 <= replicates <= 100:
        raise ValueError("Advisor corpus audit inputs are outside safe bounds.")
    run_engine = runner or BenchmarkPortfolioRunner()
    design = build_advisor_corpus_design()
    if shard_plan.design_digest != design.design_digest:
        raise ValueError("Advisor corpus audit plan does not match the versioned design.")
    generator = build_advisor_v2_generator()
    recipes = run_engine.list_recipes()
    expected_provenance = run_engine.provenance_summary()
    expected: dict[str, tuple[str, str, str, str, int, str | None]] = {}
    for shard in shard_plan.shards:
        for instance_id in shard.instance_ids:
            family_id = generator.get_instance(instance_id).family_id
            for replicate_number in range(replicates):
                replicate_id = f"replicate.{replicate_number:07d}"
                seed = benchmark_replicate_seed(
                    instance_id=instance_id,
                    replicate_number=replicate_number,
                    base_seed=base_seed,
                )
                for recipe in recipes:
                    run_id = benchmark_run_id(
                        instance_id=instance_id,
                        recipe_id=recipe.recipe_id,
                        replicate_id=replicate_id,
                    )
                    expected[run_id] = (
                        family_id,
                        instance_id,
                        recipe.recipe_digest,
                        replicate_id,
                        seed,
                        (
                            "fellegi_sunter"
                            if recipe.recipe_id == "recipe.fellegi_sunter_reference"
                            else "xgboost_classifier"
                            if recipe.recipe_id == "recipe.xgboost_classifier"
                            else "xgboost_ranker"
                            if recipe.recipe_id == "recipe.xgboost_ranker"
                            else None
                        ),
                    )

    records = registry.list_run_records()
    actual_ids = {record.run_id for record in records}
    if actual_ids - set(expected):
        raise ValueError("The advisor registry contains runs outside the approved corpus plan.")
    successful_tokens_by_family: dict[str, set[str]] = {}
    successful_tokens_by_cell: dict[tuple[str, str], set[str]] = {}
    successful_required_runs = 0
    failed_required_runs = 0
    for record in records:
        family_id, instance_id, recipe_digest, replicate_id, seed, token = expected[record.run_id]
        _validate_existing_record(
            record,
            family_id=family_id,
            instance_id=instance_id,
            recipe_digest=recipe_digest,
            replicate_id=replicate_id,
            seed=seed,
            expected_provenance=expected_provenance,
        )
        if record.status == BenchmarkRunStatus.SUCCESS and token is not None:
            successful_required_runs += 1
            successful_tokens_by_family.setdefault(record.family_id, set()).add(token)
            successful_tokens_by_cell.setdefault(
                (record.instance_id, record.replicate_id), set()
            ).add(token)
        elif token is not None:
            failed_required_runs += 1
    required_tokens = {"fellegi_sunter", "xgboost_classifier", "xgboost_ranker"}
    successful_overlap = sum(
        required_tokens.issubset(tokens) for tokens in successful_tokens_by_family.values()
    )
    expected_cells = {
        (instance_id, f"replicate.{replicate_number:07d}")
        for shard in shard_plan.shards
        for instance_id in shard.instance_ids
        for replicate_number in range(replicates)
    }
    successful_cells = sum(
        required_tokens.issubset(successful_tokens_by_cell.get(cell, set()))
        for cell in expected_cells
    )
    expected_required_runs = len(expected_cells) * len(required_tokens)
    missing_required_runs = expected_required_runs - successful_required_runs - failed_required_runs
    if missing_required_runs < 0:
        raise ValueError("Required advisor evidence counts exceed the approved grid.")
    completed = len(records)
    expected_count = len(expected)
    status: Literal["not_started", "partial", "complete"] = (
        "not_started"
        if completed == 0
        else "complete"
        if completed == expected_count
        else "partial"
    )
    initial = build_advisor_corpus_readiness(
        adapter_statuses=run_engine.adapter_statuses(),
        planned_replicates_per_instance=replicates,
    )
    readiness = AdvisorCorpusReadinessManifest.model_validate(
        {
            **initial.model_dump(mode="json"),
            "execution_status": status,
            "expected_run_count": expected_count,
            "completed_run_count": completed,
            "successful_overlap_family_count": successful_overlap,
            "successful_evidence_cell_count": successful_cells,
            "successful_required_adapter_run_count": successful_required_runs,
            "failed_required_adapter_run_count": failed_required_runs,
            "missing_required_adapter_run_count": missing_required_runs,
            "advisor_evidence_ready": (
                status == "complete"
                and replicates >= initial.minimum_replicates_per_instance
                and successful_overlap == initial.required_overlap_family_count
                and successful_cells == initial.required_evidence_cell_count
                and successful_required_runs == initial.expected_required_adapter_run_count
                and failed_required_runs == 0
                and missing_required_runs == 0
            ),
        }
    )
    _persist_exact(
        registry.root_directory
        / "governance"
        / f"readiness.audit.{readiness.readiness_digest}.json",
        readiness,
    )
    return readiness


__all__ = [
    "CorpusExecutionApproval",
    "CorpusShardExecutionReport",
    "audit_advisor_corpus",
    "execute_advisor_corpus_shard",
]
