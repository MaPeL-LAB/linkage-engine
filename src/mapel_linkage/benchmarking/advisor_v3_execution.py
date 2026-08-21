"""Serial governance preparation and concurrent-safe advisor-v3 shard execution."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, BinaryIO, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from mapel_linkage.benchmarking.advisor_v3_catalogue import (
    AdvisorV3CorpusReadinessManifest,
    AdvisorV3PreregistrationManifest,
    AdvisorV3ShardPlan,
    build_advisor_v3_corpus_design,
    build_advisor_v3_corpus_readiness,
    build_advisor_v3_generator,
    build_advisor_v3_preregistration,
    build_advisor_v3_shard_plan,
)
from mapel_linkage.benchmarking.contracts import BenchmarkRunRecord, BenchmarkRunStatus
from mapel_linkage.benchmarking.registry import BenchmarkRegistry
from mapel_linkage.benchmarking.runner import (
    BenchmarkPortfolioRunner,
    benchmark_replicate_seed,
    benchmark_run_id,
)
from mapel_linkage.governance.atomic import atomic_write_text

_MAX_PREREGISTRATION_BYTES = 4 * 1024 * 1024
_PROVENANCE_KEYS = frozenset({"engine_commit", "dependency_lock_digest", "environment_digest"})
_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_text(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def advisor_v3_execution_provenance_digest(runner: BenchmarkPortfolioRunner) -> str:
    """Bind the exact aggregate engine, dependency, and environment provenance."""

    provenance = runner.provenance_summary()
    if set(provenance) != _PROVENANCE_KEYS or any(
        _DIGEST_PATTERN.fullmatch(value) is None for value in provenance.values()
    ):
        raise ValueError("Advisor-v3 execution provenance is incomplete or invalid.")
    return _digest(
        {
            "provenance_schema_id": "advisor_v3_execution_provenance_v1",
            "engine_source_digest": provenance["engine_commit"],
            "dependency_lock_digest": provenance["dependency_lock_digest"],
            "environment_digest": provenance["environment_digest"],
        }
    )


class AdvisorV3CorpusExecutionApproval(BaseModel):
    """Human approval for the exact outcome-free preregistered synthetic run grid."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    approval_schema_version: Literal["3"] = "3"
    approval_reference: Annotated[
        StrictStr, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$", repr=False)
    ]
    human_approved: Literal[True]
    preregistration_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    design_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    catalogue_manifest_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    shard_plan_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    readiness_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    execution_provenance_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    replicates: Literal[5] = 5
    base_seed: Literal[20260816] = 20260816
    synthetic_only: Literal[True] = True
    locked_evaluation_access_authorized: Literal[False] = False
    ood_evaluation_access_authorized: Literal[False] = False
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    release_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_canonical_bindings(self) -> Self:
        preregistration = build_advisor_v3_preregistration()
        design = build_advisor_v3_corpus_design()
        plan = build_advisor_v3_shard_plan()
        if (
            self.preregistration_digest != preregistration.preregistration_digest
            or self.design_digest != design.design_digest
            or self.catalogue_manifest_digest != design.catalogue_manifest_digest
            or self.shard_plan_digest != plan.plan_digest
        ):
            raise ValueError("Advisor-v3 execution approval does not bind the canonical plan.")
        return self

    @property
    def approval_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {
            "approval_schema_version": self.approval_schema_version,
            "approval_digest": self.approval_digest,
            "human_approved": self.human_approved,
            "preregistration_digest": self.preregistration_digest,
            "design_digest": self.design_digest,
            "catalogue_manifest_digest": self.catalogue_manifest_digest,
            "shard_plan_digest": self.shard_plan_digest,
            "readiness_digest": self.readiness_digest,
            "execution_provenance_digest": self.execution_provenance_digest,
            "replicates": self.replicates,
            "base_seed": self.base_seed,
            "synthetic_only": self.synthetic_only,
            "locked_evaluation_access_authorized": self.locked_evaluation_access_authorized,
            "ood_evaluation_access_authorized": self.ood_evaluation_access_authorized,
            "automatic_promotion": self.automatic_promotion,
            "operational_validity": self.operational_validity,
        }


def build_advisor_v3_execution_approval(
    *,
    approval_reference: str,
    runner: BenchmarkPortfolioRunner | None = None,
) -> AdvisorV3CorpusExecutionApproval:
    """Build approval bindings from one exact ready runner without exposing provenance values."""

    run_engine = runner or BenchmarkPortfolioRunner()
    readiness = build_advisor_v3_corpus_readiness(adapter_statuses=run_engine.adapter_statuses())
    if not readiness.execution_ready:
        raise ValueError("Advisor-v3 execution approval requires all required adapters.")
    preregistration = build_advisor_v3_preregistration()
    design = build_advisor_v3_corpus_design()
    plan = build_advisor_v3_shard_plan()
    return AdvisorV3CorpusExecutionApproval(
        approval_reference=approval_reference,
        human_approved=True,
        preregistration_digest=preregistration.preregistration_digest,
        design_digest=design.design_digest,
        catalogue_manifest_digest=design.catalogue_manifest_digest,
        shard_plan_digest=plan.plan_digest,
        readiness_digest=readiness.readiness_digest,
        execution_provenance_digest=advisor_v3_execution_provenance_digest(run_engine),
    )


class AdvisorV3GovernancePreparationReport(BaseModel):
    """Aggregate evidence that shared immutable governance was written serially."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    report_schema_version: Literal["3"] = "3"
    preregistration_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    design_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    shard_plan_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    readiness_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    execution_provenance_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    approval_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    shared_governance_prepared_serially: Literal[True] = True
    shard_lock_count: Literal[42] = 42
    held_out_metric_values_used_for_design_fit_or_threshold: Literal[False] = False
    qualification_evaluation_accessed: Literal[False] = False
    operational_validity: Literal["not_established"] = "not_established"


class AdvisorV3ShardExecutionReport(BaseModel):
    """Aggregate result of one lock-protected, idempotent whole-family shard."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    report_schema_version: Literal["3"] = "3"
    preregistration_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    shard_plan_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    approval_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    readiness_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    execution_provenance_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    shard_index: Annotated[StrictInt, Field(ge=0, le=41)]
    shard_family_count: Literal[2] = 2
    shard_instance_count: Literal[8] = 8
    expected_run_count: Literal[280] = 280
    completed_run_count: Annotated[StrictInt, Field(ge=0, le=280)]
    newly_persisted_run_count: Annotated[StrictInt, Field(ge=0, le=280)]
    resumed_run_count: Annotated[StrictInt, Field(ge=0, le=280)]
    successful_run_count: Annotated[StrictInt, Field(ge=0, le=280)]
    retained_non_success_run_count: Annotated[StrictInt, Field(ge=0, le=280)]
    shard_complete: StrictBool
    append_only: Literal[True] = True
    idempotent_resume: Literal[True] = True
    cross_process_shard_lock: Literal[True] = True
    contains_record_values: Literal[False] = False
    contains_identifiers: Literal[False] = False
    contains_candidate_pairs: Literal[False] = False
    held_out_metric_values_used_for_design_fit_or_threshold: Literal[False] = False
    qualification_evaluation_accessed: Literal[False] = False
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.completed_run_count != (
            self.successful_run_count + self.retained_non_success_run_count
        ):
            raise ValueError("Advisor-v3 shard status counts are inconsistent.")
        if self.completed_run_count != self.newly_persisted_run_count + self.resumed_run_count:
            raise ValueError("Advisor-v3 shard resume counts are inconsistent.")
        if self.shard_complete != (self.completed_run_count == self.expected_run_count):
            raise ValueError("Advisor-v3 shard completeness must fail closed on missing evidence.")
        return self

    @property
    def report_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


def load_committed_advisor_v3_preregistration(path: Path) -> AdvisorV3PreregistrationManifest:
    """Load and verify the source-controlled, outcome-free preregistration artifact."""

    try:
        if any(component.is_symlink() for component in (path, *path.parents)) or not path.is_file():
            raise FileNotFoundError
        if path.stat().st_size > _MAX_PREREGISTRATION_BYTES:
            raise ValueError("Advisor-v3 preregistration exceeds its aggregate size bound.")
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            "The committed advisor-v3 preregistration is unavailable."
        ) from None
    except OSError:
        raise ValueError(
            "The committed advisor-v3 preregistration could not be read safely."
        ) from None
    canonical = build_advisor_v3_preregistration()
    if text != _canonical_text(canonical):
        raise ValueError("The committed advisor-v3 preregistration is not canonical.")
    manifest = AdvisorV3PreregistrationManifest.model_validate_json(text)
    if manifest != canonical:
        raise ValueError("The committed advisor-v3 preregistration is stale or conflicting.")
    return manifest


def _reject_symlink_components(root: Path, path: Path) -> None:
    try:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Advisor-v3 registry roots must be regular directories.")
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError("Advisor-v3 governed paths cannot traverse symbolic links.")
    except OSError:
        raise ValueError("Advisor-v3 governed paths could not be inspected safely.") from None


def _persist_exact(path: Path, model: BaseModel) -> None:
    try:
        _reject_symlink_components(path.parents[1], path)
        text = _canonical_text(model)
        if path.exists():
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                raise FileExistsError("A conflicting advisor-v3 governance artifact exists.")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(path.parents[1], path)
        atomic_write_text(path, text)
    except (FileExistsError, ValueError):
        raise
    except OSError:
        raise ValueError("Advisor-v3 governance could not be persisted safely.") from None


def _governance_paths(
    registry: BenchmarkRegistry,
    *,
    preregistration: AdvisorV3PreregistrationManifest,
    plan: AdvisorV3ShardPlan,
    readiness: AdvisorV3CorpusReadinessManifest,
    approval: AdvisorV3CorpusExecutionApproval,
) -> tuple[tuple[Path, BaseModel], ...]:
    governance = registry.root_directory / "governance"
    design = build_advisor_v3_corpus_design()
    return (
        (governance / f"design.v3.{design.design_digest}.json", design),
        (
            governance / f"preregistration.v3.{preregistration.preregistration_digest}.json",
            preregistration,
        ),
        (governance / f"shards.v3.{plan.plan_digest}.json", plan),
        (governance / f"readiness.v3.{readiness.readiness_digest}.json", readiness),
        (governance / f"approval.v3.{approval.approval_digest}.json", approval),
    )


def prepare_advisor_v3_execution(
    *,
    registry: BenchmarkRegistry,
    committed_preregistration_path: Path,
    approval: AdvisorV3CorpusExecutionApproval,
    runner: BenchmarkPortfolioRunner | None = None,
) -> AdvisorV3GovernancePreparationReport:
    """Serially prepare exact shared governance before any concurrent worker starts."""

    preregistration = load_committed_advisor_v3_preregistration(committed_preregistration_path)
    plan = build_advisor_v3_shard_plan()
    run_engine = runner or BenchmarkPortfolioRunner()
    readiness = build_advisor_v3_corpus_readiness(adapter_statuses=run_engine.adapter_statuses())
    if not readiness.execution_ready:
        raise ValueError("Advisor-v3 execution is not ready for all required adapters.")
    provenance_digest = advisor_v3_execution_provenance_digest(run_engine)
    if (
        approval.readiness_digest != readiness.readiness_digest
        or approval.execution_provenance_digest != provenance_digest
    ):
        raise ValueError("Advisor-v3 approval does not bind current readiness and provenance.")
    for path, model in _governance_paths(
        registry,
        preregistration=preregistration,
        plan=plan,
        readiness=readiness,
        approval=approval,
    ):
        _persist_exact(path, model)
    lock_directory = registry.root_directory / "governance" / "locks.v3"
    try:
        _reject_symlink_components(registry.root_directory, lock_directory)
        lock_directory.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(registry.root_directory, lock_directory)
    except ValueError:
        raise
    except OSError:
        raise ValueError("Advisor-v3 lock governance could not be prepared safely.") from None
    return AdvisorV3GovernancePreparationReport(
        preregistration_digest=preregistration.preregistration_digest,
        design_digest=build_advisor_v3_corpus_design().design_digest,
        shard_plan_digest=plan.plan_digest,
        readiness_digest=readiness.readiness_digest,
        execution_provenance_digest=provenance_digest,
        approval_digest=approval.approval_digest,
    )


def _require_prepared_governance(
    *,
    registry: BenchmarkRegistry,
    approval: AdvisorV3CorpusExecutionApproval,
    runner: BenchmarkPortfolioRunner,
) -> tuple[AdvisorV3PreregistrationManifest, AdvisorV3ShardPlan]:
    preregistration = build_advisor_v3_preregistration()
    plan = build_advisor_v3_shard_plan()
    readiness = build_advisor_v3_corpus_readiness(adapter_statuses=runner.adapter_statuses())
    provenance_digest = advisor_v3_execution_provenance_digest(runner)
    if (
        approval.readiness_digest != readiness.readiness_digest
        or approval.execution_provenance_digest != provenance_digest
    ):
        raise ValueError("Advisor-v3 approval is stale for current readiness or provenance.")
    for path, model in _governance_paths(
        registry,
        preregistration=preregistration,
        plan=plan,
        readiness=readiness,
        approval=approval,
    ):
        try:
            _reject_symlink_components(registry.root_directory, path)
            if not path.is_file():
                raise ValueError("Advisor-v3 shared governance was not prepared serially.")
            if path.read_text(encoding="utf-8") != _canonical_text(model):
                raise FileExistsError(
                    "Advisor-v3 shared governance conflicts with the worker plan."
                )
        except (FileExistsError, ValueError):
            raise
        except OSError:
            raise ValueError("Advisor-v3 shared governance could not be read safely.") from None
    return preregistration, plan


@contextmanager
def _shard_lock(registry: BenchmarkRegistry, shard_index: int) -> Iterator[BinaryIO]:
    try:
        import fcntl as posix_file_lock
    except ImportError:
        raise ValueError("Advisor-v3 cross-process shard locking is unavailable.") from None
    lock_path = (
        registry.root_directory / "governance" / "locks.v3" / f"shard.{shard_index:03d}.lock"
    )
    try:
        _reject_symlink_components(registry.root_directory, lock_path)
        if not lock_path.parent.is_dir():
            raise ValueError("Advisor-v3 governance preparation did not create shard locks.")
        with lock_path.open("a+b") as handle:
            posix_file_lock.flock(handle.fileno(), posix_file_lock.LOCK_EX)
            try:
                yield handle
            finally:
                posix_file_lock.flock(handle.fileno(), posix_file_lock.LOCK_UN)
    except ValueError:
        raise
    except OSError:
        raise ValueError("Advisor-v3 shard locking failed safely.") from None


def _validate_existing_record(
    record: BenchmarkRunRecord,
    *,
    family_id: str,
    instance_id: str,
    recipe_digest: str,
    replicate_id: str,
    seed: int,
    provenance: dict[str, str],
) -> None:
    if (
        record.family_id != family_id
        or record.instance_id != instance_id
        or record.pipeline_recipe_digest != recipe_digest
        or record.replicate_id != replicate_id
        or record.random_seed != seed
        or record.engine_commit != provenance["engine_commit"]
        or record.dependency_lock_digest != provenance["dependency_lock_digest"]
        or record.environment_digest != provenance["environment_digest"]
    ):
        raise FileExistsError("Existing advisor-v3 evidence conflicts with approved provenance.")


def execute_advisor_v3_corpus_shard(
    *,
    registry: BenchmarkRegistry,
    shard_index: int,
    approval: AdvisorV3CorpusExecutionApproval,
    runner: BenchmarkPortfolioRunner | None = None,
) -> AdvisorV3ShardExecutionReport:
    """Execute or resume one whole-family shard under an exclusive cross-process lock."""

    if not 0 <= shard_index < 42:
        raise ValueError("Advisor-v3 shard index is outside the preregistered plan.")
    run_engine = runner or BenchmarkPortfolioRunner()
    preregistration, plan = _require_prepared_governance(
        registry=registry,
        approval=approval,
        runner=run_engine,
    )
    generator = build_advisor_v3_generator()
    shard = plan.shards[shard_index]
    recipes = run_engine.list_recipes()
    provenance = run_engine.provenance_summary()
    with _shard_lock(registry, shard_index):
        for family in shard.families:
            registry.save_family(generator.get_family(family.family_id))
            for instance_id in family.instance_ids:
                registry.save_instance(generator.get_instance(instance_id))
        newly_persisted = 0
        resumed = 0
        final_records: dict[str, BenchmarkRunRecord] = {}
        for instance_id in shard.instance_ids:
            family_id = generator.get_instance(instance_id).family_id
            for replicate_number in range(5):
                replicate_id = f"replicate.{replicate_number:07d}"
                seed = benchmark_replicate_seed(
                    instance_id=instance_id,
                    replicate_number=replicate_number,
                    base_seed=20260816,
                )
                existing: dict[str, BenchmarkRunRecord] = {}
                for recipe in recipes:
                    run_id = benchmark_run_id(
                        instance_id=instance_id,
                        recipe_id=recipe.recipe_id,
                        replicate_id=replicate_id,
                    )
                    try:
                        record = registry.load_run_record(run_id)
                    except FileNotFoundError:
                        continue
                    _validate_existing_record(
                        record,
                        family_id=family_id,
                        instance_id=instance_id,
                        recipe_digest=recipe.recipe_digest,
                        replicate_id=replicate_id,
                        seed=seed,
                        provenance=provenance,
                    )
                    existing[run_id] = record
                    final_records[run_id] = record
                if len(existing) == len(recipes):
                    resumed += len(existing)
                    continue
                results = run_engine.run_portfolio(
                    generator,
                    instances=(instance_id,),
                    recipes=recipes,
                    replicates=1,
                    base_seed=20260816,
                    replicate_start=replicate_number,
                )
                if len(results) != len(recipes):
                    raise ValueError("Advisor-v3 portfolio output does not cover every recipe.")
                for result in results:
                    if result.record.run_id in existing:
                        resumed += 1
                        continue
                    registry.save_run_record(
                        result.record,
                        metrics=result.metrics,
                        failure=result.failure,
                    )
                    newly_persisted += 1
                    final_records[result.record.run_id] = result.record
        if len(final_records) != 280:
            raise ValueError("Advisor-v3 shard execution did not retain its complete grid.")
        successful = sum(
            record.status == BenchmarkRunStatus.SUCCESS for record in final_records.values()
        )
        report = AdvisorV3ShardExecutionReport(
            preregistration_digest=preregistration.preregistration_digest,
            shard_plan_digest=plan.plan_digest,
            approval_digest=approval.approval_digest,
            readiness_digest=approval.readiness_digest,
            execution_provenance_digest=approval.execution_provenance_digest,
            shard_index=shard_index,
            completed_run_count=len(final_records),
            newly_persisted_run_count=newly_persisted,
            resumed_run_count=resumed,
            successful_run_count=successful,
            retained_non_success_run_count=len(final_records) - successful,
            shard_complete=True,
        )
        completion = report.model_copy(
            update={"newly_persisted_run_count": 0, "resumed_run_count": 280}
        )
        _persist_exact(
            registry.root_directory
            / "governance"
            / f"completion.v3.{plan.plan_digest}.{shard_index:03d}.json",
            completion,
        )
        return report


def audit_advisor_v3_corpus(
    *,
    registry: BenchmarkRegistry,
    approval: AdvisorV3CorpusExecutionApproval,
    runner: BenchmarkPortfolioRunner | None = None,
) -> AdvisorV3CorpusReadinessManifest:
    """Check digest integrity without using held-out values for design, fit, or qualification."""

    run_engine = runner or BenchmarkPortfolioRunner()
    _, plan = _require_prepared_governance(
        registry=registry,
        approval=approval,
        runner=run_engine,
    )
    generator = build_advisor_v3_generator()
    recipes = run_engine.list_recipes()
    provenance = run_engine.provenance_summary()
    expected: dict[str, tuple[str, str, str, str, int, str | None]] = {}
    for shard in plan.shards:
        for instance_id in shard.instance_ids:
            family_id = generator.get_instance(instance_id).family_id
            for replicate_number in range(5):
                replicate_id = f"replicate.{replicate_number:07d}"
                seed = benchmark_replicate_seed(
                    instance_id=instance_id,
                    replicate_number=replicate_number,
                    base_seed=20260816,
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
    if {record.run_id for record in records} - set(expected):
        raise ValueError("The advisor-v3 registry contains evidence outside its preregistration.")
    successful_tokens_by_family: dict[str, set[str]] = {}
    successful_tokens_by_cell: dict[tuple[str, str], set[str]] = {}
    successful_required = 0
    failed_required = 0
    for record in records:
        family_id, instance_id, recipe_digest, replicate_id, seed, token = expected[record.run_id]
        _validate_existing_record(
            record,
            family_id=family_id,
            instance_id=instance_id,
            recipe_digest=recipe_digest,
            replicate_id=replicate_id,
            seed=seed,
            provenance=provenance,
        )
        if token is None:
            continue
        if record.status == BenchmarkRunStatus.SUCCESS:
            successful_required += 1
            successful_tokens_by_family.setdefault(record.family_id, set()).add(token)
            successful_tokens_by_cell.setdefault(
                (record.instance_id, record.replicate_id), set()
            ).add(token)
        else:
            failed_required += 1
    required_tokens = {"fellegi_sunter", "xgboost_classifier", "xgboost_ranker"}
    expected_cells = {
        (instance_id, f"replicate.{replicate_number:07d}")
        for shard in plan.shards
        for instance_id in shard.instance_ids
        for replicate_number in range(5)
    }
    successful_cells = sum(
        required_tokens.issubset(successful_tokens_by_cell.get(cell, set()))
        for cell in expected_cells
    )
    successful_overlap = sum(
        required_tokens.issubset(tokens) for tokens in successful_tokens_by_family.values()
    )
    missing_required = 5040 - successful_required - failed_required
    completed = len(records)
    status: Literal["not_started", "partial", "complete"] = (
        "not_started" if completed == 0 else "complete" if completed == 11760 else "partial"
    )
    initial = build_advisor_v3_corpus_readiness(adapter_statuses=run_engine.adapter_statuses())
    readiness = AdvisorV3CorpusReadinessManifest.model_validate(
        {
            **initial.model_dump(mode="json"),
            "execution_status": status,
            "completed_run_count": completed,
            "successful_evidence_cell_count": successful_cells,
            "successful_required_adapter_run_count": successful_required,
            "failed_required_adapter_run_count": failed_required,
            "missing_required_adapter_run_count": missing_required,
            "successful_overlap_family_count": successful_overlap,
            "held_out_digest_integrity_checked": True,
            "advisor_evidence_ready": (
                status == "complete"
                and successful_cells == 1680
                and successful_required == 5040
                and failed_required == 0
                and missing_required == 0
                and successful_overlap == 84
            ),
        }
    )
    _persist_exact(
        registry.root_directory
        / "governance"
        / f"readiness.audit.v3.{readiness.readiness_digest}.json",
        readiness,
    )
    return readiness


__all__ = [
    "AdvisorV3CorpusExecutionApproval",
    "AdvisorV3GovernancePreparationReport",
    "AdvisorV3ShardExecutionReport",
    "advisor_v3_execution_provenance_digest",
    "audit_advisor_v3_corpus",
    "build_advisor_v3_execution_approval",
    "execute_advisor_v3_corpus_shard",
    "load_committed_advisor_v3_preregistration",
    "prepare_advisor_v3_execution",
]
