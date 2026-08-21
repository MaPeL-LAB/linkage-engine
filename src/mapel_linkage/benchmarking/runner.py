"""Portfolio execution engine for the Linkage Engine synthetic benchmark library."""

from __future__ import annotations

import difflib
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

from mapel_linkage.benchmarking.contracts import (
    BenchmarkAggregateMetrics,
    BenchmarkFailureRecord,
    BenchmarkRunRecord,
    BenchmarkRunStatus,
)
from mapel_linkage.benchmarking.generator import (
    BenchmarkScenarioBundle,
    BenchmarkScenarioGenerator,
)
from mapel_linkage.calibration.metrics import fit_logistic_line
from mapel_linkage.comparisons import (
    ComparisonFeatureResult,
    DuckDBComparisonFeatureBuilder,
)
from mapel_linkage.configuration.models import (
    BoostedTreeModelConfig,
    ComparisonConfig,
    FellegiSunterModelConfig,
    RankingModelConfig,
)
from mapel_linkage.domain.errors import LinkageRuntimeError
from mapel_linkage.domain.sql_identifiers import quote_identifier
from mapel_linkage.governance.labels import assert_disjoint_label_partitions
from mapel_linkage.io import ColumnSpec, DuckDBStore
from mapel_linkage.models import (
    DuckDBFellegiSunterMatcher,
    DuckDBVerifiedMatrixBuilder,
    XGBoostCandidateRanker,
    XGBoostPairClassifier,
    build_ranking_matrix,
    build_ranking_scoring_matrix,
)
from mapel_linkage.models.boosted.training import (
    BoostedFeatureMatrix,
    BoostedLabelledMatrix,
)
from mapel_linkage.preprocessing import PreparedDataset
from mapel_linkage.synthetic.generator import SyntheticRecord
from mapel_linkage.validation import EntityHouseholdRecord, split_entity_household_components
from mapel_linkage.validation.splitting import build_verified_candidate_label_batches


def _digest_object(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_instance_seed(instance_id: str) -> int:
    """Derive a process-independent unsigned 32-bit seed component."""
    digest = hashlib.sha256(instance_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def benchmark_run_id(*, instance_id: str, recipe_id: str, replicate_id: str) -> str:
    """Return the stable aggregate run identifier used by planning and execution."""

    inst_slug = instance_id.replace("instance.", "")
    recipe_slug = recipe_id.replace("recipe.", "")
    return f"run.{inst_slug}.{recipe_slug}.{replicate_id}".replace("_", "-").replace(".", "-")


def benchmark_replicate_seed(*, instance_id: str, replicate_number: int, base_seed: int) -> int:
    """Return the stable unsigned seed for one instance replicate."""

    if not 0 <= replicate_number <= 9_999_999 or not 0 <= base_seed <= 4_294_967_295:
        raise ValueError("Benchmark replicate seed inputs are outside their safe bounds.")
    return (
        base_seed + replicate_number * 1000 + _stable_instance_seed(instance_id)
    ) % 4_294_967_296


class BenchmarkRecipe(BaseModel):
    """Declarative specification for a candidate linkage recipe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recipe_id: StrictStr
    model_family: Literal[
        "fellegi_sunter", "xgboost", "lightgbm", "pytorch", "dedupe", "multi_source"
    ]
    linkage_mode: Literal["link_only", "dedupe_only", "multi_source"] = "link_only"
    requires_verified_labels: StrictBool = False
    required_runtime: Literal["core", "lightgbm", "pytorch"] = "core"
    ranking_enabled: StrictBool = False
    assignment_constraint: Literal["one_to_one", "unconstrained"] = "one_to_one"
    timeout_seconds: Annotated[int, Field(ge=1)] = 60

    @property
    def recipe_digest(self) -> str:
        return _digest_object(
            {
                "recipe_id": self.recipe_id,
                "model_family": self.model_family,
                "linkage_mode": self.linkage_mode,
                "requires_verified_labels": self.requires_verified_labels,
                "required_runtime": self.required_runtime,
                "ranking_enabled": self.ranking_enabled,
                "assignment_constraint": self.assignment_constraint,
            }
        )


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Complete result bundle for a single recipe execution on an instance."""

    record: BenchmarkRunRecord
    metrics: BenchmarkAggregateMetrics | None = None
    failure: BenchmarkFailureRecord | None = None


def _get_engine_commit() -> str:
    """Bind evidence to exact installed package source bytes, including dirty worktrees."""

    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    try:
        sources = sorted(package_root.rglob("*.py"))
        if not sources:
            raise OSError
        for source in sources:
            if source.is_symlink() or not source.is_file():
                raise OSError
            relative = source.relative_to(package_root).as_posix().encode("utf-8")
            content = source.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    except OSError:
        raise ValueError(
            "The benchmark engine source provenance could not be bound safely."
        ) from None
    return digest.hexdigest()


def _get_dependency_lock_digest() -> str:
    # Hash installed package versions for repeatability
    deps = ["pydantic", "numpy", "duckdb", "xgboost", "scipy"]
    payload = {}
    for dep in deps:
        try:
            payload[dep] = importlib.metadata.version(dep)
        except importlib.metadata.PackageNotFoundError:
            payload[dep] = "missing"
    return _digest_object(payload)


def _get_environment_digest() -> str:
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    return _digest_object(env_info)


def _levenshtein_ratio(s1: str, s2: str) -> float:
    return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


@dataclass(frozen=True, slots=True)
class _BenchmarkExecutionEvidence:
    metrics: BenchmarkAggregateMetrics
    stage_artifact_manifest_digest: str


class BenchmarkPortfolioRunner:
    """Executes a portfolio of linkage recipes over generated benchmark scenarios."""

    _SUCCESS_CAPABLE_RECIPE_IDS = frozenset(
        {
            "recipe.fellegi_sunter_reference",
            "recipe.xgboost_classifier",
            "recipe.xgboost_ranker",
        }
    )

    def __init__(self, recipes: Iterable[BenchmarkRecipe] | None = None) -> None:
        self._recipes = tuple(recipes) if recipes is not None else self._standard_recipes()
        self._engine_commit = _get_engine_commit()
        self._dependency_lock_digest = _get_dependency_lock_digest()
        self._environment_digest = _get_environment_digest()

    @staticmethod
    def _standard_recipes() -> tuple[BenchmarkRecipe, ...]:
        return (
            # 1. Baseline Fellegi-Sunter reference
            BenchmarkRecipe(
                recipe_id="recipe.fellegi_sunter_reference",
                model_family="fellegi_sunter",
                linkage_mode="link_only",
                requires_verified_labels=False,
                required_runtime="core",
                ranking_enabled=False,
                assignment_constraint="one_to_one",
            ),
            # 2. Supervised XGBoost pair classifier
            BenchmarkRecipe(
                recipe_id="recipe.xgboost_classifier",
                model_family="xgboost",
                linkage_mode="link_only",
                requires_verified_labels=True,
                required_runtime="core",
                ranking_enabled=False,
                assignment_constraint="one_to_one",
            ),
            # 3. Supervised XGBoost ranker
            BenchmarkRecipe(
                recipe_id="recipe.xgboost_ranker",
                model_family="xgboost",
                linkage_mode="link_only",
                requires_verified_labels=True,
                required_runtime="core",
                ranking_enabled=True,
                assignment_constraint="one_to_one",
            ),
            # 4. LightGBM challenger
            BenchmarkRecipe(
                recipe_id="recipe.lightgbm_classifier",
                model_family="lightgbm",
                linkage_mode="link_only",
                requires_verified_labels=True,
                required_runtime="lightgbm",
                ranking_enabled=False,
                assignment_constraint="one_to_one",
            ),
            # 5. PyTorch neural pair matcher challenger
            BenchmarkRecipe(
                recipe_id="recipe.pytorch_matcher",
                model_family="pytorch",
                linkage_mode="link_only",
                requires_verified_labels=True,
                required_runtime="pytorch",
                ranking_enabled=False,
                assignment_constraint="one_to_one",
            ),
            # 6. Dedupe-only recipe
            BenchmarkRecipe(
                recipe_id="recipe.single_source_dedupe",
                model_family="dedupe",
                linkage_mode="dedupe_only",
                requires_verified_labels=False,
                required_runtime="core",
                ranking_enabled=False,
                assignment_constraint="unconstrained",
            ),
            # 7. Multi-source resolver recipe
            BenchmarkRecipe(
                recipe_id="recipe.multi_source_resolver",
                model_family="multi_source",
                linkage_mode="multi_source",
                requires_verified_labels=False,
                required_runtime="core",
                ranking_enabled=False,
                assignment_constraint="unconstrained",
            ),
        )

    def list_recipes(self) -> tuple[BenchmarkRecipe, ...]:
        return self._recipes

    def adapter_statuses(self) -> dict[str, Literal["success_capable", "ineligible"]]:
        """Return package-owned readiness without executing or inspecting record values."""

        xgboost_available = importlib.util.find_spec("xgboost") is not None
        return {
            recipe.recipe_id: (
                "success_capable"
                if recipe.recipe_id in self._SUCCESS_CAPABLE_RECIPE_IDS
                and (recipe.recipe_id == "recipe.fellegi_sunter_reference" or xgboost_available)
                else "ineligible"
            )
            for recipe in self._recipes
        }

    def provenance_summary(self) -> dict[str, str]:
        """Return the aggregate environment binding expected on retained run evidence."""

        return {
            "engine_commit": self._engine_commit,
            "dependency_lock_digest": self._dependency_lock_digest,
            "environment_digest": self._environment_digest,
        }

    def run_single(
        self,
        *,
        bundle: BenchmarkScenarioBundle,
        recipe: BenchmarkRecipe,
        replicate_id: str = "replicate.001",
        seed: int = 20260816,
    ) -> BenchmarkRunResult:
        """Execute one recipe on one scenario bundle realization."""
        run_id = benchmark_run_id(
            instance_id=bundle.instance_id,
            recipe_id=recipe.recipe_id,
            replicate_id=replicate_id,
        )

        # 1. Eligibility Checks
        if recipe.linkage_mode != bundle.task_profile.linkage_mode:
            return self._ineligible_result(
                run_id=run_id,
                bundle=bundle,
                recipe=recipe,
                replicate_id=replicate_id,
                seed=seed,
                failure_code="ML-BENCH-INELIGIBLE-MODE",
                error_message=(
                    f"Recipe mode {recipe.linkage_mode} incompatible "
                    f"with task mode {bundle.task_profile.linkage_mode}"
                ),
            )

        if recipe.requires_verified_labels and not bundle.task_profile.verified_labels_available:
            return self._ineligible_result(
                run_id=run_id,
                bundle=bundle,
                recipe=recipe,
                replicate_id=replicate_id,
                seed=seed,
                failure_code="ML-BENCH-INELIGIBLE-LABELS",
                error_message="Recipe requires verified labels unavailable in this scenario",
            )

        if recipe.required_runtime == "lightgbm" and importlib.util.find_spec("lightgbm") is None:
            return self._ineligible_result(
                run_id=run_id,
                bundle=bundle,
                recipe=recipe,
                replicate_id=replicate_id,
                seed=seed,
                failure_code="ML-BENCH-INELIGIBLE-RUNTIME",
                error_message="Optional runtime 'lightgbm' is not installed",
            )

        if recipe.required_runtime == "pytorch" and importlib.util.find_spec("torch") is None:
            return self._ineligible_result(
                run_id=run_id,
                bundle=bundle,
                recipe=recipe,
                replicate_id=replicate_id,
                seed=seed,
                failure_code="ML-BENCH-INELIGIBLE-RUNTIME",
                error_message="Optional runtime 'torch' is not installed",
            )

        if recipe.recipe_id not in self._SUCCESS_CAPABLE_RECIPE_IDS:
            return self._ineligible_result(
                run_id=run_id,
                bundle=bundle,
                recipe=recipe,
                replicate_id=replicate_id,
                seed=seed,
                failure_code="ML-BENCH-INELIGIBLE-ADAPTER",
                error_message=(
                    "No truth-safe package benchmark adapter is registered for this recipe."
                ),
            )

        # 2. Execution with profiling and error isolation
        tracemalloc.start()
        start_time = time.perf_counter()
        try:
            evidence = self._execute_linkage(bundle, recipe, seed)
            metrics = evidence.metrics
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            _current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = max(1, int(peak_mem / (1024 * 1024)))

            # Update runtime and memory in metrics
            metrics = BenchmarkAggregateMetrics(
                candidate_recall=metrics.candidate_recall,
                candidate_recall_at_k=metrics.candidate_recall_at_k,
                sensitivity=metrics.sensitivity,
                positive_predictive_value=metrics.positive_predictive_value,
                brier_score=metrics.brier_score,
                calibration_intercept=metrics.calibration_intercept,
                calibration_slope=metrics.calibration_slope,
                mean_reciprocal_rank=metrics.mean_reciprocal_rank,
                runtime_ms=elapsed_ms,
                peak_memory_mb=peak_mb,
            )

            record = BenchmarkRunRecord(
                run_id=run_id,
                family_id=bundle.family_id,
                instance_id=bundle.instance_id,
                replicate_id=replicate_id,
                task_profile_digest=bundle.task_profile.profile_digest,
                pipeline_recipe_digest=recipe.recipe_digest,
                engine_commit=self._engine_commit,
                dependency_lock_digest=self._dependency_lock_digest,
                environment_digest=self._environment_digest,
                random_seed=seed,
                status=BenchmarkRunStatus.SUCCESS,
                failure_code=None,
                aggregate_metrics_digest=metrics.metrics_digest,
                stage_artifact_manifest_digest=evidence.stage_artifact_manifest_digest,
                runtime_ms=elapsed_ms,
                peak_memory_mb=peak_mb,
            )
            return BenchmarkRunResult(record=record, metrics=metrics)

        except TimeoutError:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            tracemalloc.stop()
            return self._failure_result(
                run_id=run_id,
                bundle=bundle,
                recipe=recipe,
                replicate_id=replicate_id,
                seed=seed,
                status=BenchmarkRunStatus.TIMEOUT,
                failure_code="ML-BENCH-TIMEOUT",
                error_message="Execution exceeded configured timeout limit",
                runtime_ms=elapsed_ms,
            )
        except (FloatingPointError, np.linalg.LinAlgError):
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            tracemalloc.stop()
            return self._failure_result(
                run_id=run_id,
                bundle=bundle,
                recipe=recipe,
                replicate_id=replicate_id,
                seed=seed,
                status=BenchmarkRunStatus.NUMERICAL_FAILURE,
                failure_code="ML-BENCH-NUMERICAL-ERROR",
                error_message="Numerical failure during aggregate benchmark execution.",
                runtime_ms=elapsed_ms,
            )
        except Exception as err:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            tracemalloc.stop()
            error_str = (
                err.public_message
                if isinstance(err, LinkageRuntimeError)
                else "The benchmark adapter failed without exposing private execution values."
            )
            if "budget" in error_str.lower():
                status = BenchmarkRunStatus.CANDIDATE_BUDGET_FAILURE
                code = "ML-BENCH-CANDIDATE-BUDGET"
            elif "abstain" in error_str.lower() or "no candidates" in error_str.lower():
                status = BenchmarkRunStatus.ABSTAINED
                code = "ML-BENCH-ABSTAINED"
            else:
                status = BenchmarkRunStatus.FAILED_FIT
                code = "ML-BENCH-FAILED-FIT"
            return self._failure_result(
                run_id=run_id,
                bundle=bundle,
                recipe=recipe,
                replicate_id=replicate_id,
                seed=seed,
                status=status,
                failure_code=code,
                error_message=error_str,
                runtime_ms=elapsed_ms,
            )

    def _execute_linkage(
        self,
        bundle: BenchmarkScenarioBundle,
        recipe: BenchmarkRecipe,
        seed: int,
    ) -> _BenchmarkExecutionEvidence:
        """Run one truth-safe package adapter and mechanically evaluate locked evidence."""

        if recipe.linkage_mode != "link_only":
            raise ValueError("No truth-safe benchmark adapter is available for this mode.")
        source = bundle.datasets["source_a"]
        target = bundle.datasets["source_b"]
        candidate_pairs, rules_by_pair = self._candidate_pairs(source, target)
        true_pairs = self._cross_source_true_pairs(bundle)
        candidate_recall = (
            len(set(candidate_pairs) & true_pairs) / len(true_pairs) if true_pairs else 1.0
        )

        with DuckDBStore() as store:
            left = self._prepared_dataset(store, "source_a", source)
            right = self._prepared_dataset(store, "source_b", target)
            candidate_table = store.create_table_from_rows(
                "benchmark_candidates",
                (
                    ColumnSpec("left_record_key", "VARCHAR"),
                    ColumnSpec("right_record_key", "VARCHAR"),
                    ColumnSpec("retrieval_rule_ids", "VARCHAR"),
                    ColumnSpec("retrieval_rule_count", "INTEGER"),
                ),
                (
                    (
                        left_key,
                        right_key,
                        ",".join(rules_by_pair[(left_key, right_key)]),
                        len(rules_by_pair[(left_key, right_key)]),
                    )
                    for left_key, right_key in candidate_pairs
                ),
            )
            comparisons = self._comparisons()
            features = DuckDBComparisonFeatureBuilder(store).build(
                candidates=candidate_table,
                left=left,
                right=right,
                comparisons=comparisons,
            )
            matrices, partition_manifest_digest = self._protected_matrices(
                store=store,
                bundle=bundle,
                candidate_pairs=candidate_pairs,
                features=features,
                seed=seed,
            )
            training = matrices["training"]
            locked_test = matrices["test"]
            configuration_digest = _digest_object(
                {
                    "adapter": recipe.recipe_id,
                    "feature_schema_digest": features.table.schema_digest,
                    "seed": seed,
                }
            )

            if recipe.recipe_id == "recipe.fellegi_sunter_reference":
                training_features = self._subset_features(
                    store, features, training.pair_references, "benchmark_fs_training"
                )
                test_features = self._subset_features(
                    store, features, locked_test.pair_references, "benchmark_fs_test"
                )
                fs_config = FellegiSunterModelConfig(
                    implementation="splink_duckdb",
                    model_id="benchmark_fs_reference",
                    probability_two_random_records_match=0.01,
                    u_max_pairs=1_000_000,
                    em_max_iterations=25,
                    em_convergence=0.0001,
                    probability_smoothing=0.5,
                )
                matcher = DuckDBFellegiSunterMatcher(store)
                fs_artifact = matcher.fit(
                    u_features=training_features,
                    em_features=training_features,
                    comparisons=comparisons,
                    model=fs_config,
                    random_seed=seed,
                )
                scored = matcher.score(features=test_features, model=fs_artifact)
                rows = store._fetch_model_rows(
                    "SELECT left_record_key, right_record_key, __ml_fs_model_probability "
                    f"FROM {quote_identifier(scored.table.table_name)} "
                    "ORDER BY left_record_key, right_record_key"
                )
                score_by_pair: dict[tuple[str, str], float] = {}
                for row in rows:
                    value = row[2]
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError("Fellegi-Sunter returned invalid aggregate evidence.")
                    score_by_pair[(str(row[0]), str(row[1]))] = float(value)
                scores = np.asarray(
                    [score_by_pair[pair] for pair in locked_test.pair_references],
                    dtype=np.float64,
                )
                artifact_digest = fs_artifact.parameter_digest
            elif recipe.recipe_id == "recipe.xgboost_classifier":
                classifier_config = BoostedTreeModelConfig(
                    implementation="xgboost_classifier",
                    model_id="benchmark_xgboost_classifier",
                    n_estimators=30,
                    max_depth=4,
                    learning_rate=0.08,
                    maximum_training_pairs=1_000_000,
                )
                classifier = XGBoostPairClassifier(store)
                classifier_artifact = classifier.fit(
                    matrix=training,
                    model=classifier_config,
                    random_seed=seed,
                    configuration_digest=configuration_digest,
                )
                test_scoring = self._as_scoring_matrix(locked_test)
                scores = classifier._predict(
                    matrix=test_scoring,
                    model=classifier_artifact,
                )
                artifact_digest = classifier_artifact.model_digest
            elif recipe.recipe_id == "recipe.xgboost_ranker":
                ranking_training = build_ranking_matrix(training, query_side="source")
                ranking_test = build_ranking_scoring_matrix(
                    self._as_scoring_matrix(locked_test),
                    query_side="source",
                )
                ranker_config = RankingModelConfig(
                    implementation="xgboost_ranker",
                    model_id="benchmark_xgboost_ranker",
                    query_side="source",
                    top_k=10,
                    n_estimators=30,
                    max_depth=4,
                    learning_rate=0.08,
                    maximum_training_pairs=1_000_000,
                )
                ranker_artifact = XGBoostCandidateRanker.fit(
                    matrix=ranking_training,
                    model=ranker_config,
                    random_seed=seed,
                    configuration_digest=configuration_digest,
                )
                ranked = XGBoostCandidateRanker.score(
                    matrix=ranking_test,
                    model=ranker_artifact,
                )
                raw_score_by_pair = {
                    pair: float(score)
                    for pair, score in zip(
                        ranked.pair_references,
                        ranked.scores,
                        strict=True,
                    )
                }
                raw = np.asarray(
                    [raw_score_by_pair[pair] for pair in locked_test.pair_references],
                    dtype=np.float64,
                )
                scores = 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
                artifact_digest = ranker_artifact.artifact_digest
            else:
                raise ValueError("No truth-safe benchmark adapter is registered.")

        metrics = self._mechanical_metrics(
            pair_references=locked_test.pair_references,
            labels=locked_test.labels,
            scores=scores,
            candidate_recall=float(candidate_recall),
        )
        return _BenchmarkExecutionEvidence(
            metrics=metrics,
            stage_artifact_manifest_digest=_digest_object(
                {
                    "adapter_id": recipe.recipe_id,
                    "artifact_digest": artifact_digest,
                    "feature_schema_digest": locked_test.feature_schema_digest,
                    "partition_manifest_digest": partition_manifest_digest,
                    "truth_use": "protected_training_labels_and_post_score_mechanical_evaluation",
                    "score_authority": "evidence_only",
                    "decision_authority": "none",
                    "assignment_authority": "none",
                    "merge_authority": "none",
                    "operational_validity": "not_established",
                }
            ),
        )

    @staticmethod
    def _prepared_dataset(
        store: DuckDBStore,
        dataset_id: str,
        records: tuple[SyntheticRecord, ...],
    ) -> PreparedDataset:
        rows = []
        for item in records:
            label = item.label_value
            date_value = item.date_value
            group = item.group_value
            rows.append(
                (
                    item.record_key,
                    label,
                    label is None,
                    date.fromisoformat(date_value) if date_value is not None else None,
                    date_value is None,
                    group,
                    group is None,
                )
            )
        table = store.create_table_from_rows(
            f"benchmark_{dataset_id}",
            (
                ColumnSpec("__ml_record_key", "VARCHAR"),
                ColumnSpec("v_label", "VARCHAR"),
                ColumnSpec("m_label", "BOOLEAN"),
                ColumnSpec("v_date", "DATE"),
                ColumnSpec("m_date", "BOOLEAN"),
                ColumnSpec("v_group", "VARCHAR"),
                ColumnSpec("m_group", "BOOLEAN"),
            ),
            rows,
        )
        return PreparedDataset(
            dataset_id,
            table,
            {"label": "v_label", "date": "v_date", "group": "v_group"},
            {"label": "m_label", "date": "m_date", "group": "m_group"},
        )

    @staticmethod
    def _comparisons() -> tuple[ComparisonConfig, ...]:
        return tuple(
            ComparisonConfig.model_validate(payload)
            for payload in (
                {
                    "id": "label_similarity",
                    "variable": "label",
                    "function": {"kind": "jaro_winkler"},
                    "levels": (
                        {"kind": "missing"},
                        {"kind": "exact"},
                        {"kind": "threshold", "minimum": 0.85},
                        {"kind": "else"},
                    ),
                },
                {
                    "id": "date_distance",
                    "variable": "date",
                    "function": {"kind": "date_difference", "unit": "day"},
                    "levels": (
                        {"kind": "missing"},
                        {"kind": "exact"},
                        {"kind": "maximum_difference", "value": 3.0},
                        {"kind": "else"},
                    ),
                },
                {
                    "id": "group_exact",
                    "variable": "group",
                    "function": {"kind": "categorical"},
                    "levels": (
                        {"kind": "missing"},
                        {"kind": "exact"},
                        {"kind": "else"},
                    ),
                },
            )
        )

    @staticmethod
    def _candidate_pairs(
        source: tuple[SyntheticRecord, ...],
        target: tuple[SyntheticRecord, ...],
    ) -> tuple[tuple[tuple[str, str], ...], dict[tuple[str, str], tuple[str, ...]]]:
        by_group_left: dict[str, list[str]] = defaultdict(list)
        by_group_right: dict[str, list[str]] = defaultdict(list)
        by_prefix_left: dict[str, list[str]] = defaultdict(list)
        by_prefix_right: dict[str, list[str]] = defaultdict(list)
        for item in source:
            key = item.record_key
            group = item.group_value
            label = item.label_value
            if group is not None:
                by_group_left[str(group)].append(key)
            if label is not None and len(str(label)) >= 3:
                by_prefix_left[str(label)[:3]].append(key)
        for item in target:
            key = item.record_key
            group = item.group_value
            label = item.label_value
            if group is not None:
                by_group_right[str(group)].append(key)
            if label is not None and len(str(label)) >= 3:
                by_prefix_right[str(label)[:3]].append(key)
        rules: dict[tuple[str, str], set[str]] = defaultdict(set)
        for group, left_keys in by_group_left.items():
            for left_key in left_keys:
                for right_key in by_group_right.get(group, ()):
                    rules[(left_key, right_key)].add("group_exact")
        for prefix, left_keys in by_prefix_left.items():
            for left_key in left_keys:
                for right_key in by_prefix_right.get(prefix, ()):
                    rules[(left_key, right_key)].add("label_prefix")
        if not rules:
            raise ValueError("No candidates retrieved during blocking.")
        pairs = tuple(sorted(rules))
        return pairs, {pair: tuple(sorted(rule_ids)) for pair, rule_ids in rules.items()}

    @staticmethod
    def _cross_source_true_pairs(bundle: BenchmarkScenarioBundle) -> set[tuple[str, str]]:
        by_entity: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for item in bundle.truth:
            by_entity[item.entity_key][item.dataset_id].append(item.record_key)
        return {
            (left_key, right_key)
            for sources in by_entity.values()
            for left_key in sources.get("source_a", ())
            for right_key in sources.get("source_b", ())
        }

    @staticmethod
    def _protected_matrices(
        *,
        store: DuckDBStore,
        bundle: BenchmarkScenarioBundle,
        candidate_pairs: tuple[tuple[str, str], ...],
        features: ComparisonFeatureResult,
        seed: int,
    ) -> tuple[dict[str, BoostedLabelledMatrix], str]:
        truth_records = tuple(
            EntityHouseholdRecord(
                dataset_id=item.dataset_id,
                record_key=item.record_key,
                entity_key=item.entity_key,
                household_key=item.household_key,
            )
            for item in bundle.truth
        )
        assignment = split_entity_household_components(
            truth_records,
            fractions=(0.50, 0.15, 0.10, 0.10, 0.15),
            random_seed=seed,
        )
        truth_digest = _digest_object(
            [
                {
                    "dataset_id": item.dataset_id,
                    "record_digest": hashlib.sha256(item.record_key.encode()).hexdigest(),
                    "entity_digest": item.entity_digest,
                    "household_digest": item.household_digest,
                }
                for item in truth_records
            ]
        )
        batches = build_verified_candidate_label_batches(
            candidate_pairs=candidate_pairs,
            truth_records=truth_records,
            assignment=assignment,
            verification_protocol="synthetic_benchmark_v2",
            source_digest=truth_digest,
        )
        by_partition = {batch.partition: batch for batch in batches}
        if set(by_partition) != {"training", "validation", "calibration", "decision", "test"}:
            raise ValueError("Protected benchmark partitions are incomplete.")
        training_batch = by_partition["training"]
        test_batch = by_partition["test"]
        if (
            training_batch.positive_count <= 0
            or training_batch.negative_count <= 0
            or test_batch.positive_count <= 0
            or test_batch.negative_count <= 0
        ):
            raise ValueError("Protected benchmark training and test evidence require both classes.")
        disjointness = assert_disjoint_label_partitions(batches)
        builder = DuckDBVerifiedMatrixBuilder(store)
        return (
            {
                name: builder.build_labelled(
                    features=features,
                    labels=batch,
                    random_seed=seed,
                )
                for name, batch in by_partition.items()
            },
            disjointness.manifest_digest,
        )

    @staticmethod
    def _subset_features(
        store: DuckDBStore,
        features: ComparisonFeatureResult,
        pairs: tuple[tuple[str, str], ...],
        table_name: str,
    ) -> ComparisonFeatureResult:
        pair_table = store.create_table_from_rows(
            f"{table_name}_pairs",
            (
                ColumnSpec("left_record_key", "VARCHAR"),
                ColumnSpec("right_record_key", "VARCHAR"),
            ),
            pairs,
        )
        output = store._create_temp_table_as(
            table_name,
            f"SELECT f.* FROM {quote_identifier(features.table.table_name)} AS f "
            f"INNER JOIN {quote_identifier(pair_table.table_name)} AS p "
            "ON f.left_record_key = p.left_record_key "
            "AND f.right_record_key = p.right_record_key "
            "ORDER BY f.left_record_key, f.right_record_key",
        )
        if output.row_count != len(pairs):
            raise ValueError("Protected feature subsetting lost candidate coverage.")
        return ComparisonFeatureResult(
            table=output,
            candidate_pair_count=output.row_count,
            configured_comparison_count=features.configured_comparison_count,
            columns=features.columns,
        )

    @staticmethod
    def _as_scoring_matrix(matrix: BoostedLabelledMatrix) -> BoostedFeatureMatrix:
        """Strip all label fields before a fitted model receives evaluation features."""

        return BoostedFeatureMatrix(
            features=matrix.features,
            pair_references=matrix.pair_references,
            pair_digests=matrix.pair_digests,
            feature_names=matrix.feature_names,
            feature_schema_digest=matrix.feature_schema_digest,
        )

    @staticmethod
    def _mechanical_metrics(
        *,
        pair_references: tuple[tuple[str, str], ...],
        labels: np.ndarray,
        scores: np.ndarray,
        candidate_recall: float,
    ) -> BenchmarkAggregateMetrics:
        y_true = np.asarray(labels, dtype=np.int8)
        y_scores = np.asarray(scores, dtype=np.float64)
        if (
            len(y_true) == 0
            or len(y_true) != len(pair_references)
            or y_scores.shape != y_true.shape
            or set(int(value) for value in y_true) != {0, 1}
            or not np.all(np.isfinite(y_scores))
            or np.any(y_scores < 0.0)
            or np.any(y_scores > 1.0)
        ):
            raise ValueError("Locked mechanical evaluation evidence is not estimable.")
        predicted = y_scores >= 0.5
        true_positive = int(np.sum(predicted & (y_true == 1)))
        false_positive = int(np.sum(predicted & (y_true == 0)))
        positives = int(np.sum(y_true == 1))
        sensitivity = true_positive / positives
        ppv = true_positive / (true_positive + false_positive) if predicted.any() else 0.0
        brier = float(np.mean((y_scores - y_true) ** 2))
        intercept, slope = fit_logistic_line(y_scores, y_true)
        if not math.isfinite(intercept) or not math.isfinite(slope):
            raise ValueError("Locked calibration diagnostics are not estimable.")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, (left_key, _right_key) in enumerate(pair_references):
            grouped[left_key].append(index)
        reciprocal_ranks: list[float] = []
        hits = {1: 0, 5: 0, 10: 0}
        for indices in grouped.values():
            ordered = sorted(
                indices,
                key=lambda index: (
                    -float(y_scores[index]),
                    hashlib.sha256(
                        f"{pair_references[index][0]}\x00{pair_references[index][1]}".encode()
                    ).hexdigest(),
                ),
            )
            match_ranks = [
                rank for rank, index in enumerate(ordered, start=1) if y_true[index] == 1
            ]
            if not match_ranks:
                continue
            best = min(match_ranks)
            reciprocal_ranks.append(1.0 / best)
            for k in hits:
                hits[k] += int(best <= k)
        if not reciprocal_ranks:
            raise ValueError("Locked ranking diagnostics are not estimable.")
        query_count = len(reciprocal_ranks)
        return BenchmarkAggregateMetrics(
            candidate_recall=candidate_recall,
            candidate_recall_at_k={str(k): hits[k] / query_count for k in sorted(hits)},
            sensitivity=sensitivity,
            positive_predictive_value=ppv,
            brier_score=brier,
            calibration_intercept=float(intercept),
            calibration_slope=float(slope),
            mean_reciprocal_rank=float(np.mean(reciprocal_ranks)),
            runtime_ms=0,
            peak_memory_mb=0,
        )

    def _ineligible_result(
        self,
        *,
        run_id: str,
        bundle: BenchmarkScenarioBundle,
        recipe: BenchmarkRecipe,
        replicate_id: str,
        seed: int,
        failure_code: str,
        error_message: str,
    ) -> BenchmarkRunResult:
        record = BenchmarkRunRecord(
            run_id=run_id,
            family_id=bundle.family_id,
            instance_id=bundle.instance_id,
            replicate_id=replicate_id,
            task_profile_digest=bundle.task_profile.profile_digest,
            pipeline_recipe_digest=recipe.recipe_digest,
            engine_commit=self._engine_commit,
            dependency_lock_digest=self._dependency_lock_digest,
            environment_digest=self._environment_digest,
            random_seed=seed,
            status=BenchmarkRunStatus.INELIGIBLE,
            failure_code=failure_code,
            aggregate_metrics_digest=None,
            stage_artifact_manifest_digest=None,
            runtime_ms=0,
            peak_memory_mb=0,
        )
        failure = BenchmarkFailureRecord(
            run_id=run_id,
            family_id=bundle.family_id,
            instance_id=bundle.instance_id,
            replicate_id=replicate_id,
            recipe_id=recipe.recipe_id,
            status=BenchmarkRunStatus.INELIGIBLE,
            failure_code=failure_code,
            error_message=error_message,
        )
        return BenchmarkRunResult(record=record, failure=failure)

    def _failure_result(
        self,
        *,
        run_id: str,
        bundle: BenchmarkScenarioBundle,
        recipe: BenchmarkRecipe,
        replicate_id: str,
        seed: int,
        status: BenchmarkRunStatus,
        failure_code: str,
        error_message: str,
        runtime_ms: int = 0,
        peak_memory_mb: int = 0,
    ) -> BenchmarkRunResult:
        record = BenchmarkRunRecord(
            run_id=run_id,
            family_id=bundle.family_id,
            instance_id=bundle.instance_id,
            replicate_id=replicate_id,
            task_profile_digest=bundle.task_profile.profile_digest,
            pipeline_recipe_digest=recipe.recipe_digest,
            engine_commit=self._engine_commit,
            dependency_lock_digest=self._dependency_lock_digest,
            environment_digest=self._environment_digest,
            random_seed=seed,
            status=status,
            failure_code=failure_code,
            aggregate_metrics_digest=None,
            stage_artifact_manifest_digest=None,
            runtime_ms=runtime_ms,
            peak_memory_mb=peak_memory_mb,
        )
        failure = BenchmarkFailureRecord(
            run_id=run_id,
            family_id=bundle.family_id,
            instance_id=bundle.instance_id,
            replicate_id=replicate_id,
            recipe_id=recipe.recipe_id,
            status=status,
            failure_code=failure_code,
            error_message=error_message,
        )
        return BenchmarkRunResult(record=record, failure=failure)

    def run_portfolio(
        self,
        generator: BenchmarkScenarioGenerator,
        *,
        families: Iterable[str] | None = None,
        instances: Iterable[str] | None = None,
        recipes: Iterable[BenchmarkRecipe] | None = None,
        replicates: int = 1,
        base_seed: int = 20260816,
        replicate_start: int = 0,
    ) -> tuple[BenchmarkRunResult, ...]:
        """Execute the portfolio across requested families/instances and replicates."""
        if not 1 <= replicates <= 10_000:
            raise ValueError("Benchmark replicate count must be between 1 and 10000.")
        if not 0 <= replicate_start <= 9_999_999:
            raise ValueError("Benchmark replicate start must be between 0 and 9999999.")
        if not 0 <= base_seed <= 4_294_967_295:
            raise ValueError("Benchmark base seed must be an unsigned 32-bit integer.")

        all_family_ids = {fam.family_id for fam in generator.list_families()}
        target_families = set(families) if families else all_family_ids

        all_instances = generator.list_instances()
        if instances:
            target_instance_ids = set(instances)
            target_instances = [
                inst for inst in all_instances if inst.instance_id in target_instance_ids
            ]
        else:
            target_instances = [inst for inst in all_instances if inst.family_id in target_families]

        target_recipes = list(recipes or self._recipes)
        results: list[BenchmarkRunResult] = []

        for inst_manifest in target_instances:
            for rep_idx in range(replicates):
                replicate_number = replicate_start + rep_idx
                replicate_id = f"replicate.{replicate_number:07d}"
                seed = benchmark_replicate_seed(
                    instance_id=inst_manifest.instance_id,
                    replicate_number=replicate_number,
                    base_seed=base_seed,
                )
                bundle = generator.generate(inst_manifest.instance_id, seed=seed)

                for recipe in target_recipes:
                    res = self.run_single(
                        bundle=bundle,
                        recipe=recipe,
                        replicate_id=replicate_id,
                        seed=seed,
                    )
                    results.append(res)

        return tuple(results)


__all__ = [
    "BenchmarkPortfolioRunner",
    "BenchmarkRecipe",
    "BenchmarkRunResult",
    "benchmark_replicate_seed",
    "benchmark_run_id",
]
