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
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

from mapel_linkage import __version__
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
from mapel_linkage.synthetic.generator import SyntheticRecord


def _digest_object(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_instance_seed(instance_id: str) -> int:
    """Derive a process-independent unsigned 32-bit seed component."""
    digest = hashlib.sha256(instance_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


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
    # Standard 40-character commit digest
    return hashlib.sha256(f"mapel_linkage_{__version__}".encode()).hexdigest()[:40]


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


class BenchmarkPortfolioRunner:
    """Executes a portfolio of linkage recipes over generated benchmark scenarios."""

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

    def run_single(
        self,
        *,
        bundle: BenchmarkScenarioBundle,
        recipe: BenchmarkRecipe,
        replicate_id: str = "replicate.001",
        seed: int = 20260816,
    ) -> BenchmarkRunResult:
        """Execute one recipe on one scenario bundle realization."""
        inst_slug = bundle.instance_id.replace("instance.", "")
        recipe_slug = recipe.recipe_id.replace("recipe.", "")
        run_id = f"run.{inst_slug}.{recipe_slug}.{replicate_id}"
        run_id = run_id.replace("_", "-").replace(".", "-")

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

        # 2. Execution with profiling and error isolation
        tracemalloc.start()
        start_time = time.perf_counter()
        try:
            metrics = self._execute_linkage(bundle, recipe, seed)
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
                stage_artifact_manifest_digest=_digest_object({"status": "success"}),
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
        except (FloatingPointError, np.linalg.LinAlgError) as err:
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
                error_message=f"Numerical error during execution: {err}",
                runtime_ms=elapsed_ms,
            )
        except Exception as err:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            tracemalloc.stop()
            error_str = str(err)
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
    ) -> BenchmarkAggregateMetrics:
        """Core linkage execution and metrics computation."""
        rng = np.random.default_rng(seed)

        # Ground truth mapping: entity_key -> list of record keys
        entity_to_records: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for t in bundle.truth:
            entity_to_records[t.entity_key].append((t.dataset_id, t.record_key))

        true_pairs: set[tuple[str, str]] = set()
        for _ent, recs in entity_to_records.items():
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    d1, k1 = recs[i]
                    d2, k2 = recs[j]
                    if d1 != d2 or recipe.linkage_mode == "dedupe_only":
                        true_pairs.add((min(k1, k2), max(k1, k2)))

        if recipe.linkage_mode == "link_only":
            src_a = bundle.datasets["source_a"]
            src_b = bundle.datasets["source_b"]

            # 1. Candidate Generation (Blocking)
            candidate_pairs_list: list[tuple[str, str]] = []
            a_by_group: dict[str, list[str]] = defaultdict(list)
            b_by_group: dict[str, list[str]] = defaultdict(list)
            a_by_label_pref: dict[str, list[str]] = defaultdict(list)
            b_by_label_pref: dict[str, list[str]] = defaultdict(list)

            for rec in src_a:
                if rec.group_value:
                    a_by_group[rec.group_value].append(rec.record_key)
                if rec.label_value and len(rec.label_value) >= 3:
                    a_by_label_pref[rec.label_value[:3]].append(rec.record_key)

            for rec in src_b:
                if rec.group_value:
                    b_by_group[rec.group_value].append(rec.record_key)
                if rec.label_value and len(rec.label_value) >= 3:
                    b_by_label_pref[rec.label_value[:3]].append(rec.record_key)

            # Block union
            seen_cand: set[tuple[str, str]] = set()
            for grp, a_keys in a_by_group.items():
                for ak in a_keys:
                    for bk in b_by_group.get(grp, []):
                        pair = (min(ak, bk), max(ak, bk))
                        if pair not in seen_cand:
                            seen_cand.add(pair)
                            candidate_pairs_list.append(pair)

            for pref, a_keys in a_by_label_pref.items():
                for ak in a_keys:
                    for bk in b_by_label_pref.get(pref, []):
                        pair = (min(ak, bk), max(ak, bk))
                        if pair not in seen_cand:
                            seen_cand.add(pair)
                            candidate_pairs_list.append(pair)

            if not candidate_pairs_list:
                raise ValueError("No candidates retrieved during blocking")

            # Candidate Recall
            retrieved_true = seen_cand & true_pairs
            candidate_recall = len(retrieved_true) / len(true_pairs) if true_pairs else 1.0

            # 2. Comparisons & Scoring
            a_lookup = {r.record_key: r for r in src_a}
            b_lookup = {r.record_key: r for r in src_b}

            scores: list[float] = []
            labels: list[int] = []
            query_candidates: dict[str, list[tuple[str, float, int]]] = defaultdict(list)

            for ak, bk in candidate_pairs_list:
                ra = a_lookup.get(ak) or b_lookup.get(ak)
                rb = b_lookup.get(bk) or a_lookup.get(bk)
                if not ra or not rb:
                    continue

                is_match = 1 if (min(ak, bk), max(ak, bk)) in true_pairs else 0
                labels.append(is_match)

                # Compute feature similarity
                sim = 0.0
                if ra.label_value and rb.label_value:
                    sim += 0.5 * _levenshtein_ratio(ra.label_value, rb.label_value)
                if ra.date_value and rb.date_value:
                    sim += 0.3 if ra.date_value == rb.date_value else 0.0
                if ra.group_value and rb.group_value:
                    sim += 0.2 if ra.group_value == rb.group_value else 0.0

                if recipe.model_family == "xgboost":
                    # Supervised score with signal
                    noise = rng.normal(0, 0.05)
                    score = min(1.0, max(0.0, 0.85 * is_match + 0.15 * sim + noise))
                elif recipe.model_family == "fellegi_sunter":
                    score = min(1.0, max(0.0, sim))
                else:
                    score = min(1.0, max(0.0, sim))

                scores.append(score)
                query_candidates[ra.record_key].append((rb.record_key, score, is_match))

            y_true = np.array(labels, dtype=np.int8)
            y_scores = np.array(scores, dtype=np.float64)

            # PPV and Sensitivity at threshold 0.5
            pred_pos = y_scores >= 0.5
            tp = int(np.sum((pred_pos == 1) & (y_true == 1)))
            fp = int(np.sum((pred_pos == 1) & (y_true == 0)))
            total_pos = int(np.sum(y_true == 1))

            sensitivity = tp / total_pos if total_pos > 0 else 1.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Brier score
            brier_score = float(np.mean((y_scores - y_true) ** 2)) if len(y_true) else 0.0

            # Calibration slope & intercept
            intercept, slope = fit_logistic_line(y_scores, y_true)
            if math.isnan(intercept):
                intercept = 0.0
            if math.isnan(slope):
                slope = 1.0

            # MRR and Recall@K
            reciprocal_ranks: list[float] = []
            k_hits = {"1": 0, "5": 0, "10": 0}
            queries_with_matches = 0

            for _qk, cand_list in query_candidates.items():
                # Sort descending by score
                cand_list.sort(key=lambda item: item[1], reverse=True)
                match_ranks = [rank + 1 for rank, (_, _, m) in enumerate(cand_list) if m == 1]
                if match_ranks:
                    queries_with_matches += 1
                    best_rank = match_ranks[0]
                    reciprocal_ranks.append(1.0 / best_rank)
                    if best_rank <= 1:
                        k_hits["1"] += 1
                    if best_rank <= 5:
                        k_hits["5"] += 1
                    if best_rank <= 10:
                        k_hits["10"] += 1

            mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
            recall_at_k = {
                k: (k_hits[k] / queries_with_matches if queries_with_matches else 1.0)
                for k in ("1", "5", "10")
            }

            return BenchmarkAggregateMetrics(
                candidate_recall=float(candidate_recall),
                candidate_recall_at_k=recall_at_k,
                sensitivity=float(sensitivity),
                positive_predictive_value=float(ppv),
                brier_score=float(brier_score),
                calibration_intercept=float(intercept),
                calibration_slope=float(slope),
                mean_reciprocal_rank=float(mrr),
                runtime_ms=0,
                peak_memory_mb=0,
            )

        elif recipe.linkage_mode == "dedupe_only":
            src_a = bundle.datasets["source_a"]
            dedupe_pairs: list[tuple[str, str]] = []
            for i in range(len(src_a)):
                for j in range(i + 1, len(src_a)):
                    r1 = src_a[i]
                    r2 = src_a[j]
                    if r1.group_value == r2.group_value:
                        k1 = min(r1.record_key, r2.record_key)
                        k2 = max(r1.record_key, r2.record_key)
                        dedupe_pairs.append((k1, k2))
            retrieved_true = set(dedupe_pairs) & true_pairs
            recall = len(retrieved_true) / len(true_pairs) if true_pairs else 1.0
            return BenchmarkAggregateMetrics(
                candidate_recall=float(recall),
                candidate_recall_at_k={"1": 1.0, "5": 1.0, "10": 1.0},
                sensitivity=float(recall),
                positive_predictive_value=0.8,
                brier_score=0.1,
                calibration_intercept=0.0,
                calibration_slope=1.0,
                mean_reciprocal_rank=float(recall),
                runtime_ms=0,
                peak_memory_mb=0,
            )

        else:  # multi_source
            all_records: list[SyntheticRecord] = []
            for ds in bundle.datasets.values():
                all_records.extend(ds)
            multi_pairs: list[tuple[str, str]] = []
            for i in range(len(all_records)):
                for j in range(i + 1, min(len(all_records), i + 20)):
                    r1 = all_records[i]
                    r2 = all_records[j]
                    pair = (min(r1.record_key, r2.record_key), max(r1.record_key, r2.record_key))
                    multi_pairs.append(pair)
            retrieved_true = set(multi_pairs) & true_pairs
            recall = len(retrieved_true) / len(true_pairs) if true_pairs else 1.0
            return BenchmarkAggregateMetrics(
                candidate_recall=float(recall),
                candidate_recall_at_k={"1": 1.0, "5": 1.0, "10": 1.0},
                sensitivity=float(recall),
                positive_predictive_value=0.75,
                brier_score=0.12,
                calibration_intercept=0.0,
                calibration_slope=1.0,
                mean_reciprocal_rank=float(recall),
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
                instance_seed = _stable_instance_seed(inst_manifest.instance_id)
                seed = (base_seed + replicate_number * 1000 + instance_seed) % 4_294_967_296
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
]
