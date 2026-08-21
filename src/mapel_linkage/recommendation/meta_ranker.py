"""Stage-3 Learned Meta-Ranking Strategy Advisor with conformal uncertainty estimation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mapel_linkage.benchmarking.advisor_catalogue import (
    advisor_v2_family_roles,
    build_advisor_v2_generator,
)
from mapel_linkage.benchmarking.contracts import (
    BenchmarkRunRecord,
    BenchmarkRunStatus,
)
from mapel_linkage.benchmarking.generator import (
    BenchmarkScenarioGenerator,
)
from mapel_linkage.benchmarking.registry import (
    BenchmarkRegistry,
)
from mapel_linkage.benchmarking.runner import BenchmarkPortfolioRunner
from mapel_linkage.configuration.compiler import ExecutionPlan
from mapel_linkage.domain.errors import AdvisorError
from mapel_linkage.profiling import PreflightTaskProfile, build_preflight_task_profile
from mapel_linkage.recommendation.contracts import (
    MetaRankingAdvisoryReport,
    PredictedCandidateUtility,
    StructuralPipelineCandidate,
)
from mapel_linkage.recommendation.distance import (
    MetaFeatureDistanceComputer,
    TaskMetaFeatureVector,
    extract_family_meta_features,
)
from mapel_linkage.recommendation.eligibility import (
    AdvisorContext,
)
from mapel_linkage.recommendation.similarity_advisor import (
    SimilarityLinkageAdvisor,
)

_REQUIRED_BENCHMARK_RECIPE_TOKENS = frozenset(
    {"fellegi_sunter", "xgboost_classifier", "xgboost_ranker"}
)
_MINIMUM_COMPLETE_REPLICATES = 5


def _has_complete_required_evidence_grid(
    records: tuple[BenchmarkRunRecord, ...],
    *,
    expected_instance_ids: frozenset[str],
    recipe_token_by_digest: Mapping[str, str],
) -> bool:
    """Reject partial, failed, duplicate, mixed-provenance, or sparse adapter evidence."""

    if len(expected_instance_ids) != 280:
        return False
    relevant = tuple(record for record in records if record.instance_id in expected_instance_ids)
    if {record.instance_id for record in relevant} != expected_instance_ids:
        return False
    provenance = {
        (record.engine_commit, record.dependency_lock_digest, record.environment_digest)
        for record in relevant
    }
    if len(provenance) != 1:
        return False

    replicates_by_instance: dict[str, set[str]] = {
        instance_id: set() for instance_id in expected_instance_ids
    }
    required_by_cell: dict[tuple[str, str], dict[str, BenchmarkRunStatus]] = {}
    for record in relevant:
        replicates_by_instance[record.instance_id].add(record.replicate_id)
        token = recipe_token_by_digest.get(record.pipeline_recipe_digest)
        if token is None:
            continue
        cell = required_by_cell.setdefault((record.instance_id, record.replicate_id), {})
        if token in cell:
            return False
        cell[token] = record.status

    replicate_sets = {tuple(sorted(values)) for values in replicates_by_instance.values()}
    if len(replicate_sets) != 1:
        return False
    replicate_ids = next(iter(replicate_sets))
    if len(replicate_ids) < _MINIMUM_COMPLETE_REPLICATES or replicate_ids != tuple(
        f"replicate.{index:07d}" for index in range(len(replicate_ids))
    ):
        return False

    return all(
        set(required_by_cell.get((instance_id, replicate_id), {}))
        == _REQUIRED_BENCHMARK_RECIPE_TOKENS
        and all(
            status is BenchmarkRunStatus.SUCCESS
            for status in required_by_cell[(instance_id, replicate_id)].values()
        )
        for instance_id in expected_instance_ids
        for replicate_id in replicate_ids
    )


def _policy_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_feature_vector(candidate: StructuralPipelineCandidate) -> list[float]:
    feats = [
        float(candidate.structural_complexity),
        float(candidate.interaction_capacity),
        float(candidate.interpretability_score),
        float(candidate.artifact_portability_score),
        1.0 if candidate.requires_verified_labels else 0.0,
        1.0 if candidate.requires_protected_out_of_fold_predictions else 0.0,
    ]
    families = ["fellegi_sunter", "xgboost", "lightgbm", "pytorch_tabular", "stacking_ensemble"]
    feats.extend([1.0 if candidate.pair_model_family == fam else 0.0 for fam in families])
    rankings = ["none", "xgboost_ranker", "lightgbm_ranker"]
    feats.extend([1.0 if candidate.ranking_strategy.value == rk else 0.0 for rk in rankings])
    calibs = ["none", "sigmoid", "isotonic", "beta"]
    feats.extend([1.0 if candidate.calibration_method == cb else 0.0 for cb in calibs])
    assigns = ["one_to_one", "many_to_one", "one_to_many", "unconstrained"]
    feats.extend([1.0 if candidate.assignment_constraint == asg else 0.0 for asg in assigns])
    return feats


def _continuous_features_list(vector: TaskMetaFeatureVector) -> list[float]:
    return [
        vector.record_count_log_ratio,
        vector.missingness_mean,
        vector.missingness_max,
        vector.entropy_estimate,
        vector.candidate_edge_budget_scale,
        vector.error_estimate_approx,
        vector.label_volume_scale,
        vector.variable_count_scale,
        vector.comparison_count_scale,
        vector.blocking_rule_count_scale,
    ]


_RECIPE_TOKEN_BY_ID = {
    "recipe.fellegi_sunter_reference": "fellegi_sunter",
    "recipe.xgboost_classifier": "xgboost_classifier",
    "recipe.xgboost_ranker": "xgboost_ranker",
}


@dataclass
class LearnedMetaRankerModel:
    """Ridge-regularized meta-regressor with conformal prediction intervals."""

    weights: np.ndarray | None = None
    intercept: float = 0.5
    conformal_residual_quantile: float = 0.15
    coverage_level: float = 0.90
    trained_run_count: int = 0
    trained_family_count: int = 0
    conformal_run_count: int = 0
    conformal_family_count: int = 0
    locked_evaluation_run_count: int = 0
    locked_evaluation_family_count: int = 0
    locked_mean_absolute_error: float | None = None
    family_split_digest: str | None = None

    @property
    def model_digest(self) -> str:
        """Return a stable aggregate model digest without exposing training rows."""
        payload = {
            "weights": self.weights.tolist() if self.weights is not None else None,
            "intercept": self.intercept,
            "conformal_residual_quantile": self.conformal_residual_quantile,
            "coverage_level": self.coverage_level,
            "trained_run_count": self.trained_run_count,
            "trained_family_count": self.trained_family_count,
            "conformal_run_count": self.conformal_run_count,
            "conformal_family_count": self.conformal_family_count,
            "family_split_digest": self.family_split_digest,
        }
        return _policy_digest(payload)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_conformal: np.ndarray,
        y_conformal: np.ndarray,
        training_family_ids: tuple[str, ...],
        conformal_family_ids: tuple[str, ...],
        alpha: float = 1.0,
        coverage_level: float = 0.90,
    ) -> None:
        """Fit on training families and calibrate intervals on disjoint families."""
        n_samples, n_features = X.shape
        if (
            n_samples < 2
            or X_conformal.ndim != 2
            or X_conformal.shape[1] != n_features
            or len(y) != n_samples
            or len(y_conformal) != X_conformal.shape[0]
            or len(training_family_ids) != n_samples
            or len(conformal_family_ids) != len(y_conformal)
            or not conformal_family_ids
        ):
            raise ValueError("Meta-ranker family-partition evidence is incomplete.")
        training_families = set(training_family_ids)
        conformal_families = set(conformal_family_ids)
        if training_families & conformal_families:
            raise ValueError("Meta-ranker training and conformal families must be disjoint.")
        if len(training_families) < 2 or not np.all(np.isfinite(y)) or np.ptp(y) <= 1e-12:
            raise ValueError("Meta-ranker training evidence lacks family or utility variation.")

        X_bias = np.hstack([np.ones((n_samples, 1)), X])
        reg_matrix = alpha * np.eye(n_features + 1)
        reg_matrix[0, 0] = 0.0

        try:
            w = np.linalg.solve(X_bias.T @ X_bias + reg_matrix, X_bias.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(X_bias.T @ X_bias + reg_matrix) @ (X_bias.T @ y)

        self.intercept = float(w[0])
        self.weights = w[1:]

        conformal_predictions = X_conformal @ self.weights + self.intercept
        residuals = np.abs(y_conformal - conformal_predictions)

        conformal_count = len(residuals)
        k_idx = min(
            conformal_count - 1,
            math.ceil((conformal_count + 1) * coverage_level) - 1,
        )
        sorted_res = np.sort(residuals)
        self.conformal_residual_quantile = float(sorted_res[max(0, k_idx)])
        self.coverage_level = coverage_level
        self.trained_run_count = n_samples
        self.trained_family_count = len(training_families)
        self.conformal_run_count = conformal_count
        self.conformal_family_count = len(conformal_families)
        self.family_split_digest = _policy_digest(
            {
                "training_families": sorted(training_families),
                "conformal_families": sorted(conformal_families),
            }
        )

    def evaluate_locked(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        locked_family_ids: tuple[str, ...],
        prohibited_family_ids: frozenset[str],
    ) -> None:
        """Mechanically evaluate only after fitting; never tune on locked families."""

        if (
            not locked_family_ids
            or len(locked_family_ids) != len(y)
            or X.shape[0] != len(y)
            or set(locked_family_ids) & prohibited_family_ids
        ):
            raise ValueError("Locked meta-ranker evaluation families are invalid.")
        predictions, _, _ = self.predict_utility(X)
        self.locked_evaluation_run_count = len(y)
        self.locked_evaluation_family_count = len(set(locked_family_ids))
        self.locked_mean_absolute_error = float(np.mean(np.abs(y - predictions)))

    def predict_utility(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict expected utility and return (predictions, lower_bounds, upper_bounds)."""
        if self.weights is None:
            preds = np.full(X.shape[0], self.intercept)
        else:
            preds = X @ self.weights + self.intercept

        preds = np.clip(preds, 0.0, 1.0)
        q = self.conformal_residual_quantile
        lower = np.clip(preds - q, 0.0, 1.0)
        upper = np.clip(preds + q, 0.0, 1.0)
        return preds, lower, upper


class MetaRankingLinkageAdvisor:
    """Stage-3 Learned Meta-Ranking Advisor with Conformal Uncertainty Bounds."""

    def __init__(
        self,
        registry: BenchmarkRegistry | None = None,
        *,
        distance_computer: MetaFeatureDistanceComputer | None = None,
        generator: BenchmarkScenarioGenerator | None = None,
        max_ood_distance: float = 0.45,
        conformal_coverage: float = 0.90,
    ) -> None:
        self.registry = registry
        self.distance_computer = distance_computer or MetaFeatureDistanceComputer()
        self.generator = generator or build_advisor_v2_generator()
        self.max_ood_distance = max_ood_distance
        self.conformal_coverage = conformal_coverage
        self._family_vectors = extract_family_meta_features(self.generator)
        self._recipe_token_by_digest = {
            recipe.recipe_digest: _RECIPE_TOKEN_BY_ID[recipe.recipe_id]
            for recipe in BenchmarkPortfolioRunner().list_recipes()
            if recipe.recipe_id in _RECIPE_TOKEN_BY_ID
        }
        self.last_fitted_model: LearnedMetaRankerModel | None = None

    def advise(
        self,
        plan: ExecutionPlan,
        *,
        context: AdvisorContext,
        profile: PreflightTaskProfile | None = None,
    ) -> MetaRankingAdvisoryReport:
        """Produce a Stage-3 Learned Meta-Ranking Advisory Report."""
        self.last_fitted_model = None
        if context.test_partition_used:
            raise AdvisorError(
                "ML-ADVISOR-001",
                "The locked test partition cannot be used by the strategy advisor.",
            )

        task_profile = profile or build_preflight_task_profile(plan)
        task_vector = TaskMetaFeatureVector.from_profile(task_profile)

        similarity_advisor = SimilarityLinkageAdvisor(
            registry=self.registry,
            distance_computer=self.distance_computer,
            generator=self.generator,
            max_ood_distance=self.max_ood_distance,
        )

        sim_report = similarity_advisor.recommend(plan, context=context, profile=task_profile)

        if self.registry is None:
            return MetaRankingAdvisoryReport(
                report_id=f"meta_ranker_{task_profile.profile_digest[:16]}",
                recommendation=sim_report.recommendation,
                predicted_candidate_utilities={},
                meta_model_type="none",
                meta_model_trained_runs=0,
                fallback_to_similarity=True,
                fallback_reason="No benchmark registry provided; fell back to Stage-2 similarity.",
            )

        all_runs = self.registry.list_run_records()
        runs = [r for r in all_runs if r.status == BenchmarkRunStatus.SUCCESS]
        role_by_family = dict(advisor_v2_family_roles())
        expected_instance_ids = frozenset(
            item.instance_id
            for item in self.generator.list_instances()
            if item.family_id in role_by_family
        )
        complete_evidence_grid = _has_complete_required_evidence_grid(
            all_runs,
            expected_instance_ids=expected_instance_ids,
            recipe_token_by_digest=self._recipe_token_by_digest,
        )
        recipe_coverage: dict[str, set[str]] = {}
        recipe_token_by_run_id: dict[str, str] = {}
        for run in runs:
            token = self._recipe_token_by_digest.get(run.pipeline_recipe_digest)
            if token is not None:
                recipe_coverage.setdefault(run.family_id, set()).add(token)
                recipe_token_by_run_id[run.run_id] = token
        overlap_families = {
            family_id
            for family_id, tokens in recipe_coverage.items()
            if _REQUIRED_BENCHMARK_RECIPE_TOKENS.issubset(tokens) and family_id in role_by_family
        }
        train_overlap_families = {
            family_id
            for family_id in overlap_families
            if role_by_family[family_id] == "meta_training"
        }
        conformal_overlap_families = {
            family_id for family_id in overlap_families if role_by_family[family_id] == "conformal"
        }
        locked_overlap_families = {
            family_id
            for family_id in overlap_families
            if role_by_family[family_id] == "locked_evaluation"
        }
        expected_train_families = {
            family_id for family_id, role in role_by_family.items() if role == "meta_training"
        }
        expected_conformal_families = {
            family_id for family_id, role in role_by_family.items() if role == "conformal"
        }
        expected_locked_families = {
            family_id for family_id, role in role_by_family.items() if role == "locked_evaluation"
        }
        prospective_evidence_complete = (
            complete_evidence_grid
            and train_overlap_families == expected_train_families
            and conformal_overlap_families == expected_conformal_families
            and locked_overlap_families == expected_locked_families
        )

        nearest = self.distance_computer.find_nearest_families(
            task_vector, self._family_vectors, k=1
        )
        min_dist = nearest[0][1] if nearest else 1.0
        is_ood = min_dist > self.max_ood_distance or not prospective_evidence_complete

        if is_ood:
            if min_dist > self.max_ood_distance:
                reason = (
                    f"Task is out-of-distribution "
                    f"(distance {min_dist:.3f} > {self.max_ood_distance})"
                )
            else:
                reason = (
                    "Scenario-replicate-complete family-disjoint adapter evidence is incomplete"
                )
            return MetaRankingAdvisoryReport(
                report_id=f"meta_ranker_{task_profile.profile_digest[:16]}",
                recommendation=sim_report.recommendation,
                predicted_candidate_utilities={},
                meta_model_type="none",
                meta_model_trained_runs=len(runs),
                fallback_to_similarity=True,
                fallback_reason=f"{reason}; fell back to Stage-2 similarity.",
            )

        eligible_candidates = sim_report.recommendation.shortlist

        X_by_role: dict[str, list[list[float]]] = {
            "meta_training": [],
            "conformal": [],
            "locked_evaluation": [],
        }
        y_by_role: dict[str, list[float]] = {name: [] for name in X_by_role}
        families_by_role: dict[str, list[str]] = {name: [] for name in X_by_role}

        for run in runs:
            if run.family_id not in overlap_families:
                continue
            recipe_token = recipe_token_by_run_id.get(run.run_id)
            if recipe_token is None:
                continue
            role = role_by_family[run.family_id]
            if role == "ood_holdout":
                continue
            fam_vec = self._family_vectors.get(run.family_id)
            if fam_vec is None:
                continue

            metrics = self.registry.load_metrics(run.run_id)
            if metrics is None:
                continue

            for cand_obj in eligible_candidates:
                fam = cand_obj.pair_model_family
                r_strat = cand_obj.ranking_strategy.value
                matches = (
                    (fam == "fellegi_sunter" and recipe_token == "fellegi_sunter")
                    or (
                        fam == "xgboost"
                        and r_strat == "xgboost_ranker"
                        and recipe_token == "xgboost_ranker"
                    )
                    or (
                        fam == "xgboost"
                        and r_strat == "model_score"
                        and recipe_token == "xgboost_classifier"
                    )
                )
                if not matches:
                    continue

                recall = float(metrics.candidate_recall_at_k.get("1", metrics.candidate_recall))
                ppv = float(metrics.positive_predictive_value)
                brier = float(metrics.brier_score)
                utility = float(np.clip(0.5 * recall + 0.3 * ppv + 0.2 * (1.0 - brier), 0.0, 1.0))

                feat_vec = _continuous_features_list(fam_vec) + _candidate_feature_vector(cand_obj)
                X_by_role[role].append(feat_vec)
                y_by_role[role].append(utility)
                families_by_role[role].append(run.family_id)

        if (
            len(set(families_by_role["meta_training"])) < 2
            or not families_by_role["conformal"]
            or not families_by_role["locked_evaluation"]
        ):
            return MetaRankingAdvisoryReport(
                report_id=f"meta_ranker_{task_profile.profile_digest[:16]}",
                recommendation=sim_report.recommendation,
                predicted_candidate_utilities={},
                meta_model_type="none",
                meta_model_trained_runs=len(runs),
                fallback_to_similarity=True,
                fallback_reason=(
                    "Family-disjoint training, conformal, and locked evidence is incomplete; "
                    "fell back to Stage-2 similarity."
                ),
            )

        X_train = np.array(X_by_role["meta_training"], dtype=np.float64)
        y_train = np.array(y_by_role["meta_training"], dtype=np.float64)
        X_conformal = np.array(X_by_role["conformal"], dtype=np.float64)
        y_conformal = np.array(y_by_role["conformal"], dtype=np.float64)
        X_locked = np.array(X_by_role["locked_evaluation"], dtype=np.float64)
        y_locked = np.array(y_by_role["locked_evaluation"], dtype=np.float64)

        meta_model = LearnedMetaRankerModel()
        try:
            meta_model.fit(
                X_train,
                y_train,
                X_conformal=X_conformal,
                y_conformal=y_conformal,
                training_family_ids=tuple(families_by_role["meta_training"]),
                conformal_family_ids=tuple(families_by_role["conformal"]),
                coverage_level=self.conformal_coverage,
            )
            meta_model.evaluate_locked(
                X_locked,
                y_locked,
                locked_family_ids=tuple(families_by_role["locked_evaluation"]),
                prohibited_family_ids=frozenset(
                    families_by_role["meta_training"] + families_by_role["conformal"]
                ),
            )
        except ValueError:
            return MetaRankingAdvisoryReport(
                report_id=f"meta_ranker_{task_profile.profile_digest[:16]}",
                recommendation=sim_report.recommendation,
                predicted_candidate_utilities={},
                meta_model_type="none",
                meta_model_trained_runs=len(runs),
                fallback_to_similarity=True,
                fallback_reason=(
                    "Meta-learning evidence lacks estimable family or utility variation; "
                    "fell back to Stage-2 similarity."
                ),
            )
        self.last_fitted_model = meta_model

        task_cont = _continuous_features_list(task_vector)
        X_test_list = [task_cont + _candidate_feature_vector(c) for c in eligible_candidates]
        X_test = np.array(X_test_list, dtype=np.float64)

        preds, lowers, uppers = meta_model.predict_utility(X_test)

        predicted_utilities: dict[str, PredictedCandidateUtility] = {}
        for c, pred, low, up in zip(eligible_candidates, preds, lowers, uppers, strict=True):
            predicted_utilities[c.candidate_id] = PredictedCandidateUtility(
                candidate_id=c.candidate_id,
                predicted_utility=round(float(pred), 4),
                uncertainty_lower_bound=round(float(low), 4),
                uncertainty_upper_bound=round(float(up), 4),
                conformal_coverage_level=self.conformal_coverage,
            )

        return MetaRankingAdvisoryReport(
            report_id=f"meta_ranker_{task_profile.profile_digest[:16]}",
            recommendation=sim_report.recommendation,
            predicted_candidate_utilities=predicted_utilities,
            meta_model_type="ridge_meta_ranker_v1",
            meta_model_trained_runs=meta_model.trained_run_count,
            fallback_to_similarity=False,
            fallback_reason=None,
        )


__all__ = [
    "LearnedMetaRankerModel",
    "MetaRankingLinkageAdvisor",
]
