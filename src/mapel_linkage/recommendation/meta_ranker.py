"""Stage-3 Learned Meta-Ranking Strategy Advisor with conformal uncertainty estimation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mapel_linkage.benchmarking.contracts import (
    BenchmarkRunStatus,
)
from mapel_linkage.benchmarking.generator import (
    BenchmarkScenarioGenerator,
)
from mapel_linkage.benchmarking.registry import (
    BenchmarkRegistry,
)
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

_MIN_FAMILIES_FOR_META_LEARNING = 3


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


@dataclass
class LearnedMetaRankerModel:
    """Ridge-regularized meta-regressor with conformal prediction intervals."""

    weights: np.ndarray | None = None
    intercept: float = 0.5
    conformal_residual_quantile: float = 0.15
    coverage_level: float = 0.90
    trained_run_count: int = 0
    trained_family_count: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        alpha: float = 1.0,
        coverage_level: float = 0.90,
    ) -> None:
        """Fit Ridge regression model on historical run meta-features."""
        n_samples, n_features = X.shape
        if n_samples < 2:
            self.weights = np.zeros(n_features)
            self.intercept = float(np.mean(y)) if len(y) > 0 else 0.5
            self.conformal_residual_quantile = 0.20
            self.coverage_level = coverage_level
            return

        X_bias = np.hstack([np.ones((n_samples, 1)), X])
        reg_matrix = alpha * np.eye(n_features + 1)
        reg_matrix[0, 0] = 0.0

        try:
            w = np.linalg.solve(X_bias.T @ X_bias + reg_matrix, X_bias.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(X_bias.T @ X_bias + reg_matrix) @ (X_bias.T @ y)

        self.intercept = float(w[0])
        self.weights = w[1:]

        predictions = X_bias @ w
        residuals = np.abs(y - predictions)

        k_idx = min(n_samples - 1, math.ceil((n_samples + 1) * coverage_level) - 1)
        sorted_res = np.sort(residuals)
        self.conformal_residual_quantile = float(sorted_res[max(0, k_idx)])
        self.coverage_level = coverage_level
        self.trained_run_count = n_samples

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
        self.generator = generator or BenchmarkScenarioGenerator()
        self.max_ood_distance = max_ood_distance
        self.conformal_coverage = conformal_coverage
        self._family_vectors = extract_family_meta_features(self.generator)

    def advise(
        self,
        plan: ExecutionPlan,
        *,
        context: AdvisorContext,
        profile: PreflightTaskProfile | None = None,
    ) -> MetaRankingAdvisoryReport:
        """Produce a Stage-3 Learned Meta-Ranking Advisory Report."""
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

        runs = [
            r for r in self.registry.list_run_records() if r.status == BenchmarkRunStatus.SUCCESS
        ]
        families_in_runs = {r.family_id for r in runs}

        nearest = self.distance_computer.find_nearest_families(
            task_vector, self._family_vectors, k=1
        )
        min_dist = nearest[0][1] if nearest else 1.0
        is_ood = (
            min_dist > self.max_ood_distance
            or len(families_in_runs) < _MIN_FAMILIES_FOR_META_LEARNING
        )

        if is_ood:
            if min_dist > self.max_ood_distance:
                reason = (
                    f"Task is out-of-distribution "
                    f"(distance {min_dist:.3f} > {self.max_ood_distance})"
                )
            else:
                reason = (
                    f"Insufficient historical families "
                    f"({len(families_in_runs)} < {_MIN_FAMILIES_FOR_META_LEARNING})"
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

        X_train_list: list[list[float]] = []
        y_train_list: list[float] = []

        for run in runs:
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
                    (fam == "fellegi_sunter" and "fellegi" in run.run_id)
                    or (
                        fam == "xgboost"
                        and r_strat == "xgboost_ranker"
                        and "xgboost-ranker" in run.run_id
                    )
                    or (
                        fam == "xgboost"
                        and r_strat == "model_score"
                        and "xgboost-classifier" in run.run_id
                    )
                    or (fam == "lightgbm" and "lightgbm" in run.run_id)
                    or (fam == "pytorch" and "pytorch" in run.run_id)
                    or (fam == "stacking")
                )
                if not matches:
                    continue

                recall = float(metrics.candidate_recall_at_k.get("1", metrics.candidate_recall))
                ppv = float(metrics.positive_predictive_value)
                brier = float(metrics.brier_score)
                utility = float(np.clip(0.5 * recall + 0.3 * ppv + 0.2 * (1.0 - brier), 0.0, 1.0))

                feat_vec = _continuous_features_list(fam_vec) + _candidate_feature_vector(cand_obj)
                X_train_list.append(feat_vec)
                y_train_list.append(utility)

        if len(X_train_list) < _MIN_FAMILIES_FOR_META_LEARNING:
            return MetaRankingAdvisoryReport(
                report_id=f"meta_ranker_{task_profile.profile_digest[:16]}",
                recommendation=sim_report.recommendation,
                predicted_candidate_utilities={},
                meta_model_type="none",
                meta_model_trained_runs=len(runs),
                fallback_to_similarity=True,
                fallback_reason="Insufficient training runs; fell back to Stage-2 similarity.",
            )

        X_train = np.array(X_train_list, dtype=np.float64)
        y_train = np.array(y_train_list, dtype=np.float64)

        meta_model = LearnedMetaRankerModel()
        meta_model.fit(X_train, y_train, coverage_level=self.conformal_coverage)

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
            meta_model_trained_runs=len(runs),
            fallback_to_similarity=False,
            fallback_reason=None,
        )


__all__ = [
    "LearnedMetaRankerModel",
    "MetaRankingLinkageAdvisor",
]
