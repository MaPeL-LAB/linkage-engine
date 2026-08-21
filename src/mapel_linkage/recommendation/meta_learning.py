"""Family-level evidence and split-conformal model for the learned advisor."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from mapel_linkage.benchmarking.advisor_catalogue import AdvisorFamilyRole
from mapel_linkage.benchmarking.contracts import BenchmarkRunRecord, BenchmarkRunStatus
from mapel_linkage.benchmarking.registry import BenchmarkRegistry
from mapel_linkage.recommendation.distance import TaskMetaFeatureVector
from mapel_linkage.recommendation.utility import (
    REQUIRED_ADVISOR_RECIPE_TOKENS,
    AdvisorRecipeToken,
    benchmark_utility,
)

_MINIMUM_COMPLETE_REPLICATES = 5
_MODELLED_ROLES = frozenset({"meta_training", "conformal", "locked_evaluation"})


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def has_complete_required_evidence_grid(
    records: tuple[BenchmarkRunRecord, ...],
    *,
    expected_instance_ids: frozenset[str],
    recipe_token_by_digest: Mapping[str, AdvisorRecipeToken],
) -> bool:
    """Reject partial, failed, duplicate, mixed-provenance, or sparse evidence."""

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
    required_by_cell: dict[tuple[str, str], dict[AdvisorRecipeToken, BenchmarkRunStatus]] = {}
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

    required_tokens = frozenset(REQUIRED_ADVISOR_RECIPE_TOKENS)
    return all(
        set(required_by_cell.get((instance_id, replicate_id), {})) == required_tokens
        and all(
            status is BenchmarkRunStatus.SUCCESS
            for status in required_by_cell[(instance_id, replicate_id)].values()
        )
        for instance_id in expected_instance_ids
        for replicate_id in replicate_ids
    )


@dataclass(frozen=True, slots=True, repr=False)
class FamilyRecipeUtilityEvidence:
    """One family-level utility unit aggregated across instances and replicates."""

    family_id: str
    family_role: AdvisorFamilyRole
    recipe_token: AdvisorRecipeToken
    mean_utility: float
    run_count: int
    evidence_digest: str


def aggregate_family_recipe_evidence(
    *,
    registry: BenchmarkRegistry,
    records: tuple[BenchmarkRunRecord, ...],
    role_by_family: Mapping[str, AdvisorFamilyRole],
    recipe_token_by_digest: Mapping[str, AdvisorRecipeToken],
) -> tuple[FamilyRecipeUtilityEvidence, ...]:
    """Aggregate complete run evidence to family-by-recipe statistical units."""

    grouped: dict[tuple[str, AdvisorRecipeToken], list[tuple[float, str, str]]] = defaultdict(list)
    for record in records:
        role = role_by_family.get(record.family_id)
        token = recipe_token_by_digest.get(record.pipeline_recipe_digest)
        if role not in _MODELLED_ROLES or token is None:
            continue
        if record.status is not BenchmarkRunStatus.SUCCESS:
            raise ValueError("Required advisor evidence contains an unsuccessful run.")
        metrics = registry.load_metrics(record.run_id)
        if metrics is None or record.aggregate_metrics_digest != metrics.metrics_digest:
            raise ValueError("Required advisor metrics failed their integrity check.")
        grouped[(record.family_id, token)].append(
            (benchmark_utility(metrics), record.run_digest, metrics.metrics_digest)
        )

    expected_families = {
        family_id for family_id, role in role_by_family.items() if role in _MODELLED_ROLES
    }
    expected_keys = {
        (family_id, token)
        for family_id in expected_families
        for token in REQUIRED_ADVISOR_RECIPE_TOKENS
    }
    if set(grouped) != expected_keys:
        raise ValueError("Family-level advisor evidence is incomplete or outside the design.")

    result: list[FamilyRecipeUtilityEvidence] = []
    for family_id, token in sorted(grouped):
        values = grouped[(family_id, token)]
        utilities = [item[0] for item in values]
        if not utilities or not np.all(np.isfinite(utilities)):
            raise ValueError("Family-level advisor utility evidence is invalid.")
        result.append(
            FamilyRecipeUtilityEvidence(
                family_id=family_id,
                family_role=role_by_family[family_id],
                recipe_token=token,
                mean_utility=float(np.mean(utilities)),
                run_count=len(values),
                evidence_digest=_digest(
                    {
                        "family_id": family_id,
                        "recipe_token": token,
                        "run_evidence": sorted(
                            (run_digest, metric_digest) for _, run_digest, metric_digest in values
                        ),
                    }
                ),
            )
        )
    return tuple(result)


def continuous_family_features(vector: TaskMetaFeatureVector) -> list[float]:
    """Return the stable numeric family features used by the meta-regressor."""

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


def recipe_features(token: AdvisorRecipeToken) -> list[float]:
    """Return a fixed one-hot recipe representation with no learned ID lookup."""

    return [1.0 if token == item else 0.0 for item in REQUIRED_ADVISOR_RECIPE_TOKENS]


def family_recipe_features(vector: TaskMetaFeatureVector, token: AdvisorRecipeToken) -> list[float]:
    return continuous_family_features(vector) + recipe_features(token)


@dataclass
class LearnedMetaRankerModel:
    """Ridge meta-regressor fit by family with split-conformal intervals."""

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
        """Bind fitted and conformal state while excluding locked outcomes."""

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
        return _digest(payload)

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
        """Fit on training families and calibrate only on disjoint families."""

        if X.ndim != 2:
            raise ValueError("Meta-ranker training features must be a matrix.")
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
            or not np.all(np.isfinite(X))
            or not np.all(np.isfinite(X_conformal))
            or not np.all(np.isfinite(y))
            or not np.all(np.isfinite(y_conformal))
            or not math.isfinite(alpha)
            or alpha <= 0.0
            or not 0.5 <= coverage_level <= 0.99
        ):
            raise ValueError("Meta-ranker family-partition evidence is incomplete.")
        training_families = set(training_family_ids)
        conformal_families = set(conformal_family_ids)
        if training_families & conformal_families:
            raise ValueError("Meta-ranker training and conformal families must be disjoint.")
        if len(training_families) < 2 or np.ptp(y) <= 1e-12:
            raise ValueError("Meta-ranker training evidence lacks family or utility variation.")

        X_bias = np.hstack([np.ones((n_samples, 1)), X])
        reg_matrix = alpha * np.eye(n_features + 1)
        reg_matrix[0, 0] = 0.0
        try:
            fitted = np.linalg.solve(X_bias.T @ X_bias + reg_matrix, X_bias.T @ y)
        except np.linalg.LinAlgError:
            fitted = np.linalg.pinv(X_bias.T @ X_bias + reg_matrix) @ (X_bias.T @ y)

        self.intercept = float(fitted[0])
        self.weights = fitted[1:]
        conformal_predictions = np.clip(X_conformal @ self.weights + self.intercept, 0.0, 1.0)
        residuals = np.abs(y_conformal - conformal_predictions)
        conformal_count = len(residuals)
        quantile_index = min(
            conformal_count - 1,
            math.ceil((conformal_count + 1) * coverage_level) - 1,
        )
        self.conformal_residual_quantile = float(np.sort(residuals)[max(0, quantile_index)])
        self.coverage_level = coverage_level
        self.trained_run_count = n_samples
        self.trained_family_count = len(training_families)
        self.conformal_run_count = conformal_count
        self.conformal_family_count = len(conformal_families)
        self.family_split_digest = _digest(
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
        """Mechanically evaluate after fitting without changing model identity."""

        if (
            self.weights is None
            or X.ndim != 2
            or not locked_family_ids
            or len(locked_family_ids) != len(y)
            or X.shape[0] != len(y)
            or X.shape[1] != len(self.weights)
            or set(locked_family_ids) & prohibited_family_ids
            or not np.all(np.isfinite(X))
            or not np.all(np.isfinite(y))
        ):
            raise ValueError("Locked meta-ranker evaluation families are invalid.")
        predictions, _, _ = self.predict_utility(X)
        self.locked_evaluation_run_count = len(y)
        self.locked_evaluation_family_count = len(set(locked_family_ids))
        self.locked_mean_absolute_error = float(np.mean(np.abs(y - predictions)))

    def predict_utility(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict expected utility and bounded split-conformal intervals."""

        if (
            self.weights is None
            or X.ndim != 2
            or X.shape[1] != len(self.weights)
            or not np.all(np.isfinite(X))
        ):
            raise ValueError("The fitted meta-ranker cannot score this feature matrix.")
        predictions = np.clip(X @ self.weights + self.intercept, 0.0, 1.0)
        lower = np.clip(predictions - self.conformal_residual_quantile, 0.0, 1.0)
        upper = np.clip(predictions + self.conformal_residual_quantile, 0.0, 1.0)
        return predictions, lower, upper


__all__ = [
    "FamilyRecipeUtilityEvidence",
    "LearnedMetaRankerModel",
    "aggregate_family_recipe_evidence",
    "continuous_family_features",
    "family_recipe_features",
    "has_complete_required_evidence_grid",
    "recipe_features",
]
