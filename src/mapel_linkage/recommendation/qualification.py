"""Prospective family-held-out qualification for the advisory-only strategy stack."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from mapel_linkage.benchmarking.advisor_catalogue import (
    AdvisorCorpusReadinessManifest,
    AdvisorFamilyRole,
    BenchmarkShardPlan,
    advisor_v2_family_roles,
    build_advisor_corpus_design,
    build_advisor_corpus_readiness,
    build_advisor_v2_generator,
)
from mapel_linkage.benchmarking.contracts import BenchmarkRunRecord, BenchmarkRunStatus
from mapel_linkage.benchmarking.registry import BenchmarkRegistry, build_registry_snapshot
from mapel_linkage.benchmarking.runner import (
    BenchmarkPortfolioRunner,
    benchmark_replicate_seed,
    benchmark_run_id,
)
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.recommendation.distance import (
    MetaFeatureDistanceComputer,
    TaskMetaFeatureVector,
    extract_family_meta_features,
)
from mapel_linkage.recommendation.meta_learning import (
    FamilyRecipeUtilityEvidence,
    LearnedMetaRankerModel,
    aggregate_family_recipe_evidence,
    family_recipe_features,
    has_complete_required_evidence_grid,
)
from mapel_linkage.recommendation.utility import (
    ADVISOR_UTILITY_POLICY_DIGEST,
    REQUIRED_ADVISOR_RECIPE_TOKENS,
    AdvisorRecipeToken,
    recipe_token_by_digest,
)

Digest = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _metric(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError("Advisor qualification produced a non-finite aggregate metric.")
    return round(float(value), 12)


class AdvisorQualificationPolicy(BaseModel):
    """Immutable thresholds fixed before the locked family outcomes are inspected."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    policy_schema_version: Literal["1"] = "1"
    policy_id: Literal["advisor_family_qualification_v1"] = "advisor_family_qualification_v1"
    utility_policy_digest: Digest = ADVISOR_UTILITY_POLICY_DIGEST
    nearest_family_count: StrictInt = 3
    oracle_coverage_k: StrictInt = 2
    ridge_alpha: StrictFloat = 1.0
    conformal_coverage_level: StrictFloat = 0.90
    ood_distance_threshold: StrictFloat = 0.45
    minimum_stage2_regret_improvement: StrictFloat = 0.005
    minimum_stage3_regret_improvement: StrictFloat = 0.01
    minimum_top2_oracle_coverage: StrictFloat = 0.875
    minimum_locked_interval_coverage: StrictFloat = 0.80
    maximum_mean_interval_width: StrictFloat = 0.50
    minimum_selection_stability: StrictFloat = 0.80
    minimum_ood_detection_rate: StrictFloat = 0.75
    maximum_locked_false_abstention_rate: StrictFloat = 0.125
    maximum_learning_curve_tail_regret_range: StrictFloat = 0.02
    learning_curve_family_counts: tuple[StrictInt, ...] = (8, 16, 24, 32, 40)
    meta_training_family_count: StrictInt = 40
    conformal_family_count: StrictInt = 8
    locked_evaluation_family_count: StrictInt = 8
    ood_holdout_family_count: StrictInt = 8
    required_recipe_tokens: tuple[AdvisorRecipeToken, ...] = REQUIRED_ADVISOR_RECIPE_TOKENS
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    release_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_fixed_policy(self) -> Self:
        expected: dict[str, object] = {
            "utility_policy_digest": ADVISOR_UTILITY_POLICY_DIGEST,
            "nearest_family_count": 3,
            "oracle_coverage_k": 2,
            "ridge_alpha": 1.0,
            "conformal_coverage_level": 0.90,
            "ood_distance_threshold": 0.45,
            "minimum_stage2_regret_improvement": 0.005,
            "minimum_stage3_regret_improvement": 0.01,
            "minimum_top2_oracle_coverage": 0.875,
            "minimum_locked_interval_coverage": 0.80,
            "maximum_mean_interval_width": 0.50,
            "minimum_selection_stability": 0.80,
            "minimum_ood_detection_rate": 0.75,
            "maximum_locked_false_abstention_rate": 0.125,
            "maximum_learning_curve_tail_regret_range": 0.02,
            "learning_curve_family_counts": (8, 16, 24, 32, 40),
            "meta_training_family_count": 40,
            "conformal_family_count": 8,
            "locked_evaluation_family_count": 8,
            "ood_holdout_family_count": 8,
            "required_recipe_tokens": REQUIRED_ADVISOR_RECIPE_TOKENS,
        }
        if {name: getattr(self, name) for name in expected} != expected:
            raise ValueError("Advisor qualification policy thresholds are package-fixed.")
        return self

    @property
    def policy_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {**self.model_dump(mode="json"), "policy_digest": self.policy_digest}


class AdvisorQualificationApproval(BaseModel):
    """Human approval binding one locked evaluation to exact aggregate evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    approval_schema_version: Literal["1"] = "1"
    approval_reference: Annotated[
        StrictStr, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$", repr=False)
    ]
    human_approved: Literal[True]
    design_digest: Digest
    readiness_digest: Digest
    registry_snapshot_digest: Digest
    policy_digest: Digest
    locked_evaluation_authorized: Literal[True] = True
    synthetic_only: Literal[True] = True
    release_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    @property
    def approval_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class AdvisorMethodQualificationMetrics(BaseModel):
    """Aggregate locked-family ranking behavior for one advisory method."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    method: Literal["similarity", "meta_ranker"]
    locked_family_count: Literal[8] = 8
    mean_regret: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    median_regret: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    top1_oracle_coverage: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    top2_oracle_coverage: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    selection_stability: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]


class FixedRecipeBaselineMetrics(BaseModel):
    """Locked-family regret for a prespecified always-use recipe baseline."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    recipe_token: Literal["fellegi_sunter", "xgboost_classifier", "xgboost_ranker"]
    locked_family_count: Literal[8] = 8
    mean_regret: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    median_regret: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    oracle_coverage: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]


class ConformalQualificationMetrics(BaseModel):
    """Locked family-by-recipe coverage for the split-conformal interval."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    nominal_coverage: StrictFloat = 0.90
    locked_family_count: Literal[8] = 8
    locked_family_recipe_count: Literal[24] = 24
    empirical_coverage: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    mean_interval_width: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    mean_absolute_error: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def validate_nominal_coverage(self) -> Self:
        if self.nominal_coverage != 0.90:
            raise ValueError("The conformal qualification target is fixed at 90 percent.")
        return self


class OODQualificationMetrics(BaseModel):
    """Prospective true-mechanism OOD detection and abstention behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    ood_family_count: Literal[8] = 8
    locked_family_count: Literal[8] = 8
    detected_ood_family_count: Annotated[StrictInt, Field(ge=0, le=8)]
    falsely_abstained_locked_family_count: Annotated[StrictInt, Field(ge=0, le=8)]
    ood_detection_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    locked_false_abstention_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    balanced_abstention_accuracy: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]


class AdvisorLearningCurvePoint(BaseModel):
    """One prespecified nested training-family learning-curve point."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    training_family_count: StrictInt
    similarity_mean_regret: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    meta_ranker_mean_regret: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def validate_family_count(self) -> Self:
        if self.training_family_count not in {8, 16, 24, 32, 40}:
            raise ValueError("Advisor learning-curve family counts are package-fixed.")
        return self


class AdvisorQualificationReport(BaseModel):
    """Aggregate qualification result with no family IDs, rows, pairs, or local paths."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    qualification_schema_version: Literal["1"] = "1"
    qualification_status: Literal["qualified", "not_qualified"]
    design_digest: Digest
    readiness_digest: Digest
    registry_snapshot_digest: Digest
    evidence_digest: Digest
    evidence_provenance_digest: Digest
    policy_digest: Digest
    approval_digest: Digest
    family_split_digest: Digest
    meta_model_digest: Digest
    expected_run_count: Literal[9800] = 9800
    required_success_run_count: Literal[4200] = 4200
    retained_ineligible_run_count: Literal[5600] = 5600
    meta_training_family_count: Literal[40] = 40
    conformal_family_count: Literal[8] = 8
    locked_evaluation_family_count: Literal[8] = 8
    ood_holdout_family_count: Literal[8] = 8
    fixed_baselines: Annotated[
        tuple[FixedRecipeBaselineMetrics, ...], Field(min_length=3, max_length=3)
    ]
    stage2_similarity: AdvisorMethodQualificationMetrics
    stage3_meta_ranker: AdvisorMethodQualificationMetrics
    conformal: ConformalQualificationMetrics
    ood: OODQualificationMetrics
    learning_curve: Annotated[
        tuple[AdvisorLearningCurvePoint, ...], Field(min_length=5, max_length=5)
    ]
    stage2_regret_improvement_over_best_fixed: StrictFloat
    stage3_regret_improvement_over_stage2: StrictFloat
    learning_curve_tail_regret_range: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    stage2_qualified: StrictBool
    stage3_qualified: StrictBool
    ood_qualified: StrictBool
    conformal_qualified: StrictBool
    learning_curve_qualified: StrictBool
    fallback_to_similarity_required: StrictBool
    failed_gate_codes: tuple[StrictStr, ...]
    hard_constraint_violation_count: Literal[0] = 0
    locked_evaluation_accessed: Literal[True] = True
    locked_evaluation_used_for_fit: Literal[False] = False
    ood_evidence_used_for_fit: Literal[False] = False
    ood_evidence_used_for_interval_calibration: Literal[False] = False
    contains_record_values: Literal[False] = False
    contains_identifiers: Literal[False] = False
    contains_candidate_pairs: Literal[False] = False
    contains_local_paths: Literal[False] = False
    recommendation_authority: Literal["advisory_only"] = "advisory_only"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    automatic_promotion: Literal["prohibited"] = "prohibited"
    release_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_qualification(self) -> Self:
        policy = AdvisorQualificationPolicy()
        if self.policy_digest != policy.policy_digest:
            raise ValueError("Advisor qualification policy binding is not current.")
        if (
            tuple(item.recipe_token for item in self.fixed_baselines)
            != REQUIRED_ADVISOR_RECIPE_TOKENS
        ):
            raise ValueError("Fixed advisor baselines must retain their prespecified order.")
        if tuple(item.training_family_count for item in self.learning_curve) != (
            8,
            16,
            24,
            32,
            40,
        ):
            raise ValueError("Advisor learning-curve points must follow the fixed nested design.")
        if len(self.failed_gate_codes) != len(set(self.failed_gate_codes)):
            raise ValueError("Advisor qualification gate codes must be unique.")
        stage2_improvement = (
            min(item.mean_regret for item in self.fixed_baselines)
            - self.stage2_similarity.mean_regret
        )
        stage3_improvement = (
            self.stage2_similarity.mean_regret - self.stage3_meta_ranker.mean_regret
        )
        tail_range = abs(
            self.learning_curve[-1].meta_ranker_mean_regret
            - self.learning_curve[-2].meta_ranker_mean_regret
        )
        if (
            self.stage2_regret_improvement_over_best_fixed != _metric(stage2_improvement)
            or self.stage3_regret_improvement_over_stage2 != _metric(stage3_improvement)
            or self.learning_curve_tail_regret_range != _metric(tail_range)
        ):
            raise ValueError("Advisor qualification derived metrics are inconsistent.")
        expected_ood = (
            self.ood.ood_detection_rate >= policy.minimum_ood_detection_rate
            and self.ood.locked_false_abstention_rate <= policy.maximum_locked_false_abstention_rate
        )
        expected_conformal = (
            self.conformal.empirical_coverage >= policy.minimum_locked_interval_coverage
            and self.conformal.mean_interval_width <= policy.maximum_mean_interval_width
        )
        expected_learning_curve = (
            tail_range <= policy.maximum_learning_curve_tail_regret_range
            and self.learning_curve[-1].meta_ranker_mean_regret
            <= self.learning_curve[0].meta_ranker_mean_regret
        )
        expected_stage2 = (
            stage2_improvement >= policy.minimum_stage2_regret_improvement
            and self.stage2_similarity.top2_oracle_coverage >= policy.minimum_top2_oracle_coverage
            and self.stage2_similarity.selection_stability >= policy.minimum_selection_stability
            and expected_ood
        )
        expected_stage3 = (
            expected_stage2
            and stage3_improvement >= policy.minimum_stage3_regret_improvement
            and self.stage3_meta_ranker.top2_oracle_coverage >= policy.minimum_top2_oracle_coverage
            and self.stage3_meta_ranker.selection_stability >= policy.minimum_selection_stability
            and expected_conformal
            and expected_learning_curve
        )
        if (
            self.ood_qualified != expected_ood
            or self.conformal_qualified != expected_conformal
            or self.learning_curve_qualified != expected_learning_curve
            or self.stage2_qualified != expected_stage2
            or self.stage3_qualified != expected_stage3
        ):
            raise ValueError("Advisor qualification gate outcomes are inconsistent.")
        expected_failed = tuple(
            code
            for code, passed in (
                (
                    "stage2_regret_improvement",
                    stage2_improvement >= policy.minimum_stage2_regret_improvement,
                ),
                (
                    "stage2_top2_oracle_coverage",
                    self.stage2_similarity.top2_oracle_coverage
                    >= policy.minimum_top2_oracle_coverage,
                ),
                (
                    "stage2_selection_stability",
                    self.stage2_similarity.selection_stability
                    >= policy.minimum_selection_stability,
                ),
                (
                    "stage3_regret_improvement",
                    stage3_improvement >= policy.minimum_stage3_regret_improvement,
                ),
                (
                    "stage3_top2_oracle_coverage",
                    self.stage3_meta_ranker.top2_oracle_coverage
                    >= policy.minimum_top2_oracle_coverage,
                ),
                (
                    "stage3_selection_stability",
                    self.stage3_meta_ranker.selection_stability
                    >= policy.minimum_selection_stability,
                ),
                (
                    "locked_interval_coverage",
                    self.conformal.empirical_coverage >= policy.minimum_locked_interval_coverage,
                ),
                (
                    "mean_interval_width",
                    self.conformal.mean_interval_width <= policy.maximum_mean_interval_width,
                ),
                (
                    "ood_detection_rate",
                    self.ood.ood_detection_rate >= policy.minimum_ood_detection_rate,
                ),
                (
                    "locked_false_abstention_rate",
                    self.ood.locked_false_abstention_rate
                    <= policy.maximum_locked_false_abstention_rate,
                ),
                ("learning_curve_stability", expected_learning_curve),
            )
            if not passed
        )
        if self.failed_gate_codes != expected_failed:
            raise ValueError("Advisor qualification failed-gate evidence is inconsistent.")
        expected_status = (
            "qualified"
            if self.stage2_qualified
            and self.stage3_qualified
            and self.ood_qualified
            and self.conformal_qualified
            and self.learning_curve_qualified
            else "not_qualified"
        )
        if self.qualification_status != expected_status:
            raise ValueError("Advisor qualification status does not match its fail-closed gates.")
        if self.fallback_to_similarity_required != (not self.stage3_qualified):
            raise ValueError("Stage-3 fallback status does not match its qualification gate.")
        return self

    @property
    def report_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {
            **self.model_dump(mode="json"),
            "report_digest": self.report_digest,
        }


class AdvisorQualificationArtifact(BaseModel):
    """Canonical envelope binding the exact semantic qualification report."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    artifact_schema_version: Literal["1"] = "1"
    report: AdvisorQualificationReport
    report_digest: Digest
    contains_record_values: Literal[False] = False
    contains_identifiers: Literal[False] = False
    contains_candidate_pairs: Literal[False] = False
    contains_local_paths: Literal[False] = False
    operational_validity: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def validate_report_digest(self) -> Self:
        if self.report_digest != self.report.report_digest:
            raise ValueError("Advisor qualification report integrity verification failed.")
        return self

    @property
    def artifact_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def safe_summary(self) -> dict[str, object]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "artifact_digest": self.artifact_digest,
            "report": self.report.safe_summary(),
            "contains_record_values": self.contains_record_values,
            "contains_identifiers": self.contains_identifiers,
            "contains_candidate_pairs": self.contains_candidate_pairs,
            "contains_local_paths": self.contains_local_paths,
            "operational_validity": self.operational_validity,
        }


def _utility_lookup(
    evidence: tuple[FamilyRecipeUtilityEvidence, ...],
) -> dict[tuple[str, AdvisorRecipeToken], float]:
    result = {(item.family_id, item.recipe_token): item.mean_utility for item in evidence}
    if len(result) != len(evidence):
        raise ValueError("Family-by-recipe advisor evidence must be unique.")
    return result


def _role_families(
    role_by_family: Mapping[str, AdvisorFamilyRole], role: AdvisorFamilyRole
) -> tuple[str, ...]:
    return tuple(
        sorted(family_id for family_id, item_role in role_by_family.items() if item_role == role)
    )


def _similarity_predictions(
    *,
    target_family_ids: tuple[str, ...],
    training_family_ids: tuple[str, ...],
    family_vectors: Mapping[str, TaskMetaFeatureVector],
    utilities: Mapping[tuple[str, AdvisorRecipeToken], float],
    policy: AdvisorQualificationPolicy,
    distance_computer: MetaFeatureDistanceComputer,
) -> tuple[dict[str, dict[AdvisorRecipeToken, float]], dict[str, float]]:
    training_vectors = {family_id: family_vectors[family_id] for family_id in training_family_ids}
    predictions: dict[str, dict[AdvisorRecipeToken, float]] = {}
    minimum_distances: dict[str, float] = {}
    for family_id in target_family_ids:
        nearest = distance_computer.find_nearest_families(
            family_vectors[family_id],
            training_vectors,
            k=policy.nearest_family_count,
        )
        if len(nearest) != policy.nearest_family_count:
            raise ValueError("Similarity qualification lacks the prespecified neighbour count.")
        predictions[family_id] = {
            token: float(np.mean([utilities[(near_family, token)] for near_family, _ in nearest]))
            for token in REQUIRED_ADVISOR_RECIPE_TOKENS
        }
        minimum_distances[family_id] = float(nearest[0][1])
    return predictions, minimum_distances


def _fit_meta_model(
    *,
    training_family_ids: tuple[str, ...],
    conformal_family_ids: tuple[str, ...],
    target_family_ids: tuple[str, ...],
    family_vectors: Mapping[str, TaskMetaFeatureVector],
    utilities: Mapping[tuple[str, AdvisorRecipeToken], float],
    policy: AdvisorQualificationPolicy,
    evaluate_locked: bool,
    prohibited_family_ids: frozenset[str],
) -> tuple[
    LearnedMetaRankerModel,
    dict[str, dict[AdvisorRecipeToken, tuple[float, float, float]]],
]:
    training_rows = tuple(
        (family_id, token)
        for family_id in training_family_ids
        for token in REQUIRED_ADVISOR_RECIPE_TOKENS
    )
    conformal_rows = tuple(
        (family_id, token)
        for family_id in conformal_family_ids
        for token in REQUIRED_ADVISOR_RECIPE_TOKENS
    )
    target_rows = tuple(
        (family_id, token)
        for family_id in target_family_ids
        for token in REQUIRED_ADVISOR_RECIPE_TOKENS
    )
    X_train = np.asarray(
        [
            family_recipe_features(family_vectors[family_id], token)
            for family_id, token in training_rows
        ],
        dtype=np.float64,
    )
    y_train = np.asarray([utilities[item] for item in training_rows], dtype=np.float64)
    X_conformal = np.asarray(
        [
            family_recipe_features(family_vectors[family_id], token)
            for family_id, token in conformal_rows
        ],
        dtype=np.float64,
    )
    y_conformal = np.asarray([utilities[item] for item in conformal_rows], dtype=np.float64)
    X_target = np.asarray(
        [
            family_recipe_features(family_vectors[family_id], token)
            for family_id, token in target_rows
        ],
        dtype=np.float64,
    )
    y_target = np.asarray([utilities[item] for item in target_rows], dtype=np.float64)

    model = LearnedMetaRankerModel()
    model.fit(
        X_train,
        y_train,
        X_conformal=X_conformal,
        y_conformal=y_conformal,
        training_family_ids=tuple(family_id for family_id, _ in training_rows),
        conformal_family_ids=tuple(family_id for family_id, _ in conformal_rows),
        alpha=policy.ridge_alpha,
        coverage_level=policy.conformal_coverage_level,
    )
    if evaluate_locked:
        model.evaluate_locked(
            X_target,
            y_target,
            locked_family_ids=tuple(family_id for family_id, _ in target_rows),
            prohibited_family_ids=prohibited_family_ids,
        )
    predicted, lower, upper = model.predict_utility(X_target)
    result: dict[str, dict[AdvisorRecipeToken, tuple[float, float, float]]] = {
        family_id: {} for family_id in target_family_ids
    }
    for (family_id, token), pred, low, high in zip(
        target_rows, predicted, lower, upper, strict=True
    ):
        result[family_id][token] = (float(pred), float(low), float(high))
    return model, result


def _method_metrics(
    *,
    method: Literal["similarity", "meta_ranker"],
    family_ids: tuple[str, ...],
    predicted_utilities: Mapping[str, Mapping[AdvisorRecipeToken, float]],
    actual_utilities: Mapping[tuple[str, AdvisorRecipeToken], float],
    selection_stability: float,
) -> tuple[AdvisorMethodQualificationMetrics, dict[str, AdvisorRecipeToken]]:
    regrets: list[float] = []
    top1 = 0
    top2 = 0
    selections: dict[str, AdvisorRecipeToken] = {}
    for family_id in family_ids:
        ranked = sorted(
            REQUIRED_ADVISOR_RECIPE_TOKENS,
            key=lambda token: (-predicted_utilities[family_id][token], token),
        )
        oracle_utility = max(
            actual_utilities[(family_id, token)] for token in REQUIRED_ADVISOR_RECIPE_TOKENS
        )
        oracle_tokens = {
            token
            for token in REQUIRED_ADVISOR_RECIPE_TOKENS
            if abs(actual_utilities[(family_id, token)] - oracle_utility) <= 1e-12
        }
        selected = ranked[0]
        selections[family_id] = selected
        regrets.append(oracle_utility - actual_utilities[(family_id, selected)])
        top1 += selected in oracle_tokens
        top2 += bool(set(ranked[:2]) & oracle_tokens)
    return (
        AdvisorMethodQualificationMetrics(
            method=method,
            mean_regret=_metric(float(np.mean(regrets))),
            median_regret=_metric(float(statistics.median(regrets))),
            top1_oracle_coverage=_metric(top1 / len(family_ids)),
            top2_oracle_coverage=_metric(top2 / len(family_ids)),
            selection_stability=_metric(selection_stability),
        ),
        selections,
    )


def _selection_agreement(
    baseline: Mapping[str, AdvisorRecipeToken],
    alternatives: tuple[Mapping[str, AdvisorRecipeToken], ...],
) -> float:
    comparisons = len(baseline) * len(alternatives)
    if comparisons == 0:
        raise ValueError("Advisor stability requires non-empty perturbation evidence.")
    agreed = sum(
        alternative[family_id] == token
        for family_id, token in baseline.items()
        for alternative in alternatives
    )
    return agreed / comparisons


def _fixed_baselines(
    *,
    family_ids: tuple[str, ...],
    utilities: Mapping[tuple[str, AdvisorRecipeToken], float],
) -> tuple[FixedRecipeBaselineMetrics, ...]:
    result: list[FixedRecipeBaselineMetrics] = []
    for token in REQUIRED_ADVISOR_RECIPE_TOKENS:
        regrets: list[float] = []
        oracle_count = 0
        for family_id in family_ids:
            oracle = max(utilities[(family_id, item)] for item in REQUIRED_ADVISOR_RECIPE_TOKENS)
            selected = utilities[(family_id, token)]
            regrets.append(oracle - selected)
            oracle_count += abs(oracle - selected) <= 1e-12
        result.append(
            FixedRecipeBaselineMetrics(
                recipe_token=token,
                mean_regret=_metric(float(np.mean(regrets))),
                median_regret=_metric(float(statistics.median(regrets))),
                oracle_coverage=_metric(oracle_count / len(family_ids)),
            )
        )
    return tuple(result)


def _deterministic_training_order(
    family_ids: tuple[str, ...], policy_digest: str
) -> tuple[str, ...]:
    return tuple(
        sorted(
            family_ids,
            key=lambda family_id: hashlib.sha256(
                f"{policy_digest}:{family_id}".encode()
            ).hexdigest(),
        )
    )


def _evaluate_complete_family_evidence(
    *,
    evidence: tuple[FamilyRecipeUtilityEvidence, ...],
    role_by_family: Mapping[str, AdvisorFamilyRole],
    family_vectors: Mapping[str, TaskMetaFeatureVector],
    policy: AdvisorQualificationPolicy,
) -> tuple[
    tuple[FixedRecipeBaselineMetrics, ...],
    AdvisorMethodQualificationMetrics,
    AdvisorMethodQualificationMetrics,
    ConformalQualificationMetrics,
    OODQualificationMetrics,
    tuple[AdvisorLearningCurvePoint, ...],
    LearnedMetaRankerModel,
]:
    training_families = _role_families(role_by_family, "meta_training")
    conformal_families = _role_families(role_by_family, "conformal")
    locked_families = _role_families(role_by_family, "locked_evaluation")
    ood_families = _role_families(role_by_family, "ood_holdout")
    if tuple(map(len, (training_families, conformal_families, locked_families, ood_families))) != (
        policy.meta_training_family_count,
        policy.conformal_family_count,
        policy.locked_evaluation_family_count,
        policy.ood_holdout_family_count,
    ):
        raise ValueError("Advisor family roles do not match the prospective qualification policy.")
    if set(role_by_family) != set(family_vectors):
        raise ValueError("Advisor family vectors do not exactly cover the prospective design.")
    utilities = _utility_lookup(evidence)
    expected_utility_keys = {
        (family_id, token)
        for family_id in (*training_families, *conformal_families, *locked_families)
        for token in REQUIRED_ADVISOR_RECIPE_TOKENS
    }
    if set(utilities) != expected_utility_keys:
        raise ValueError("Advisor family utility evidence crosses or omits a protected role.")

    distance_computer = MetaFeatureDistanceComputer()
    similarity_predictions, _ = _similarity_predictions(
        target_family_ids=locked_families,
        training_family_ids=training_families,
        family_vectors=family_vectors,
        utilities=utilities,
        policy=policy,
        distance_computer=distance_computer,
    )
    preliminary_similarity, full_similarity_selections = _method_metrics(
        method="similarity",
        family_ids=locked_families,
        predicted_utilities=similarity_predictions,
        actual_utilities=utilities,
        selection_stability=1.0,
    )

    prohibited = frozenset((*training_families, *conformal_families, *ood_families))
    meta_model, meta_intervals = _fit_meta_model(
        training_family_ids=training_families,
        conformal_family_ids=conformal_families,
        target_family_ids=locked_families,
        family_vectors=family_vectors,
        utilities=utilities,
        policy=policy,
        evaluate_locked=True,
        prohibited_family_ids=prohibited,
    )
    meta_predictions = {
        family_id: {token: values[0] for token, values in token_values.items()}
        for family_id, token_values in meta_intervals.items()
    }
    preliminary_meta, full_meta_selections = _method_metrics(
        method="meta_ranker",
        family_ids=locked_families,
        predicted_utilities=meta_predictions,
        actual_utilities=utilities,
        selection_stability=1.0,
    )

    similarity_perturbations: list[Mapping[str, AdvisorRecipeToken]] = []
    meta_perturbations: list[Mapping[str, AdvisorRecipeToken]] = []
    for omitted_family in training_families:
        subset = tuple(item for item in training_families if item != omitted_family)
        subset_similarity, _ = _similarity_predictions(
            target_family_ids=locked_families,
            training_family_ids=subset,
            family_vectors=family_vectors,
            utilities=utilities,
            policy=policy,
            distance_computer=distance_computer,
        )
        _, subset_similarity_selections = _method_metrics(
            method="similarity",
            family_ids=locked_families,
            predicted_utilities=subset_similarity,
            actual_utilities=utilities,
            selection_stability=1.0,
        )
        similarity_perturbations.append(subset_similarity_selections)
        _, subset_meta_intervals = _fit_meta_model(
            training_family_ids=subset,
            conformal_family_ids=conformal_families,
            target_family_ids=locked_families,
            family_vectors=family_vectors,
            utilities=utilities,
            policy=policy,
            evaluate_locked=False,
            prohibited_family_ids=prohibited,
        )
        subset_meta_predictions = {
            family_id: {token: values[0] for token, values in token_values.items()}
            for family_id, token_values in subset_meta_intervals.items()
        }
        _, subset_meta_selections = _method_metrics(
            method="meta_ranker",
            family_ids=locked_families,
            predicted_utilities=subset_meta_predictions,
            actual_utilities=utilities,
            selection_stability=1.0,
        )
        meta_perturbations.append(subset_meta_selections)

    similarity_metrics = preliminary_similarity.model_copy(
        update={
            "selection_stability": _metric(
                _selection_agreement(full_similarity_selections, tuple(similarity_perturbations))
            )
        }
    )
    meta_metrics = preliminary_meta.model_copy(
        update={
            "selection_stability": _metric(
                _selection_agreement(full_meta_selections, tuple(meta_perturbations))
            )
        }
    )

    coverage_values: list[bool] = []
    interval_widths: list[float] = []
    absolute_errors: list[float] = []
    for family_id in locked_families:
        for token in REQUIRED_ADVISOR_RECIPE_TOKENS:
            predicted, lower, upper = meta_intervals[family_id][token]
            actual = utilities[(family_id, token)]
            coverage_values.append(lower <= actual <= upper)
            interval_widths.append(upper - lower)
            absolute_errors.append(abs(actual - predicted))
    conformal = ConformalQualificationMetrics(
        empirical_coverage=_metric(sum(coverage_values) / len(coverage_values)),
        mean_interval_width=_metric(float(np.mean(interval_widths))),
        mean_absolute_error=_metric(float(np.mean(absolute_errors))),
    )

    training_vectors = {family_id: family_vectors[family_id] for family_id in training_families}
    locked_distances = {
        family_id: distance_computer.find_nearest_families(
            family_vectors[family_id], training_vectors, k=1
        )[0][1]
        for family_id in locked_families
    }
    ood_distances = {
        family_id: distance_computer.find_nearest_families(
            family_vectors[family_id], training_vectors, k=1
        )[0][1]
        for family_id in ood_families
    }
    detected = sum(value > policy.ood_distance_threshold for value in ood_distances.values())
    false_abstentions = sum(
        value > policy.ood_distance_threshold for value in locked_distances.values()
    )
    detection_rate = detected / len(ood_families)
    false_rate = false_abstentions / len(locked_families)
    ood = OODQualificationMetrics(
        detected_ood_family_count=detected,
        falsely_abstained_locked_family_count=false_abstentions,
        ood_detection_rate=_metric(detection_rate),
        locked_false_abstention_rate=_metric(false_rate),
        balanced_abstention_accuracy=_metric((detection_rate + (1.0 - false_rate)) / 2.0),
    )

    order = _deterministic_training_order(training_families, policy.policy_digest)
    curve: list[AdvisorLearningCurvePoint] = []
    for family_count in policy.learning_curve_family_counts:
        subset = order[:family_count]
        curve_similarity, _ = _similarity_predictions(
            target_family_ids=locked_families,
            training_family_ids=subset,
            family_vectors=family_vectors,
            utilities=utilities,
            policy=policy,
            distance_computer=distance_computer,
        )
        curve_similarity_metrics, _ = _method_metrics(
            method="similarity",
            family_ids=locked_families,
            predicted_utilities=curve_similarity,
            actual_utilities=utilities,
            selection_stability=1.0,
        )
        _, curve_meta_intervals = _fit_meta_model(
            training_family_ids=subset,
            conformal_family_ids=conformal_families,
            target_family_ids=locked_families,
            family_vectors=family_vectors,
            utilities=utilities,
            policy=policy,
            evaluate_locked=False,
            prohibited_family_ids=prohibited,
        )
        curve_meta_predictions = {
            family_id: {token: values[0] for token, values in token_values.items()}
            for family_id, token_values in curve_meta_intervals.items()
        }
        curve_meta_metrics, _ = _method_metrics(
            method="meta_ranker",
            family_ids=locked_families,
            predicted_utilities=curve_meta_predictions,
            actual_utilities=utilities,
            selection_stability=1.0,
        )
        curve.append(
            AdvisorLearningCurvePoint(
                training_family_count=family_count,
                similarity_mean_regret=curve_similarity_metrics.mean_regret,
                meta_ranker_mean_regret=curve_meta_metrics.mean_regret,
            )
        )

    return (
        _fixed_baselines(family_ids=locked_families, utilities=utilities),
        similarity_metrics,
        meta_metrics,
        conformal,
        ood,
        tuple(curve),
        meta_model,
    )


def _validate_registry_evidence(
    *,
    registry: BenchmarkRegistry,
    shard_plan: BenchmarkShardPlan,
    replicates: int,
    base_seed: int,
    runner: BenchmarkPortfolioRunner,
) -> tuple[
    tuple[BenchmarkRunRecord, ...],
    AdvisorCorpusReadinessManifest,
    str,
    str,
]:
    design = build_advisor_corpus_design()
    if shard_plan.design_digest != design.design_digest or replicates != 5 or base_seed != 20260816:
        raise ValueError("Advisor qualification inputs do not match the prospective corpus design.")
    generator = build_advisor_v2_generator()
    expected_families = {
        family.family_id: family
        for family in generator.list_families()
        if family.family_id.startswith("family.advisor_v2.")
    }
    expected_instances = {
        instance.instance_id: instance
        for instance in generator.list_instances()
        if instance.family_id in expected_families
    }
    retained_families = {family.family_id: family for family in registry.list_families()}
    retained_instances = {instance.instance_id: instance for instance in registry.list_instances()}
    if retained_families != expected_families or retained_instances != expected_instances:
        raise ValueError("Advisor registry manifests do not match the versioned v2 catalogue.")

    recipes = runner.list_recipes()
    token_by_digest = recipe_token_by_digest(runner)
    expected: dict[str, tuple[str, str, str, str, int, AdvisorRecipeToken | None]] = {}
    for shard in shard_plan.shards:
        for instance_id in shard.instance_ids:
            family_id = expected_instances[instance_id].family_id
            for replicate_number in range(replicates):
                replicate_id = f"replicate.{replicate_number:07d}"
                seed = benchmark_replicate_seed(
                    instance_id=instance_id,
                    replicate_number=replicate_number,
                    base_seed=base_seed,
                )
                for recipe in recipes:
                    expected[
                        benchmark_run_id(
                            instance_id=instance_id,
                            recipe_id=recipe.recipe_id,
                            replicate_id=replicate_id,
                        )
                    ] = (
                        family_id,
                        instance_id,
                        recipe.recipe_digest,
                        replicate_id,
                        seed,
                        token_by_digest.get(recipe.recipe_digest),
                    )

    records = registry.list_run_records()
    if len(records) != 9800 or {record.run_id for record in records} != set(expected):
        raise ValueError("Advisor qualification requires the exact complete 9,800-run registry.")
    provenance = {
        (record.engine_commit, record.dependency_lock_digest, record.environment_digest)
        for record in records
    }
    if len(provenance) != 1:
        raise ValueError("Advisor qualification evidence has mixed engine provenance.")
    successful_required = 0
    retained_ineligible = 0
    evidence_bindings: list[tuple[str, str]] = []
    successful_tokens_by_cell: dict[tuple[str, str], set[AdvisorRecipeToken]] = {}
    successful_tokens_by_family: dict[str, set[AdvisorRecipeToken]] = {}
    for record in records:
        family_id, instance_id, recipe_digest, replicate_id, seed, token = expected[record.run_id]
        if (
            record.family_id != family_id
            or record.instance_id != instance_id
            or record.pipeline_recipe_digest != recipe_digest
            or record.replicate_id != replicate_id
            or record.random_seed != seed
        ):
            raise ValueError("Advisor run evidence conflicts with the prospective execution grid.")
        if token is not None:
            if (
                record.status is not BenchmarkRunStatus.SUCCESS
                or record.aggregate_metrics_digest is None
            ):
                raise ValueError("A required advisor adapter lacks successful aggregate evidence.")
            successful_required += 1
            successful_tokens_by_cell.setdefault((instance_id, replicate_id), set()).add(token)
            successful_tokens_by_family.setdefault(family_id, set()).add(token)
            evidence_bindings.append((record.run_digest, record.aggregate_metrics_digest))
        else:
            failure = registry.load_failure_record(record.run_id)
            if record.status is not BenchmarkRunStatus.INELIGIBLE or failure is None:
                raise ValueError(
                    "Non-qualified portfolio adapters must remain explicitly ineligible."
                )
            retained_ineligible += 1
            evidence_bindings.append((record.run_digest, failure.failure_digest))

    expected_cells = {
        (instance_id, f"replicate.{replicate_number:07d}")
        for instance_id in expected_instances
        for replicate_number in range(replicates)
    }
    required_tokens = set(REQUIRED_ADVISOR_RECIPE_TOKENS)
    successful_cells = sum(
        successful_tokens_by_cell.get(cell) == required_tokens for cell in expected_cells
    )
    successful_families = sum(
        successful_tokens_by_family.get(family_id) == required_tokens
        for family_id in expected_families
    )
    if successful_required != 4200 or retained_ineligible != 5600:
        raise ValueError("Advisor retained status counts do not match the approved portfolio.")
    initial = build_advisor_corpus_readiness(
        adapter_statuses=runner.adapter_statuses(), planned_replicates_per_instance=replicates
    )
    readiness = AdvisorCorpusReadinessManifest.model_validate(
        {
            **initial.model_dump(mode="json"),
            "execution_status": "complete",
            "expected_run_count": len(expected),
            "completed_run_count": len(records),
            "successful_overlap_family_count": successful_families,
            "successful_evidence_cell_count": successful_cells,
            "successful_required_adapter_run_count": successful_required,
            "failed_required_adapter_run_count": 0,
            "missing_required_adapter_run_count": 0,
            "advisor_evidence_ready": True,
        }
    )
    if not readiness.advisor_evidence_ready or not has_complete_required_evidence_grid(
        records,
        expected_instance_ids=frozenset(expected_instances),
        recipe_token_by_digest=token_by_digest,
    ):
        raise ValueError("Advisor evidence readiness failed closed.")
    provenance_digest = _digest(sorted(provenance))
    evidence_digest = _digest(sorted(evidence_bindings))
    return records, readiness, provenance_digest, evidence_digest


def qualify_advisor_registry(
    *,
    registry: BenchmarkRegistry,
    shard_plan: BenchmarkShardPlan,
    approval_reference: str,
    policy: AdvisorQualificationPolicy | None = None,
    replicates: int = 5,
    base_seed: int = 20260816,
    runner: BenchmarkPortfolioRunner | None = None,
) -> AdvisorQualificationArtifact:
    """Run one approved, prospective, aggregate-only qualification evaluation."""

    active_policy = policy or AdvisorQualificationPolicy()
    run_engine = runner or BenchmarkPortfolioRunner()
    design = build_advisor_corpus_design()
    records, readiness, provenance_digest, evidence_digest = _validate_registry_evidence(
        registry=registry,
        shard_plan=shard_plan,
        replicates=replicates,
        base_seed=base_seed,
        runner=run_engine,
    )
    snapshot = build_registry_snapshot(
        snapshot_id="snapshot.advisor_v2_qualification_v1", records=records
    )
    approval = AdvisorQualificationApproval(
        approval_reference=approval_reference,
        human_approved=True,
        design_digest=design.design_digest,
        readiness_digest=readiness.readiness_digest,
        registry_snapshot_digest=snapshot.registry_digest,
        policy_digest=active_policy.policy_digest,
    )
    generator = build_advisor_v2_generator()
    role_by_family = dict(advisor_v2_family_roles())
    family_vectors = {
        family_id: vector
        for family_id, vector in extract_family_meta_features(generator).items()
        if family_id in role_by_family
    }
    family_evidence = aggregate_family_recipe_evidence(
        registry=registry,
        records=records,
        role_by_family=role_by_family,
        recipe_token_by_digest=recipe_token_by_digest(run_engine),
    )
    (
        fixed_baselines,
        stage2,
        stage3,
        conformal,
        ood,
        learning_curve,
        meta_model,
    ) = _evaluate_complete_family_evidence(
        evidence=family_evidence,
        role_by_family=role_by_family,
        family_vectors=family_vectors,
        policy=active_policy,
    )

    best_fixed_regret = min(item.mean_regret for item in fixed_baselines)
    stage2_improvement = best_fixed_regret - stage2.mean_regret
    stage3_improvement = stage2.mean_regret - stage3.mean_regret
    tail_range = abs(
        learning_curve[-1].meta_ranker_mean_regret - learning_curve[-2].meta_ranker_mean_regret
    )
    ood_qualified = (
        ood.ood_detection_rate >= active_policy.minimum_ood_detection_rate
        and ood.locked_false_abstention_rate <= active_policy.maximum_locked_false_abstention_rate
    )
    conformal_qualified = (
        conformal.empirical_coverage >= active_policy.minimum_locked_interval_coverage
        and conformal.mean_interval_width <= active_policy.maximum_mean_interval_width
    )
    learning_curve_qualified = (
        tail_range <= active_policy.maximum_learning_curve_tail_regret_range
        and learning_curve[-1].meta_ranker_mean_regret <= learning_curve[0].meta_ranker_mean_regret
    )
    stage2_qualified = (
        stage2_improvement >= active_policy.minimum_stage2_regret_improvement
        and stage2.top2_oracle_coverage >= active_policy.minimum_top2_oracle_coverage
        and stage2.selection_stability >= active_policy.minimum_selection_stability
        and ood_qualified
    )
    stage3_qualified = (
        stage2_qualified
        and stage3_improvement >= active_policy.minimum_stage3_regret_improvement
        and stage3.top2_oracle_coverage >= active_policy.minimum_top2_oracle_coverage
        and stage3.selection_stability >= active_policy.minimum_selection_stability
        and conformal_qualified
        and learning_curve_qualified
    )
    failed_gates: list[str] = []
    gate_values = {
        "stage2_regret_improvement": stage2_improvement
        >= active_policy.minimum_stage2_regret_improvement,
        "stage2_top2_oracle_coverage": stage2.top2_oracle_coverage
        >= active_policy.minimum_top2_oracle_coverage,
        "stage2_selection_stability": stage2.selection_stability
        >= active_policy.minimum_selection_stability,
        "stage3_regret_improvement": stage3_improvement
        >= active_policy.minimum_stage3_regret_improvement,
        "stage3_top2_oracle_coverage": stage3.top2_oracle_coverage
        >= active_policy.minimum_top2_oracle_coverage,
        "stage3_selection_stability": stage3.selection_stability
        >= active_policy.minimum_selection_stability,
        "locked_interval_coverage": conformal.empirical_coverage
        >= active_policy.minimum_locked_interval_coverage,
        "mean_interval_width": conformal.mean_interval_width
        <= active_policy.maximum_mean_interval_width,
        "ood_detection_rate": ood.ood_detection_rate >= active_policy.minimum_ood_detection_rate,
        "locked_false_abstention_rate": ood.locked_false_abstention_rate
        <= active_policy.maximum_locked_false_abstention_rate,
        "learning_curve_stability": learning_curve_qualified,
    }
    failed_gates.extend(code for code, passed in gate_values.items() if not passed)
    status: Literal["qualified", "not_qualified"] = (
        "qualified" if stage2_qualified and stage3_qualified else "not_qualified"
    )
    if meta_model.family_split_digest is None:
        raise ValueError("The qualified meta-model lacks its protected family split binding.")
    report = AdvisorQualificationReport(
        qualification_status=status,
        design_digest=design.design_digest,
        readiness_digest=readiness.readiness_digest,
        registry_snapshot_digest=snapshot.registry_digest,
        evidence_digest=evidence_digest,
        evidence_provenance_digest=provenance_digest,
        policy_digest=active_policy.policy_digest,
        approval_digest=approval.approval_digest,
        family_split_digest=meta_model.family_split_digest,
        meta_model_digest=meta_model.model_digest,
        fixed_baselines=fixed_baselines,
        stage2_similarity=stage2,
        stage3_meta_ranker=stage3,
        conformal=conformal,
        ood=ood,
        learning_curve=learning_curve,
        stage2_regret_improvement_over_best_fixed=_metric(stage2_improvement),
        stage3_regret_improvement_over_stage2=_metric(stage3_improvement),
        learning_curve_tail_regret_range=_metric(tail_range),
        stage2_qualified=stage2_qualified,
        stage3_qualified=stage3_qualified,
        ood_qualified=ood_qualified,
        conformal_qualified=conformal_qualified,
        learning_curve_qualified=learning_curve_qualified,
        fallback_to_similarity_required=not stage3_qualified,
        failed_gate_codes=tuple(failed_gates),
    )
    return AdvisorQualificationArtifact(report=report, report_digest=report.report_digest)


def serialize_advisor_qualification_artifact(artifact: AdvisorQualificationArtifact) -> str:
    """Return the exact canonical JSON encoding used at the artifact boundary."""

    return json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def deserialize_advisor_qualification_artifact(text: str) -> AdvisorQualificationArtifact:
    """Reject non-canonical, oversized, unknown, or tampered qualification artifacts."""

    if len(text.encode("utf-8")) > _MAX_ARTIFACT_BYTES:
        raise ValueError("The advisor qualification artifact exceeds its aggregate size limit.")
    artifact = AdvisorQualificationArtifact.model_validate_json(text)
    if text != serialize_advisor_qualification_artifact(artifact):
        raise ValueError("The advisor qualification artifact is not canonical JSON.")
    return artifact


def _reject_existing_symlink_components(path: Path) -> None:
    lexical = path.absolute()
    for component in (*reversed(lexical.parents), lexical):
        if component.is_symlink():
            raise ValueError("Advisor qualification artifact paths cannot use symbolic links.")


def write_advisor_qualification_artifact(
    path: Path, artifact: AdvisorQualificationArtifact
) -> None:
    """Write once, or accept an exact idempotent aggregate-artifact replay."""

    try:
        _reject_existing_symlink_components(path)
        canonical = serialize_advisor_qualification_artifact(artifact)
        if path.exists():
            if not path.is_file() or path.read_text(encoding="utf-8") != canonical:
                raise FileExistsError("A conflicting advisor qualification artifact exists.")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, canonical)
    except (FileExistsError, ValueError):
        raise
    except OSError:
        raise ValueError(
            "The advisor qualification artifact could not be written safely."
        ) from None


def load_advisor_qualification_artifact(path: Path) -> AdvisorQualificationArtifact:
    """Load and strictly verify one aggregate qualification artifact."""

    try:
        _reject_existing_symlink_components(path)
        if not path.is_file() or path.stat().st_size > _MAX_ARTIFACT_BYTES:
            raise ValueError("The advisor qualification artifact path is invalid.")
        text = path.read_text(encoding="utf-8")
    except ValueError:
        raise
    except (OSError, UnicodeError):
        raise ValueError("The advisor qualification artifact could not be read safely.") from None
    return deserialize_advisor_qualification_artifact(text)


__all__ = [
    "AdvisorLearningCurvePoint",
    "AdvisorMethodQualificationMetrics",
    "AdvisorQualificationApproval",
    "AdvisorQualificationArtifact",
    "AdvisorQualificationPolicy",
    "AdvisorQualificationReport",
    "ConformalQualificationMetrics",
    "FixedRecipeBaselineMetrics",
    "OODQualificationMetrics",
    "deserialize_advisor_qualification_artifact",
    "load_advisor_qualification_artifact",
    "qualify_advisor_registry",
    "serialize_advisor_qualification_artifact",
    "write_advisor_qualification_artifact",
]
