"""Stage-3 Learned Meta-Ranking Strategy Advisor with conformal uncertainty estimation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from mapel_linkage.benchmarking.advisor_catalogue import (
    advisor_v2_family_roles,
    build_advisor_corpus_design,
    build_advisor_v2_generator,
)
from mapel_linkage.benchmarking.contracts import BenchmarkRunRecord, BenchmarkRunStatus
from mapel_linkage.benchmarking.generator import (
    BenchmarkScenarioGenerator,
)
from mapel_linkage.benchmarking.registry import (
    BenchmarkRegistry,
    build_registry_snapshot,
)
from mapel_linkage.configuration.compiler import ExecutionPlan
from mapel_linkage.domain.errors import AdvisorError
from mapel_linkage.profiling import PreflightTaskProfile, build_preflight_task_profile
from mapel_linkage.recommendation.contracts import (
    MetaRankingAdvisoryReport,
    PipelineRecommendation,
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
from mapel_linkage.recommendation.meta_learning import (
    LearnedMetaRankerModel,
    aggregate_family_recipe_evidence,
    family_recipe_features,
    has_complete_required_evidence_grid,
)
from mapel_linkage.recommendation.qualification import (
    AdvisorQualificationArtifact,
    AdvisorQualificationPolicy,
)
from mapel_linkage.recommendation.similarity_advisor import (
    SimilarityLinkageAdvisor,
)
from mapel_linkage.recommendation.utility import (
    ADVISOR_UTILITY_POLICY_DIGEST,
    candidate_recipe_token,
    recipe_token_by_digest,
)

_has_complete_required_evidence_grid = has_complete_required_evidence_grid


def _policy_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rank_evidence_backed_shortlist(
    candidates: tuple[StructuralPipelineCandidate, ...],
    predicted_utilities: Mapping[str, float],
) -> tuple[StructuralPipelineCandidate, ...]:
    """Order supported candidates by utility and preserve unsupported candidates."""

    supported = sorted(
        (candidate for candidate in candidates if candidate.candidate_id in predicted_utilities),
        key=lambda candidate: (
            -predicted_utilities[candidate.candidate_id],
            candidate.candidate_id,
        ),
    )
    supported_ids = {candidate.candidate_id for candidate in supported}
    return tuple(supported) + tuple(
        candidate for candidate in candidates if candidate.candidate_id not in supported_ids
    )


class MetaRankingLinkageAdvisor:
    """Stage-3 Learned Meta-Ranking Advisor with Conformal Uncertainty Bounds."""

    def __init__(
        self,
        registry: BenchmarkRegistry | None = None,
        *,
        qualification_artifact: AdvisorQualificationArtifact | None = None,
        distance_computer: MetaFeatureDistanceComputer | None = None,
        generator: BenchmarkScenarioGenerator | None = None,
        max_ood_distance: float = 0.45,
        conformal_coverage: float = 0.90,
    ) -> None:
        self.registry = registry
        self.qualification_artifact = qualification_artifact
        self.distance_computer = distance_computer or MetaFeatureDistanceComputer()
        self.generator = generator or build_advisor_v2_generator()
        self.max_ood_distance = max_ood_distance
        self.conformal_coverage = conformal_coverage
        self._family_vectors = extract_family_meta_features(self.generator)
        self._recipe_token_by_digest = recipe_token_by_digest()
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
        prospective_evidence_complete = complete_evidence_grid

        training_family_vectors = {
            family_id: vector
            for family_id, vector in self._family_vectors.items()
            if role_by_family.get(family_id) == "meta_training"
        }
        nearest = self.distance_computer.find_nearest_families(
            task_vector, training_family_vectors, k=1
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

        qualification_reason = self._qualification_failure_reason(all_runs)
        if qualification_reason is not None:
            return MetaRankingAdvisoryReport(
                report_id=f"meta_ranker_{task_profile.profile_digest[:16]}",
                recommendation=sim_report.recommendation,
                predicted_candidate_utilities={},
                meta_model_type="none",
                meta_model_trained_runs=len(runs),
                fallback_to_similarity=True,
                fallback_reason=f"{qualification_reason}; fell back to Stage-2 similarity.",
            )

        eligible_candidates = sim_report.recommendation.shortlist
        X_by_role: dict[str, list[list[float]]] = {
            "meta_training": [],
            "conformal": [],
            "locked_evaluation": [],
        }
        y_by_role: dict[str, list[float]] = {name: [] for name in X_by_role}
        families_by_role: dict[str, list[str]] = {name: [] for name in X_by_role}

        try:
            family_evidence = aggregate_family_recipe_evidence(
                registry=self.registry,
                records=all_runs,
                role_by_family=role_by_family,
                recipe_token_by_digest=self._recipe_token_by_digest,
            )
        except ValueError:
            family_evidence = ()
        for item in family_evidence:
            family_vector = self._family_vectors.get(item.family_id)
            if family_vector is None:
                continue
            role = item.family_role
            X_by_role[role].append(family_recipe_features(family_vector, item.recipe_token))
            y_by_role[role].append(item.mean_utility)
            families_by_role[role].append(item.family_id)

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

        supported_candidates = tuple(
            (candidate, token)
            for candidate in eligible_candidates
            if (token := candidate_recipe_token(candidate)) is not None
        )
        if not supported_candidates:
            return MetaRankingAdvisoryReport(
                report_id=f"meta_ranker_{task_profile.profile_digest[:16]}",
                recommendation=sim_report.recommendation,
                predicted_candidate_utilities={},
                meta_model_type="none",
                meta_model_trained_runs=meta_model.trained_run_count,
                fallback_to_similarity=True,
                fallback_reason=(
                    "No evidence-backed candidate maps to the qualified core adapters; "
                    "fell back to Stage-2 similarity."
                ),
            )
        X_test_list = [
            family_recipe_features(task_vector, token) for _, token in supported_candidates
        ]
        X_test = np.array(X_test_list, dtype=np.float64)

        preds, lowers, uppers = meta_model.predict_utility(X_test)

        predicted_utilities: dict[str, PredictedCandidateUtility] = {}
        full_predictions: dict[str, float] = {}
        for (candidate, _), pred, low, up in zip(
            supported_candidates, preds, lowers, uppers, strict=True
        ):
            full_predictions[candidate.candidate_id] = float(pred)
            predicted_utilities[candidate.candidate_id] = PredictedCandidateUtility(
                candidate_id=candidate.candidate_id,
                predicted_utility=round(float(pred), 4),
                uncertainty_lower_bound=round(float(low), 4),
                uncertainty_upper_bound=round(float(up), 4),
                conformal_coverage_level=self.conformal_coverage,
            )

        ranked_shortlist = _rank_evidence_backed_shortlist(
            eligible_candidates,
            full_predictions,
        )
        recommendation_seed = {
            "task_profile_digest": task_profile.profile_digest,
            "registry_snapshot_digest": sim_report.recommendation.registry_snapshot_digest,
            "model_digest": meta_model.model_digest,
            "ranked_candidate_ids": [candidate.candidate_id for candidate in ranked_shortlist],
        }
        recommendation_payload = sim_report.recommendation.model_dump(mode="json")
        recommendation_payload.update(
            {
                "recommendation_id": (
                    f"advisor.meta_ranker.{_policy_digest(recommendation_seed)[:24]}"
                ),
                "utility_policy_digest": ADVISOR_UTILITY_POLICY_DIGEST,
                "shortlist": [candidate.model_dump(mode="json") for candidate in ranked_shortlist],
                "abstained_from_empirical_ranking": False,
            }
        )
        ranked_recommendation = PipelineRecommendation.model_validate(recommendation_payload)
        report_seed = {
            "recommendation_digest": ranked_recommendation.recommendation_digest,
            "model_digest": meta_model.model_digest,
        }

        return MetaRankingAdvisoryReport(
            report_id=f"report.meta_ranker.{_policy_digest(report_seed)[:24]}",
            recommendation=ranked_recommendation,
            predicted_candidate_utilities=predicted_utilities,
            meta_model_type="ridge_meta_ranker_v1",
            meta_model_trained_runs=meta_model.trained_run_count,
            fallback_to_similarity=False,
            fallback_reason=None,
        )

    def _qualification_failure_reason(self, all_runs: tuple[BenchmarkRunRecord, ...]) -> str | None:
        artifact = self.qualification_artifact
        if artifact is None:
            return "No approved empirical qualification artifact was supplied"
        validated = AdvisorQualificationArtifact.model_validate(artifact.model_dump(mode="json"))
        report = validated.report
        if report.qualification_status != "qualified" or report.fallback_to_similarity_required:
            return "The prospective empirical qualification did not authorize learned ranking"
        current_snapshot = build_registry_snapshot(
            snapshot_id="snapshot.advisor_v2_qualification_v1",
            records=all_runs,
        )
        if (
            report.registry_snapshot_digest != current_snapshot.registry_digest
            or report.design_digest != build_advisor_corpus_design().design_digest
            or report.policy_digest != AdvisorQualificationPolicy().policy_digest
        ):
            return "The empirical qualification artifact is stale or bound to different evidence"
        return None


__all__ = [
    "LearnedMetaRankerModel",
    "MetaRankingLinkageAdvisor",
]
