"""Stage-2 Similarity and Coverage Linkage Strategy Advisor."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

from mapel_linkage.benchmarking.advisor_catalogue import (
    advisor_v2_family_roles,
    build_advisor_v2_generator,
)
from mapel_linkage.benchmarking.contracts import (
    BenchmarkRegistrySnapshot,
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
from mapel_linkage.pipeline.model_portfolio import (
    compile_model_portfolio,
)
from mapel_linkage.profiling import PreflightTaskProfile, build_preflight_task_profile
from mapel_linkage.recommendation.advisor import (
    _LOCAL_CONFIRMATION,
    _empty_evidence_contributions,
    _evaluate_all,
    _explanation_rules,
    build_structural_pipeline_candidates,
)
from mapel_linkage.recommendation.contracts import (
    AbstentionReason,
    CandidateExplanation,
    CoverageStatus,
    DisqualifiedCandidate,
    EmpiricalMetricDistribution,
    EvidenceContribution,
    EvidenceScope,
    PipelineRecommendation,
    RecommendationIntent,
    SimilarityAdvisoryReport,
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
from mapel_linkage.recommendation.structural_pareto import (
    build_diverse_shortlist,
    structural_pareto_frontier,
)
from mapel_linkage.recommendation.utility import (
    ADVISOR_UTILITY_POLICY_DIGEST,
    candidate_recipe_token,
    empirical_distribution_utility,
    recipe_token_by_digest,
)

_ELIGIBILITY_POLICY: dict[str, Any] = {
    "policy": "stage1-hard-eligibility-v1",
    "test_partition_used": False,
    "supervised_training_requires_verified_labels": True,
    "inference_requires_approved_recipe": True,
    "stacking_requires_protected_oof": True,
    "runtime_dependencies_are_hard_constraints": True,
}


def _policy_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SimilarityLinkageAdvisor:
    """Stage-2 Evidence-Based Linkage Strategy Advisor using similarity retrieval."""

    def __init__(
        self,
        registry: BenchmarkRegistry | None = None,
        *,
        distance_computer: MetaFeatureDistanceComputer | None = None,
        generator: BenchmarkScenarioGenerator | None = None,
        max_ood_distance: float = 0.45,
        k_nearest_families: int = 3,
    ) -> None:
        self.registry = registry
        self.distance_computer = distance_computer or MetaFeatureDistanceComputer()
        self.generator = generator or build_advisor_v2_generator()
        self.max_ood_distance = max_ood_distance
        self.k_nearest_families = k_nearest_families
        self._family_vectors = extract_family_meta_features(self.generator)
        self._role_by_family = dict(advisor_v2_family_roles())
        self._recipe_token_by_digest = recipe_token_by_digest()

    def recommend(
        self,
        plan: ExecutionPlan,
        *,
        context: AdvisorContext,
        profile: PreflightTaskProfile | None = None,
    ) -> SimilarityAdvisoryReport:
        """Produce an empirical similarity advisory report and family-diverse recommendation."""
        if context.test_partition_used:
            raise AdvisorError(
                "ML-ADVISOR-001",
                "The locked test partition cannot be used by the strategy advisor.",
            )

        task_profile = profile or build_preflight_task_profile(plan)
        portfolio = compile_model_portfolio(plan.config)
        candidates = build_structural_pipeline_candidates(
            plan=plan,
            portfolio=portfolio,
            context=context,
        )
        decisions = _evaluate_all(candidates, context)
        decision_by_id = {item.candidate_id: item for item in decisions}
        eligible = tuple(
            candidate for candidate in candidates if decision_by_id[candidate.candidate_id].eligible
        )
        frontier = structural_pareto_frontier(eligible)
        frontier_ids = tuple(candidate.candidate_id for candidate in frontier)

        disqualified = tuple(
            DisqualifiedCandidate(
                candidate_id=candidate.candidate_id,
                candidate_digest=candidate.candidate_digest,
                reasons=tuple(
                    reason.value for reason in decision_by_id[candidate.candidate_id].reasons
                ),
            )
            for candidate in candidates
            if not decision_by_id[candidate.candidate_id].eligible
        )

        explanations = tuple(
            CandidateExplanation(
                candidate_id=candidate.candidate_id,
                rule_codes=_explanation_rules(
                    candidate,
                    decision_by_id[candidate.candidate_id],
                    on_frontier=candidate.candidate_id in frontier_ids,
                ),
            )
            for candidate in candidates
        )

        # Base abstention reasons from lifecycle intent and eligibility
        base_abstentions: list[AbstentionReason] = [AbstentionReason.LOCAL_CONFIRMATION_REQUIRED]
        if not eligible:
            base_abstentions.append(AbstentionReason.NO_ELIGIBLE_PIPELINES)
        if (
            context.intent is RecommendationIntent.INFER_WITH_APPROVED_RECIPE
            and not context.approved_recipe_available
        ):
            base_abstentions.append(AbstentionReason.APPROVED_RECIPE_REQUIRED)
        if (
            context.intent is RecommendationIntent.SHADOW_SCORE_CHALLENGER
            and not context.approved_artifact_model_ids
        ):
            base_abstentions.append(AbstentionReason.APPROVED_ARTIFACT_REQUIRED)
        if context.candidate_retrieval_status.value == "failed":
            base_abstentions.append(AbstentionReason.CANDIDATE_RETRIEVAL_FAILED)

        # Meta-feature vector extraction
        target_meta = TaskMetaFeatureVector.from_profile(task_profile)

        # Check if benchmark evidence exists in registry
        has_benchmark_evidence = False
        snapshot: BenchmarkRegistrySnapshot | None = None
        reg = self.registry
        if reg is not None:
            records = reg.list_run_records()
            if records:
                has_benchmark_evidence = True
                snapshot = reg.build_snapshot()

        # Handle no benchmark evidence fallback
        if not has_benchmark_evidence or snapshot is None or reg is None:
            shortlist = build_diverse_shortlist(
                eligible,
                mandatory_baseline_id=portfolio.mandatory_baseline_id,
                maximum_challengers=portfolio.maximum_challengers,
            )
            abstention_reasons = tuple(
                dict.fromkeys([AbstentionReason.NO_BENCHMARK_EVIDENCE, *base_abstentions])
            )
            baseline_cand = next(
                (
                    item
                    for item in shortlist
                    if item.pair_model_id == portfolio.mandatory_baseline_id
                ),
                None,
            )
            rec_seed: dict[str, Any] = {
                "profile": task_profile.profile_digest,
                "state": "no_evidence",
            }
            recommendation_id = f"advisor.similarity.{_policy_digest(rec_seed)[:24]}"
            recommendation = PipelineRecommendation(
                recommendation_id=recommendation_id,
                intent=context.intent,
                task_profile_digest=task_profile.profile_digest,
                utility_policy_digest=ADVISOR_UTILITY_POLICY_DIGEST,
                eligibility_policy_digest=_policy_digest(_ELIGIBILITY_POLICY),
                registry_snapshot_digest=None,
                coverage_status=CoverageStatus.STRUCTURAL_ONLY,
                out_of_distribution_score=None,
                abstention_reasons=abstention_reasons,
                mandatory_baseline_candidate_id=(
                    baseline_cand.candidate_id if baseline_cand is not None else None
                ),
                shortlist=shortlist,
                structural_pareto_candidate_ids=frontier_ids,
                disqualified_candidates=disqualified,
                explanations=explanations,
                evidence_contributions=_empty_evidence_contributions(),
                required_local_confirmation=_LOCAL_CONFIRMATION,
            )
            report_seed: dict[str, Any] = {"rec": recommendation.recommendation_digest}
            report_id = f"report.similarity.{_policy_digest(report_seed)[:24]}"
            return SimilarityAdvisoryReport(
                report_id=report_id,
                recommendation=recommendation,
                target_task_profile_digest=task_profile.profile_digest,
                nearest_family_ids=(),
                nearest_family_distances={},
                synthetic_evidence_retrieved=False,
                out_of_distribution=True,
                out_of_distribution_score=1.0,
                empirical_metric_distributions={},
            )

        # Find nearest scenario families
        available_family_ids = {record.family_id for record in snapshot.records}
        v2_evidence_present = bool(available_family_ids & set(self._role_by_family))
        eligible_family_vectors = {
            family_id: vector
            for family_id, vector in self._family_vectors.items()
            if family_id in available_family_ids
            and (not v2_evidence_present or self._role_by_family.get(family_id) == "meta_training")
        }
        nearest = self.distance_computer.find_nearest_families(
            target_meta,
            eligible_family_vectors,
            k=self.k_nearest_families,
        )
        nearest_family_ids = tuple(fam_id for fam_id, _ in nearest)
        nearest_distances = {fam_id: dist for fam_id, dist in nearest}
        best_distance = nearest[0][1] if nearest else 1.0

        is_ood = best_distance > self.max_ood_distance

        if is_ood:
            # Out of distribution fallback
            shortlist = build_diverse_shortlist(
                eligible,
                mandatory_baseline_id=portfolio.mandatory_baseline_id,
                maximum_challengers=portfolio.maximum_challengers,
            )
            abstention_reasons = tuple(
                dict.fromkeys([AbstentionReason.OUT_OF_DISTRIBUTION, *base_abstentions])
            )
            baseline_cand = next(
                (
                    item
                    for item in shortlist
                    if item.pair_model_id == portfolio.mandatory_baseline_id
                ),
                None,
            )
            rec_seed = {
                "profile": task_profile.profile_digest,
                "dist": best_distance,
                "ood": True,
            }
            recommendation_id = f"advisor.similarity.{_policy_digest(rec_seed)[:24]}"
            recommendation = PipelineRecommendation(
                recommendation_id=recommendation_id,
                intent=context.intent,
                task_profile_digest=task_profile.profile_digest,
                utility_policy_digest=ADVISOR_UTILITY_POLICY_DIGEST,
                eligibility_policy_digest=_policy_digest(_ELIGIBILITY_POLICY),
                registry_snapshot_digest=snapshot.registry_digest,
                coverage_status=CoverageStatus.OUT_OF_DISTRIBUTION,
                out_of_distribution_score=float(best_distance),
                abstention_reasons=abstention_reasons,
                mandatory_baseline_candidate_id=(
                    baseline_cand.candidate_id if baseline_cand is not None else None
                ),
                shortlist=shortlist,
                structural_pareto_candidate_ids=frontier_ids,
                disqualified_candidates=disqualified,
                explanations=explanations,
                evidence_contributions=_empty_evidence_contributions(),
                required_local_confirmation=_LOCAL_CONFIRMATION,
            )
            report_seed = {"rec": recommendation.recommendation_digest}
            report_id = f"report.similarity.{_policy_digest(report_seed)[:24]}"
            return SimilarityAdvisoryReport(
                report_id=report_id,
                recommendation=recommendation,
                target_task_profile_digest=task_profile.profile_digest,
                nearest_family_ids=nearest_family_ids,
                nearest_family_distances=nearest_distances,
                synthetic_evidence_retrieved=False,
                out_of_distribution=True,
                out_of_distribution_score=float(best_distance),
                empirical_metric_distributions={},
            )

        # Within Distribution: aggregate empirical metrics across nearest runs
        runs_by_family: list[Any] = []
        for fam_id in nearest_family_ids:
            runs_by_family.extend(reg.list_run_records(family_id=fam_id))

        # Group runs by model family / recipe
        metrics_by_candidate: dict[str, list[Any]] = defaultdict(list)
        all_runs_by_candidate: dict[str, list[Any]] = defaultdict(list)

        for run in runs_by_family:
            for candidate in eligible:
                candidate_token = candidate_recipe_token(candidate)
                run_token = self._recipe_token_by_digest.get(run.pipeline_recipe_digest)
                matches = candidate_token is not None and candidate_token == run_token

                if matches:
                    all_runs_by_candidate[candidate.candidate_id].append(run)
                    if run.status == BenchmarkRunStatus.SUCCESS:
                        m = reg.load_metrics(run.run_id)
                        if m is not None:
                            metrics_by_candidate[candidate.candidate_id].append(m)

        empirical_distributions: dict[str, EmpiricalMetricDistribution] = {}
        for candidate in eligible:
            cid = candidate.candidate_id
            m_list = metrics_by_candidate.get(cid, [])
            total_runs = all_runs_by_candidate.get(cid, [])
            sample_count = len(m_list)

            if sample_count > 0:
                mean_rec = float(np.mean([m.candidate_recall for m in m_list]))
                mean_r1 = float(
                    np.mean([m.candidate_recall_at_k.get("1", m.candidate_recall) for m in m_list])
                )
                mean_r5 = float(
                    np.mean([m.candidate_recall_at_k.get("5", m.candidate_recall) for m in m_list])
                )
                mean_ppv = float(np.mean([m.positive_predictive_value for m in m_list]))
                mean_brier = float(np.mean([m.brier_score for m in m_list]))
                mean_rt = float(np.mean([m.runtime_ms for m in m_list]))
                mean_mem = float(np.mean([m.peak_memory_mb for m in m_list]))
                fail_rate = (
                    float((len(total_runs) - sample_count) / len(total_runs)) if total_runs else 0.0
                )
            else:
                mean_rec = 0.0
                mean_r1 = 0.0
                mean_r5 = 0.0
                mean_ppv = 0.0
                mean_brier = 0.0
                mean_rt = 0.0
                mean_mem = 0.0
                fail_rate = 1.0 if total_runs else 0.0

            empirical_distributions[cid] = EmpiricalMetricDistribution(
                sample_count=sample_count,
                mean_candidate_recall=mean_rec,
                mean_recall_at_1=mean_r1,
                mean_recall_at_5=mean_r5,
                mean_positive_predictive_value=mean_ppv,
                mean_brier_score=mean_brier,
                mean_runtime_ms=mean_rt,
                mean_peak_memory_mb=mean_mem,
                failure_rate=fail_rate,
            )

        # Rank eligible challengers using empirical utility
        def _candidate_empirical_score(c: StructuralPipelineCandidate) -> float:
            dist = empirical_distributions.get(c.candidate_id)
            if dist is None:
                return -1.0
            return empirical_distribution_utility(dist)

        baseline_candidate = next(
            (c for c in eligible if c.pair_model_id == portfolio.mandatory_baseline_id),
            None,
        )
        non_baseline = [c for c in eligible if c.pair_model_id != portfolio.mandatory_baseline_id]
        non_baseline.sort(key=_candidate_empirical_score, reverse=True)

        selected: list[StructuralPipelineCandidate] = []
        if baseline_candidate is not None:
            selected.append(baseline_candidate)

        seen_families = (
            {baseline_candidate.pair_model_family} if baseline_candidate is not None else set()
        )
        for cand in non_baseline:
            if len(selected) >= portfolio.maximum_challengers + int(baseline_candidate is not None):
                break
            if cand.pair_model_family not in seen_families:
                selected.append(cand)
                seen_families.add(cand.pair_model_family)

        for cand in non_baseline:
            if len(selected) >= portfolio.maximum_challengers + int(baseline_candidate is not None):
                break
            if cand not in selected:
                selected.append(cand)

        shortlist = tuple(selected)

        # Build evidence contribution
        total_approved_runs = sum(
            1 for run in runs_by_family if run.status == BenchmarkRunStatus.SUCCESS
        )
        evidence_digest = _policy_digest(
            {
                "nearest_families": list(nearest_family_ids),
                "snapshot": snapshot.registry_digest,
                "run_count": total_approved_runs,
            }
        )
        evidence_contributions = (
            EvidenceContribution(
                scope=EvidenceScope.GLOBAL_SYNTHETIC,
                scenario_family_count=len(nearest_family_ids),
                approved_run_count=total_approved_runs,
                current=True,
                eligible=True,
                evidence_digest=evidence_digest,
            ),
            EvidenceContribution(scope=EvidenceScope.LOCAL_SCHEMA_MATCHED_SYNTHETIC),
            EvidenceContribution(scope=EvidenceScope.LOCAL_VERIFIED_VALIDATION),
            EvidenceContribution(scope=EvidenceScope.LOCAL_OPERATIONAL_MONITORING),
        )

        abstention_reasons = tuple(dict.fromkeys(base_abstentions))
        rec_seed = {
            "profile": task_profile.profile_digest,
            "dist": best_distance,
            "snapshot": snapshot.registry_digest,
        }
        recommendation_id = f"advisor.similarity.{_policy_digest(rec_seed)[:24]}"
        recommendation = PipelineRecommendation(
            recommendation_id=recommendation_id,
            intent=context.intent,
            task_profile_digest=task_profile.profile_digest,
            utility_policy_digest=ADVISOR_UTILITY_POLICY_DIGEST,
            eligibility_policy_digest=_policy_digest(_ELIGIBILITY_POLICY),
            registry_snapshot_digest=snapshot.registry_digest,
            coverage_status=CoverageStatus.WITHIN_BENCHMARK_ENVELOPE,
            out_of_distribution_score=float(best_distance),
            abstained_from_empirical_ranking=False,
            abstention_reasons=abstention_reasons,
            mandatory_baseline_candidate_id=(
                baseline_candidate.candidate_id if baseline_candidate is not None else None
            ),
            shortlist=shortlist,
            structural_pareto_candidate_ids=frontier_ids,
            disqualified_candidates=disqualified,
            explanations=explanations,
            evidence_contributions=evidence_contributions,
            required_local_confirmation=_LOCAL_CONFIRMATION,
        )

        report_seed = {
            "rec": recommendation.recommendation_digest,
            "dist": best_distance,
        }
        report_id = f"report.similarity.{_policy_digest(report_seed)[:24]}"
        return SimilarityAdvisoryReport(
            report_id=report_id,
            recommendation=recommendation,
            target_task_profile_digest=task_profile.profile_digest,
            nearest_family_ids=nearest_family_ids,
            nearest_family_distances=nearest_distances,
            synthetic_evidence_retrieved=True,
            out_of_distribution=False,
            out_of_distribution_score=float(best_distance),
            empirical_metric_distributions=empirical_distributions,
        )


def recommend_with_similarity(
    plan: ExecutionPlan,
    *,
    context: AdvisorContext,
    registry: BenchmarkRegistry | None = None,
    profile: PreflightTaskProfile | None = None,
    distance_computer: MetaFeatureDistanceComputer | None = None,
    generator: BenchmarkScenarioGenerator | None = None,
    max_ood_distance: float = 0.45,
    k_nearest_families: int = 3,
) -> SimilarityAdvisoryReport:
    """Convenience function for similarity-based recommendation."""
    advisor = SimilarityLinkageAdvisor(
        registry=registry,
        distance_computer=distance_computer,
        generator=generator,
        max_ood_distance=max_ood_distance,
        k_nearest_families=k_nearest_families,
    )
    return advisor.recommend(plan, context=context, profile=profile)


__all__ = [
    "SimilarityAdvisoryReport",
    "SimilarityLinkageAdvisor",
    "recommend_with_similarity",
]
