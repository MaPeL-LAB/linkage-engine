"""Stage-1 rule-based Linkage Strategy Advisor."""

from __future__ import annotations

import hashlib
import json

from mapel_linkage.configuration.compiler import ExecutionPlan
from mapel_linkage.domain.errors import AdvisorError
from mapel_linkage.pipeline.model_portfolio import (
    ModelPortfolioDeclaration,
    PairModelCandidateDeclaration,
    RankingCandidateDeclaration,
    compile_model_portfolio,
)
from mapel_linkage.profiling import PreflightTaskProfile, build_preflight_task_profile
from mapel_linkage.recommendation.contracts import (
    AbstentionReason,
    CandidateExplanation,
    CoverageStatus,
    DisqualifiedCandidate,
    EvidenceContribution,
    EvidenceScope,
    PipelineRecommendation,
    RankingStrategy,
    RecommendationIntent,
    RuntimeDependency,
    StructuralPipelineCandidate,
)
from mapel_linkage.recommendation.eligibility import (
    AdvisorContext,
    EligibilityDecision,
    evaluate_candidate,
)
from mapel_linkage.recommendation.structural_pareto import (
    build_diverse_shortlist,
    structural_pareto_frontier,
)

_ELIGIBILITY_POLICY = {
    "policy": "stage1-hard-eligibility-v1",
    "test_partition_used": False,
    "supervised_training_requires_verified_labels": True,
    "inference_requires_approved_recipe": True,
    "stacking_requires_protected_oof": True,
    "runtime_dependencies_are_hard_constraints": True,
}
_UTILITY_POLICY = {
    "policy": "stage1-structural-pareto-v1",
    "empirical_metrics_used": False,
    "costs_minimized": ["verified_label_requirement", "runtime_count", "complexity"],
    "attributes_maximized": [
        "interaction_capacity",
        "interpretability",
        "artifact_portability",
    ],
    "mandatory_baseline_retained": True,
    "shortlist_family_diversity": True,
}
_LOCAL_CONFIRMATION = (
    "run_bounded_local_champion_challenger",
    "fit_calibrator_on_protected_partition",
    "select_thresholds_on_protected_decision_partition",
    "evaluate_once_on_locked_test_partition",
    "obtain_explicit_local_approval",
)


def _policy_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ranking_choice(
    rankers: tuple[RankingCandidateDeclaration, ...],
    context: AdvisorContext,
) -> RankingCandidateDeclaration | None:
    for ranker in sorted(rankers, key=lambda item: item.model_id):
        if not ranker.enabled:
            continue
        runtime = (
            RuntimeDependency.CORE if ranker.family == "xgboost" else RuntimeDependency.LIGHTGBM
        )
        labels_permitted = context.verified_labels_available or context.intent in {
            RecommendationIntent.INFER_WITH_APPROVED_RECIPE,
            RecommendationIntent.SHADOW_SCORE_CHALLENGER,
        }
        if runtime in context.available_runtimes and labels_permitted:
            return ranker
    return None


def _structural_attributes(
    candidate: PairModelCandidateDeclaration,
) -> tuple[int, int, int, int]:
    by_family = {
        "fellegi_sunter": (0, 0, 3, 3),
        "xgboost": (1, 2, 2, 3),
        "lightgbm": (1, 2, 2, 2),
        "pytorch": (2, 3, 1, 1),
        "stacking": (3, 3, 1, 3),
    }
    return by_family[candidate.family]


def _required_runtimes(
    candidate: PairModelCandidateDeclaration,
    ranker: RankingCandidateDeclaration | None,
) -> tuple[RuntimeDependency, ...]:
    required = [RuntimeDependency.CORE]
    if candidate.family == "lightgbm":
        required.append(RuntimeDependency.LIGHTGBM)
    elif candidate.family == "pytorch":
        required.append(RuntimeDependency.PYTORCH)
    if ranker is not None and ranker.family == "lightgbm":
        required.append(RuntimeDependency.LIGHTGBM)
    return tuple(dict.fromkeys(required))


def build_structural_pipeline_candidates(
    *,
    plan: ExecutionPlan,
    portfolio: ModelPortfolioDeclaration,
    context: AdvisorContext,
) -> tuple[StructuralPipelineCandidate, ...]:
    """Compile one complete structural template per pair-model candidate."""

    ranker = _ranking_choice(portfolio.ranking_candidates, context)
    if ranker is None:
        ranking_strategy = RankingStrategy.MODEL_SCORE
        ranking_model_id = None
    elif ranker.family == "xgboost":
        ranking_strategy = RankingStrategy.XGBOOST_RANKER
        ranking_model_id = ranker.model_id
    else:
        ranking_strategy = RankingStrategy.LIGHTGBM_RANKER
        ranking_model_id = ranker.model_id

    candidates: list[StructuralPipelineCandidate] = []
    for pair_candidate in portfolio.pair_candidates:
        if not pair_candidate.enabled:
            continue
        complexity, interaction, interpretability, portability = _structural_attributes(
            pair_candidate
        )
        requires_labels = pair_candidate.require_verified_labels or ranker is not None
        candidate_payload: dict[str, object] = {
            "configuration_digest": plan.configuration_digest,
            "portfolio_digest": portfolio.portfolio_digest,
            "pair_model_id": pair_candidate.model_id,
            "ranking_strategy": ranking_strategy.value,
            "ranking_model_id": ranking_model_id,
            "calibration_method": plan.config.calibration.method,
        }
        candidate_id = f"strategy.{_policy_digest(candidate_payload)[:24]}"
        candidates.append(
            StructuralPipelineCandidate(
                candidate_id=candidate_id,
                configuration_digest=plan.configuration_digest,
                portfolio_digest=portfolio.portfolio_digest,
                pair_model_id=pair_candidate.model_id,
                pair_model_family=pair_candidate.family,
                pair_model_role=pair_candidate.role,
                base_model_ids=pair_candidate.base_model_ids,
                ranking_strategy=ranking_strategy,
                ranking_model_id=ranking_model_id,
                calibration_method=plan.config.calibration.method,
                linkage_mode=plan.config.project.linkage_mode,
                assignment_constraint=plan.config.project.assignment_constraint,
                requires_verified_labels=requires_labels,
                requires_protected_out_of_fold_predictions=(pair_candidate.family == "stacking"),
                required_runtimes=_required_runtimes(pair_candidate, ranker),
                structural_complexity=min(3, complexity + int(ranker is not None)),
                interaction_capacity=interaction,
                interpretability_score=interpretability,
                artifact_portability_score=portability,
            )
        )
    return tuple(candidates)


def _evaluate_all(
    candidates: tuple[StructuralPipelineCandidate, ...],
    context: AdvisorContext,
) -> tuple[EligibilityDecision, ...]:
    decisions: dict[str, EligibilityDecision] = {}
    for candidate in candidates:
        if candidate.pair_model_family == "stacking":
            continue
        decisions[candidate.candidate_id] = evaluate_candidate(candidate, context=context)
    eligible_base_ids = frozenset(
        candidate.pair_model_id
        for candidate in candidates
        if candidate.pair_model_family != "stacking" and decisions[candidate.candidate_id].eligible
    )
    for candidate in candidates:
        if candidate.pair_model_family != "stacking":
            continue
        decisions[candidate.candidate_id] = evaluate_candidate(
            candidate,
            context=context,
            eligible_base_model_ids=eligible_base_ids,
        )
    return tuple(decisions[candidate.candidate_id] for candidate in candidates)


def _explanation_rules(
    candidate: StructuralPipelineCandidate,
    decision: EligibilityDecision,
    *,
    on_frontier: bool,
) -> tuple[str, ...]:
    rules = [f"eligibility.{reason.value}" for reason in decision.reasons]
    rules.append(f"model_family.{candidate.pair_model_family}")
    rules.append(f"ranking_strategy.{candidate.ranking_strategy.value}")
    rules.append(f"calibration.{candidate.calibration_method}")
    if candidate.pair_model_role == "baseline":
        rules.append("mandatory_baseline.retained")
    if on_frontier:
        rules.append("structural_pareto.member")
    rules.append("empirical_performance.not_claimed")
    return tuple(dict.fromkeys(rules))


def _empty_evidence_contributions() -> tuple[EvidenceContribution, ...]:
    return tuple(EvidenceContribution(scope=scope) for scope in EvidenceScope)


def recommend_pipeline(
    plan: ExecutionPlan,
    *,
    context: AdvisorContext,
    profile: PreflightTaskProfile | None = None,
) -> PipelineRecommendation:
    """Return a transparent structural shortlist and abstain from empirical ranking."""

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
    shortlist = build_diverse_shortlist(
        eligible,
        mandatory_baseline_id=portfolio.mandatory_baseline_id,
        maximum_challengers=portfolio.maximum_challengers,
    )

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
    frontier_ids = tuple(candidate.candidate_id for candidate in frontier)
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

    abstention_reasons: list[AbstentionReason] = [
        AbstentionReason.NO_BENCHMARK_EVIDENCE,
        AbstentionReason.LOCAL_CONFIRMATION_REQUIRED,
    ]
    if not eligible:
        abstention_reasons.append(AbstentionReason.NO_ELIGIBLE_PIPELINES)
    if (
        context.intent is RecommendationIntent.INFER_WITH_APPROVED_RECIPE
        and not context.approved_recipe_available
    ):
        abstention_reasons.append(AbstentionReason.APPROVED_RECIPE_REQUIRED)
    if (
        context.intent is RecommendationIntent.SHADOW_SCORE_CHALLENGER
        and not context.approved_artifact_model_ids
    ):
        abstention_reasons.append(AbstentionReason.APPROVED_ARTIFACT_REQUIRED)
    if context.candidate_retrieval_status.value == "failed":
        abstention_reasons.append(AbstentionReason.CANDIDATE_RETRIEVAL_FAILED)

    digest_seed: dict[str, object] = {
        "profile": task_profile.profile_digest,
        "intent": context.intent.value,
        "portfolio": portfolio.portfolio_digest,
        "eligibility_policy": _ELIGIBILITY_POLICY,
        "utility_policy": _UTILITY_POLICY,
    }
    recommendation_id = f"advisor.{_policy_digest(digest_seed)[:24]}"
    baseline_candidate = next(
        (item for item in shortlist if item.pair_model_id == portfolio.mandatory_baseline_id),
        None,
    )
    return PipelineRecommendation(
        recommendation_id=recommendation_id,
        intent=context.intent,
        task_profile_digest=task_profile.profile_digest,
        utility_policy_digest=_policy_digest(_UTILITY_POLICY),
        eligibility_policy_digest=_policy_digest(_ELIGIBILITY_POLICY),
        registry_snapshot_digest=None,
        coverage_status=CoverageStatus.STRUCTURAL_ONLY,
        out_of_distribution_score=None,
        abstention_reasons=tuple(dict.fromkeys(abstention_reasons)),
        mandatory_baseline_candidate_id=(
            baseline_candidate.candidate_id if baseline_candidate is not None else None
        ),
        shortlist=shortlist,
        structural_pareto_candidate_ids=frontier_ids,
        disqualified_candidates=disqualified,
        explanations=explanations,
        evidence_contributions=_empty_evidence_contributions(),
        required_local_confirmation=_LOCAL_CONFIRMATION,
    )


__all__ = ["build_structural_pipeline_candidates", "recommend_pipeline"]
