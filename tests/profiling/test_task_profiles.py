from __future__ import annotations

from mapel_linkage.configuration import compile_config, load_config
from mapel_linkage.configuration.compiler import ExecutionPlan
from mapel_linkage.profiling import (
    CalibrationEvidenceStatus,
    CandidateBudgetStatus,
    CandidateGraphProfile,
    CandidateRecallStatus,
    CountBand,
    EvidenceProfile,
    ProfileScope,
    RateBand,
    build_preflight_task_profile,
)
from tests.helpers import EXAMPLE_CONFIG, ROOT


def _plan() -> ExecutionPlan:
    loaded = load_config(EXAMPLE_CONFIG)
    return compile_config(loaded.config, project_root=ROOT)


def test_preflight_profile_is_deterministic_aggregate_and_value_safe() -> None:
    plan = _plan()
    first = build_preflight_task_profile(plan)
    second = build_preflight_task_profile(plan)

    assert first == second
    assert first.profile_digest == second.profile_digest
    assert first.profile_scope is ProfileScope.LOCAL_RESTRICTED
    assert first.record_count_band is CountBand.NOT_OBSERVED
    assert first.verified_labels_available is True
    assert first.contains_record_values is False
    assert first.contains_source_field_names is False
    summary = first.safe_summary()
    rendered = repr(summary)
    assert plan.config.project.project_id not in rendered
    assert "record_key_a" not in rendered
    assert str(EXAMPLE_CONFIG) not in rendered


def test_candidate_and_evidence_profiles_expose_bands_not_rows() -> None:
    preflight = build_preflight_task_profile(_plan())
    candidate = CandidateGraphProfile(
        profile_scope=ProfileScope.GLOBAL_SYNTHETIC,
        preflight_profile_digest=preflight.profile_digest,
        candidate_pair_count_band=CountBand.MEDIUM,
        mean_candidate_set_size_band=CountBand.SMALL,
        p95_candidate_set_size_band=CountBand.MEDIUM,
        zero_candidate_rate_band=RateBand.LOW,
        conflict_density_band=RateBand.MODERATE,
        candidate_budget_status=CandidateBudgetStatus.WITHIN_BUDGET,
        candidate_recall_status=CandidateRecallStatus.ESTIMATED,
    )
    evidence = EvidenceProfile(
        profile_scope=ProfileScope.GLOBAL_SYNTHETIC,
        preflight_profile_digest=preflight.profile_digest,
        candidate_graph_profile_digest=candidate.profile_digest,
        pair_model_count=3,
        top_score_margin_band=RateBand.MODERATE,
        model_disagreement_band=RateBand.LOW,
        review_burden_band=RateBand.MODERATE,
        assignment_change_band=RateBand.LOW,
        calibration_status=CalibrationEvidenceStatus.CALIBRATED_SYNTHETIC,
    )

    assert candidate.safe_summary()["contains_candidate_pairs"] is False
    assert evidence.safe_summary()["contains_score_vectors"] is False
    assert "record_key" not in repr(candidate.safe_summary())
    evidence_summary = evidence.safe_summary()
    assert evidence_summary["contains_candidate_pairs"] is False
    assert "left_record_key" not in repr(evidence_summary)
    assert "right_record_key" not in repr(evidence_summary)
