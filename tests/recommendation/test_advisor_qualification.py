"""Prospective, aggregate-only qualification tests for the advisory stack."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from mapel_linkage.benchmarking.advisor_catalogue import (
    AdvisorCorpusReadinessManifest,
    AdvisorFamilyRole,
    advisor_v2_family_roles,
    build_advisor_corpus_readiness,
    build_advisor_v2_generator,
    build_benchmark_shard_plan,
)
from mapel_linkage.benchmarking.advisor_v3_catalogue import advisor_v3_family_roles
from mapel_linkage.benchmarking.registry import BenchmarkRegistry
from mapel_linkage.benchmarking.runner import BenchmarkPortfolioRunner
from mapel_linkage.configuration import compile_config, load_config
from mapel_linkage.profiling import build_preflight_task_profile
from mapel_linkage.recommendation import AdvisorContext, RecommendationIntent, recommend_pipeline
from mapel_linkage.recommendation.distance import (
    TaskMetaFeatureVector,
    extract_family_meta_features,
)
from mapel_linkage.recommendation.distance_v3 import (
    MechanismAwareTaskMetaFeatureVector,
)
from mapel_linkage.recommendation.meta_learning import FamilyRecipeUtilityEvidence
from mapel_linkage.recommendation.meta_ranker import _rank_evidence_backed_shortlist
from mapel_linkage.recommendation.qualification import (
    AdvisorQualificationPolicy,
    _evaluate_complete_family_evidence,
    deserialize_advisor_qualification_artifact,
    load_advisor_qualification_artifact,
    qualify_advisor_registry,
    serialize_advisor_qualification_artifact,
    write_advisor_qualification_artifact,
)
from mapel_linkage.recommendation.qualification_v3 import (
    AdvisorV3QualificationPolicy,
    AdvisorV31FamilyUtilityEvidence,
    AdvisorV31QualificationApproval,
    AdvisorV31QualificationReadinessArtifact,
    build_advisor_v31_qualification_readiness,
    evaluate_advisor_v31_aggregate_evidence,
)
from mapel_linkage.recommendation.utility import (
    ADVISOR_UTILITY_POLICY_DIGEST,
    REQUIRED_ADVISOR_RECIPE_TOKENS,
)
from tests.helpers import EXAMPLE_CONFIG, ROOT


def _complete_readiness() -> AdvisorCorpusReadinessManifest:
    runner = BenchmarkPortfolioRunner()
    initial = build_advisor_corpus_readiness(
        adapter_statuses=runner.adapter_statuses(), planned_replicates_per_instance=5
    )
    return AdvisorCorpusReadinessManifest.model_validate(
        {
            **initial.model_dump(mode="json"),
            "execution_status": "complete",
            "expected_run_count": 9800,
            "completed_run_count": 9800,
            "successful_overlap_family_count": 64,
            "successful_evidence_cell_count": 1400,
            "successful_required_adapter_run_count": 4200,
            "failed_required_adapter_run_count": 0,
            "missing_required_adapter_run_count": 0,
            "advisor_evidence_ready": True,
        }
    )


def _family_fixture() -> tuple[
    dict[str, AdvisorFamilyRole],
    dict[str, TaskMetaFeatureVector],
    tuple[FamilyRecipeUtilityEvidence, ...],
]:
    roles = dict(advisor_v2_family_roles())
    vectors = {
        family_id: vector
        for family_id, vector in extract_family_meta_features(build_advisor_v2_generator()).items()
        if family_id in roles
    }
    evidence: list[FamilyRecipeUtilityEvidence] = []
    for family_id, role in sorted(roles.items()):
        if role == "ood_holdout":
            continue
        vector = vectors[family_id]
        utilities = {
            "fellegi_sunter": (
                0.72 - 0.12 * vector.error_estimate_approx - 0.06 * vector.missingness_mean
            ),
            "xgboost_classifier": (
                0.58 + 0.30 * vector.label_volume_scale - 0.05 * vector.missingness_mean
            ),
            "xgboost_ranker": (
                0.57 + 0.22 * vector.error_estimate_approx + 0.08 * vector.label_volume_scale
            ),
        }
        for token in REQUIRED_ADVISOR_RECIPE_TOKENS:
            evidence.append(
                FamilyRecipeUtilityEvidence(
                    family_id=family_id,
                    family_role=role,
                    recipe_token=token,
                    mean_utility=max(0.0, min(1.0, utilities[token])),
                    run_count=20,
                    evidence_digest=hashlib.sha256(f"{family_id}:{token}".encode()).hexdigest(),
                )
            )
    return roles, vectors, tuple(evidence)


def test_qualification_policy_is_prespecified_and_immutable() -> None:
    policy = AdvisorQualificationPolicy()

    assert policy.utility_policy_digest == ADVISOR_UTILITY_POLICY_DIGEST
    assert policy.learning_curve_family_counts == (8, 16, 24, 32, 40)
    assert policy.required_recipe_tokens == REQUIRED_ADVISOR_RECIPE_TOKENS
    assert policy.operational_validity == "not_established"
    with pytest.raises(ValidationError, match="package-fixed"):
        AdvisorQualificationPolicy(minimum_stage3_regret_improvement=0.0)


def test_family_qualification_keeps_locked_and_ood_evidence_out_of_fit() -> None:
    roles, vectors, evidence = _family_fixture()
    policy = AdvisorQualificationPolicy()
    first = _evaluate_complete_family_evidence(
        evidence=evidence,
        role_by_family=roles,
        family_vectors=vectors,
        policy=policy,
    )
    first_model = first[-1]

    changed_locked = tuple(
        replace(item, mean_utility=min(1.0, item.mean_utility + 0.10))
        if item.family_role == "locked_evaluation"
        else item
        for item in evidence
    )
    second = _evaluate_complete_family_evidence(
        evidence=changed_locked,
        role_by_family=roles,
        family_vectors=vectors,
        policy=policy,
    )
    assert second[-1].model_digest == first_model.model_digest

    changed_vectors = dict(vectors)
    for family_id, role in roles.items():
        if role == "ood_holdout":
            changed_vectors[family_id] = replace(
                changed_vectors[family_id], error_estimate_approx=1.0
            )
    third = _evaluate_complete_family_evidence(
        evidence=evidence,
        role_by_family=roles,
        family_vectors=changed_vectors,
        policy=policy,
    )
    assert third[-1].model_digest == first_model.model_digest
    assert first_model.trained_family_count == 40
    assert first_model.conformal_family_count == 8
    assert first_model.locked_evaluation_family_count == 8


def test_family_qualification_fails_closed_on_missing_or_constant_evidence() -> None:
    roles, vectors, evidence = _family_fixture()
    policy = AdvisorQualificationPolicy()

    with pytest.raises(ValueError, match="omits"):
        _evaluate_complete_family_evidence(
            evidence=evidence[:-1],
            role_by_family=roles,
            family_vectors=vectors,
            policy=policy,
        )

    constant = tuple(replace(item, mean_utility=0.5) for item in evidence)
    with pytest.raises(ValueError, match="variation"):
        _evaluate_complete_family_evidence(
            evidence=constant,
            role_by_family=roles,
            family_vectors=vectors,
            policy=policy,
        )


def test_meta_ranking_order_is_applied_and_keeps_the_baseline() -> None:
    plan = compile_config(load_config(EXAMPLE_CONFIG).config, project_root=ROOT)
    recommendation = recommend_pipeline(
        plan,
        context=AdvisorContext(
            intent=RecommendationIntent.DEVELOP_NEW_RECIPE,
            verified_labels_available=True,
        ),
        profile=build_preflight_task_profile(plan),
    )
    assert len(recommendation.shortlist) >= 2
    first, second, *rest = recommendation.shortlist
    ranked = _rank_evidence_backed_shortlist(
        recommendation.shortlist,
        {first.candidate_id: 0.1, second.candidate_id: 0.9},
    )

    assert ranked[:2] == (second, first)
    assert tuple(ranked[2:]) == tuple(rest)
    assert recommendation.mandatory_baseline_candidate_id in {item.candidate_id for item in ranked}


def test_qualification_artifact_is_aggregate_canonical_and_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mapel_linkage.recommendation.qualification as module

    roles, vectors, evidence = _family_fixture()
    readiness = _complete_readiness()
    monkeypatch.setattr(
        module,
        "_validate_registry_evidence",
        lambda **_kwargs: ((), readiness, "a" * 64, "b" * 64),
    )
    monkeypatch.setattr(module, "aggregate_family_recipe_evidence", lambda **_kwargs: evidence)
    monkeypatch.setattr(module, "advisor_v2_family_roles", lambda: tuple(roles.items()))
    monkeypatch.setattr(module, "extract_family_meta_features", lambda _generator: vectors)

    artifact = qualify_advisor_registry(
        registry=BenchmarkRegistry(tmp_path / "registry"),
        shard_plan=build_benchmark_shard_plan(shard_count=32),
        approval_reference="owner-approved-synthetic-qualification",
    )
    text = serialize_advisor_qualification_artifact(artifact)
    loaded = deserialize_advisor_qualification_artifact(text)

    assert loaded == artifact
    assert loaded.report.qualification_status == "not_qualified"
    assert loaded.report.operational_validity == "not_established"
    assert loaded.report.hard_constraint_violation_count == 0
    safe_text = json.dumps(loaded.safe_summary(), sort_keys=True)
    assert "family.advisor_v2" not in safe_text
    assert str(tmp_path) not in safe_text
    with pytest.raises(ValueError, match="canonical"):
        deserialize_advisor_qualification_artifact(" " + text)

    payload = json.loads(text)
    payload["report"]["stage2_similarity"]["mean_regret"] = 0.5
    tampered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with pytest.raises(ValidationError, match=r"inconsistent|integrity"):
        deserialize_advisor_qualification_artifact(tampered)

    payload = json.loads(text)
    payload["unexpected"] = True
    unknown = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with pytest.raises(ValidationError):
        deserialize_advisor_qualification_artifact(unknown)

    payload = json.loads(text)
    payload["report"]["ood_qualified"] = True
    payload["report_digest"] = hashlib.sha256(
        json.dumps(payload["report"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    semantically_forged = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with pytest.raises(ValidationError, match="gate outcomes"):
        deserialize_advisor_qualification_artifact(semantically_forged)

    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(outside, target_is_directory=True)
    linked_artifact = linked_parent / "qualification.json"
    with pytest.raises(ValueError, match="symbolic"):
        write_advisor_qualification_artifact(linked_artifact, artifact)
    assert not (outside / "qualification.json").exists()

    stored = tmp_path / "qualification.json"
    write_advisor_qualification_artifact(stored, artifact)
    assert load_advisor_qualification_artifact(stored) == artifact
    write_advisor_qualification_artifact(stored, artifact)


def test_committed_advisor_v2_qualification_evidence_is_canonical_and_not_qualified() -> None:
    evidence_path = ROOT / "docs" / "evidence" / "advisor_v2_qualification_20260821.json"
    artifact = load_advisor_qualification_artifact(evidence_path)

    assert artifact.artifact_digest == (
        "ffb6f2b5b29856e0e40fba0999803a931fd967d9027523abedd330f2c135a4cd"
    )
    assert artifact.report.qualification_status == "not_qualified"
    assert artifact.report.failed_gate_codes == (
        "stage2_regret_improvement",
        "stage3_regret_improvement",
        "locked_interval_coverage",
        "ood_detection_rate",
    )
    assert artifact.report.fallback_to_similarity_required is True
    assert artifact.report.automatic_promotion == "prohibited"
    assert artifact.report.operational_validity == "not_established"
    assert evidence_path.read_text(encoding="utf-8") == serialize_advisor_qualification_artifact(
        artifact
    )


def _v31_qualification_inputs() -> tuple[
    AdvisorV31QualificationReadinessArtifact,
    AdvisorV31QualificationApproval,
    tuple[AdvisorV31FamilyUtilityEvidence, ...],
    dict[str, AdvisorFamilyRole],
    dict[str, MechanismAwareTaskMetaFeatureVector],
]:
    digest = "a" * 64
    readiness = build_advisor_v31_qualification_readiness(
        amendment_digest=digest,
        source_execution_approval_digest=digest,
        source_execution_provenance_digest=digest,
        source_v3_readiness_digest=digest,
        source_registry_snapshot_digest=digest,
        analysis_provenance_digest=digest,
        remediation_approval_digest=digest,
        remediation_readiness_digest=digest,
        advisor_evidence_ready=True,
    )
    approval = AdvisorV31QualificationApproval(
        approval_reference="owner_approved_v31_synthetic",
        human_approved=True,
        locked_evaluation_access_authorized=True,
        ood_evaluation_access_authorized=True,
        amendment_digest=digest,
        source_execution_approval_digest=digest,
        source_execution_provenance_digest=digest,
        source_v3_readiness_digest=digest,
        source_registry_snapshot_digest=digest,
        analysis_provenance_digest=digest,
        remediation_approval_digest=digest,
        remediation_readiness_digest=digest,
        policy_digest=AdvisorV3QualificationPolicy().policy_digest,
        evaluation_algorithm_digest=readiness.evaluation_algorithm_digest,
    )
    roles = dict(advisor_v3_family_roles())
    vectors = {
        family_id: MechanismAwareTaskMetaFeatureVector(
            linkage_mode="link_only",
            assignment_constraint="one_to_one",
            label_volume_class="sparse",
            script_variation_rate=(index % 4) / 3,
            punctuation_variation_rate=(index % 5) / 4,
            tokenization_variation_rate=(index % 3) / 2,
            missingness_mean=(index % 7) / 6,
            missingness_asymmetry=(index % 6) / 5,
            frequency_concentration=(index % 8) / 7,
            candidate_ambiguity_scale=(index % 9) / 8,
            duplicate_signature_rate=(index % 10) / 9,
            planned_training_label_budget_scale=(index % 11) / 10,
        )
        for index, family_id in enumerate(sorted(roles))
    }
    evidence = tuple(
        AdvisorV31FamilyUtilityEvidence(
            family_id=family_id,
            family_role=role,
            recipe_token=token,
            mean_utility=0.25 + 0.1 * token_index + (family_index % 5) / 100,
            run_count=20,
            evidence_digest=hashlib.sha256(f"{family_id}:{token}".encode()).hexdigest(),
        )
        for family_index, (family_id, role) in enumerate(sorted(roles.items()))
        if role != "ood_holdout"
        for token_index, token in enumerate(REQUIRED_ADVISOR_RECIPE_TOKENS)
    )
    return readiness, approval, evidence, roles, vectors


def test_v31_qualification_requires_distinct_bound_approval_and_excludes_ood_metrics() -> None:
    readiness, approval, evidence, roles, vectors = _v31_qualification_inputs()
    with pytest.raises(ValidationError):
        AdvisorV31QualificationApproval.model_validate(
            {**approval.model_dump(mode="json"), "locked_evaluation_access_authorized": False}
        )
    report = evaluate_advisor_v31_aggregate_evidence(
        readiness=readiness,
        approval=approval,
        evidence=evidence,
        role_by_family=roles,
        family_vectors=vectors,
    )
    assert report.locked_evaluation_accessed is True
    assert report.ood_evaluation_accessed is True
    assert report.ood_recipe_metric_payloads_parsed_for_digest_integrity_only is True
    assert report.ood_recipe_metric_values_used_for_fit_threshold_or_qualification is False
    assert report.automatic_promotion == "prohibited"
    assert "family.advisor_v3" not in json.dumps(report.safe_summary(), sort_keys=True)

    with pytest.raises(ValueError, match="exact readiness evidence"):
        evaluate_advisor_v31_aggregate_evidence(
            readiness=readiness,
            approval=approval.model_copy(update={"analysis_provenance_digest": "b" * 64}),
            evidence=evidence,
            role_by_family=roles,
            family_vectors=vectors,
        )
    with pytest.raises(ValueError, match="OOD recipe metrics"):
        evaluate_advisor_v31_aggregate_evidence(
            readiness=readiness,
            approval=approval,
            evidence=(
                *evidence,
                AdvisorV31FamilyUtilityEvidence(
                    family_id=next(key for key, value in roles.items() if value == "ood_holdout"),
                    family_role="ood_holdout",
                    recipe_token=REQUIRED_ADVISOR_RECIPE_TOKENS[0],
                    mean_utility=0.5,
                    run_count=20,
                    evidence_digest="c" * 64,
                ),
            ),
            role_by_family=roles,
            family_vectors=vectors,
        )
