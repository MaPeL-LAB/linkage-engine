"""Assurance tests for advisory-only active synthetic benchmark planning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mapel_linkage.benchmarking import (
    BenchmarkPortfolioRunner,
    BenchmarkRegistry,
    BenchmarkScenarioGenerator,
)
from mapel_linkage.configuration import compile_config, load_config
from mapel_linkage.configuration.compiler import ExecutionPlan
from mapel_linkage.domain.errors import AdvisorError
from mapel_linkage.recommendation import (
    ActiveBenchmarkPlanner,
    AdvisorContext,
    BenchmarkGapAnalyzer,
    CoverageDimension,
    ExperimentExecutionApproval,
    ExperimentPlan,
    ExperimentPlanningStatus,
    ExperimentPlanningTrigger,
    MetaModelRefitStatus,
    MetaRankingAdvisoryReport,
    MetaRankingLinkageAdvisor,
    RecommendationIntent,
    RuntimeDependency,
    SimilarityAdvisoryReport,
    SimilarityLinkageAdvisor,
    execute_planned_experiments,
)
from tests.helpers import EXAMPLE_CONFIG, ROOT

_SYNTHETIC_SEED = 20260816


def _linkage_plan() -> ExecutionPlan:
    loaded = load_config(EXAMPLE_CONFIG)
    return compile_config(loaded.config, project_root=ROOT)


def _advisor_context() -> AdvisorContext:
    return AdvisorContext(
        intent=RecommendationIntent.PLAN_BENCHMARK,
        verified_labels_available=True,
        available_runtimes=(RuntimeDependency.CORE,),
    )


def test_gap_analyzer_uses_catalogue_without_generating_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = BenchmarkRegistry(tmp_path / "registry")
    generator = BenchmarkScenarioGenerator()

    def reject_record_generation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Coverage analysis must not generate record-level bundles.")

    monkeypatch.setattr(generator, "generate", reject_record_generation)
    analysis = BenchmarkGapAnalyzer(registry, generator=generator).analyze()

    assert {cell.dimension for cell in analysis.cells} == set(CoverageDimension)
    assert all(cell.density == 0.0 for cell in analysis.cells)
    assert all(cell.covered_instance_count == 0 for cell in analysis.cells)
    assert analysis.retained_run_count == 0
    assert analysis.contains_record_values is False
    assert analysis.contains_record_identifiers is False
    assert analysis.contains_candidate_pairs is False
    assert analysis.latent_values_persisted is False


def test_planner_is_stable_advisory_only_and_protects_held_out_families(
    tmp_path: Path,
) -> None:
    registry = BenchmarkRegistry(tmp_path / "registry")
    generator = BenchmarkScenarioGenerator()
    profile = generator.build_task_profile("instance.typo_low")

    first = ActiveBenchmarkPlanner(registry, generator=generator).plan(target_profile=profile)
    second = ActiveBenchmarkPlanner(registry, generator=generator).plan(target_profile=profile)

    assert first.plan_digest == second.plan_digest
    assert first.experiments == second.experiments
    assert first.planning_status is ExperimentPlanningStatus.READY_FOR_HUMAN_APPROVAL
    assert first.trigger is ExperimentPlanningTrigger.COVERAGE_GAP
    assert first.recommendation_authority == "advisory_only"
    assert first.decision_authority == "none"
    assert first.assignment_authority == "none"
    assert first.merge_authority == "none"
    assert first.automatic_promotion == "prohibited"
    assert first.execution_authority == "explicit_human_approval_required"
    held_out = {
        family.family_id for family in generator.list_families() if family.prospectively_held_out
    }
    assert held_out.isdisjoint(item.family_id for item in first.experiments)
    assert all(0 <= item.base_seed <= 4_294_967_295 for item in first.experiments)

    unsafe = first.model_dump(mode="json")
    unsafe["decision_authority"] = "relationship_decisions"
    with pytest.raises(ValidationError):
        ExperimentPlan.model_validate(unsafe)


def test_planner_triggers_on_similarity_ood_and_meta_ranker_fallback(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path / "registry")
    generator = BenchmarkScenarioGenerator()
    profile = generator.build_task_profile("instance.typo_low")
    linkage_plan = _linkage_plan()
    context = _advisor_context()

    similarity = SimilarityLinkageAdvisor(registry=None, generator=generator).recommend(
        linkage_plan,
        context=context,
        profile=profile,
    )
    similarity_plan = ActiveBenchmarkPlanner(registry, generator=generator).plan(
        similarity,
        target_profile=profile,
    )
    assert similarity_plan.trigger is ExperimentPlanningTrigger.SIMILARITY_OUT_OF_DISTRIBUTION
    assert similarity_plan.experiments

    meta = MetaRankingLinkageAdvisor(registry=None, generator=generator).advise(
        linkage_plan,
        context=context,
        profile=profile,
    )
    meta_plan = ActiveBenchmarkPlanner(registry, generator=generator).plan(
        meta,
        target_profile=profile,
    )
    assert meta_plan.trigger is ExperimentPlanningTrigger.META_RANKER_FALLBACK
    assert meta_plan.experiments

    wide_payload = meta.model_dump(mode="json")
    wide_payload["fallback_to_similarity"] = False
    wide_payload["fallback_reason"] = None
    wide_payload["meta_model_type"] = "ridge_meta_ranker_v1"
    wide_payload["meta_model_trained_runs"] = 3
    wide_payload["predicted_candidate_utilities"] = {
        "candidate.synthetic": {
            "candidate_id": "candidate.synthetic",
            "predicted_utility": 0.5,
            "uncertainty_lower_bound": 0.1,
            "uncertainty_upper_bound": 0.9,
            "conformal_coverage_level": 0.9,
        }
    }
    wide_report = MetaRankingAdvisoryReport.model_validate(wide_payload)
    wide_plan = ActiveBenchmarkPlanner(registry, generator=generator).plan(
        wide_report,
        target_profile=profile,
    )
    assert wide_plan.trigger is ExperimentPlanningTrigger.META_RANKER_WIDE_INTERVAL
    assert wide_plan.experiments


def test_planner_does_not_trigger_when_similarity_uncertainty_is_low(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path / "registry")
    generator = BenchmarkScenarioGenerator()
    profile = generator.build_task_profile("instance.typo_low")
    baseline = SimilarityLinkageAdvisor(registry=None, generator=generator).recommend(
        _linkage_plan(),
        context=_advisor_context(),
        profile=profile,
    )
    payload = baseline.model_dump(mode="json")
    payload["out_of_distribution"] = False
    payload["out_of_distribution_score"] = 0.1
    low_uncertainty = SimilarityAdvisoryReport.model_validate(payload)

    plan = ActiveBenchmarkPlanner(registry, generator=generator).plan(
        low_uncertainty,
        target_profile=profile,
    )

    assert plan.planning_status is ExperimentPlanningStatus.NOT_TRIGGERED
    assert plan.trigger is ExperimentPlanningTrigger.NOT_TRIGGERED
    assert plan.experiments == ()


def test_analyzer_rejects_run_evidence_without_matching_manifests(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path / "registry")
    generator = BenchmarkScenarioGenerator()
    runner = BenchmarkPortfolioRunner()
    bundle = generator.generate("instance.typo_low", seed=_SYNTHETIC_SEED)
    recipe = runner.list_recipes()[0]
    result = runner.run_single(
        bundle=bundle,
        recipe=recipe,
        replicate_id="replicate.0000000",
        seed=_SYNTHETIC_SEED,
    )
    registry.save_run_record(result.record, metrics=result.metrics, failure=result.failure)

    with pytest.raises(AdvisorError, match="ML-ADVISOR-042"):
        BenchmarkGapAnalyzer(registry, generator=generator).analyze()


def test_execution_requires_bound_approval_appends_and_refits(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path / "registry")
    generator = BenchmarkScenarioGenerator()
    profile = generator.build_task_profile("instance.typo_low")
    plan = ActiveBenchmarkPlanner(registry, generator=generator).plan(target_profile=profile)

    with pytest.raises(ValidationError):
        ExperimentExecutionApproval(
            approval_id="approval.synthetic_denied",
            plan_digest=plan.plan_digest,
            approved_by_human=False,  # type: ignore[arg-type]
        )

    wrong_approval = ExperimentExecutionApproval(
        approval_id="approval.synthetic_wrong",
        plan_digest="0" * 64,
        approved_by_human=True,
    )
    with pytest.raises(AdvisorError, match="ML-ADVISOR-045"):
        execute_planned_experiments(
            plan,
            approval=wrong_approval,
            registry=registry,
            linkage_plan=_linkage_plan(),
            advisor_context=_advisor_context(),
            target_profile=profile,
            generator=generator,
        )
    assert registry.list_run_records() == ()

    approval = ExperimentExecutionApproval(
        approval_id="approval.synthetic_test",
        plan_digest=plan.plan_digest,
        approved_by_human=True,
    )
    report = execute_planned_experiments(
        plan,
        approval=approval,
        registry=registry,
        linkage_plan=_linkage_plan(),
        advisor_context=_advisor_context(),
        target_profile=profile,
        generator=generator,
    )

    records = registry.list_run_records()
    assert report.appended_run_count == len(records) > 0
    assert len({record.run_id for record in records}) == len(records)
    assert report.registry_snapshot_digest_before != report.registry_snapshot_digest_after
    assert report.meta_model_refit_status is MetaModelRefitStatus.FITTED
    assert report.refitted_meta_model_digest is not None
    assert report.meta_ranking_report_digest is not None
    assert report.meta_model_trained_run_count > 0
    assert report.recommendation_authority == "advisory_only"
    assert report.decision_authority == "none"
    assert report.assignment_authority == "none"
    assert report.merge_authority == "none"
    assert report.automatic_promotion == "prohibited"

    run_count = len(records)
    with pytest.raises(AdvisorError, match="ML-ADVISOR-047"):
        execute_planned_experiments(
            plan,
            approval=approval,
            registry=registry,
            linkage_plan=_linkage_plan(),
            advisor_context=_advisor_context(),
            target_profile=profile,
            generator=generator,
        )
    assert len(registry.list_run_records()) == run_count

    with pytest.raises(FileExistsError, match="already exists"):
        registry.save_run_record(records[0])


def test_plan_benchmarks_cli_emits_safe_output_and_rejects_invalid_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mapel_linkage.cli.main import main

    registry_dir = tmp_path / "registry"
    profile_path = tmp_path / "target-profile.json"
    assert (
        main(
            [
                "profile-job",
                "--config",
                str(EXAMPLE_CONFIG),
                "--project-root",
                str(ROOT),
            ]
        )
        == 0
    )
    profile_output = capsys.readouterr().out
    profile_path.write_text(profile_output, encoding="utf-8")

    assert (
        main(
            [
                "plan-benchmarks",
                "--registry-dir",
                str(registry_dir),
                "--target-profile",
                str(profile_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["planning_status"] == "ready_for_human_approval"
    assert payload["recommendation_authority"] == "advisory_only"
    assert payload["decision_authority"] == "none"
    assert payload["assignment_authority"] == "none"
    assert payload["merge_authority"] == "none"
    assert payload["automatic_promotion"] == "prohibited"
    assert payload["contains_record_values"] is False
    rendered = json.dumps(payload, sort_keys=True)
    assert str(registry_dir) not in rendered
    assert str(profile_path) not in rendered

    invalid_path = tmp_path / "invalid-profile.json"
    invalid_path.write_text("not-json", encoding="utf-8")
    assert (
        main(
            [
                "plan-benchmarks",
                "--registry-dir",
                str(registry_dir),
                "--target-profile",
                str(invalid_path),
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "ML-ADVISOR-054" in error
    assert "not-json" not in error
    assert str(invalid_path) not in error


def test_registry_rejects_symlinked_managed_directories(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_root = tmp_path / "registry"
    redirected_runs = tmp_path / "redirected-runs"
    registry_root.mkdir()
    redirected_runs.mkdir()
    (registry_root / "runs").symlink_to(redirected_runs, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic links"):
        BenchmarkRegistry(registry_root)

    from mapel_linkage.cli.main import main

    assert main(["plan-benchmarks", "--registry-dir", str(registry_root)]) == 2
    error = capsys.readouterr().err
    assert "ML-ADVISOR-054" in error
    assert str(registry_root) not in error
    assert "Traceback" not in error
