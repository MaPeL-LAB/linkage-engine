"""Unit tests for Stage-3 Learned Meta-Ranking Strategy Advisor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mapel_linkage.benchmarking.contracts import BenchmarkRunRecord, BenchmarkRunStatus
from mapel_linkage.benchmarking.generator import BenchmarkScenarioGenerator
from mapel_linkage.benchmarking.registry import BenchmarkRegistry
from mapel_linkage.benchmarking.runner import BenchmarkPortfolioRunner
from mapel_linkage.benchmarking.seed_corpus import generate_and_run_seed_corpus
from mapel_linkage.configuration import compile_config, load_config
from mapel_linkage.configuration.compiler import ExecutionPlan
from mapel_linkage.domain.errors import AdvisorError
from mapel_linkage.profiling import build_preflight_task_profile
from mapel_linkage.recommendation.contracts import (
    MetaRankingAdvisoryReport,
    RecommendationIntent,
    RuntimeDependency,
)
from mapel_linkage.recommendation.eligibility import AdvisorContext
from mapel_linkage.recommendation.meta_ranker import (
    LearnedMetaRankerModel,
    MetaRankingLinkageAdvisor,
    _has_complete_required_evidence_grid,
)
from tests.helpers import EXAMPLE_CONFIG, ROOT


def _plan() -> ExecutionPlan:
    loaded = load_config(EXAMPLE_CONFIG)
    return compile_config(loaded.config, project_root=ROOT)


def _complete_required_grid(
    *, replicates: int = 5
) -> tuple[tuple[BenchmarkRunRecord, ...], frozenset[str], dict[str, str]]:
    recipe_tokens = {
        "a" * 64: "fellegi_sunter",
        "b" * 64: "xgboost_classifier",
        "c" * 64: "xgboost_ranker",
    }
    instance_ids = frozenset(f"instance.synthetic.{index:03d}" for index in range(280))
    records = tuple(
        BenchmarkRunRecord(
            run_id=f"run.i{instance_index:03d}.r{replicate_index:02d}.m{model_index}",
            family_id=f"family.synthetic.{instance_index // 5:03d}",
            instance_id=f"instance.synthetic.{instance_index:03d}",
            replicate_id=f"replicate.{replicate_index:07d}",
            task_profile_digest="d" * 64,
            pipeline_recipe_digest=recipe_digest,
            engine_commit="e" * 64,
            dependency_lock_digest="f" * 64,
            environment_digest="1" * 64,
            random_seed=instance_index * 10 + replicate_index,
            status=BenchmarkRunStatus.SUCCESS,
            aggregate_metrics_digest="2" * 64,
            stage_artifact_manifest_digest="3" * 64,
            runtime_ms=1,
            peak_memory_mb=1,
        )
        for instance_index in range(280)
        for replicate_index in range(replicates)
        for model_index, recipe_digest in enumerate(recipe_tokens)
    )
    return records, instance_ids, recipe_tokens


@pytest.fixture(scope="module")
def populated_registry(tmp_path_factory: pytest.TempPathFactory) -> Path:
    reg_dir = tmp_path_factory.mktemp("meta_ranker_registry")
    generate_and_run_seed_corpus(
        registry_directory=reg_dir,
        generator=BenchmarkScenarioGenerator(),
        runner=BenchmarkPortfolioRunner(),
        families=("family.typo_stress", "family.missingness_regime", "family.date_variation"),
        instances=("instance.typo_low", "instance.missing_zero", "instance.date_shift_low"),
        replicates=1,
    )
    return reg_dir


def test_learned_meta_ranker_model_fit_predict() -> None:
    model = LearnedMetaRankerModel()
    np.random.seed(42)
    X = np.random.randn(16, 5)
    true_w = np.array([0.2, -0.1, 0.4, 0.05, -0.3])
    y = np.clip(0.6 + X @ true_w + np.random.randn(16) * 0.05, 0.0, 1.0)
    X_conformal = np.random.randn(4, 5)
    y_conformal = np.clip(0.6 + X_conformal @ true_w, 0.0, 1.0)

    model.fit(
        X,
        y,
        X_conformal=X_conformal,
        y_conformal=y_conformal,
        training_family_ids=tuple(f"family.train.{index // 4}" for index in range(16)),
        conformal_family_ids=("family.conformal.1",) * 4,
        coverage_level=0.90,
    )

    assert model.weights is not None
    assert model.trained_run_count == 16
    assert model.trained_family_count == 4
    assert model.conformal_run_count == 4
    assert model.conformal_family_count == 1
    assert 0.0 < model.conformal_residual_quantile < 0.5

    X_test = np.random.randn(5, 5)
    preds, lowers, uppers = model.predict_utility(X_test)

    assert len(preds) == 5
    assert all(0.0 <= p <= 1.0 for p in preds)
    assert all(low <= p <= up for p, low, up in zip(preds, lowers, uppers, strict=True))

    X_locked = np.random.randn(3, 5)
    y_locked = np.clip(0.6 + X_locked @ true_w, 0.0, 1.0)
    model.evaluate_locked(
        X_locked,
        y_locked,
        locked_family_ids=("family.locked.1",) * 3,
        prohibited_family_ids=frozenset({"family.train.0", "family.train.1", "family.conformal.1"}),
    )
    assert model.locked_evaluation_run_count == 3
    assert model.locked_evaluation_family_count == 1
    assert model.locked_mean_absolute_error is not None


def test_meta_ranker_rejects_family_leakage_and_constant_evidence() -> None:
    X_train = np.asarray([[0.0], [1.0], [0.5], [0.2]], dtype=np.float64)
    y_train = np.asarray([0.1, 0.8, 0.6, 0.3], dtype=np.float64)
    X_conformal = np.asarray([[0.3], [0.7]], dtype=np.float64)
    y_conformal = np.asarray([0.4, 0.7], dtype=np.float64)
    model = LearnedMetaRankerModel()

    with pytest.raises(ValueError, match="disjoint"):
        model.fit(
            X_train,
            y_train,
            X_conformal=X_conformal,
            y_conformal=y_conformal,
            training_family_ids=("family.a", "family.a", "family.b", "family.b"),
            conformal_family_ids=("family.b", "family.c"),
        )

    with pytest.raises(ValueError, match="variation"):
        model.fit(
            X_train,
            np.full(4, 0.5),
            X_conformal=X_conformal,
            y_conformal=y_conformal,
            training_family_ids=("family.a", "family.a", "family.b", "family.b"),
            conformal_family_ids=("family.c", "family.c"),
        )


def test_meta_ranker_grid_rejects_failure_missingness_and_mixed_provenance() -> None:
    records, instance_ids, recipe_tokens = _complete_required_grid()
    assert _has_complete_required_evidence_grid(
        records,
        expected_instance_ids=instance_ids,
        recipe_token_by_digest=recipe_tokens,
    )

    failed_payload = records[0].model_dump(mode="json")
    failed_payload.update(
        {
            "status": "failed_fit",
            "failure_code": "ML-BENCH-FAILED-FIT",
            "aggregate_metrics_digest": None,
            "stage_artifact_manifest_digest": None,
        }
    )
    failed = BenchmarkRunRecord.model_validate(failed_payload)
    assert not _has_complete_required_evidence_grid(
        (failed, *records[1:]),
        expected_instance_ids=instance_ids,
        recipe_token_by_digest=recipe_tokens,
    )
    assert not _has_complete_required_evidence_grid(
        records[:-1],
        expected_instance_ids=instance_ids,
        recipe_token_by_digest=recipe_tokens,
    )

    mixed_payload = records[0].model_dump(mode="json")
    mixed_payload["engine_commit"] = "4" * 64
    mixed = BenchmarkRunRecord.model_validate(mixed_payload)
    assert not _has_complete_required_evidence_grid(
        (mixed, *records[1:]),
        expected_instance_ids=instance_ids,
        recipe_token_by_digest=recipe_tokens,
    )

    sparse_records, sparse_instances, sparse_tokens = _complete_required_grid(replicates=4)
    assert not _has_complete_required_evidence_grid(
        sparse_records,
        expected_instance_ids=sparse_instances,
        recipe_token_by_digest=sparse_tokens,
    )


def test_locked_evaluation_rejects_training_or_conformal_family_reuse() -> None:
    model = LearnedMetaRankerModel()
    model.fit(
        np.asarray([[0.0], [1.0], [0.2], [0.8]], dtype=np.float64),
        np.asarray([0.1, 0.9, 0.3, 0.7], dtype=np.float64),
        X_conformal=np.asarray([[0.4], [0.6]], dtype=np.float64),
        y_conformal=np.asarray([0.4, 0.6], dtype=np.float64),
        training_family_ids=("family.a", "family.a", "family.b", "family.b"),
        conformal_family_ids=("family.c", "family.c"),
    )

    with pytest.raises(ValueError, match="Locked"):
        model.evaluate_locked(
            np.asarray([[0.5]], dtype=np.float64),
            np.asarray([0.5], dtype=np.float64),
            locked_family_ids=("family.c",),
            prohibited_family_ids=frozenset({"family.a", "family.b", "family.c"}),
        )


def test_meta_ranker_fallback_on_empty_registry(tmp_path: Path) -> None:
    plan = _plan()
    profile = build_preflight_task_profile(plan)
    context = AdvisorContext(
        intent=RecommendationIntent.DEVELOP_NEW_RECIPE,
        verified_labels_available=True,
        approved_recipe_available=False,
        protected_out_of_fold_predictions_available=False,
        available_runtimes=(RuntimeDependency.CORE,),
        approved_artifact_model_ids=(),
    )

    advisor = MetaRankingLinkageAdvisor(registry=None)
    report = advisor.advise(plan, context=context, profile=profile)

    assert report.fallback_to_similarity is True
    assert report.recommendation_authority == "advisory_only"
    assert report.decision_authority == "none"
    assert report.assignment_authority == "none"
    assert report.merge_authority == "none"
    assert report.automatic_promotion == "prohibited"
    assert report.operational_validity == "not_established"


def test_meta_ranker_with_populated_registry(populated_registry: Path) -> None:
    registry = BenchmarkRegistry(populated_registry)
    plan = _plan()
    profile = build_preflight_task_profile(plan)
    context = AdvisorContext(
        intent=RecommendationIntent.DEVELOP_NEW_RECIPE,
        verified_labels_available=False,
        approved_recipe_available=False,
        protected_out_of_fold_predictions_available=False,
        available_runtimes=(RuntimeDependency.CORE,),
        approved_artifact_model_ids=(),
    )

    advisor = MetaRankingLinkageAdvisor(
        registry=registry,
        max_ood_distance=0.60,
    )
    report = advisor.advise(plan, context=context, profile=profile)

    assert isinstance(report, MetaRankingAdvisoryReport)
    assert report.fallback_to_similarity is True
    assert report.meta_model_type == "none"
    assert "scenario-replicate-complete" in str(report.fallback_reason).lower()
    assert advisor.last_fitted_model is None
    assert report.recommendation.mandatory_baseline_candidate_id in {
        item.candidate_id for item in report.recommendation.shortlist
    }
    assert any(
        item.pair_model_family == "fellegi_sunter" for item in report.recommendation.shortlist
    )
    assert len(report.recommendation.shortlist) > 0

    # Safe summary check
    summary = report.safe_summary()
    assert summary["report_schema_version"] == "1"
    assert summary["recommendation_authority"] == "advisory_only"


def test_meta_ranker_test_partition_rejection() -> None:
    advisor = MetaRankingLinkageAdvisor(registry=None)
    plan = _plan()
    mock_context: Any = type("MockContext", (), {"test_partition_used": True})()

    with pytest.raises(AdvisorError, match="ML-ADVISOR-001"):
        advisor.advise(plan, context=mock_context)
