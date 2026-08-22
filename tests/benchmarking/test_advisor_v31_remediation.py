from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mapel_linkage.benchmarking import advisor_v31_remediation as remediation
from mapel_linkage.benchmarking.advisor_v3_catalogue import (
    build_advisor_v3_geometry_coherence,
)
from mapel_linkage.benchmarking.advisor_v3_execution import (
    advisor_v3_execution_provenance_digest,
)
from mapel_linkage.benchmarking.advisor_v31_remediation import (
    AdvisorV31ProtocolAmendmentManifest,
    AdvisorV31RemediationReadinessManifest,
    advisor_v31_analysis_provenance_digest,
    build_advisor_v31_protocol_amendment,
    frozen_advisor_v3_provenance_digest,
    load_committed_advisor_v31_protocol_amendment,
)
from mapel_linkage.benchmarking.contracts import (
    BenchmarkFailureRecord,
    BenchmarkRunRecord,
    BenchmarkRunStatus,
)
from mapel_linkage.benchmarking.registry import BenchmarkRegistry
from mapel_linkage.benchmarking.runner import BenchmarkPortfolioRunner
from mapel_linkage.cli.main import main

ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = ROOT / "docs" / "evidence" / "advisor_v31_protocol_amendment_20260821.json"


def _readiness_payload() -> dict[str, object]:
    digest = "0" * 64
    return {
        "amendment_digest": digest,
        "source_execution_approval_digest": digest,
        "source_execution_provenance_digest": digest,
        "source_v3_preregistration_digest": digest,
        "source_v3_readiness_digest": digest,
        "source_registry_snapshot_digest": digest,
        "analysis_provenance_digest": digest,
        "remediation_approval_digest": digest,
        "recomputed_geometry_coherence_digest": (
            build_advisor_v31_protocol_amendment().source_geometry_coherence_digest
        ),
        "source_completed_run_count": 11_760,
        "source_family_manifest_count": 84,
        "source_instance_manifest_count": 336,
        "source_catalogue_integrity_checked": True,
        "source_failure_sidecar_count": 6_730,
        "successful_qualification_required_family_count": 72,
        "successful_qualification_required_evidence_cell_count": 1_440,
        "successful_qualification_required_adapter_run_count": 4_320,
        "failed_qualification_required_adapter_run_count": 0,
        "missing_qualification_required_adapter_run_count": 0,
        "complete_ood_geometry_family_count": 12,
        "completed_ood_diagnostic_adapter_run_count": 720,
        "non_success_ood_diagnostic_adapter_run_count": 10,
        "ineligible_nonrequired_recipe_run_count": 6_720,
        "advisor_evidence_ready": True,
    }


def _run_record(*, engine_commit: str, run_id: str) -> BenchmarkRunRecord:
    digest = "1" * 64
    return BenchmarkRunRecord(
        run_id=run_id,
        family_id="family.synthetic.unit",
        instance_id="scenario.synthetic.unit",
        replicate_id="replicate.0000000",
        task_profile_digest=digest,
        pipeline_recipe_digest="2" * 64,
        engine_commit=engine_commit,
        dependency_lock_digest="3" * 64,
        environment_digest="4" * 64,
        random_seed=20260816,
        status=BenchmarkRunStatus.SUCCESS,
        aggregate_metrics_digest="5" * 64,
        runtime_ms=1,
    )


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, bytes], ...]:
    return tuple(
        (
            str(path.relative_to(root)),
            "directory" if path.is_dir() else "file",
            b"" if path.is_dir() else path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
    )


def _create_v3_source_layout(path: Path) -> None:
    for name in (
        "families",
        "instances",
        "runs",
        "metrics",
        "failures",
        "snapshots",
        "reports",
        "governance",
    ):
        (path / name).mkdir(parents=True, exist_ok=True)
    (path / "source-marker.txt").write_text("immutable synthetic source\n", encoding="utf-8")


def test_v31_amendment_is_canonical_and_does_not_claim_preregistration() -> None:
    loaded = load_committed_advisor_v31_protocol_amendment(AMENDMENT)
    expected = build_advisor_v31_protocol_amendment()

    assert loaded == expected
    assert loaded.amendment_id == "advisor_v31_role_evidence_20260821"
    assert loaded.performance_metric_values_accessed_to_select_amendment is False
    assert loaded.adapter_status_metadata_accessed_to_select_amendment is True
    assert loaded.failure_code_metadata_accessed_to_select_amendment is True
    assert loaded.ood_recipe_metric_use_for_fit_threshold_or_qualification == "prohibited"
    assert loaded.locked_and_ood_evaluation_requires_later_human_approval is True
    assert "preregistration" not in loaded.amendment_id


def test_v31_recomputes_geometry_and_fails_closed_on_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amendment = build_advisor_v31_protocol_amendment()
    assert remediation._recompute_and_validate_observable_geometry(amendment) == (
        amendment.source_geometry_coherence_digest
    )
    observed = build_advisor_v3_geometry_coherence().model_copy(
        update={"selected_distance_threshold": 0.0}
    )
    monkeypatch.setattr(
        remediation,
        "validate_advisor_v3_geometry_coherence",
        lambda: observed,
    )
    with pytest.raises(ValueError, match="observable geometry conflicts"):
        remediation._recompute_and_validate_observable_geometry(amendment)


def test_v31_readiness_allows_ood_diagnostics_but_not_required_role_gaps() -> None:
    ready = AdvisorV31RemediationReadinessManifest.model_validate(_readiness_payload())
    assert ready.advisor_evidence_ready is True
    assert ready.non_success_ood_diagnostic_adapter_run_count == 10

    required_gap = _readiness_payload()
    required_gap.update(
        {
            "successful_qualification_required_adapter_run_count": 4_319,
            "failed_qualification_required_adapter_run_count": 1,
            "source_failure_sidecar_count": 6_731,
        }
    )
    with pytest.raises(ValidationError, match="fail closed"):
        AdvisorV31RemediationReadinessManifest.model_validate(required_gap)


def test_v31_amendment_rejects_ood_metric_use_and_stale_bindings() -> None:
    payload = build_advisor_v31_protocol_amendment().model_dump(mode="json")
    payload["ood_recipe_metric_use_for_fit_threshold_or_qualification"] = "allowed"
    with pytest.raises(ValidationError):
        AdvisorV31ProtocolAmendmentManifest.model_validate(payload)

    payload = build_advisor_v31_protocol_amendment().model_dump(mode="json")
    payload["source_design_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="stale or conflicting"):
        AdvisorV31ProtocolAmendmentManifest.model_validate(payload)


def test_frozen_source_provenance_does_not_substitute_current_source() -> None:
    old_records = (
        _run_record(engine_commit="a" * 64, run_id="run.synthetic.one"),
        _run_record(engine_commit="a" * 64, run_id="run.synthetic.two"),
    )
    frozen_digest = frozen_advisor_v3_provenance_digest(old_records)
    current_runner = BenchmarkPortfolioRunner()
    current_runner._engine_commit = "b" * 64

    assert frozen_digest != advisor_v3_execution_provenance_digest(current_runner)
    assert frozen_advisor_v3_provenance_digest(old_records) == frozen_digest

    mixed = (old_records[0], _run_record(engine_commit="c" * 64, run_id="run.synthetic.three"))
    with pytest.raises(ValueError, match="mixed execution provenance"):
        frozen_advisor_v3_provenance_digest(mixed)
    assert advisor_v31_analysis_provenance_digest() == advisor_v3_execution_provenance_digest(
        BenchmarkPortfolioRunner()
    )


def test_frozen_failure_sidecar_must_match_run_identity_recipe_and_code() -> None:
    record = BenchmarkRunRecord(
        run_id="run.synthetic.failure",
        family_id="family.synthetic.unit",
        instance_id="scenario.synthetic.unit",
        replicate_id="replicate.0000000",
        task_profile_digest="1" * 64,
        pipeline_recipe_digest="2" * 64,
        engine_commit="a" * 64,
        dependency_lock_digest="3" * 64,
        environment_digest="4" * 64,
        random_seed=20260816,
        status=BenchmarkRunStatus.FAILED_FIT,
        failure_code="ML-RANK-016",
    )
    matching = BenchmarkFailureRecord(
        run_id=record.run_id,
        family_id=record.family_id,
        instance_id=record.instance_id,
        replicate_id=record.replicate_id,
        recipe_id="recipe.xgboost_ranker",
        status=record.status,
        failure_code="ML-RANK-016",
        error_message="Aggregate synthetic fit failure.",
    )
    remediation._validate_failure_sidecar_binding(
        record,
        matching,
        recipe_id="recipe.xgboost_ranker",
    )

    conflicting = matching.model_copy(update={"failure_code": "ML-RANK-999"})
    with pytest.raises(ValueError, match="conflicts with its run record"):
        remediation._validate_failure_sidecar_binding(
            record,
            conflicting,
            recipe_id="recipe.xgboost_ranker",
        )


def test_v31_destination_rejects_non_governance_content_and_symlinks(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path / "remediation")
    unexpected = registry.runs_dir / "unexpected.json"
    unexpected.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="governance-only"):
        remediation._reject_non_governance_destination_content(registry)
    unexpected.unlink()

    linked = registry.root_directory / "linked"
    linked.symlink_to(registry.runs_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        remediation._reject_non_governance_destination_content(registry)


def test_v31_sidecar_coverage_rejects_orphans_non_json_and_symlinks(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "metrics"
    directory.mkdir()
    retained = directory / "run.synthetic.one.json"
    retained.write_text("{}\n", encoding="utf-8")
    expected = frozenset({"run.synthetic.one"})
    remediation._validate_exact_artifact_coverage(
        directory,
        expected_run_ids=expected,
        artifact_name="metric-sidecar",
    )

    extra = directory / "orphan.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="coverage is not exact"):
        remediation._validate_exact_artifact_coverage(
            directory,
            expected_run_ids=expected,
            artifact_name="metric-sidecar",
        )
    extra.unlink()

    non_json = directory / "unexpected.txt"
    non_json.write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-JSON"):
        remediation._validate_exact_artifact_coverage(
            directory,
            expected_run_ids=expected,
            artifact_name="metric-sidecar",
        )
    non_json.unlink()

    linked = directory / "linked.json"
    linked.symlink_to(retained)
    with pytest.raises(ValueError, match="path-unsafe"):
        remediation._validate_exact_artifact_coverage(
            directory,
            expected_run_ids=expected,
            artifact_name="metric-sidecar",
        )


def test_v31_amendment_rejects_noncanonical_and_symlink_paths(tmp_path: Path) -> None:
    expected = build_advisor_v31_protocol_amendment()
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(expected.model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        load_committed_advisor_v31_protocol_amendment(noncanonical)

    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps(expected.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    linked = tmp_path / "linked.json"
    linked.symlink_to(canonical)
    with pytest.raises(FileNotFoundError, match="unavailable"):
        load_committed_advisor_v31_protocol_amendment(linked)


def test_v31_cli_requires_approval_and_private_distinct_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    without_approval = main(
        (
            "audit-advisor-v31-remediation",
            "--project-root",
            str(ROOT),
            "--source-registry-dir",
            "private/benchmark_registry/source",
            "--remediation-registry-dir",
            "private/benchmark_registry/remediation",
            "--approval-reference",
            "synthetic-v31-negative",
        )
    )
    assert without_approval == 2
    assert "Explicit human remediation approval" in capsys.readouterr().err

    outside = main(
        (
            "audit-advisor-v31-remediation",
            "--project-root",
            str(ROOT),
            "--source-registry-dir",
            str(tmp_path),
            "--remediation-registry-dir",
            "private/benchmark_registry/remediation",
            "--approve-remediation",
            "--approval-reference",
            "synthetic-v31-path-negative",
        )
    )
    assert outside == 2
    assert "failed closed" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("source_relative", "remediation_relative"),
    (
        (
            Path("private/benchmark_registry/source"),
            Path("private/benchmark_registry/source/remediation"),
        ),
        (
            Path("private/benchmark_registry/remediation/source"),
            Path("private/benchmark_registry/remediation"),
        ),
    ),
)
def test_v31_cli_rejects_nested_registry_paths_without_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    source_relative: Path,
    remediation_relative: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname = 'synthetic-v31-test'\n",
        encoding="utf-8",
    )
    _create_v3_source_layout(project_root / source_relative)
    before = _tree_snapshot(project_root)

    result = main(
        (
            "audit-advisor-v31-remediation",
            "--project-root",
            str(project_root),
            "--source-registry-dir",
            str(source_relative),
            "--remediation-registry-dir",
            str(remediation_relative),
            "--approve-remediation",
            "--approval-reference",
            "synthetic-v31-overlap-negative",
        )
    )

    assert result == 2
    assert "failed closed" in capsys.readouterr().err
    assert _tree_snapshot(project_root) == before
