from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import mapel_linkage.pipeline.synthetic_mode_workflow as workflow_module
from mapel_linkage.calibration import write_calibrator_artifact
from mapel_linkage.configuration import compile_config, load_config_text
from mapel_linkage.domain.errors import BoostedTreeError, PipelineError
from mapel_linkage.models.boosted import (
    BoostedFeatureMatrix,
    read_xgboost_artifact,
    write_xgboost_artifact,
)
from mapel_linkage.pipeline.mode_artifacts import (
    deserialize_mode_orchestration_artifact,
    deserialize_mode_run_artifact,
)
from mapel_linkage.pipeline.recipe_io import deserialize_pipeline_recipe
from mapel_linkage.pipeline.synthetic_mode_workflow import SyntheticModeWorkflowRunner
from mapel_linkage.synthetic import SyntheticGenerationConfig
from tests.helpers import valid_payload, yaml_text


def _link_mode_payload(constraint: str) -> dict[str, Any]:
    payload = valid_payload()
    payload["project"]["assignment_constraint"] = constraint
    payload["assignment"]["constraint"] = constraint
    payload["assignment"]["solver"] = (
        "unconstrained" if constraint == "unconstrained" else "ortools_min_cost_flow"
    )
    payload["calibration"]["source_model"] = "xgb_pair_classifier"
    payload["mode_orchestration"] = {
        "artifact_schema_version": "1",
        "implementation": "synthetic_mode_v1",
        "pair_model_id": "xgb_pair_classifier",
    }
    return payload


def _generation(*, right_duplicate_count: int = 100) -> SyntheticGenerationConfig:
    return SyntheticGenerationConfig(
        seed=20260816,
        entity_count=100,
        left_only_count=4,
        right_only_count=4,
        duplicate_count=100,
        right_duplicate_count=right_duplicate_count,
        competing_candidate_count=12,
        source_a_missing_rate=0.05,
        source_b_missing_rate=0.20,
        source_b_typo_rate=0.35,
        source_b_date_shift_rate=0.20,
    )


def _dedupe_payload() -> dict[str, Any]:
    payload = valid_payload()
    payload["project"]["linkage_mode"] = "dedupe_only"
    payload["project"]["assignment_constraint"] = "unconstrained"
    payload["assignment"]["constraint"] = "unconstrained"
    payload["assignment"]["solver"] = "unconstrained"
    payload["calibration"]["source_model"] = "xgb_pair_classifier"
    payload["datasets"] = [payload["datasets"][0]]
    for variable in payload["variables"]:
        variable["source_columns"] = {"source_a": variable["source_columns"]["source_a"]}
    payload["labels"]["source"]["entity_group_columns"] = {"source_a": "synthetic_entity_id_a"}
    payload["labels"]["source"]["household_group_columns"] = {
        "source_a": "synthetic_household_id_a"
    }
    payload["mode_orchestration"] = {
        "artifact_schema_version": "1",
        "implementation": "synthetic_mode_v1",
        "pair_model_id": "xgb_pair_classifier",
        "deduplication": {
            "algorithm": "clique",
            "minimum_probability": 0.75,
            "no_match_utility": 0.0,
            "maximum_cluster_size": 100,
            "maximum_candidate_edges": 100000,
            "deterministic_tie_breaking": True,
        },
    }
    return payload


def _link_and_dedupe_payload() -> dict[str, Any]:
    payload = valid_payload()
    payload["project"]["linkage_mode"] = "link_and_dedupe"
    payload["calibration"]["source_model"] = "xgb_pair_classifier"
    payload["mode_orchestration"] = {
        "artifact_schema_version": "1",
        "implementation": "synthetic_mode_v1",
        "pair_model_id": "xgb_pair_classifier",
        "deduplication": {
            "algorithm": "clique",
            "minimum_probability": 0.75,
            "no_match_utility": 0.0,
            "maximum_cluster_size": 100,
            "maximum_candidate_edges": 100000,
            "deterministic_tie_breaking": True,
        },
    }
    return payload


@pytest.mark.parametrize("constraint", ["many_to_one", "one_to_many", "unconstrained"])
def test_link_only_extended_modes_replay_fitted_evidence(
    tmp_path: Path,
    constraint: str,
) -> None:
    config = load_config_text(
        yaml_text(_link_mode_payload(constraint)), source_format="yaml"
    ).config
    plan = compile_config(config, project_root=tmp_path)

    result = SyntheticModeWorkflowRunner.run_link_only(
        plan,
        generation=_generation(),
    )

    assert result.assignment_constraint == constraint
    assert result.recipe.configuration_digest == plan.configuration_digest
    assert result.recipe.champion_artifact_digest == result.model_artifact.model_digest
    assert result.recipe.calibrator_digest == result.calibrator_artifact.calibrator_digest
    assert result.model_artifact.decision_authority == "evidence_only"
    assert result.calibrator_artifact.probability_status == "calibrated_probability"
    assert result.inference.assignment_result.source_record_count < 204
    assert result.inference.pair_count == len(result.evidence_audit.prepared_inference_pair_digests)
    assert result.evidence_audit.pair_digests_for("decision") == frozenset(
        result.evidence_audit.prepared_inference_pair_digests
    )
    for protected_partition in ("training", "validation", "calibration", "test"):
        assert not (
            result.evidence_audit.pair_digests_for("decision")
            & result.evidence_audit.pair_digests_for(protected_partition)
        )
        assert not (
            result.evidence_audit.component_digests_for("decision")
            & result.evidence_audit.component_digests_for(protected_partition)
        )
    assert result.inference.output_path is None
    assert result.operational_validation == "not_established"
    summary_text = json.dumps(result.safe_summary(), sort_keys=True)
    for forbidden in (
        "A000000",
        "B000000",
        "source_a.jsonl",
        "source_b.jsonl",
        str(tmp_path),
    ):
        assert forbidden not in summary_text

    if constraint == "unconstrained":
        selected = tuple(
            decision
            for decision in result.inference.decisions
            if decision.candidate_rank is not None and decision.candidate_rank > 1
        )
        assert selected
        assert all(item.relationship_status != "confirmed" for item in selected)
        assert (
            result.inference.assignment_result.real_assignment_count
            > result.inference.assignment_result.source_record_count
        )


def test_link_mode_rejects_incomplete_cardinality_fixture_before_generation(
    tmp_path: Path,
) -> None:
    config = load_config_text(
        yaml_text(_link_mode_payload("many_to_one")), source_format="yaml"
    ).config
    plan = compile_config(config, project_root=tmp_path)

    with pytest.raises(PipelineError, match="ML-MODE-013"):
        SyntheticModeWorkflowRunner.run_link_only(
            plan,
            generation=_generation(right_duplicate_count=0),
        )

    assert not (tmp_path / "data/synthetic/source_a.jsonl").exists()


def test_one_to_many_replay_is_deterministic_and_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    config = load_config_text(
        yaml_text(_link_mode_payload("one_to_many")), source_format="yaml"
    ).config
    plan = compile_config(config, project_root=tmp_path)
    first = SyntheticModeWorkflowRunner.run_link_only(plan, generation=_generation())
    second = SyntheticModeWorkflowRunner.run_link_only(plan, generation=_generation())

    assert second.workflow_digest == first.workflow_digest
    assert second.recipe.recipe_digest == first.recipe.recipe_digest
    assert second.model_artifact.model_digest == first.model_artifact.model_digest
    assert (
        second.calibrator_artifact.calibrator_digest == first.calibrator_artifact.calibrator_digest
    )

    base = tmp_path / f"artifacts/runs/{first.run_id}/synthetic_mode"
    recipe_path = base / "recipe-v1.json"
    recipe_payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe_payload["assignment_constraint"] = "unconstrained"
    with pytest.raises(PipelineError, match="ML-RECIPE-012"):
        deserialize_pipeline_recipe(json.dumps(recipe_payload))

    model_path = base / "champion.json"
    model_path.write_bytes(model_path.read_bytes() + b" ")
    with pytest.raises(BoostedTreeError, match="ML-BOOST-048"):
        read_xgboost_artifact(
            model_path=f"artifacts/runs/{first.run_id}/synthetic_mode/champion.json",
            manifest_path=(f"artifacts/runs/{first.run_id}/synthetic_mode/champion.manifest.json"),
            policy=plan.path_policy,
        )


def test_dedupe_only_uses_reloaded_decision_partition_evidence(
    tmp_path: Path,
) -> None:
    config = load_config_text(yaml_text(_dedupe_payload()), source_format="yaml").config
    plan = compile_config(config, project_root=tmp_path)
    result = SyntheticModeWorkflowRunner.run_dedupe_only(
        plan,
        generation=_generation(right_duplicate_count=0),
    )

    assert result.decision_pair_count > 1
    assert result.input_record_count > 1
    assert result.run_artifact.candidate_pair_count == result.decision_pair_count
    assert result.orchestration_artifact.champion_artifact_digest == (
        result.model_artifact.model_digest
    )
    assert result.orchestration_artifact.calibrator_digest == (
        result.calibrator_artifact.calibrator_digest
    )
    assert result.orchestration_artifact.decision_authority == "none"
    assert result.run_artifact.decision_authority == "none"
    assert result.decision_authority == "none"
    assert result.merge_authority == "none"
    for protected_partition in ("training", "validation", "calibration", "test"):
        assert not (
            result.evidence_audit.pair_digests_for("decision")
            & result.evidence_audit.pair_digests_for(protected_partition)
        )
        assert not (
            result.evidence_audit.component_digests_for("decision")
            & result.evidence_audit.component_digests_for(protected_partition)
        )
    summary_text = json.dumps(result.safe_summary(), sort_keys=True)
    assert "relationship_status" not in summary_text
    for forbidden in ("A000000", "source_a.jsonl", str(tmp_path)):
        assert forbidden not in summary_text


def test_dedupe_artifacts_are_deterministic_and_reject_tampering(tmp_path: Path) -> None:
    config = load_config_text(yaml_text(_dedupe_payload()), source_format="yaml").config
    plan = compile_config(config, project_root=tmp_path)
    first = SyntheticModeWorkflowRunner.run_dedupe_only(
        plan,
        generation=_generation(right_duplicate_count=0),
    )
    second = SyntheticModeWorkflowRunner.run_dedupe_only(
        plan,
        generation=_generation(right_duplicate_count=0),
    )
    assert second.qualification_digest == first.qualification_digest
    assert second.orchestration_artifact.artifact_digest == (
        first.orchestration_artifact.artifact_digest
    )
    assert second.run_artifact.run_digest == first.run_artifact.run_digest

    base = tmp_path / f"artifacts/runs/{first.run_id}/synthetic_mode"
    orchestration_path = base / "mode-orchestration-v1.json"
    orchestration_payload = json.loads(orchestration_path.read_text(encoding="utf-8"))
    orchestration_payload["decision_authority"] = "relationship_status"
    with pytest.raises(PipelineError, match="ML-MODE-006"):
        deserialize_mode_orchestration_artifact(json.dumps(orchestration_payload))

    run_path = base / "mode-run-v1.json"
    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    run_payload["candidate_pair_count"] += 1
    with pytest.raises(PipelineError, match="ML-MODE-007"):
        deserialize_mode_run_artifact(json.dumps(run_payload))


def test_link_and_dedupe_requires_combined_surface_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config_text(yaml_text(_link_and_dedupe_payload()), source_format="yaml").config
    plan = compile_config(config, project_root=tmp_path)
    result = SyntheticModeWorkflowRunner.run_link_and_dedupe(
        plan,
        generation=_generation(),
    )

    assert result.orchestration_artifact.linkage_mode == "link_and_dedupe"
    assert len(result.orchestration_artifact.candidate_plan_digests) == 3
    assert len(result.orchestration_artifact.calibrated_evidence_digests) == 3
    assert result.calibration_binding.surfaces == ("cross", "intra_a", "intra_b")
    assert result.run_artifact.decision_authority == "none"
    assert result.decision_authority == "none"
    assert result.merge_authority == "none"
    for _, audit in result.evidence_audits:
        for protected_partition in ("training", "validation", "calibration", "test"):
            assert not (
                audit.pair_digests_for("decision") & audit.pair_digests_for(protected_partition)
            )
            assert not (
                audit.component_digests_for("decision")
                & audit.component_digests_for(protected_partition)
            )
    summary_text = json.dumps(result.safe_summary(), sort_keys=True)
    assert "relationship_status" not in summary_text
    for forbidden in ("A000000", "B000000", "source_a.jsonl", str(tmp_path)):
        assert forbidden not in summary_text

    relabelled = list(result.calibration_binding.surface_label_authority_digests)
    relabelled[1] = (
        "intra_a",
        *relabelled[0][1:],
    )
    with pytest.raises(PipelineError, match="ML-MODE-024"):
        replace(
            result.calibration_binding,
            surface_label_authority_digests=tuple(relabelled),
        )
    with pytest.raises(PipelineError, match="ML-MODE-024"):
        replace(
            result.calibration_binding,
            authority_construction_digest="0" * 64,
        )

    empty_decision_matrix = BoostedFeatureMatrix(
        features=np.empty((0, len(result.model_artifact.feature_names)), dtype=np.float64),
        pair_references=(),
        pair_digests=(),
        feature_names=result.model_artifact.feature_names,
        feature_schema_digest=result.model_artifact.feature_schema_digest,
    )
    for substituted_calibrator in (
        replace(result.calibrator_artifact, source_model_id="substituted"),
        replace(result.calibrator_artifact, source_evidence_digest="0" * 64),
    ):
        with pytest.raises(PipelineError, match="ML-MODE-026"):
            result.calibration_binding.assert_authorizes(
                surface="cross",
                matrix=empty_decision_matrix,
                model=result.model_artifact,
                calibrator=substituted_calibrator,
            )

    original_model_writer = write_xgboost_artifact

    def tamper_model_manifest(**kwargs: Any) -> Any:
        written = original_model_writer(**kwargs)
        manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
        manifest["parameter_digest"] = "0" * 64
        written.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return written

    monkeypatch.setattr(
        workflow_module,
        "write_xgboost_artifact",
        tamper_model_manifest,
        raising=False,
    )
    with pytest.raises(PipelineError, match="ML-MODE-014"):
        workflow_module._persist_reload_model(
            artifact=result.model_artifact,
            base="artifacts/runs/semantic-model-tamper",
            plan=plan,
        )
    original_calibrator_writer = write_calibrator_artifact

    def tamper_calibrator_manifest(**kwargs: Any) -> Any:
        written = original_calibrator_writer(**kwargs)
        manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
        current_brier = float(manifest["diagnostics"]["brier_score"])
        manifest["diagnostics"]["brier_score"] = 0.0 if current_brier != 0.0 else 0.1
        written.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return written

    monkeypatch.setattr(
        workflow_module,
        "write_calibrator_artifact",
        tamper_calibrator_manifest,
        raising=False,
    )
    with pytest.raises(PipelineError, match="ML-MODE-015"):
        workflow_module._persist_reload_calibrator(
            artifact=result.calibrator_artifact,
            base="artifacts/runs/semantic-calibrator-tamper",
            plan=plan,
        )


def test_link_and_dedupe_rejects_mismatched_champion_selection_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_factory = workflow_module.CombinedSurfaceCalibrationBinding.from_protected_evidence

    def mismatched_factory(**kwargs: Any) -> Any:
        kwargs["selection"] = replace(kwargs["selection"], selected_model_id="substituted")
        return original_factory(**kwargs)

    monkeypatch.setattr(
        workflow_module.CombinedSurfaceCalibrationBinding,
        "from_protected_evidence",
        staticmethod(mismatched_factory),
    )
    config = load_config_text(yaml_text(_link_and_dedupe_payload()), source_format="yaml").config
    plan = compile_config(config, project_root=tmp_path)

    with pytest.raises(PipelineError, match="ML-MODE-024"):
        SyntheticModeWorkflowRunner.run_link_and_dedupe(
            plan,
            generation=_generation(),
        )


def test_link_and_dedupe_replay_is_deterministic_and_three_surface_bound(
    tmp_path: Path,
) -> None:
    config = load_config_text(yaml_text(_link_and_dedupe_payload()), source_format="yaml").config
    plan = compile_config(config, project_root=tmp_path)
    first = SyntheticModeWorkflowRunner.run_link_and_dedupe(
        plan,
        generation=_generation(),
    )
    second = SyntheticModeWorkflowRunner.run_link_and_dedupe(
        plan,
        generation=_generation(),
    )
    assert second.qualification_digest == first.qualification_digest
    assert second.calibration_binding.binding_digest == (first.calibration_binding.binding_digest)
    assert second.orchestration_artifact.artifact_digest == (
        first.orchestration_artifact.artifact_digest
    )
    assert second.run_artifact.run_digest == first.run_artifact.run_digest

    artifact_path = (
        tmp_path / f"artifacts/runs/{first.run_id}/synthetic_mode/mode-orchestration-v1.json"
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["calibrated_evidence_digests"].pop()
    with pytest.raises(PipelineError, match="ML-MODE-006"):
        deserialize_mode_orchestration_artifact(json.dumps(payload))


def test_link_and_dedupe_enforces_one_aggregate_candidate_budget_before_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _link_and_dedupe_payload()
    payload["runtime"]["maximum_candidate_pairs"] = 10_000
    payload["mode_orchestration"]["deduplication"]["maximum_candidate_edges"] = 10_000
    payload["models"]["fellegi_sunter"]["u_max_pairs"] = 10_000
    payload["models"]["boosted_tree"]["maximum_training_pairs"] = 10_000
    payload["models"]["ranking"]["maximum_training_pairs"] = 10_000
    config = load_config_text(yaml_text(payload), source_format="yaml").config
    plan = compile_config(config, project_root=tmp_path)

    def forbid_feature_construction(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("feature construction crossed the aggregate candidate budget gate")

    monkeypatch.setattr(
        "mapel_linkage.pipeline.synthetic_mode_workflow.DuckDBComparisonFeatureBuilder.build",
        forbid_feature_construction,
    )
    with pytest.raises(PipelineError, match="ML-MODE-027"):
        SyntheticModeWorkflowRunner.run_link_and_dedupe(
            plan,
            generation=_generation(),
        )

    assert not (tmp_path / "artifacts").exists()
