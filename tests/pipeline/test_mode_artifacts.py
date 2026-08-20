from __future__ import annotations

import json

import pytest

from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.pipeline.mode_artifacts import (
    SyntheticModeOrchestrationArtifact,
    SyntheticModeRunArtifact,
    deserialize_mode_orchestration_artifact,
    deserialize_mode_run_artifact,
    serialize_mode_orchestration_artifact,
    serialize_mode_run_artifact,
)


def _orchestration_artifact() -> SyntheticModeOrchestrationArtifact:
    return SyntheticModeOrchestrationArtifact(
        linkage_mode="dedupe_only",
        assignment_constraint="unconstrained",
        configuration_digest="1" * 64,
        registry_digest="2" * 64,
        synthetic_bundle_digest="3" * 64,
        generator_version="0.1",
        random_seed=20260816,
        candidate_plan_digests=("4" * 64,),
        calibrated_evidence_digests=("5" * 64,),
        feature_schema_digest="6" * 64,
        champion_model_id="mode_xgb",
        champion_model_version="v1",
        champion_artifact_digest="7" * 64,
        calibrator_digest="8" * 64,
        partition_manifest_digest="9" * 64,
        deduplication_plan_digest="a" * 64,
        assignment_plan_digest="b" * 64,
    )


def test_mode_artifacts_strictly_round_trip_aggregate_contracts() -> None:
    orchestration = _orchestration_artifact()
    loaded = deserialize_mode_orchestration_artifact(
        serialize_mode_orchestration_artifact(orchestration)
    )
    assert loaded == orchestration
    assert "record" not in repr(loaded).lower()

    run = SyntheticModeRunArtifact(
        linkage_mode="dedupe_only",
        orchestration_artifact_digest=loaded.artifact_digest,
        configuration_digest=loaded.configuration_digest,
        result_digest="c" * 64,
        input_record_count=120,
        candidate_pair_count=240,
        cluster_count=100,
        selected_edge_count=20,
    )
    assert deserialize_mode_run_artifact(serialize_mode_run_artifact(run)) == run
    assert run.safe_summary()["operational_validation"] == "not_established"
    assert run.safe_summary()["merge_authority"] == "none"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "2"),
        ("decision_authority", "model"),
        ("operational_validation", "established"),
        ("candidate_plan_digests", ["4" * 64, "d" * 64]),
    ],
)
def test_mode_orchestration_artifact_rejects_tampering(field: str, value: object) -> None:
    payload = json.loads(serialize_mode_orchestration_artifact(_orchestration_artifact()))
    payload[field] = value
    with pytest.raises(PipelineError):
        deserialize_mode_orchestration_artifact(json.dumps(payload))


def test_mode_run_artifact_rejects_unknown_fields_and_digest_drift() -> None:
    run = SyntheticModeRunArtifact(
        linkage_mode="dedupe_only",
        orchestration_artifact_digest="1" * 64,
        configuration_digest="2" * 64,
        result_digest="3" * 64,
        input_record_count=1,
        candidate_pair_count=1,
        cluster_count=1,
        selected_edge_count=0,
    )
    payload = json.loads(serialize_mode_run_artifact(run))
    payload["private_path"] = "restricted"
    with pytest.raises(PipelineError):
        deserialize_mode_run_artifact(json.dumps(payload))

    payload.pop("private_path")
    payload["candidate_pair_count"] = 2
    with pytest.raises(PipelineError, match="ML-MODE-007"):
        deserialize_mode_run_artifact(json.dumps(payload))


@pytest.mark.parametrize(
    "encoding",
    ["leading_whitespace", "extra_newline", "indented", "reversed_keys"],
)
def test_mode_artifact_loaders_reject_semantically_equivalent_noncanonical_bytes(
    encoding: str,
) -> None:
    orchestration = _orchestration_artifact()
    run = SyntheticModeRunArtifact(
        linkage_mode="dedupe_only",
        orchestration_artifact_digest=orchestration.artifact_digest,
        configuration_digest=orchestration.configuration_digest,
        result_digest="c" * 64,
        input_record_count=1,
        candidate_pair_count=1,
        cluster_count=1,
        selected_edge_count=0,
    )

    for canonical, loader in (
        (
            serialize_mode_orchestration_artifact(orchestration),
            deserialize_mode_orchestration_artifact,
        ),
        (serialize_mode_run_artifact(run), deserialize_mode_run_artifact),
    ):
        payload = json.loads(canonical)
        if encoding == "leading_whitespace":
            noncanonical = " " + canonical
        elif encoding == "extra_newline":
            noncanonical = canonical + "\n"
        elif encoding == "indented":
            noncanonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        else:
            reversed_payload = dict(reversed(tuple(payload.items())))
            noncanonical = json.dumps(reversed_payload, separators=(",", ":")) + "\n"
        with pytest.raises(PipelineError, match="ML-MODE-007"):
            loader(noncanonical)
