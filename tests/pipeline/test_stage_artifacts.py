from __future__ import annotations

import pytest
from pydantic import ValidationError

from mapel_linkage.pipeline.stage_artifacts import (
    OutOfFoldPredictionManifest,
    StageArtifactLedger,
    StageArtifactRef,
)


def digest(character: str) -> str:
    return character * 64


def artifact(
    *,
    artifact_id: str,
    artifact_digest: str,
    upstream: tuple[str, ...] = (),
    row_level: bool = False,
    restricted: bool = False,
) -> StageArtifactRef:
    return StageArtifactRef(
        artifact_id=artifact_id,
        stage="comparison_features",
        kind="feature_table",
        run_id="run_demo",
        engine_version="0.2.0.dev2",
        artifact_digest=artifact_digest,
        configuration_digest=digest("a"),
        schema_digest=digest("b"),
        upstream_artifact_digests=upstream,
        contains_row_level_data=row_level,
        restricted=restricted,
        decision_authority="evidence_only",
    )


def test_row_level_artifact_must_be_restricted_and_summary_is_aggregate_only() -> None:
    with pytest.raises(ValidationError):
        artifact(
            artifact_id="unsafe_features",
            artifact_digest=digest("c"),
            row_level=True,
            restricted=False,
        )

    safe = artifact(
        artifact_id="restricted_features",
        artifact_digest=digest("c"),
        row_level=True,
        restricted=True,
    )
    summary = safe.safe_summary()
    assert summary["restricted"] is True
    assert summary["upstream_artifact_count"] == 0
    assert "candidate_pair" not in repr(safe)
    assert "private/" not in repr(summary)


def test_ledger_requires_ordered_available_upstream_evidence() -> None:
    first = artifact(artifact_id="first", artifact_digest=digest("c"))
    second = artifact(
        artifact_id="second",
        artifact_digest=digest("d"),
        upstream=(digest("c"),),
    )
    ledger = StageArtifactLedger(artifacts=(first, second))
    assert ledger.safe_summary()["artifact_count"] == 2
    assert len(ledger.ledger_digest) == 64

    with pytest.raises(ValidationError):
        StageArtifactLedger(artifacts=(second, first))


def test_oof_manifest_prohibits_test_calibration_and_decision_partitions() -> None:
    manifest = OutOfFoldPredictionManifest(
        model_id="xgb_candidate",
        model_version="v1",
        model_artifact_digest=digest("a"),
        feature_schema_digest=digest("b"),
        label_authority_digest=digest("c"),
        split_manifest_digest=digest("d"),
        fold_count=5,
        group_count=40,
        pair_count=120,
        prediction_digest=digest("e"),
        group_assignment_digest=digest("f"),
    )
    summary = manifest.safe_summary()
    assert summary["partition"] == "training_oof"
    assert summary["test_partition_used"] is False
    assert summary["calibration_partition_used"] is False
    assert summary["decision_partition_used"] is False
    assert summary["grouping_method"] == "source_entity_household_connected_components"
    assert summary["decision_authority"] == "evidence_only"
    assert "pair_references" not in repr(manifest)
