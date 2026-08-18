from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from mapel_linkage.calibration import (
    ChampionChallengerSelector,
    ChampionSelection,
    IsotonicCalibrator,
    ModelEvaluationCandidate,
    PairScoreBatch,
    SigmoidCalibrator,
    apply_calibrator,
    read_calibrator_artifact,
    write_calibrator_artifact,
)
from mapel_linkage.configuration.models import ModelSelectionConfig
from mapel_linkage.domain.errors import (
    CalibrationArtifactError,
    CalibrationError,
    ModelSelectionError,
)
from mapel_linkage.governance.labels import LabelPartition
from mapel_linkage.governance.paths import PathPolicy


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def candidate(
    family: str,
    model_id: str,
    *,
    ap: float,
    brier: float,
) -> ModelEvaluationCandidate:
    return ModelEvaluationCandidate(
        model_family=family,
        model_id=model_id,
        model_version="v1",
        evidence_digest=digest(f"evidence-{family}"),
        feature_schema_digest=digest("features"),
        validation_label_authority_digest=digest("validation-labels"),
        partition_manifest_digest=digest("partition-manifest"),
        average_precision=ap,
        brier_score=brier,
        pair_count=8,
        training_label_authority_digest=(
            None if family == "fellegi_sunter" else digest("training-labels")
        ),
    )


def selection() -> ChampionSelection:
    return ChampionChallengerSelector.select(
        (
            candidate("fellegi_sunter", "fs_baseline", ap=0.81, brier=0.16),
            candidate("xgboost", "xgb_pair_classifier", ap=0.91, brier=0.12),
        ),
        ModelSelectionConfig(),
    )


def batch(partition: LabelPartition = "calibration") -> PairScoreBatch:
    pairs = tuple((f"left-{index}", f"right-{index}") for index in range(8))
    pair_digests = tuple(digest(f"{left}\x00{right}") for left, right in pairs)
    scores = np.asarray([0.03, 0.08, 0.15, 0.30, 0.68, 0.79, 0.91, 0.97], dtype=np.float64)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    chosen = selection()
    return PairScoreBatch(
        pair_references=pairs,
        pair_digests=pair_digests,
        scores=scores,
        labels=labels,
        partition=partition,
        source_model_family="xgboost",
        source_model_id="xgb_pair_classifier",
        source_model_version="v1",
        source_evidence_digest=digest("evidence-xgboost"),
        feature_schema_digest=digest("features"),
        label_authority_digest=digest(f"{partition}-labels"),
        partition_manifest_digest=digest("partition-manifest"),
        champion_selection_digest=chosen.selection_digest,
    )


def test_selection_uses_validation_metrics_and_deterministic_tie_break() -> None:
    selected = selection()
    assert selected.selected_model_family == "xgboost"
    assert selected.test_partition_used is False
    assert selected.calibration_partition_used is False
    tied = ChampionChallengerSelector.select(
        (
            candidate("z_model", "z", ap=0.9, brier=0.1),
            candidate("a_model", "a", ap=0.9, brier=0.1),
        ),
        ModelSelectionConfig(),
    )
    assert tied.selected_model_family == "a_model"


def test_selection_rejects_incompatible_validation_authority() -> None:
    first = candidate("fellegi_sunter", "fs_baseline", ap=0.8, brier=0.2)
    second = replace(
        candidate("xgboost", "xgb_pair_classifier", ap=0.9, brier=0.1),
        validation_label_authority_digest=digest("other-labels"),
    )
    with pytest.raises(ModelSelectionError):
        ChampionChallengerSelector.select((first, second), ModelSelectionConfig())


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_is_monotone_bounded_and_evidence_only(
    method: Literal["sigmoid", "isotonic"],
) -> None:
    selected = selection()
    protected = batch()
    if method == "sigmoid":
        artifact = SigmoidCalibrator.fit(protected, selected)
    else:
        artifact = IsotonicCalibrator.fit(protected, selected)
    calibrated = apply_calibrator(protected, artifact)
    assert np.all(calibrated.probabilities >= 0.0)
    assert np.all(calibrated.probabilities <= 1.0)
    assert np.all(np.diff(calibrated.probabilities) >= -1e-12)
    assert artifact.decision_authority == "evidence_only"
    assert artifact.threshold_authority == "none"
    assert artifact.real_data_validation_status == "not_established"
    assert "left-0" not in repr(artifact)
    assert "right-0" not in repr(calibrated)


def test_calibration_rejects_validation_partition_and_reused_authority() -> None:
    selected = selection()
    with pytest.raises(CalibrationError):
        SigmoidCalibrator.fit(batch("validation"), selected)
    reused = replace(
        batch(),
        label_authority_digest=selected.validation_label_authority_digest,
    )
    with pytest.raises(CalibrationError):
        SigmoidCalibrator.fit(reused, selected)


def test_calibrator_artifact_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    selected = selection()
    artifact = SigmoidCalibrator.fit(batch(), selected)
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    written = write_calibrator_artifact(
        artifact=artifact,
        payload_path="artifacts/models/calibrator.json",
        manifest_path="artifacts/models/calibrator.manifest.json",
        policy=policy,
    )
    restored = read_calibrator_artifact(
        payload_path="artifacts/models/calibrator.json",
        manifest_path="artifacts/models/calibrator.manifest.json",
        policy=policy,
    )
    assert restored.calibrator_digest == artifact.calibrator_digest
    written.payload_path.write_text('{"method":"sigmoid","slope":99}', encoding="utf-8")
    with pytest.raises(CalibrationArtifactError):
        read_calibrator_artifact(
            payload_path="artifacts/models/calibrator.json",
            manifest_path="artifacts/models/calibrator.manifest.json",
            policy=policy,
        )


def test_calibrator_manifest_tamper_is_rejected(tmp_path: Path) -> None:
    import json

    selected = selection()
    artifact = SigmoidCalibrator.fit(batch(), selected)
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    written = write_calibrator_artifact(
        artifact=artifact,
        payload_path="artifacts/models/calibrator.json",
        manifest_path="artifacts/models/calibrator.manifest.json",
        policy=policy,
    )
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    manifest["positive_count"] = int(manifest["positive_count"]) + 1
    written.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CalibrationArtifactError):
        read_calibrator_artifact(
            payload_path="artifacts/models/calibrator.json",
            manifest_path="artifacts/models/calibrator.manifest.json",
            policy=policy,
        )


def test_apply_calibrator_rejects_partition_provenance_mismatch() -> None:
    protected = batch()
    artifact = SigmoidCalibrator.fit(protected, selection())
    with pytest.raises(CalibrationError, match="ML-CAL-054"):
        apply_calibrator(
            replace(protected, partition_manifest_digest=digest("other-partition")),
            artifact,
        )


def test_calibrator_writer_rejects_colliding_output_paths(tmp_path: Path) -> None:
    artifact = SigmoidCalibrator.fit(batch(), selection())
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    with pytest.raises(CalibrationArtifactError, match="ML-CAL-ART-009"):
        write_calibrator_artifact(
            artifact=artifact,
            payload_path="artifacts/models/calibrator.json",
            manifest_path="artifacts/models/calibrator.json",
            policy=policy,
        )
