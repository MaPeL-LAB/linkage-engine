from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mapel_linkage.calibration import (
    BetaCalibrator,
    ChampionCalibratorSelector,
    ChampionChallengerSelector,
    ChampionSelection,
    ModelEvaluationCandidate,
    PairScoreBatch,
    apply_calibrator,
    read_calibrator_artifact,
    write_calibrator_artifact,
)
from mapel_linkage.configuration.models import ModelSelectionConfig
from mapel_linkage.domain.errors import (
    CalibrationArtifactError,
    CalibrationError,
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


def test_beta_calibration_is_monotone_bounded_and_evidence_only() -> None:
    selected = selection()
    protected = batch()
    artifact = BetaCalibrator.fit(protected, selected)
    assert artifact.method == "beta"
    assert artifact.decision_authority == "evidence_only"
    assert artifact.threshold_authority == "none"
    assert artifact.real_data_validation_status == "not_established"
    assert "left-0" not in repr(artifact)

    calibrated = apply_calibrator(protected, artifact)
    assert np.all(calibrated.probabilities >= 0.0)
    assert np.all(calibrated.probabilities <= 1.0)
    assert np.all(np.diff(calibrated.probabilities) >= -1e-12)
    assert calibrated.calibrator_method == "beta"
    assert "right-0" not in repr(calibrated)


def test_beta_calibrator_rejects_invalid_contracts() -> None:
    selected = selection()
    with pytest.raises(CalibrationError, match="ML-CAL-019"):
        BetaCalibrator.fit(batch("validation"), selected)

    with pytest.raises(CalibrationError, match="ML-CAL-040"):
        BetaCalibrator.fit(batch(), selected, max_iterations=0)

    with pytest.raises(CalibrationError, match="ML-CAL-041"):
        BetaCalibrator.fit(batch(), selected, tolerance=0.0)

    reused = replace(
        batch(),
        label_authority_digest=selected.validation_label_authority_digest,
    )
    with pytest.raises(CalibrationError, match="ML-CAL-026"):
        BetaCalibrator.fit(reused, selected)


def test_beta_calibrator_artifact_round_trip_and_tamper(tmp_path: Path) -> None:
    selected = selection()
    artifact = BetaCalibrator.fit(batch(), selected)
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    written = write_calibrator_artifact(
        artifact=artifact,
        payload_path="artifacts/models/beta_calibrator.json",
        manifest_path="artifacts/models/beta_calibrator.manifest.json",
        policy=policy,
    )
    restored = read_calibrator_artifact(
        payload_path="artifacts/models/beta_calibrator.json",
        manifest_path="artifacts/models/beta_calibrator.manifest.json",
        policy=policy,
    )
    assert restored.calibrator_digest == artifact.calibrator_digest
    assert restored.method == "beta"
    assert restored.payload["alpha"] == artifact.payload["alpha"]

    # Tamper payload
    written.payload_path.write_text(
        '{"method":"beta","alpha":-1.0,"beta":1.0,"gamma":0.0,"score_clip":1e-9,"iterations":5,"converged":true}',
        encoding="utf-8",
    )
    with pytest.raises(CalibrationArtifactError):
        read_calibrator_artifact(
            payload_path="artifacts/models/beta_calibrator.json",
            manifest_path="artifacts/models/beta_calibrator.manifest.json",
            policy=policy,
        )


def test_beta_calibrator_manifest_tamper_rejected(tmp_path: Path) -> None:
    selected = selection()
    artifact = BetaCalibrator.fit(batch(), selected)
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    written = write_calibrator_artifact(
        artifact=artifact,
        payload_path="artifacts/models/beta_cal.json",
        manifest_path="artifacts/models/beta_cal.manifest.json",
        policy=policy,
    )
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    manifest["calibration_pair_count"] = int(manifest["calibration_pair_count"]) + 5
    written.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CalibrationArtifactError):
        read_calibrator_artifact(
            payload_path="artifacts/models/beta_cal.json",
            manifest_path="artifacts/models/beta_cal.manifest.json",
            policy=policy,
        )


def test_champion_calibrator_selector() -> None:
    selected = selection()
    protected = batch()
    champion_brier = ChampionCalibratorSelector.select(
        protected,
        selected,
        methods=("sigmoid", "isotonic", "beta"),
        primary_metric="brier_score",
    )
    assert champion_brier.method in {"sigmoid", "isotonic", "beta"}
    assert champion_brier.diagnostics.brier_score <= 0.25

    champion_ece = ChampionCalibratorSelector.select(
        protected,
        selected,
        methods=("sigmoid", "isotonic", "beta"),
        primary_metric="expected_calibration_error",
    )
    assert champion_ece.method in {"sigmoid", "isotonic", "beta"}

    with pytest.raises(CalibrationError, match="ML-CAL-044"):
        ChampionCalibratorSelector.select(protected, selected, methods=())
