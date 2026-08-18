"""Native JSON persistence for probability calibrators."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mapel_linkage.calibration.contracts import (
    CalibrationDiagnostics,
    CalibratorArtifact,
    ReliabilityBin,
    canonical_digest,
)
from mapel_linkage.domain.errors import CalibrationArtifactError, CalibrationError
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.paths import PathPolicy

_MAX_PAYLOAD_BYTES = 2 * 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class WrittenCalibratorArtifact:
    payload_path: Path
    manifest_path: Path
    calibrator_digest: str


def _payload_dict(artifact: CalibratorArtifact) -> dict[str, object]:
    return dict(artifact.payload)


def _manifest_dict(artifact: CalibratorArtifact) -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "method": artifact.method,
        "calibrator_version": artifact.calibrator_version,
        "engine_version": artifact.engine_version,
        "numpy_version": artifact.numpy_version,
        "source_model_family": artifact.source_model_family,
        "source_model_id": artifact.source_model_id,
        "source_model_version": artifact.source_model_version,
        "source_evidence_digest": artifact.source_evidence_digest,
        "feature_schema_digest": artifact.feature_schema_digest,
        "champion_selection_digest": artifact.champion_selection_digest,
        "validation_label_authority_digest": artifact.validation_label_authority_digest,
        "calibration_label_authority_digest": artifact.calibration_label_authority_digest,
        "partition_manifest_digest": artifact.partition_manifest_digest,
        "calibration_pair_count": artifact.calibration_pair_count,
        "positive_count": artifact.positive_count,
        "negative_count": artifact.negative_count,
        "payload_digest": artifact.payload_digest,
        "calibrator_digest": artifact.calibrator_digest,
        "diagnostics": {
            **artifact.diagnostics.safe_summary(),
            "mean_probability": artifact.diagnostics.mean_probability,
            "observed_fraction": artifact.diagnostics.observed_fraction,
            "reliability_bins": [asdict(item) for item in artifact.diagnostics.reliability_bins],
        },
        "probability_status": artifact.probability_status,
        "calibration_status": artifact.calibration_status,
        "decision_authority": artifact.decision_authority,
        "threshold_authority": artifact.threshold_authority,
        "assignment_authority": artifact.assignment_authority,
        "real_data_validation_status": artifact.real_data_validation_status,
    }


def write_calibrator_artifact(
    *,
    artifact: CalibratorArtifact,
    payload_path: str,
    manifest_path: str,
    policy: PathPolicy,
) -> WrittenCalibratorArtifact:
    payload_destination = policy.resolve_output(payload_path)
    manifest_destination = policy.resolve_output(manifest_path)
    if payload_destination == manifest_destination:
        raise CalibrationArtifactError(
            "ML-CAL-ART-009", "Calibrator payload and manifest paths must differ."
        )
    payload_destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_text(
            payload_destination,
            json.dumps(_payload_dict(artifact), indent=2, sort_keys=True) + "\n",
        )
        atomic_write_text(
            manifest_destination,
            json.dumps(_manifest_dict(artifact), indent=2, sort_keys=True) + "\n",
        )
    except (OSError, TypeError, ValueError):
        raise CalibrationArtifactError(
            "ML-CAL-ART-001", "A calibrator artifact could not be written safely."
        ) from None
    return WrittenCalibratorArtifact(
        payload_path=payload_destination,
        manifest_path=manifest_destination,
        calibrator_digest=artifact.calibrator_digest,
    )


def read_calibrator_artifact(
    *,
    payload_path: str,
    manifest_path: str,
    policy: PathPolicy,
) -> CalibratorArtifact:
    payload_source = policy.resolve_output(payload_path)
    manifest_source = policy.resolve_output(manifest_path)
    try:
        if (
            payload_source.suffix != ".json"
            or manifest_source.suffix != ".json"
            or not payload_source.is_file()
            or not manifest_source.is_file()
            or payload_source.stat().st_size > _MAX_PAYLOAD_BYTES
            or manifest_source.stat().st_size > _MAX_MANIFEST_BYTES
        ):
            raise OSError
        payload_raw = json.loads(payload_source.read_text(encoding="utf-8"))
        manifest_raw = json.loads(manifest_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise CalibrationArtifactError(
            "ML-CAL-ART-002", "A calibrator artifact could not be read safely."
        ) from None
    if not isinstance(payload_raw, dict) or not isinstance(manifest_raw, dict):
        raise CalibrationArtifactError("ML-CAL-ART-003", "A calibrator artifact is invalid.")
    expected_manifest_keys = {
        "schema_version",
        "method",
        "calibrator_version",
        "engine_version",
        "numpy_version",
        "source_model_family",
        "source_model_id",
        "source_model_version",
        "source_evidence_digest",
        "feature_schema_digest",
        "champion_selection_digest",
        "validation_label_authority_digest",
        "calibration_label_authority_digest",
        "partition_manifest_digest",
        "calibration_pair_count",
        "positive_count",
        "negative_count",
        "payload_digest",
        "calibrator_digest",
        "diagnostics",
        "probability_status",
        "calibration_status",
        "decision_authority",
        "threshold_authority",
        "assignment_authority",
        "real_data_validation_status",
    }
    if set(manifest_raw) != expected_manifest_keys or manifest_raw.get("schema_version") != "0.1":
        raise CalibrationArtifactError("ML-CAL-ART-003", "A calibrator artifact is invalid.")
    expected_constants = {
        "probability_status": "calibrated_probability",
        "calibration_status": "calibrated_on_protected_partition",
        "decision_authority": "evidence_only",
        "threshold_authority": "none",
        "assignment_authority": "none",
        "real_data_validation_status": "not_established",
    }
    if any(manifest_raw.get(key) != value for key, value in expected_constants.items()):
        raise CalibrationArtifactError("ML-CAL-ART-003", "A calibrator artifact is invalid.")
    method = manifest_raw.get("method")
    payload_keys = set(payload_raw)
    expected_payload_keys = {
        "sigmoid": {"method", "slope", "intercept", "score_clip", "iterations", "converged"},
        "isotonic": {"method", "lower_bounds", "upper_bounds", "probabilities"},
    }
    if (
        method not in expected_payload_keys
        or payload_raw.get("method") != method
        or payload_keys != expected_payload_keys[method]
    ):
        raise CalibrationArtifactError("ML-CAL-ART-003", "A calibrator artifact is invalid.")
    payload_digest = canonical_digest(payload_raw)
    if payload_digest != manifest_raw.get("payload_digest"):
        raise CalibrationArtifactError("ML-CAL-ART-004", "Calibrator payload integrity failed.")
    diagnostic_raw = manifest_raw.get("diagnostics")
    if not isinstance(diagnostic_raw, dict):
        raise CalibrationArtifactError("ML-CAL-ART-005", "Calibrator diagnostics are invalid.")
    bins_raw = diagnostic_raw.get("reliability_bins")
    if not isinstance(bins_raw, list):
        raise CalibrationArtifactError("ML-CAL-ART-006", "Calibrator reliability data are invalid.")
    try:
        diagnostics = CalibrationDiagnostics(
            pair_count=int(diagnostic_raw["pair_count"]),
            positive_count=int(diagnostic_raw["positive_count"]),
            negative_count=int(diagnostic_raw["negative_count"]),
            brier_score=float(diagnostic_raw["brier_score"]),
            expected_calibration_error=float(diagnostic_raw["expected_calibration_error"]),
            maximum_calibration_error=float(diagnostic_raw["maximum_calibration_error"]),
            calibration_intercept=float(diagnostic_raw["calibration_intercept"]),
            calibration_slope=float(diagnostic_raw["calibration_slope"]),
            mean_probability=float(diagnostic_raw["mean_probability"]),
            observed_fraction=float(diagnostic_raw["observed_fraction"]),
            reliability_bins=tuple(ReliabilityBin(**item) for item in bins_raw),
        )
        artifact = CalibratorArtifact(
            method=manifest_raw["method"],
            calibrator_version=str(manifest_raw["calibrator_version"]),
            engine_version=str(manifest_raw["engine_version"]),
            numpy_version=str(manifest_raw["numpy_version"]),
            source_model_family=str(manifest_raw["source_model_family"]),
            source_model_id=str(manifest_raw["source_model_id"]),
            source_model_version=str(manifest_raw["source_model_version"]),
            source_evidence_digest=str(manifest_raw["source_evidence_digest"]),
            feature_schema_digest=str(manifest_raw["feature_schema_digest"]),
            champion_selection_digest=str(manifest_raw["champion_selection_digest"]),
            validation_label_authority_digest=str(
                manifest_raw["validation_label_authority_digest"]
            ),
            calibration_label_authority_digest=str(
                manifest_raw["calibration_label_authority_digest"]
            ),
            partition_manifest_digest=str(manifest_raw["partition_manifest_digest"]),
            calibration_pair_count=int(manifest_raw["calibration_pair_count"]),
            positive_count=int(manifest_raw["positive_count"]),
            negative_count=int(manifest_raw["negative_count"]),
            payload=payload_raw,
            payload_digest=payload_digest,
            calibrator_digest=str(manifest_raw["calibrator_digest"]),
            diagnostics=diagnostics,
        )
    except (CalibrationError, KeyError, TypeError, ValueError):
        raise CalibrationArtifactError(
            "ML-CAL-ART-007", "A calibrator artifact is invalid."
        ) from None
    digest_payload = {
        "method": artifact.method,
        "calibrator_version": artifact.calibrator_version,
        "source_model_family": artifact.source_model_family,
        "source_model_id": artifact.source_model_id,
        "source_model_version": artifact.source_model_version,
        "source_evidence_digest": artifact.source_evidence_digest,
        "feature_schema_digest": artifact.feature_schema_digest,
        "champion_selection_digest": artifact.champion_selection_digest,
        "validation_label_authority_digest": artifact.validation_label_authority_digest,
        "calibration_label_authority_digest": artifact.calibration_label_authority_digest,
        "partition_manifest_digest": artifact.partition_manifest_digest,
        "calibration_pair_count": artifact.calibration_pair_count,
        "positive_count": artifact.positive_count,
        "negative_count": artifact.negative_count,
        "payload_digest": artifact.payload_digest,
        "engine_version": artifact.engine_version,
        "numpy_version": artifact.numpy_version,
        "probability_status": artifact.probability_status,
        "decision_authority": artifact.decision_authority,
        "threshold_authority": artifact.threshold_authority,
        "real_data_validation_status": artifact.real_data_validation_status,
    }
    if canonical_digest(digest_payload) != artifact.calibrator_digest:
        raise CalibrationArtifactError("ML-CAL-ART-008", "Calibrator manifest integrity failed.")
    return artifact
