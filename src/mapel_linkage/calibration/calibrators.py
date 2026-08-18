"""Deterministic sigmoid and isotonic probability calibrators."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from mapel_linkage import __version__
from mapel_linkage.calibration.contracts import (
    CalibratedScoreBatch,
    CalibrationMethod,
    CalibratorArtifact,
    ChampionSelection,
    PairScoreBatch,
    canonical_digest,
    immutable_float_vector,
)
from mapel_linkage.calibration.metrics import calibration_diagnostics
from mapel_linkage.domain.errors import CalibrationError

_EPSILON = 1e-9
_CALIBRATOR_VERSION = "1"


def _logit(values: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(values, _EPSILON, 1.0 - _EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _negative_log_likelihood(
    parameters: NDArray[np.float64],
    predictor: NDArray[np.float64],
    labels: NDArray[np.float64],
) -> float:
    probability = np.clip(
        _sigmoid(parameters[0] * predictor + parameters[1]), _EPSILON, 1 - _EPSILON
    )
    return float(-np.sum(labels * np.log(probability) + (1.0 - labels) * np.log(1.0 - probability)))


def _validate_fit_contract(batch: PairScoreBatch, selection: ChampionSelection) -> None:
    if batch.partition != "calibration":
        raise CalibrationError(
            "ML-CAL-019", "A calibrator may fit only on the calibration partition."
        )
    if (
        batch.source_model_family != selection.selected_model_family
        or batch.source_model_id != selection.selected_model_id
        or batch.source_model_version != selection.selected_model_version
    ):
        raise CalibrationError(
            "ML-CAL-020", "Calibration scores do not belong to the selected champion."
        )
    if batch.source_evidence_digest != selection.selected_evidence_digest:
        raise CalibrationError(
            "ML-CAL-021", "Calibration evidence does not match the selected champion."
        )
    if batch.feature_schema_digest != selection.selected_feature_schema_digest:
        raise CalibrationError(
            "ML-CAL-022", "Calibration features do not match the selected champion."
        )
    if batch.partition_manifest_digest != selection.partition_manifest_digest:
        raise CalibrationError("ML-CAL-023", "Calibration partition provenance is inconsistent.")
    if batch.champion_selection_digest != selection.selection_digest:
        raise CalibrationError(
            "ML-CAL-024", "Calibration scores are not bound to champion selection."
        )
    if batch.positive_count <= 0 or batch.negative_count <= 0:
        raise CalibrationError("ML-CAL-025", "Calibration requires both verified outcome classes.")
    if batch.label_authority_digest == selection.validation_label_authority_digest:
        raise CalibrationError("ML-CAL-026", "Validation labels cannot be reused for calibration.")


def _artifact(
    *,
    method: CalibrationMethod,
    payload: dict[str, object],
    batch: PairScoreBatch,
    selection: ChampionSelection,
    probabilities: NDArray[np.float64],
) -> CalibratorArtifact:
    payload_digest = canonical_digest(payload)
    diagnostic = calibration_diagnostics(probabilities, batch.labels)
    manifest_payload = {
        "method": method,
        "calibrator_version": _CALIBRATOR_VERSION,
        "source_model_family": batch.source_model_family,
        "source_model_id": batch.source_model_id,
        "source_model_version": batch.source_model_version,
        "source_evidence_digest": batch.source_evidence_digest,
        "feature_schema_digest": batch.feature_schema_digest,
        "champion_selection_digest": selection.selection_digest,
        "validation_label_authority_digest": selection.validation_label_authority_digest,
        "calibration_label_authority_digest": batch.label_authority_digest,
        "partition_manifest_digest": batch.partition_manifest_digest,
        "calibration_pair_count": batch.pair_count,
        "positive_count": batch.positive_count,
        "negative_count": batch.negative_count,
        "payload_digest": payload_digest,
        "engine_version": __version__,
        "numpy_version": np.__version__,
        "probability_status": "calibrated_probability",
        "decision_authority": "evidence_only",
        "threshold_authority": "none",
        "real_data_validation_status": "not_established",
    }
    calibrator_digest = canonical_digest(manifest_payload)
    return CalibratorArtifact(
        method=method,
        calibrator_version=_CALIBRATOR_VERSION,
        engine_version=__version__,
        numpy_version=np.__version__,
        source_model_family=batch.source_model_family,
        source_model_id=batch.source_model_id,
        source_model_version=batch.source_model_version,
        source_evidence_digest=batch.source_evidence_digest,
        feature_schema_digest=batch.feature_schema_digest,
        champion_selection_digest=selection.selection_digest,
        validation_label_authority_digest=selection.validation_label_authority_digest,
        calibration_label_authority_digest=batch.label_authority_digest,
        partition_manifest_digest=batch.partition_manifest_digest,
        calibration_pair_count=batch.pair_count,
        positive_count=batch.positive_count,
        negative_count=batch.negative_count,
        payload=payload,
        payload_digest=payload_digest,
        calibrator_digest=calibrator_digest,
        diagnostics=diagnostic,
    )


class SigmoidCalibrator:
    """Fit a monotone logistic map on the protected calibration partition."""

    @staticmethod
    def fit(
        batch: PairScoreBatch,
        selection: ChampionSelection,
        *,
        max_iterations: int = 100,
        tolerance: float = 1e-10,
    ) -> CalibratorArtifact:
        _validate_fit_contract(batch, selection)
        predictor = _logit(batch.scores)
        labels = batch.labels.astype(np.float64)
        parameters = np.asarray([1.0, 0.0], dtype=np.float64)
        ridge = np.asarray([1e-8, 1e-8], dtype=np.float64)
        current_loss = _negative_log_likelihood(parameters, predictor, labels)
        converged = False
        if max_iterations < 1 or max_iterations > 10_000:
            raise CalibrationError(
                "ML-CAL-040", "Sigmoid calibration iteration limits are invalid."
            )
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise CalibrationError("ML-CAL-041", "Sigmoid calibration tolerance is invalid.")
        for _iteration_count in range(1, max_iterations + 1):
            probability = _sigmoid(parameters[0] * predictor + parameters[1])
            residual = labels - probability
            weight = np.clip(probability * (1.0 - probability), 1e-8, None)
            design = np.column_stack((predictor, np.ones_like(predictor)))
            gradient = design.T @ residual - ridge * parameters
            information = design.T @ (weight[:, None] * design) + np.diag(ridge)
            try:
                step = np.linalg.solve(information, gradient)
            except np.linalg.LinAlgError:
                raise CalibrationError(
                    "ML-CAL-027", "Sigmoid calibration did not have a stable fit."
                ) from None
            scale = 1.0
            accepted = False
            while scale >= 1e-6:
                candidate = parameters + scale * step
                if candidate[0] <= 0.0 or not np.all(np.isfinite(candidate)):
                    scale *= 0.5
                    continue
                candidate_loss = _negative_log_likelihood(candidate, predictor, labels)
                if candidate_loss <= current_loss:
                    parameters = candidate
                    current_loss = candidate_loss
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                break
            if float(np.max(np.abs(scale * step))) < tolerance:
                converged = True
                break
        if parameters[0] <= 0.0 or not np.all(np.isfinite(parameters)):
            raise CalibrationError("ML-CAL-028", "Sigmoid calibration violated monotonicity.")
        probabilities = immutable_float_vector(_sigmoid(parameters[0] * predictor + parameters[1]))
        payload: dict[str, object] = {
            "method": "sigmoid",
            "slope": float(parameters[0]),
            "intercept": float(parameters[1]),
            "score_clip": _EPSILON,
            "iterations": _iteration_count,
            "converged": converged,
        }
        return _artifact(
            method="sigmoid",
            payload=payload,
            batch=batch,
            selection=selection,
            probabilities=probabilities,
        )

    @staticmethod
    def apply(scores: NDArray[np.float64], artifact: CalibratorArtifact) -> NDArray[np.float64]:
        if artifact.method != "sigmoid":
            raise CalibrationError("ML-CAL-029", "A non-sigmoid artifact was rejected.")
        try:
            raw_slope = artifact.payload["slope"]
            raw_intercept = artifact.payload["intercept"]
            if (
                isinstance(raw_slope, bool)
                or not isinstance(raw_slope, (int, float))
                or isinstance(raw_intercept, bool)
                or not isinstance(raw_intercept, (int, float))
            ):
                raise TypeError
            slope = float(raw_slope)
            intercept = float(raw_intercept)
        except (KeyError, TypeError, ValueError):
            raise CalibrationError(
                "ML-CAL-030", "A sigmoid calibrator payload is invalid."
            ) from None
        if slope <= 0.0 or not math.isfinite(slope) or not math.isfinite(intercept):
            raise CalibrationError("ML-CAL-031", "A sigmoid calibrator payload is invalid.")
        values = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise CalibrationError("ML-CAL-032", "Scores for sigmoid calibration are invalid.")
        return immutable_float_vector(_sigmoid(slope * _logit(values) + intercept))


class IsotonicCalibrator:
    """Fit a deterministic pool-adjacent-violators monotone step map."""

    @staticmethod
    def fit(batch: PairScoreBatch, selection: ChampionSelection) -> CalibratorArtifact:
        _validate_fit_contract(batch, selection)
        order = np.lexsort((np.asarray(batch.pair_digests), batch.scores))
        scores = batch.scores[order]
        labels = batch.labels[order].astype(np.float64)
        unique_scores: list[float] = []
        sums: list[float] = []
        weights: list[int] = []
        for score, label in zip(scores, labels, strict=True):
            value = float(score)
            if unique_scores and value == unique_scores[-1]:
                sums[-1] += float(label)
                weights[-1] += 1
            else:
                unique_scores.append(value)
                sums.append(float(label))
                weights.append(1)
        block_starts = list(unique_scores)
        block_ends = list(unique_scores)
        block_sums = list(sums)
        block_weights = list(weights)
        index = 0
        while index < len(block_sums) - 1:
            left_mean = block_sums[index] / block_weights[index]
            right_mean = block_sums[index + 1] / block_weights[index + 1]
            if left_mean <= right_mean:
                index += 1
                continue
            block_ends[index] = block_ends[index + 1]
            block_sums[index] += block_sums[index + 1]
            block_weights[index] += block_weights[index + 1]
            for collection in (block_starts, block_ends, block_sums, block_weights):
                del collection[index + 1]
            if index > 0:
                index -= 1
        values = [block_sums[i] / block_weights[i] for i in range(len(block_sums))]
        payload: dict[str, object] = {
            "method": "isotonic",
            "lower_bounds": block_starts,
            "upper_bounds": block_ends,
            "probabilities": values,
        }
        upper_bounds = np.asarray(block_ends, dtype=np.float64)
        fitted_probabilities = np.asarray(values, dtype=np.float64)
        indices = np.searchsorted(upper_bounds, batch.scores, side="left")
        indices = np.clip(indices, 0, len(fitted_probabilities) - 1)
        probabilities = immutable_float_vector(fitted_probabilities[indices])
        return _artifact(
            method="isotonic",
            payload=payload,
            batch=batch,
            selection=selection,
            probabilities=probabilities,
        )

    @staticmethod
    def apply(scores: NDArray[np.float64], artifact: CalibratorArtifact) -> NDArray[np.float64]:
        if artifact.method != "isotonic":
            raise CalibrationError("ML-CAL-033", "A non-isotonic artifact was rejected.")
        try:
            upper_bounds = np.asarray(artifact.payload["upper_bounds"], dtype=np.float64)
            probabilities = np.asarray(artifact.payload["probabilities"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            raise CalibrationError(
                "ML-CAL-034", "An isotonic calibrator payload is invalid."
            ) from None
        if (
            upper_bounds.ndim != 1
            or probabilities.ndim != 1
            or len(upper_bounds) == 0
            or len(upper_bounds) != len(probabilities)
        ):
            raise CalibrationError("ML-CAL-035", "An isotonic calibrator payload is invalid.")
        if (
            not np.all(np.isfinite(upper_bounds))
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
            or np.any(np.diff(upper_bounds) < 0.0)
            or np.any(np.diff(probabilities) < -1e-12)
        ):
            raise CalibrationError("ML-CAL-036", "An isotonic calibrator is not monotone.")
        values = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise CalibrationError("ML-CAL-037", "Scores for isotonic calibration are invalid.")
        indices = np.searchsorted(upper_bounds, values, side="left")
        indices = np.clip(indices, 0, len(probabilities) - 1)
        return immutable_float_vector(probabilities[indices])


def apply_calibrator(batch: PairScoreBatch, artifact: CalibratorArtifact) -> CalibratedScoreBatch:
    if (
        batch.source_model_family != artifact.source_model_family
        or batch.source_model_id != artifact.source_model_id
        or batch.source_model_version != artifact.source_model_version
    ):
        raise CalibrationError("ML-CAL-038", "Scores do not belong to the calibrator source model.")
    if (
        batch.source_evidence_digest != artifact.source_evidence_digest
        or batch.feature_schema_digest != artifact.feature_schema_digest
    ):
        raise CalibrationError("ML-CAL-039", "Scores violate the calibrator evidence contract.")
    if (
        batch.champion_selection_digest != artifact.champion_selection_digest
        or batch.partition_manifest_digest != artifact.partition_manifest_digest
    ):
        raise CalibrationError("ML-CAL-054", "Scores violate calibrator provenance boundaries.")
    if artifact.method == "sigmoid":
        probabilities = SigmoidCalibrator.apply(batch.scores, artifact)
    else:
        probabilities = IsotonicCalibrator.apply(batch.scores, artifact)
    return CalibratedScoreBatch(
        pair_references=batch.pair_references,
        pair_digests=batch.pair_digests,
        probabilities=probabilities,
        source_model_family=batch.source_model_family,
        source_model_id=batch.source_model_id,
        source_model_version=batch.source_model_version,
        source_evidence_digest=batch.source_evidence_digest,
        feature_schema_digest=batch.feature_schema_digest,
        calibrator_method=artifact.method,
        calibrator_version=artifact.calibrator_version,
        calibrator_digest=artifact.calibrator_digest,
        champion_selection_digest=artifact.champion_selection_digest,
    )
