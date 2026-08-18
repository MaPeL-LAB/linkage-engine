"""Aggregate calibration diagnostics without exposing pair-level values."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from mapel_linkage.calibration.contracts import CalibrationDiagnostics, ReliabilityBin
from mapel_linkage.domain.errors import CalibrationError

_EPSILON = 1e-9


def _logit(values: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(values, _EPSILON, 1.0 - _EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def fit_logistic_line(
    predictor: NDArray[np.float64],
    labels: NDArray[np.int8],
    *,
    max_iterations: int = 100,
) -> tuple[float, float]:
    """Fit intercept and slope for a one-predictor logistic regression."""

    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise CalibrationError("ML-CAL-015", "Calibration-regression inputs are invalid.")
    if np.unique(y).size < 2:
        return (math.nan, math.nan)
    design = np.column_stack((np.ones_like(x), x))
    beta = np.zeros(2, dtype=np.float64)
    ridge = np.diag(np.asarray([1e-8, 1e-8], dtype=np.float64))
    for _ in range(max_iterations):
        probability = _sigmoid(design @ beta)
        weight = np.clip(probability * (1.0 - probability), 1e-8, None)
        gradient = design.T @ (y - probability) - ridge @ beta
        hessian = -(design.T @ (weight[:, None] * design)) - ridge
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return (math.nan, math.nan)
        candidate = beta - step
        if not np.all(np.isfinite(candidate)):
            return (math.nan, math.nan)
        if float(np.max(np.abs(candidate - beta))) < 1e-10:
            beta = candidate
            break
        beta = candidate
    return (float(beta[0]), float(beta[1]))


def calibration_diagnostics(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int8],
    *,
    bin_count: int = 10,
) -> CalibrationDiagnostics:
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int8)
    if probs.ndim != 1 or targets.ndim != 1 or len(probs) != len(targets) or len(probs) == 0:
        raise CalibrationError("ML-CAL-016", "Calibration diagnostic inputs are invalid.")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0) or np.any(probs > 1.0):
        raise CalibrationError(
            "ML-CAL-017", "Calibration diagnostics received invalid probabilities."
        )
    if bin_count < 2 or bin_count > 100:
        raise CalibrationError("ML-CAL-018", "Calibration bin count is outside the safe range.")

    targets_float = targets.astype(np.float64)
    brier = float(np.mean((probs - targets_float) ** 2))
    boundaries = np.linspace(0.0, 1.0, bin_count + 1)
    bins: list[ReliabilityBin] = []
    weighted_gap = 0.0
    max_gap = 0.0
    for index in range(bin_count):
        lower = float(boundaries[index])
        upper = float(boundaries[index + 1])
        if index == bin_count - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        count = int(mask.sum())
        if count == 0:
            mean_probability = 0.0
            observed = 0.0
            gap = 0.0
        else:
            mean_probability = float(np.mean(probs[mask]))
            observed = float(np.mean(targets_float[mask]))
            gap = abs(mean_probability - observed)
            weighted_gap += gap * count
            max_gap = max(max_gap, gap)
        bins.append(
            ReliabilityBin(
                lower_bound=lower,
                upper_bound=upper,
                pair_count=count,
                mean_probability=mean_probability,
                observed_fraction=observed,
                absolute_gap=gap,
            )
        )
    intercept, slope = fit_logistic_line(_logit(probs), targets)
    return CalibrationDiagnostics(
        pair_count=len(probs),
        positive_count=int(targets.sum()),
        negative_count=len(targets) - int(targets.sum()),
        brier_score=brier,
        expected_calibration_error=weighted_gap / len(probs),
        maximum_calibration_error=max_gap,
        calibration_intercept=intercept,
        calibration_slope=slope,
        mean_probability=float(np.mean(probs)),
        observed_fraction=float(np.mean(targets_float)),
        reliability_bins=tuple(bins),
    )
