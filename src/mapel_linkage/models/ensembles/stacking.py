"""Deterministic stacking ensemble pair-classifier combining multiple model evidences."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from mapel_linkage import __version__
from mapel_linkage.domain.errors import DataPlaneError, EnsembleError
from mapel_linkage.domain.sql_identifiers import validate_identifier
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.labels import (
    LabelSourceKind,
    PartitionDisjointnessReport,
)
from mapel_linkage.governance.paths import PathPolicy
from mapel_linkage.io.duckdb_store import ColumnSpec, DuckDBStore
from mapel_linkage.models.boosted.xgboost_classifier import BoostedTreeScoreResult
from mapel_linkage.validation import PairValidationReport, evaluate_binary_scores

_EPSILON = 1e-9
_PROBABILITY_STATUS: Final[Literal["model_score_uncalibrated"]] = "model_score_uncalibrated"
_CALIBRATION_STATUS: Final[Literal["not_calibrated"]] = "not_calibrated"
_DECISION_AUTHORITY: Final[Literal["evidence_only"]] = "evidence_only"
_MODEL_VERSION: Final[Literal["m5-stacking-v1"]] = "m5-stacking-v1"


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _immutable_float_vector(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64).copy()
    vector.setflags(write=False)
    return vector


class StackingArtifactManifest(BaseModel):
    """Strict unrestricted metadata for reloading a native stacking artifact."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True
    )

    model_id: Annotated[StrictStr, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")]
    model_version: Literal["m5-stacking-v1"]
    engine_version: StrictStr
    configuration_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    random_seed: Annotated[StrictInt, Field(ge=0)]
    training_pair_count: Annotated[StrictInt, Field(gt=0)]
    positive_count: Annotated[StrictInt, Field(gt=0)]
    negative_count: Annotated[StrictInt, Field(gt=0)]
    base_model_count: Annotated[StrictInt, Field(ge=2)]
    label_authority_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    training_selection_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    parameter_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    model_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    label_source_kind: LabelSourceKind
    training_partition: Literal["training"]
    probability_status: Literal["model_score_uncalibrated"]
    calibration_status: Literal["not_calibrated"]
    decision_authority: Literal["evidence_only"]
    real_data_validation_status: Literal["not_established"]

    def validate_counts(self) -> None:
        if self.training_pair_count != self.positive_count + self.negative_count:
            raise ValueError("training class counts do not sum to the training pair count")


@dataclass(frozen=True, slots=True)
class StackingModelArtifact:
    """Stacking meta-learner model parameters and aggregate provenance metadata."""

    model_id: str
    model_version: Literal["m5-stacking-v1"]
    engine_version: str
    configuration_digest: str
    random_seed: int
    training_pair_count: int
    positive_count: int
    negative_count: int
    base_model_ids: tuple[str, ...]
    base_model_weights: tuple[float, ...]
    intercept: float
    label_authority_digest: str
    training_selection_digest: str
    parameter_digest: str
    model_digest: str
    label_source_kind: LabelSourceKind
    training_partition: Literal["training"] = "training"
    probability_status: Literal["model_score_uncalibrated"] = _PROBABILITY_STATUS
    calibration_status: Literal["not_calibrated"] = _CALIBRATION_STATUS
    decision_authority: Literal["evidence_only"] = _DECISION_AUTHORITY
    real_data_validation_status: Literal["not_established"] = "not_established"

    def __post_init__(self) -> None:
        if self.random_seed < 0:
            raise ValueError("stacking model random seed must be non-negative")
        if len(self.base_model_ids) < 2:
            raise ValueError("stacking ensemble requires at least two base models")
        if len(self.base_model_ids) != len(set(self.base_model_ids)):
            raise ValueError("base model IDs must be unique")
        if len(self.base_model_ids) != len(self.base_model_weights):
            raise ValueError("base model weights must align with base model IDs")
        for model_id in self.base_model_ids:
            validate_identifier(model_id)
        if any(not math.isfinite(w) or w < 0.0 for w in self.base_model_weights):
            raise ValueError("stacking base model weights must be non-negative and finite")
        if not math.isfinite(self.intercept):
            raise ValueError("stacking intercept must be finite")
        if self.training_pair_count != self.positive_count + self.negative_count:
            raise ValueError("training class counts do not sum to the training pair count")
        if self.training_pair_count <= 0 or self.positive_count <= 0 or self.negative_count <= 0:
            raise ValueError("stacking model training requires both classes")
        digests = (
            self.configuration_digest,
            self.label_authority_digest,
            self.training_selection_digest,
            self.parameter_digest,
            self.model_digest,
        )
        if any(
            len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            for digest in digests
        ):
            raise ValueError("stacking model artifact digests must be lowercase SHA-256 values")
        payload = {
            "base_model_ids": self.base_model_ids,
            "base_model_weights": self.base_model_weights,
            "intercept": self.intercept,
        }
        if _canonical_digest(payload) != self.model_digest:
            raise ValueError("model digest does not match parameter payload")

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "engine_version": self.engine_version,
            "configuration_digest": self.configuration_digest,
            "random_seed": self.random_seed,
            "training_pair_count": self.training_pair_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "base_model_count": len(self.base_model_ids),
            "label_authority_digest": self.label_authority_digest,
            "training_selection_digest": self.training_selection_digest,
            "parameter_digest": self.parameter_digest,
            "model_digest": self.model_digest,
            "label_source_kind": self.label_source_kind,
            "training_partition": self.training_partition,
            "probability_status": self.probability_status,
            "calibration_status": self.calibration_status,
            "decision_authority": self.decision_authority,
            "real_data_validation_status": self.real_data_validation_status,
        }

    def manifest(self) -> dict[str, int | str]:
        StackingArtifactManifest.model_validate(self.safe_summary()).validate_counts()
        return self.safe_summary()


@dataclass(frozen=True, slots=True)
class WrittenStackingArtifact:
    model_path: Path = field(repr=False)
    manifest_path: Path = field(repr=False)
    model_digest: str

    def safe_summary(self) -> dict[str, str]:
        return {"model_digest": self.model_digest, "artifact_format": "stacking_json"}


class StackingPairClassifier:
    """Deterministic meta-learner combining Fellegi-Sunter, XGBoost, and LightGBM probabilities."""

    __slots__ = ("_store",)

    def __init__(self, store: DuckDBStore | None = None) -> None:
        self._store = store

    def fit(
        self,
        *,
        base_scores: dict[str, NDArray[np.float64]] | NDArray[np.float64],
        labels: NDArray[np.int8],
        base_model_ids: tuple[str, ...],
        random_seed: int = 42,
        model_id: str = "stacking_ensemble",
        configuration_digest: str = "0" * 64,
        label_authority_digest: str = "0" * 64,
        selection_digest: str = "0" * 64,
        label_source_kind: LabelSourceKind = "synthetic_truth",
        partition: Literal["training"] = "training",
        max_iterations: int = 100,
        tolerance: float = 1e-8,
    ) -> StackingModelArtifact:
        if partition != "training":
            raise EnsembleError(
                "ML-ENS-001", "Stacking fitting is restricted to training partition."
            )
        if len(base_model_ids) < 2:
            raise EnsembleError("ML-ENS-002", "Stacking requires at least two base models.")
        if random_seed < 0:
            raise EnsembleError("ML-ENS-003", "Random seed must be non-negative.")

        matrix = self._extract_matrix(base_scores, base_model_ids)
        y = np.asarray(labels, dtype=np.float64)
        if len(matrix) != len(y):
            raise EnsembleError("ML-ENS-004", "Base score matrix does not align with label count.")
        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        if pos_count == 0 or neg_count == 0 or pos_count + neg_count != len(y):
            raise EnsembleError(
                "ML-ENS-005", "Stacking fitting requires binary labels for both classes."
            )

        z = _logit(matrix)
        design = np.column_stack((z, np.ones(len(z), dtype=np.float64)))
        num_features = design.shape[1]
        params = np.ones(num_features, dtype=np.float64)
        params[:-1] = 1.0 / len(base_model_ids)
        params[-1] = 0.0
        ridge = np.full(num_features, 1e-5, dtype=np.float64)

        def _nll(p: NDArray[np.float64]) -> float:
            prob = np.clip(_sigmoid(design @ p), _EPSILON, 1.0 - _EPSILON)
            return float(-np.sum(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob)))

        current_loss = _nll(params)
        for _ in range(max_iterations):
            prob = _sigmoid(design @ params)
            residual = y - prob
            weight = np.clip(prob * (1.0 - prob), 1e-8, None)
            gradient = design.T @ residual - ridge * params
            information = design.T @ (weight[:, None] * design) + np.diag(ridge)
            try:
                step = np.linalg.solve(information, gradient)
            except np.linalg.LinAlgError:
                break
            scale = 1.0
            accepted = False
            while scale >= 1e-6:
                candidate = params + scale * step
                if np.any(candidate[:-1] < 0.0) or not np.all(np.isfinite(candidate)):
                    scale *= 0.5
                    continue
                candidate_loss = _nll(candidate)
                if candidate_loss <= current_loss:
                    params = candidate
                    current_loss = candidate_loss
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                break
            if float(np.max(np.abs(scale * step))) < tolerance:
                break

        weights = tuple(float(max(0.0, w)) for w in params[:-1])
        intercept = float(params[-1])
        model_payload = {
            "base_model_ids": base_model_ids,
            "base_model_weights": weights,
            "intercept": intercept,
        }
        model_digest = _canonical_digest(model_payload)
        parameter_digest = _canonical_digest(
            {"weights": weights, "intercept": intercept, "seed": random_seed}
        )

        return StackingModelArtifact(
            model_id=model_id,
            model_version=_MODEL_VERSION,
            engine_version=__version__,
            configuration_digest=configuration_digest,
            random_seed=random_seed,
            training_pair_count=len(y),
            positive_count=pos_count,
            negative_count=neg_count,
            base_model_ids=base_model_ids,
            base_model_weights=weights,
            intercept=intercept,
            label_authority_digest=label_authority_digest,
            training_selection_digest=selection_digest,
            parameter_digest=parameter_digest,
            model_digest=model_digest,
            label_source_kind=label_source_kind,
        )

    def predict(
        self,
        *,
        base_scores: dict[str, NDArray[np.float64]] | NDArray[np.float64],
        model: StackingModelArtifact,
    ) -> NDArray[np.float64]:
        matrix = self._extract_matrix(base_scores, model.base_model_ids)
        z = _logit(matrix)
        weights = np.asarray(model.base_model_weights, dtype=np.float64)
        eta = z @ weights + model.intercept
        scores = _immutable_float_vector(_sigmoid(eta))
        return scores

    def score(
        self,
        *,
        base_scores: dict[str, NDArray[np.float64]] | NDArray[np.float64],
        pair_references: tuple[tuple[str, str], ...],
        model: StackingModelArtifact,
    ) -> BoostedTreeScoreResult:
        if self._store is None:
            raise EnsembleError("ML-ENS-006", "A DuckDBStore is required to materialize scores.")
        scores = self.predict(base_scores=base_scores, model=model)
        if len(scores) != len(pair_references):
            raise EnsembleError("ML-ENS-007", "Score count does not match pair references.")
        table_name = f"__ml_stacking_scores_{model.model_digest[:12]}"
        rows = tuple(
            (
                left,
                right,
                float(score),
                model.model_id,
                model.model_version,
                model.model_digest,
                model.probability_status,
                model.calibration_status,
                model.decision_authority,
            )
            for (left, right), score in zip(pair_references, scores, strict=True)
        )
        try:
            table = self._store.create_table_from_rows(
                table_name,
                (
                    ColumnSpec("left_record_key", "VARCHAR"),
                    ColumnSpec("right_record_key", "VARCHAR"),
                    ColumnSpec("__ml_bt_model_score", "DOUBLE"),
                    ColumnSpec("__ml_bt_model_id", "VARCHAR"),
                    ColumnSpec("__ml_bt_model_version", "VARCHAR"),
                    ColumnSpec("__ml_bt_model_digest", "VARCHAR"),
                    ColumnSpec("__ml_bt_probability_status", "VARCHAR"),
                    ColumnSpec("__ml_bt_calibration_status", "VARCHAR"),
                    ColumnSpec("__ml_bt_decision_authority", "VARCHAR"),
                ),
                rows,
            )
        except DataPlaneError:
            raise EnsembleError(
                "ML-ENS-008", "Stacking pair scores could not be materialized."
            ) from None
        return BoostedTreeScoreResult(
            table=table,
            pair_count=table.row_count,
            model_id=model.model_id,
            model_version=model.model_version,
            model_digest=model.model_digest,
        )

    def evaluate(
        self,
        *,
        base_scores: dict[str, NDArray[np.float64]] | NDArray[np.float64],
        labels: NDArray[np.int8],
        model: StackingModelArtifact,
        disjointness: PartitionDisjointnessReport,
        partition: Literal["validation", "test"] = "validation",
        diagnostic_threshold: float = 0.5,
    ) -> PairValidationReport:
        if partition not in ("validation", "test"):
            raise EnsembleError(
                "ML-ENS-009", "Training partition cannot be reported as independent validation."
            )
        scores = self.predict(base_scores=base_scores, model=model)
        scope = (
            "synthetic_mechanical_evaluation"
            if model.label_source_kind == "synthetic_truth"
            else "verified_label_evaluation"
        )
        return evaluate_binary_scores(
            labels=labels,
            scores=scores,
            diagnostic_threshold=diagnostic_threshold,
            evaluation_scope=scope,
            partition_manifest_digest=disjointness.manifest_digest,
        )

    @staticmethod
    def _extract_matrix(
        base_scores: dict[str, NDArray[np.float64]] | NDArray[np.float64],
        base_model_ids: tuple[str, ...],
    ) -> NDArray[np.float64]:
        if isinstance(base_scores, dict):
            columns: list[NDArray[np.float64]] = []
            for model_id in base_model_ids:
                if model_id not in base_scores:
                    raise EnsembleError("ML-ENS-010", f"Missing base model scores for {model_id}")
                col = np.asarray(base_scores[model_id], dtype=np.float64)
                if (
                    col.ndim != 1
                    or not np.all(np.isfinite(col))
                    or np.any(col < 0.0)
                    or np.any(col > 1.0)
                ):
                    raise EnsembleError(
                        "ML-ENS-011", f"Base model scores for {model_id} are invalid."
                    )
                columns.append(col)
            matrix = np.column_stack(columns)
        else:
            matrix = np.asarray(base_scores, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[1] != len(base_model_ids):
                raise EnsembleError("ML-ENS-012", "Base score matrix shape is invalid.")
            if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0) or np.any(matrix > 1.0):
                raise EnsembleError("ML-ENS-011", "Base model scores are invalid.")
        return matrix


def write_stacking_artifact(
    *,
    artifact: StackingModelArtifact,
    model_path: str,
    manifest_path: str,
    policy: PathPolicy,
) -> WrittenStackingArtifact:
    destination_model = policy.resolve_output(model_path)
    destination_manifest = policy.resolve_output(manifest_path)
    if (
        destination_model.suffix.lower() != ".json"
        or destination_manifest.suffix.lower() != ".json"
    ):
        raise EnsembleError(
            "ML-ENS-013", "Stacking model and manifest artifacts must use JSON paths."
        )
    if destination_model == destination_manifest:
        raise EnsembleError("ML-ENS-014", "Stacking model and manifest paths must differ.")
    destination_model.parent.mkdir(parents=True, exist_ok=True)
    destination_manifest.parent.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "base_model_ids": artifact.base_model_ids,
        "base_model_weights": artifact.base_model_weights,
        "intercept": artifact.intercept,
    }
    try:
        atomic_write_text(
            destination_model,
            json.dumps(model_payload, indent=2, sort_keys=True) + "\n",
        )
        atomic_write_text(
            destination_manifest,
            json.dumps(artifact.manifest(), indent=2, sort_keys=True) + "\n",
        )
    except OSError:
        raise EnsembleError("ML-ENS-015", "Stacking artifact could not be written.") from None
    return WrittenStackingArtifact(
        model_path=destination_model,
        manifest_path=destination_manifest,
        model_digest=artifact.model_digest,
    )


def read_stacking_artifact(
    *,
    model_path: str,
    manifest_path: str,
    policy: PathPolicy,
) -> StackingModelArtifact:
    source_model = policy.resolve_output(model_path)
    source_manifest = policy.resolve_output(manifest_path)
    if source_model == source_manifest:
        raise EnsembleError("ML-ENS-014", "Stacking model and manifest paths must differ.")
    try:
        model_raw = json.loads(source_model.read_text(encoding="utf-8"))
        manifest_raw = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise EnsembleError("ML-ENS-016", "Stacking model artifact could not be read.") from None
    if not isinstance(model_raw, dict) or not isinstance(manifest_raw, dict):
        raise EnsembleError("ML-ENS-017", "Stacking model artifact is invalid.")
    manifest = StackingArtifactManifest.model_validate(manifest_raw)
    manifest.validate_counts()
    if _canonical_digest(model_raw) != manifest.model_digest:
        raise EnsembleError("ML-ENS-018", "Stacking model payload failed integrity check.")
    return StackingModelArtifact(
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        engine_version=manifest.engine_version,
        configuration_digest=manifest.configuration_digest,
        random_seed=manifest.random_seed,
        training_pair_count=manifest.training_pair_count,
        positive_count=manifest.positive_count,
        negative_count=manifest.negative_count,
        base_model_ids=tuple(str(x) for x in model_raw["base_model_ids"]),
        base_model_weights=tuple(float(x) for x in model_raw["base_model_weights"]),
        intercept=float(model_raw["intercept"]),
        label_authority_digest=manifest.label_authority_digest,
        training_selection_digest=manifest.training_selection_digest,
        parameter_digest=manifest.parameter_digest,
        model_digest=manifest.model_digest,
        label_source_kind=manifest.label_source_kind,
        training_partition=manifest.training_partition,
    )
