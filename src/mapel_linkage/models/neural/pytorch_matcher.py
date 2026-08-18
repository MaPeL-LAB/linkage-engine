"""Deterministic PyTorch tabular MLP pair-matcher over canonical comparison features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, ClassVar, Final, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from mapel_linkage import __version__
from mapel_linkage.configuration.models import NeuralModelConfig
from mapel_linkage.domain.errors import DataPlaneError, NeuralModelError
from mapel_linkage.domain.sql_identifiers import validate_identifier
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.labels import (
    LabelSourceKind,
    PartitionDisjointnessReport,
)
from mapel_linkage.governance.paths import PathPolicy
from mapel_linkage.io.duckdb_store import ColumnSpec, DuckDBStore
from mapel_linkage.models.boosted.training import BoostedFeatureMatrix, BoostedLabelledMatrix
from mapel_linkage.models.boosted.xgboost_classifier import BoostedTreeScoreResult
from mapel_linkage.validation import PairValidationReport, evaluate_binary_scores

_torch: Any
_nn: Any
try:
    import torch as _torch
    import torch.nn as _nn
except ModuleNotFoundError:  # pragma: no cover - optional dependency boundary
    _torch = None
    _nn = None

_PROBABILITY_STATUS: Final[Literal["model_score_uncalibrated"]] = "model_score_uncalibrated"
_CALIBRATION_STATUS: Final[Literal["not_calibrated"]] = "not_calibrated"
_DECISION_AUTHORITY: Final[Literal["evidence_only"]] = "evidence_only"
_MODEL_VERSION: Final[Literal["m5-pytorch-mlp-v1"]] = "m5-pytorch-mlp-v1"


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_torch() -> Any:
    if _torch is None or _nn is None:
        raise NeuralModelError("ML-NEUR-001", "The PyTorch pair-matcher dependency is unavailable.")
    return _torch


class PyTorchArtifactManifest(BaseModel):
    """Strict unrestricted metadata used to reload a native PyTorch artifact."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True
    )

    model_id: Annotated[StrictStr, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")]
    model_version: Literal["m5-pytorch-mlp-v1"]
    engine_version: StrictStr
    configuration_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    random_seed: Annotated[StrictInt, Field(ge=0)]
    training_pair_count: Annotated[StrictInt, Field(gt=0)]
    positive_count: Annotated[StrictInt, Field(gt=0)]
    negative_count: Annotated[StrictInt, Field(gt=0)]
    feature_count: Annotated[StrictInt, Field(gt=0)]
    feature_schema_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    label_authority_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    training_selection_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    parameter_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    model_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    torch_version: StrictStr
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
class PyTorchModelArtifact:
    """Portable JSON-serialized PyTorch model weights and aggregate provenance."""

    model_id: str
    model_version: Literal["m5-pytorch-mlp-v1"]
    engine_version: str
    configuration_digest: str
    random_seed: int
    training_pair_count: int
    positive_count: int
    negative_count: int
    feature_schema_digest: str
    label_authority_digest: str
    training_selection_digest: str
    parameter_digest: str
    model_digest: str
    torch_version: str
    label_source_kind: LabelSourceKind
    feature_names: tuple[str, ...] = field(repr=False)
    weights_json: str = field(repr=False)
    training_partition: Literal["training"] = "training"
    probability_status: Literal["model_score_uncalibrated"] = _PROBABILITY_STATUS
    calibration_status: Literal["not_calibrated"] = _CALIBRATION_STATUS
    decision_authority: Literal["evidence_only"] = _DECISION_AUTHORITY
    real_data_validation_status: Literal["not_established"] = "not_established"

    def __post_init__(self) -> None:
        if self.random_seed < 0:
            raise ValueError("neural model random seed must be non-negative")
        if not self.feature_names or len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("neural model feature names must be non-empty and unique")
        for feature_name in self.feature_names:
            validate_identifier(feature_name)
        if self.training_pair_count != self.positive_count + self.negative_count:
            raise ValueError("training class counts do not sum to the training pair count")
        if self.training_pair_count <= 0 or self.positive_count <= 0 or self.negative_count <= 0:
            raise ValueError("neural model training requires both verified classes")
        digests = (
            self.configuration_digest,
            self.feature_schema_digest,
            self.label_authority_digest,
            self.training_selection_digest,
            self.parameter_digest,
            self.model_digest,
        )
        if any(
            len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            for digest in digests
        ):
            raise ValueError("neural model artifact digests must be lowercase SHA-256 values")
        if hashlib.sha256(self.weights_json.encode("utf-8")).hexdigest() != self.model_digest:
            raise ValueError("model digest does not match the weights payload")
        if self.training_partition != "training":
            raise ValueError("the neural model artifact must record the training partition")

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
            "feature_count": len(self.feature_names),
            "feature_schema_digest": self.feature_schema_digest,
            "label_authority_digest": self.label_authority_digest,
            "training_selection_digest": self.training_selection_digest,
            "parameter_digest": self.parameter_digest,
            "model_digest": self.model_digest,
            "torch_version": self.torch_version,
            "label_source_kind": self.label_source_kind,
            "training_partition": self.training_partition,
            "probability_status": self.probability_status,
            "calibration_status": self.calibration_status,
            "decision_authority": self.decision_authority,
            "real_data_validation_status": self.real_data_validation_status,
        }

    def manifest(self) -> dict[str, int | str]:
        PyTorchArtifactManifest.model_validate(self.safe_summary()).validate_counts()
        return self.safe_summary()


@dataclass(frozen=True, slots=True)
class WrittenPyTorchArtifact:
    model_path: Path = field(repr=False)
    manifest_path: Path = field(repr=False)
    model_digest: str

    def safe_summary(self) -> dict[str, str]:
        return {"model_digest": self.model_digest, "artifact_format": "pytorch_json"}


def _build_torch_mlp(in_features: int) -> Any:
    torch = _require_torch()
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
        torch.nn.Sigmoid(),
    )


class PyTorchPairMatcher:
    """Fit and score a deterministic, CPU-only tabular MLP pair-matcher in PyTorch."""

    __slots__ = ("_store",)

    def __init__(self, store: DuckDBStore | None = None) -> None:
        self._store = store

    def fit(
        self,
        *,
        matrix: BoostedLabelledMatrix,
        model: NeuralModelConfig | None = None,
        random_seed: int = 42,
        configuration_digest: str = "0" * 64,
        epochs: int = 50,
        learning_rate: float = 0.01,
    ) -> PyTorchModelArtifact:
        if matrix.partition != "training":
            raise NeuralModelError(
                "ML-NEUR-002", "PyTorch fitting is restricted to training partition."
            )
        if random_seed < 0:
            raise NeuralModelError("ML-NEUR-003", "The PyTorch random seed must be non-negative.")
        if len(configuration_digest) != 64 or any(
            c not in "0123456789abcdef" for c in configuration_digest
        ):
            raise NeuralModelError("ML-NEUR-004", "The configuration digest is invalid.")

        torch = _require_torch()
        torch.manual_seed(random_seed)
        model_id = model.model_id if model is not None else "pytorch_pair_mlp"

        # Handle NaNs in features
        features = np.nan_to_num(matrix.features, nan=0.0)
        x_tensor = torch.tensor(features, dtype=torch.float32, device="cpu")
        y_tensor = torch.tensor(matrix.labels, dtype=torch.float32, device="cpu").unsqueeze(1)

        net = _build_torch_mlp(len(matrix.feature_names))
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

        net.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = net(x_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        net.eval()
        weights_dict: dict[str, list[list[float]] | list[float]] = {}
        for name, param in net.state_dict().items():
            weights_dict[name] = param.cpu().numpy().tolist()

        weights_json = json.dumps(weights_dict, sort_keys=True)
        model_digest = hashlib.sha256(weights_json.encode("utf-8")).hexdigest()
        parameter_digest = _canonical_digest(
            {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "seed": random_seed,
                "architecture": "32-16-1-mlp",
                "feature_names": matrix.feature_names,
            }
        )

        return PyTorchModelArtifact(
            model_id=model_id,
            model_version=_MODEL_VERSION,
            engine_version=__version__,
            configuration_digest=configuration_digest,
            random_seed=random_seed,
            training_pair_count=matrix.pair_count,
            positive_count=matrix.positive_count,
            negative_count=matrix.negative_count,
            feature_schema_digest=matrix.feature_schema_digest,
            label_authority_digest=matrix.label_authority_digest,
            training_selection_digest=matrix.selection_digest,
            parameter_digest=parameter_digest,
            model_digest=model_digest,
            torch_version=str(torch.__version__),
            label_source_kind=matrix.label_source_kind,
            feature_names=matrix.feature_names,
            weights_json=weights_json,
        )

    def score(
        self,
        *,
        matrix: BoostedFeatureMatrix,
        model: PyTorchModelArtifact,
    ) -> BoostedTreeScoreResult:
        if self._store is None:
            raise NeuralModelError("ML-NEUR-005", "DuckDBStore is required to materialize scores.")
        scores = self._predict(matrix=matrix, model=model)
        table_name = f"__ml_pt_scores_{model.model_digest[:12]}_{matrix.feature_schema_digest[:8]}"
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
            for (left, right), score in zip(matrix.pair_references, scores, strict=True)
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
            raise NeuralModelError(
                "ML-NEUR-006", "Neural pair evidence could not be materialized safely."
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
        matrix: BoostedLabelledMatrix,
        model: PyTorchModelArtifact,
        disjointness: PartitionDisjointnessReport,
        diagnostic_threshold: float = 0.5,
    ) -> PairValidationReport:
        if matrix.partition == "training":
            raise NeuralModelError(
                "ML-NEUR-007", "Training labels cannot be reported as validation."
            )
        if not disjointness.covers("training", model.label_authority_digest) or not (
            disjointness.covers(matrix.partition, matrix.label_authority_digest)
        ):
            raise NeuralModelError(
                "ML-NEUR-008", "Evaluation labels are not covered by protected partition proof."
            )
        scores = self._predict(matrix=matrix, model=model)
        scope = (
            "synthetic_mechanical_evaluation"
            if matrix.label_source_kind == "synthetic_truth"
            else "verified_label_evaluation"
        )
        return evaluate_binary_scores(
            labels=matrix.labels,
            scores=scores,
            diagnostic_threshold=diagnostic_threshold,
            evaluation_scope=scope,
            partition_manifest_digest=disjointness.manifest_digest,
        )

    @staticmethod
    def _predict(
        *,
        matrix: BoostedFeatureMatrix,
        model: PyTorchModelArtifact,
    ) -> NDArray[np.float64]:
        if matrix.feature_schema_digest != model.feature_schema_digest:
            raise NeuralModelError("ML-NEUR-009", "Scoring feature schema does not match model.")
        if matrix.feature_names != model.feature_names:
            raise NeuralModelError("ML-NEUR-010", "Scoring feature order does not match model.")

        torch = _require_torch()
        try:
            weights_dict = json.loads(model.weights_json)
            net = _build_torch_mlp(len(model.feature_names))
            state_dict = {
                name: torch.tensor(val, dtype=torch.float32, device="cpu")
                for name, val in weights_dict.items()
            }
            net.load_state_dict(state_dict)
            net.eval()

            features = np.nan_to_num(matrix.features, nan=0.0)
            x_tensor = torch.tensor(features, dtype=torch.float32, device="cpu")
            with torch.no_grad():
                out = net(x_tensor).squeeze(1).cpu().numpy()
            scores = np.asarray(out, dtype=np.float64)
        except Exception:
            raise NeuralModelError(
                "ML-NEUR-011", "PyTorch model could not score features."
            ) from None

        if scores.ndim != 1 or len(scores) != matrix.pair_count:
            raise NeuralModelError("ML-NEUR-012", "PyTorch scoring output violates pair contract.")
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise NeuralModelError("ML-NEUR-013", "PyTorch model returned invalid scores.")
        scores.setflags(write=False)
        return scores


def write_pytorch_artifact(
    *,
    artifact: PyTorchModelArtifact,
    model_path: str,
    manifest_path: str,
    policy: PathPolicy,
) -> WrittenPyTorchArtifact:
    destination_model = policy.resolve_output(model_path)
    destination_manifest = policy.resolve_output(manifest_path)
    if (
        destination_model.suffix.lower() != ".json"
        or destination_manifest.suffix.lower() != ".json"
    ):
        raise NeuralModelError(
            "ML-NEUR-014", "PyTorch model and manifest artifacts must use JSON paths."
        )
    if destination_model == destination_manifest:
        raise NeuralModelError("ML-NEUR-015", "PyTorch model and manifest paths must differ.")
    destination_model.parent.mkdir(parents=True, exist_ok=True)
    destination_manifest.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_text(destination_model, artifact.weights_json)
        atomic_write_text(
            destination_manifest,
            json.dumps(artifact.manifest(), indent=2, sort_keys=True) + "\n",
        )
    except OSError:
        raise NeuralModelError(
            "ML-NEUR-016", "PyTorch model artifact could not be written."
        ) from None
    return WrittenPyTorchArtifact(
        model_path=destination_model,
        manifest_path=destination_manifest,
        model_digest=artifact.model_digest,
    )


def read_pytorch_artifact(
    *,
    model_path: str,
    manifest_path: str,
    policy: PathPolicy,
) -> PyTorchModelArtifact:
    source_model = policy.resolve_output(model_path)
    source_manifest = policy.resolve_output(manifest_path)
    if source_model == source_manifest:
        raise NeuralModelError("ML-NEUR-015", "PyTorch model and manifest paths must differ.")
    try:
        weights_json = source_model.read_text(encoding="utf-8")
        manifest_text = source_manifest.read_text(encoding="utf-8")
        manifest_payload = json.loads(manifest_text)
        manifest = PyTorchArtifactManifest.model_validate(manifest_payload)
        manifest.validate_counts()
    except (OSError, json.JSONDecodeError, ValueError):
        raise NeuralModelError("ML-NEUR-017", "PyTorch artifact manifest is invalid.") from None
    if hashlib.sha256(weights_json.encode("utf-8")).hexdigest() != manifest.model_digest:
        raise NeuralModelError("ML-NEUR-018", "PyTorch model failed integrity check.")
    weights_dict = json.loads(weights_json)
    # Extract input feature count from first layer weight
    first_layer = weights_dict.get("0.weight", [])
    feature_count = len(first_layer[0]) if first_layer else manifest.feature_count
    feature_names = tuple(f"feature_{i}" for i in range(feature_count))

    return PyTorchModelArtifact(
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        engine_version=manifest.engine_version,
        configuration_digest=manifest.configuration_digest,
        random_seed=manifest.random_seed,
        training_pair_count=manifest.training_pair_count,
        positive_count=manifest.positive_count,
        negative_count=manifest.negative_count,
        feature_schema_digest=manifest.feature_schema_digest,
        label_authority_digest=manifest.label_authority_digest,
        training_selection_digest=manifest.training_selection_digest,
        parameter_digest=manifest.parameter_digest,
        model_digest=manifest.model_digest,
        torch_version=manifest.torch_version,
        label_source_kind=manifest.label_source_kind,
        feature_names=feature_names,
        weights_json=weights_json,
        training_partition=manifest.training_partition,
    )
