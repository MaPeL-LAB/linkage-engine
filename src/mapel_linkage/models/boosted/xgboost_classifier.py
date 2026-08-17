"""Deterministic XGBoost pair-classifier challenger over canonical features."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, ClassVar, Final, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from mapel_linkage import __version__
from mapel_linkage.configuration.models import BoostedTreeModelConfig
from mapel_linkage.domain.errors import BoostedTreeBudgetExceeded, BoostedTreeError, DataPlaneError
from mapel_linkage.domain.sql_identifiers import validate_identifier
from mapel_linkage.domain.table_refs import TableRef
from mapel_linkage.governance.labels import (
    LabelSourceKind,
    PartitionDisjointnessReport,
)
from mapel_linkage.governance.paths import PathPolicy
from mapel_linkage.io.duckdb_store import ColumnSpec, DuckDBStore
from mapel_linkage.models.boosted.training import BoostedFeatureMatrix, BoostedLabelledMatrix
from mapel_linkage.validation import PairValidationReport, evaluate_binary_scores

_xgboost: Any
try:
    import xgboost as _xgboost
except ModuleNotFoundError:  # pragma: no cover - exercised by optional-dependency tests.
    _xgboost = None

_PROBABILITY_STATUS: Final[Literal["model_score_uncalibrated"]] = "model_score_uncalibrated"
_CALIBRATION_STATUS: Final[Literal["not_calibrated"]] = "not_calibrated"
_DECISION_AUTHORITY: Final[Literal["evidence_only"]] = "evidence_only"
_MODEL_VERSION: Final[Literal["m2e-xgboost-v1"]] = "m2e-xgboost-v1"


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_xgboost() -> Any:
    if _xgboost is None:
        raise BoostedTreeError(
            "ML-BOOST-020", "The XGBoost pair-classifier dependency is unavailable."
        )
    return _xgboost


class XGBoostArtifactManifest(BaseModel):
    """Strict unrestricted metadata used to reload a native XGBoost artifact."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True
    )

    model_id: Annotated[StrictStr, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")]
    model_version: Literal["m2e-xgboost-v1"]
    engine_version: StrictStr
    configuration_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    random_seed: Annotated[StrictInt, Field(ge=0)]
    training_pair_count: Annotated[StrictInt, Field(gt=0)]
    positive_count: Annotated[StrictInt, Field(gt=0)]
    negative_count: Annotated[StrictInt, Field(gt=0)]
    hard_negative_count: Annotated[StrictInt, Field(ge=0)]
    feature_count: Annotated[StrictInt, Field(gt=0)]
    feature_schema_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    label_authority_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    training_selection_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    parameter_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    model_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    xgboost_version: StrictStr
    label_source_kind: LabelSourceKind
    training_partition: Literal["training"]
    probability_status: Literal["model_score_uncalibrated"]
    calibration_status: Literal["not_calibrated"]
    decision_authority: Literal["evidence_only"]
    real_data_validation_status: Literal["not_established"]

    def validate_counts(self) -> None:
        if self.training_pair_count != self.positive_count + self.negative_count:
            raise ValueError("training class counts do not sum to the training pair count")
        if self.hard_negative_count > self.negative_count:
            raise ValueError("hard-negative count exceeds selected verified nonmatches")


@dataclass(frozen=True, slots=True)
class XGBoostModelArtifact:
    """Native JSON model plus privacy-safe aggregate training metadata."""

    model_id: str
    model_version: Literal["m2e-xgboost-v1"]
    engine_version: str
    configuration_digest: str
    random_seed: int
    training_pair_count: int
    positive_count: int
    negative_count: int
    hard_negative_count: int
    feature_schema_digest: str
    label_authority_digest: str
    training_selection_digest: str
    parameter_digest: str
    model_digest: str
    xgboost_version: str
    label_source_kind: LabelSourceKind
    feature_names: tuple[str, ...] = field(repr=False)
    model_json: bytes = field(repr=False)
    training_partition: Literal["training"] = "training"
    probability_status: Literal["model_score_uncalibrated"] = _PROBABILITY_STATUS
    calibration_status: Literal["not_calibrated"] = _CALIBRATION_STATUS
    decision_authority: Literal["evidence_only"] = _DECISION_AUTHORITY
    real_data_validation_status: Literal["not_established"] = "not_established"

    def __post_init__(self) -> None:
        if self.random_seed < 0:
            raise ValueError("boosted model random seed must be non-negative")
        if not self.feature_names or len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("boosted model feature names must be non-empty and unique")
        for feature_name in self.feature_names:
            validate_identifier(feature_name)
        if self.training_pair_count != self.positive_count + self.negative_count:
            raise ValueError("training class counts do not sum to the training pair count")
        if self.training_pair_count <= 0 or self.positive_count <= 0 or self.negative_count <= 0:
            raise ValueError("boosted model training requires both verified classes")
        if not 0 <= self.hard_negative_count <= self.negative_count:
            raise ValueError("hard negative count exceeds selected verified nonmatches")
        digests = (
            self.configuration_digest,
            self.feature_schema_digest,
            self.label_authority_digest,
            self.training_selection_digest,
            self.parameter_digest,
            self.model_digest,
        )
        if any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        ):
            raise ValueError("boosted model artifact digests must be lowercase SHA-256 values")
        if hashlib.sha256(self.model_json).hexdigest() != self.model_digest:
            raise ValueError("model digest does not match the native model payload")
        if self.training_partition != "training":
            raise ValueError("the boosted model artifact must record the training partition")
        try:
            json.loads(self.model_json.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("native XGBoost artifact must be JSON") from None

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
            "hard_negative_count": self.hard_negative_count,
            "feature_count": len(self.feature_names),
            "feature_schema_digest": self.feature_schema_digest,
            "label_authority_digest": self.label_authority_digest,
            "training_selection_digest": self.training_selection_digest,
            "parameter_digest": self.parameter_digest,
            "model_digest": self.model_digest,
            "xgboost_version": self.xgboost_version,
            "label_source_kind": self.label_source_kind,
            "training_partition": self.training_partition,
            "probability_status": self.probability_status,
            "calibration_status": self.calibration_status,
            "decision_authority": self.decision_authority,
            "real_data_validation_status": self.real_data_validation_status,
        }

    def manifest(self) -> dict[str, int | str]:
        """Return JSON-safe unrestricted metadata without model bytes or row values."""

        XGBoostArtifactManifest.model_validate(self.safe_summary()).validate_counts()
        return self.safe_summary()


@dataclass(frozen=True, slots=True)
class BoostedTreeScoreResult:
    """Structural reference to local uncalibrated boosted pair evidence."""

    table: TableRef
    pair_count: int
    model_id: str
    model_version: str
    model_digest: str
    probability_status: str = _PROBABILITY_STATUS
    calibration_status: str = _CALIBRATION_STATUS
    decision_authority: str = _DECISION_AUTHORITY

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "pair_count": self.pair_count,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_digest": self.model_digest,
            "probability_status": self.probability_status,
            "calibration_status": self.calibration_status,
            "decision_authority": self.decision_authority,
            "schema_digest": self.table.schema_digest,
        }


@dataclass(frozen=True, slots=True)
class WrittenXGBoostArtifact:
    """Restricted local artifact paths hidden from public representation."""

    model_path: Path = field(repr=False)
    manifest_path: Path = field(repr=False)
    model_digest: str

    def safe_summary(self) -> dict[str, str]:
        return {"model_digest": self.model_digest, "artifact_format": "xgboost_json"}


class XGBoostPairClassifier:
    """Fit and score a deterministic, evidence-only XGBoost challenger."""

    __slots__ = ("_store",)

    def __init__(self, store: DuckDBStore) -> None:
        self._store = store

    def fit(
        self,
        *,
        matrix: BoostedLabelledMatrix,
        model: BoostedTreeModelConfig,
        random_seed: int,
        configuration_digest: str,
    ) -> XGBoostModelArtifact:
        if model.implementation != "xgboost_classifier":
            raise BoostedTreeError(
                "ML-BOOST-021", "The boosted model plan is not an XGBoost classifier."
            )
        if matrix.partition != "training":
            raise BoostedTreeError(
                "ML-BOOST-022", "XGBoost fitting is restricted to the training partition."
            )
        if matrix.pair_count > model.maximum_training_pairs:
            raise BoostedTreeBudgetExceeded(
                "ML-BOOST-023", "The verified training matrix exceeds its configured pair budget."
            )
        if random_seed < 0:
            raise BoostedTreeError("ML-BOOST-024", "The XGBoost random seed must be non-negative.")

        if len(configuration_digest) != 64 or any(
            character not in "0123456789abcdef" for character in configuration_digest
        ):
            raise BoostedTreeError("ML-BOOST-052", "The XGBoost configuration digest is invalid.")

        xgb = _require_xgboost()
        parameters: dict[str, object] = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "max_depth": model.max_depth,
            "eta": model.learning_rate,
            "subsample": model.subsample,
            "colsample_bytree": model.column_sample,
            "seed": random_seed,
            "nthread": model.n_jobs,
            "verbosity": 0,
        }
        training_contract = {
            "parameters": parameters,
            "num_boost_round": model.n_estimators,
            "feature_names": matrix.feature_names,
        }
        parameter_digest = _canonical_digest(training_contract)
        try:
            training_data = xgb.DMatrix(
                matrix.features,
                label=matrix.labels,
                feature_names=list(matrix.feature_names),
                missing=np.nan,
            )
            booster = xgb.train(
                parameters,
                training_data,
                num_boost_round=model.n_estimators,
                verbose_eval=False,
            )
            model_json = bytes(booster.save_raw(raw_format="json"))
        except Exception:
            raise BoostedTreeError(
                "ML-BOOST-025", "The XGBoost pair classifier could not be fitted."
            ) from None

        return XGBoostModelArtifact(
            model_id=model.model_id,
            model_version=_MODEL_VERSION,
            engine_version=__version__,
            configuration_digest=configuration_digest,
            random_seed=random_seed,
            training_pair_count=matrix.pair_count,
            positive_count=matrix.positive_count,
            negative_count=matrix.negative_count,
            hard_negative_count=matrix.hard_negative_count,
            feature_schema_digest=matrix.feature_schema_digest,
            label_authority_digest=matrix.label_authority_digest,
            training_selection_digest=matrix.selection_digest,
            parameter_digest=parameter_digest,
            model_digest=hashlib.sha256(model_json).hexdigest(),
            xgboost_version=str(xgb.__version__),
            label_source_kind=matrix.label_source_kind,
            feature_names=matrix.feature_names,
            model_json=model_json,
        )

    def score(
        self,
        *,
        matrix: BoostedFeatureMatrix,
        model: XGBoostModelArtifact,
    ) -> BoostedTreeScoreResult:
        scores = self._predict(matrix=matrix, model=model)
        table_name = f"__ml_xgb_scores_{model.model_digest[:12]}_{matrix.feature_schema_digest[:8]}"
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
            raise BoostedTreeError(
                "ML-BOOST-026", "Boosted pair evidence could not be materialised safely."
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
        model: XGBoostModelArtifact,
        disjointness: PartitionDisjointnessReport,
        diagnostic_threshold: float = 0.5,
    ) -> PairValidationReport:
        if matrix.partition == "training":
            raise BoostedTreeError(
                "ML-BOOST-027", "Training labels cannot be reported as independent validation."
            )
        if not disjointness.covers("training", model.label_authority_digest) or not (
            disjointness.covers(matrix.partition, matrix.label_authority_digest)
        ):
            raise BoostedTreeError(
                "ML-BOOST-041",
                "The evaluation labels are not covered by the protected partition proof.",
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
        model: XGBoostModelArtifact,
    ) -> NDArray[np.float64]:
        if matrix.feature_schema_digest != model.feature_schema_digest:
            raise BoostedTreeError(
                "ML-BOOST-028", "The scoring feature schema does not match the model artifact."
            )
        if matrix.feature_names != model.feature_names:
            raise BoostedTreeError(
                "ML-BOOST-029", "The scoring feature order does not match the model artifact."
            )
        xgb = _require_xgboost()
        try:
            booster = xgb.Booster()
            booster.load_model(bytearray(model.model_json))
            data = xgb.DMatrix(
                matrix.features,
                feature_names=list(matrix.feature_names),
                missing=np.nan,
            )
            scores = np.asarray(booster.predict(data), dtype=np.float64)
        except Exception:
            raise BoostedTreeError(
                "ML-BOOST-035", "The XGBoost pair classifier could not score the feature matrix."
            ) from None
        if scores.ndim != 1 or len(scores) != matrix.pair_count:
            raise BoostedTreeError(
                "ML-BOOST-036", "The XGBoost scoring output violates the pair contract."
            )
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise BoostedTreeError(
                "ML-BOOST-037", "The XGBoost model returned invalid evidence scores."
            )
        scores.setflags(write=False)
        return scores


def read_xgboost_artifact(
    *,
    model_path: str,
    manifest_path: str,
    policy: PathPolicy,
) -> XGBoostModelArtifact:
    """Load a digest-checked native JSON model from approved local artifact roots."""

    source_model = policy.resolve_output(model_path)
    source_manifest = policy.resolve_output(manifest_path)
    if source_model == source_manifest:
        raise BoostedTreeError(
            "ML-BOOST-043", "The XGBoost model and manifest require distinct paths."
        )
    try:
        if source_model.stat().st_size > 256 * 1024 * 1024:
            raise BoostedTreeError(
                "ML-BOOST-044", "The XGBoost model artifact exceeds its safe size limit."
            )
        if source_manifest.stat().st_size > 2 * 1024 * 1024:
            raise BoostedTreeError(
                "ML-BOOST-045", "The XGBoost manifest exceeds its safe size limit."
            )
        model_json = source_model.read_bytes()
        manifest_text = source_manifest.read_text(encoding="utf-8")
    except BoostedTreeError:
        raise
    except OSError:
        raise BoostedTreeError(
            "ML-BOOST-046", "The XGBoost model artifact could not be read."
        ) from None

    try:
        manifest_payload = json.loads(manifest_text)
        manifest = XGBoostArtifactManifest.model_validate(manifest_payload)
        manifest.validate_counts()
    except (json.JSONDecodeError, ValueError):
        raise BoostedTreeError(
            "ML-BOOST-047", "The XGBoost artifact manifest is invalid."
        ) from None
    if hashlib.sha256(model_json).hexdigest() != manifest.model_digest:
        raise BoostedTreeError(
            "ML-BOOST-048", "The XGBoost model artifact failed its integrity check."
        )

    xgb = _require_xgboost()
    if manifest.xgboost_version.split(".", 1)[0] != str(xgb.__version__).split(".", 1)[0]:
        raise BoostedTreeError(
            "ML-BOOST-051", "The XGBoost artifact is incompatible with this runtime."
        )
    try:
        booster = xgb.Booster()
        booster.load_model(bytearray(model_json))
        raw_feature_names = booster.feature_names or []
        if not all(isinstance(name, str) for name in raw_feature_names):
            raise ValueError("invalid feature-name metadata")
        feature_names = tuple(str(name) for name in raw_feature_names)
    except Exception:
        raise BoostedTreeError(
            "ML-BOOST-049", "The XGBoost native model artifact is invalid."
        ) from None
    if len(feature_names) != manifest.feature_count:
        raise BoostedTreeError(
            "ML-BOOST-050", "The XGBoost artifact feature contract is inconsistent."
        )

    try:
        return XGBoostModelArtifact(
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            engine_version=manifest.engine_version,
            configuration_digest=manifest.configuration_digest,
            random_seed=manifest.random_seed,
            training_pair_count=manifest.training_pair_count,
            positive_count=manifest.positive_count,
            negative_count=manifest.negative_count,
            hard_negative_count=manifest.hard_negative_count,
            feature_schema_digest=manifest.feature_schema_digest,
            label_authority_digest=manifest.label_authority_digest,
            training_selection_digest=manifest.training_selection_digest,
            parameter_digest=manifest.parameter_digest,
            model_digest=manifest.model_digest,
            xgboost_version=manifest.xgboost_version,
            label_source_kind=manifest.label_source_kind,
            feature_names=feature_names,
            model_json=model_json,
            training_partition=manifest.training_partition,
        )
    except (DataPlaneError, ValueError):
        raise BoostedTreeError(
            "ML-BOOST-050", "The XGBoost artifact feature contract is inconsistent."
        ) from None


def write_xgboost_artifact(
    *,
    artifact: XGBoostModelArtifact,
    model_path: str,
    manifest_path: str,
    policy: PathPolicy,
) -> WrittenXGBoostArtifact:
    """Write a native JSON model and safe manifest under approved output roots."""

    destination_model = policy.resolve_output(model_path)
    destination_manifest = policy.resolve_output(manifest_path)
    if (
        destination_model.suffix.lower() != ".json"
        or destination_manifest.suffix.lower() != ".json"
    ):
        raise BoostedTreeError(
            "ML-BOOST-038", "XGBoost model and manifest artifacts must use JSON paths."
        )
    if destination_model == destination_manifest:
        raise BoostedTreeError(
            "ML-BOOST-040", "The XGBoost model and manifest require distinct paths."
        )
    destination_model.parent.mkdir(parents=True, exist_ok=True)
    destination_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = destination_model.with_suffix(destination_model.suffix + ".tmp")
    temporary_manifest = destination_manifest.with_suffix(destination_manifest.suffix + ".tmp")
    try:
        temporary_model.write_bytes(artifact.model_json)
        temporary_manifest.write_text(
            json.dumps(artifact.manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_model.replace(destination_model)
        temporary_manifest.replace(destination_manifest)
    except OSError:
        with suppress(OSError):
            temporary_model.unlink(missing_ok=True)
        with suppress(OSError):
            temporary_manifest.unlink(missing_ok=True)
        raise BoostedTreeError(
            "ML-BOOST-039", "The XGBoost model artifact could not be written."
        ) from None
    return WrittenXGBoostArtifact(
        model_path=destination_model,
        manifest_path=destination_manifest,
        model_digest=artifact.model_digest,
    )
