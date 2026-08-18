"""Deterministic LightGBM pair-classifier challenger over canonical features."""

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
from mapel_linkage.configuration.models import BoostedTreeModelConfig
from mapel_linkage.domain.errors import BoostedTreeBudgetExceeded, BoostedTreeError, DataPlaneError
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

_lightgbm: Any
try:
    import lightgbm as _lightgbm
except ModuleNotFoundError:  # pragma: no cover - exercised by optional-dependency tests.
    _lightgbm = None

_PROBABILITY_STATUS: Final[Literal["model_score_uncalibrated"]] = "model_score_uncalibrated"
_CALIBRATION_STATUS: Final[Literal["not_calibrated"]] = "not_calibrated"
_DECISION_AUTHORITY: Final[Literal["evidence_only"]] = "evidence_only"
_MODEL_VERSION: Final[Literal["m5-lightgbm-v1"]] = "m5-lightgbm-v1"


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_lightgbm() -> Any:
    if _lightgbm is None:
        raise BoostedTreeError(
            "ML-BOOST-020", "The LightGBM pair-classifier dependency is unavailable."
        )
    return _lightgbm


class LightGBMArtifactManifest(BaseModel):
    """Strict unrestricted metadata used to reload a native LightGBM artifact."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True
    )

    model_id: Annotated[StrictStr, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")]
    model_version: Literal["m5-lightgbm-v1"]
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
    lightgbm_version: StrictStr
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
class LightGBMModelArtifact:
    """Native LightGBM model text plus privacy-safe aggregate training metadata."""

    model_id: str
    model_version: Literal["m5-lightgbm-v1"]
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
    lightgbm_version: str
    label_source_kind: LabelSourceKind
    feature_names: tuple[str, ...] = field(repr=False)
    model_str: str = field(repr=False)
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
        if hashlib.sha256(self.model_str.encode("utf-8")).hexdigest() != self.model_digest:
            raise ValueError("model digest does not match the native model payload")
        if self.training_partition != "training":
            raise ValueError("the boosted model artifact must record the training partition")
        if not self.model_str.strip():
            raise ValueError("native LightGBM artifact model string cannot be empty")

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
            "lightgbm_version": self.lightgbm_version,
            "label_source_kind": self.label_source_kind,
            "training_partition": self.training_partition,
            "probability_status": self.probability_status,
            "calibration_status": self.calibration_status,
            "decision_authority": self.decision_authority,
            "real_data_validation_status": self.real_data_validation_status,
        }

    def manifest(self) -> dict[str, int | str]:
        """Return JSON-safe unrestricted metadata without model bytes or row values."""
        LightGBMArtifactManifest.model_validate(self.safe_summary()).validate_counts()
        return self.safe_summary()


@dataclass(frozen=True, slots=True)
class WrittenLightGBMArtifact:
    """Restricted local artifact paths hidden from public representation."""

    model_path: Path = field(repr=False)
    manifest_path: Path = field(repr=False)
    model_digest: str

    def safe_summary(self) -> dict[str, str]:
        return {"model_digest": self.model_digest, "artifact_format": "lightgbm_txt"}


class LightGBMPairClassifier:
    """Fit and score a deterministic, evidence-only LightGBM challenger."""

    __slots__ = ("_store",)

    def __init__(self, store: DuckDBStore | None = None) -> None:
        self._store = store

    def fit(
        self,
        *,
        matrix: BoostedLabelledMatrix,
        model: BoostedTreeModelConfig,
        random_seed: int,
        configuration_digest: str,
    ) -> LightGBMModelArtifact:
        if model.implementation != "lightgbm_classifier":
            raise BoostedTreeError(
                "ML-BOOST-021", "The boosted model plan is not a LightGBM classifier."
            )
        if matrix.partition != "training":
            raise BoostedTreeError(
                "ML-BOOST-022", "LightGBM fitting is restricted to the training partition."
            )
        if matrix.pair_count > model.maximum_training_pairs:
            raise BoostedTreeBudgetExceeded(
                "ML-BOOST-023", "The verified training matrix exceeds its configured pair budget."
            )
        if random_seed < 0:
            raise BoostedTreeError("ML-BOOST-024", "The LightGBM random seed must be non-negative.")

        if len(configuration_digest) != 64 or any(
            character not in "0123456789abcdef" for character in configuration_digest
        ):
            raise BoostedTreeError("ML-BOOST-052", "The LightGBM configuration digest is invalid.")

        lgb = _require_lightgbm()
        parameters: dict[str, object] = {
            "objective": "binary",
            "metric": "binary_logloss",
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "subsample": model.subsample,
            "colsample_bytree": model.column_sample,
            "random_state": random_seed,
            "num_threads": model.n_jobs,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        }
        training_contract = {
            "parameters": parameters,
            "num_boost_round": model.n_estimators,
            "feature_names": matrix.feature_names,
        }
        parameter_digest = _canonical_digest(training_contract)
        try:
            dataset = lgb.Dataset(
                matrix.features,
                label=matrix.labels,
                feature_name=list(matrix.feature_names),
                free_raw_data=False,
            )
            booster = lgb.train(
                parameters,
                dataset,
                num_boost_round=model.n_estimators,
            )
            model_str = booster.model_to_string()
        except Exception:
            raise BoostedTreeError(
                "ML-BOOST-025", "The LightGBM pair classifier could not be fitted."
            ) from None

        return LightGBMModelArtifact(
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
            model_digest=hashlib.sha256(model_str.encode("utf-8")).hexdigest(),
            lightgbm_version=str(lgb.__version__),
            label_source_kind=matrix.label_source_kind,
            feature_names=matrix.feature_names,
            model_str=model_str,
        )

    def score(
        self,
        *,
        matrix: BoostedFeatureMatrix,
        model: LightGBMModelArtifact,
    ) -> BoostedTreeScoreResult:
        if self._store is None:
            raise BoostedTreeError(
                "ML-BOOST-026", "A DuckDBStore is required to materialize scores."
            )
        scores = self._predict(matrix=matrix, model=model)
        table_name = f"__ml_lgb_scores_{model.model_digest[:12]}_{matrix.feature_schema_digest[:8]}"
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
        model: LightGBMModelArtifact,
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
        model: LightGBMModelArtifact,
    ) -> NDArray[np.float64]:
        if matrix.feature_schema_digest != model.feature_schema_digest:
            raise BoostedTreeError(
                "ML-BOOST-028", "The scoring feature schema does not match the model artifact."
            )
        if matrix.feature_names != model.feature_names:
            raise BoostedTreeError(
                "ML-BOOST-029", "The scoring feature order does not match the model artifact."
            )
        lgb = _require_lightgbm()
        try:
            booster = lgb.Booster(model_str=model.model_str)
            raw_scores = booster.predict(matrix.features)
            scores = np.asarray(raw_scores, dtype=np.float64)
        except Exception:
            raise BoostedTreeError(
                "ML-BOOST-035", "The LightGBM pair classifier could not score the feature matrix."
            ) from None
        if scores.ndim != 1 or len(scores) != matrix.pair_count:
            raise BoostedTreeError(
                "ML-BOOST-036", "The LightGBM scoring output violates the pair contract."
            )
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise BoostedTreeError(
                "ML-BOOST-037", "The LightGBM model returned invalid evidence scores."
            )
        scores.setflags(write=False)
        return scores


def read_lightgbm_artifact(
    *,
    model_path: str,
    manifest_path: str,
    policy: PathPolicy,
) -> LightGBMModelArtifact:
    """Load a digest-checked native LightGBM model from approved local artifact roots."""
    source_model = policy.resolve_output(model_path)
    source_manifest = policy.resolve_output(manifest_path)
    if source_model == source_manifest:
        raise BoostedTreeError(
            "ML-BOOST-043", "The LightGBM model and manifest require distinct paths."
        )
    try:
        if source_model.stat().st_size > 256 * 1024 * 1024:
            raise BoostedTreeError(
                "ML-BOOST-044", "The LightGBM model artifact exceeds its safe size limit."
            )
        if source_manifest.stat().st_size > 2 * 1024 * 1024:
            raise BoostedTreeError(
                "ML-BOOST-045", "The LightGBM manifest exceeds its safe size limit."
            )
        model_str = source_model.read_text(encoding="utf-8")
        manifest_text = source_manifest.read_text(encoding="utf-8")
    except BoostedTreeError:
        raise
    except OSError:
        raise BoostedTreeError(
            "ML-BOOST-046", "The LightGBM model artifact could not be read."
        ) from None

    try:
        manifest_payload = json.loads(manifest_text)
        manifest = LightGBMArtifactManifest.model_validate(manifest_payload)
        manifest.validate_counts()
    except (json.JSONDecodeError, ValueError):
        raise BoostedTreeError(
            "ML-BOOST-047", "The LightGBM artifact manifest is invalid."
        ) from None
    if hashlib.sha256(model_str.encode("utf-8")).hexdigest() != manifest.model_digest:
        raise BoostedTreeError(
            "ML-BOOST-048", "The LightGBM model artifact failed its integrity check."
        )

    lgb = _require_lightgbm()
    if manifest.lightgbm_version.split(".", 1)[0] != str(lgb.__version__).split(".", 1)[0]:
        raise BoostedTreeError(
            "ML-BOOST-051", "The LightGBM artifact is incompatible with this runtime."
        )
    try:
        booster = lgb.Booster(model_str=model_str)
        raw_feature_names = booster.feature_name() or []
        if not all(isinstance(name, str) for name in raw_feature_names):
            raise ValueError("invalid feature-name metadata")
        feature_names = tuple(str(name) for name in raw_feature_names)
    except Exception:
        raise BoostedTreeError(
            "ML-BOOST-049", "The LightGBM native model artifact is invalid."
        ) from None
    if len(feature_names) != manifest.feature_count:
        raise BoostedTreeError(
            "ML-BOOST-050", "The LightGBM artifact feature contract is inconsistent."
        )

    try:
        return LightGBMModelArtifact(
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
            lightgbm_version=manifest.lightgbm_version,
            label_source_kind=manifest.label_source_kind,
            feature_names=feature_names,
            model_str=model_str,
            training_partition=manifest.training_partition,
        )
    except (DataPlaneError, ValueError):
        raise BoostedTreeError(
            "ML-BOOST-050", "The LightGBM artifact feature contract is inconsistent."
        ) from None


def write_lightgbm_artifact(
    *,
    artifact: LightGBMModelArtifact,
    model_path: str,
    manifest_path: str,
    policy: PathPolicy,
) -> WrittenLightGBMArtifact:
    """Write a native LightGBM model and safe manifest under approved output roots."""
    destination_model = policy.resolve_output(model_path)
    destination_manifest = policy.resolve_output(manifest_path)
    if (
        destination_model.suffix.lower() not in (".txt", ".json")
        or destination_manifest.suffix.lower() != ".json"
    ):
        raise BoostedTreeError(
            "ML-BOOST-038", "LightGBM model and manifest artifacts must use approved paths."
        )
    if destination_model == destination_manifest:
        raise BoostedTreeError(
            "ML-BOOST-040", "The LightGBM model and manifest require distinct paths."
        )
    destination_model.parent.mkdir(parents=True, exist_ok=True)
    destination_manifest.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_text(destination_model, artifact.model_str)
        atomic_write_text(
            destination_manifest,
            json.dumps(artifact.manifest(), indent=2, sort_keys=True) + "\n",
        )
    except OSError:
        raise BoostedTreeError(
            "ML-BOOST-039", "The LightGBM model artifact could not be written."
        ) from None
    return WrittenLightGBMArtifact(
        model_path=destination_model,
        manifest_path=destination_manifest,
        model_digest=artifact.model_digest,
    )
