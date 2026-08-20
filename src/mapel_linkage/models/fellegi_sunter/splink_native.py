"""Deterministic, value-hidden Splink 4 model estimation and reload.

Splink is used only to estimate and score Fellegi-Sunter pair evidence.  The
package-owned candidate generator remains retrieval authority, and every native
prediction must reproduce its exact bounded pair set before scores are exposed.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import io
import json
import logging
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol, cast

from mapel_linkage import __version__
from mapel_linkage.configuration.models import FellegiSunterModelConfig
from mapel_linkage.domain.errors import FellegiSunterBudgetExceeded, FellegiSunterError
from mapel_linkage.domain.table_refs import TableRef
from mapel_linkage.io import ColumnSpec, DuckDBStore
from mapel_linkage.models.fellegi_sunter.splink_adapter import (
    SplinkCandidateParityChecker,
    SplinkSettingsPlan,
)
from mapel_linkage.preprocessing import PreparedDataset

_ARTIFACT_SCHEMA_VERSION: Final[str] = "1"
_MODEL_VERSION: Final[str] = "i1-splink-native-v1"
_SMOOTHING_METHOD: Final[Literal["additive_pseudocount_v1"]] = "additive_pseudocount_v1"
SUPPORTED_SPLINK_VERSION: Final[str] = "4.0.16"
_MAX_ARTIFACT_BYTES: Final[int] = 8 * 1024 * 1024
_DIGEST_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_INPUT_ALIASES: Final[tuple[str, str]] = ("mapel_source_a", "mapel_source_b")
_EXPECTED_ARTIFACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "model_id",
        "model_version",
        "engine_version",
        "splink_version",
        "random_seed",
        "configuration_digest",
        "feature_schema_digest",
        "settings_digest",
        "training_candidate_pair_set_digest",
        "candidate_pair_count",
        "u_sample_pair_limit",
        "probability_smoothing",
        "smoothing_method",
        "comparison_count",
        "blocking_rule_count",
        "parameter_digest",
        "model_digest",
        "artifact_digest",
        "model_parameters",
        "probability_status",
        "decision_authority",
        "relationship_authority",
        "assignment_authority",
        "merge_authority",
        "operational_validation",
    }
)


def _canonical_json(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise FellegiSunterError(
            "ML-FS-050", "A native Splink artifact contains invalid JSON parameters."
        ) from None


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_digest(value: object, *, code: str = "ML-FS-051") -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise FellegiSunterError(code, "A native Splink artifact digest is invalid.")
    return value


def _reject_duplicate_keys(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FellegiSunterError(
                "ML-FS-052", "A native Splink artifact contains duplicate JSON keys."
            )
        result[key] = value
    return result


def _load_canonical_object(payload: str) -> dict[str, object]:
    if not isinstance(payload, str) or len(payload.encode("utf-8")) > _MAX_ARTIFACT_BYTES:
        raise FellegiSunterError(
            "ML-FS-053", "The native Splink artifact exceeds its safe size limit."
        )
    try:
        raw = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except FellegiSunterError:
        raise
    except (TypeError, ValueError):
        raise FellegiSunterError(
            "ML-FS-054", "The native Splink artifact is not valid JSON."
        ) from None
    if not isinstance(raw, dict):
        raise FellegiSunterError("ML-FS-055", "The native Splink artifact must be a JSON object.")
    if payload != _canonical_json(raw) + "\n":
        raise FellegiSunterError("ML-FS-056", "The native Splink artifact is not canonical JSON.")
    return cast(dict[str, object], raw)


def _require_supported_runtime() -> tuple[Any, Any, Any, str]:
    try:
        runtime_version = importlib.metadata.version("splink")
        splink = importlib.import_module("splink")
        linker_type = splink.Linker
        settings_type = splink.SettingsCreator
        duckdb_api_type = splink.DuckDBAPI
    except (ImportError, AttributeError, importlib.metadata.PackageNotFoundError):
        raise FellegiSunterError(
            "ML-FS-057", "The required Splink runtime is unavailable."
        ) from None
    if runtime_version != SUPPORTED_SPLINK_VERSION:
        raise FellegiSunterError(
            "ML-FS-058", "The installed Splink runtime is outside the supported contract."
        )
    return linker_type, settings_type, duckdb_api_type, runtime_version


@contextmanager
def _quiet_splink() -> Iterator[None]:
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(previous_disable)


def _prepared_records(
    store: DuckDBStore,
    dataset: PreparedDataset,
    *,
    safe_alias: str,
) -> list[dict[str, object]]:
    # Reuse the adapter's deliberately private row bridge without adding a new
    # public row-preview capability.
    from mapel_linkage.models.fellegi_sunter.splink_adapter import _prepared_records

    records = _prepared_records(store, dataset)
    for record in records:
        record["source_dataset"] = safe_alias
    return records


def _pair_digest(left_record_key: str, right_record_key: str) -> str:
    return hashlib.sha256(f"{left_record_key}\x1f{right_record_key}".encode()).hexdigest()


def _pair_set_digest(pairs: Sequence[tuple[str, str]]) -> str:
    return _digest(sorted(_pair_digest(*pair) for pair in pairs))


def splink_native_feature_schema_digest(
    *,
    left: PreparedDataset,
    right: PreparedDataset,
    settings_plan: SplinkSettingsPlan,
) -> str:
    """Bind prepared schemas and generated comparison-vector shapes without values."""

    raw_comparisons = settings_plan.settings.get("comparisons")
    if not isinstance(raw_comparisons, Sequence) or isinstance(raw_comparisons, (str, bytes)):
        raise FellegiSunterError("ML-FS-059", "The native Splink comparison schema is invalid.")
    comparison_shapes: list[dict[str, object]] = []
    for comparison in raw_comparisons:
        if not isinstance(comparison, Mapping):
            raise FellegiSunterError("ML-FS-059", "The native Splink comparison schema is invalid.")
        output = comparison.get("output_column_name")
        levels = comparison.get("comparison_levels")
        if (
            not isinstance(output, str)
            or not output
            or not isinstance(levels, Sequence)
            or isinstance(levels, (str, bytes))
            or len(levels) < 2
        ):
            raise FellegiSunterError("ML-FS-059", "The native Splink comparison schema is invalid.")
        comparison_shapes.append({"output": output, "level_count": len(levels)})
    return _digest(
        {
            "contract": "splink_native_feature_schema_v1",
            "left_prepared_schema_digest": left.table.schema_digest,
            "right_prepared_schema_digest": right.table.schema_digest,
            "comparisons": comparison_shapes,
        }
    )


def _expected_untrained_payload(settings_plan: SplinkSettingsPlan) -> dict[str, object]:
    _, settings_type, _, _ = _require_supported_runtime()
    try:
        settings = settings_type(**_runtime_settings(settings_plan)).get_settings("duckdb")
        raw = settings.as_dict()
    except Exception:
        raise FellegiSunterError(
            "ML-FS-060", "The package-owned Splink settings contract is invalid."
        ) from None
    if not isinstance(raw, dict):
        raise FellegiSunterError(
            "ML-FS-060", "The package-owned Splink settings contract is invalid."
        )
    return cast(dict[str, object], raw)


def _runtime_settings(settings_plan: SplinkSettingsPlan) -> dict[str, object]:
    try:
        settings = json.loads(_canonical_json(dict(settings_plan.settings)))
    except (TypeError, ValueError):
        raise FellegiSunterError(
            "ML-FS-060", "The package-owned Splink settings contract is invalid."
        ) from None
    rules = settings.get("blocking_rules_to_generate_predictions")
    if not isinstance(rules, list) or not rules:
        raise FellegiSunterError("ML-FS-069", "Native Splink rejects Cartesian EM training.")
    settings["linker_uid"] = "mapel001"
    # Splink 4.0.16 random-u estimation internally references the default
    # source column name.  Supply that package-owned alias rather than a
    # dataset-configured identifier.
    settings["source_dataset_column_name"] = "source_dataset"
    return cast(dict[str, object], settings)


def _without_learned_parameters(payload: Mapping[str, object]) -> dict[str, object]:
    try:
        copied = json.loads(_canonical_json(payload))
        comparisons = copied["comparisons"]
        for comparison in comparisons:
            for level in comparison["comparison_levels"]:
                level.pop("m_probability", None)
                level.pop("u_probability", None)
    except (KeyError, TypeError):
        raise FellegiSunterError(
            "ML-FS-061", "The learned Splink parameter structure is invalid."
        ) from None
    if not isinstance(copied, dict):  # pragma: no cover - mapping input guarantees this.
        raise FellegiSunterError("ML-FS-061", "The learned Splink parameter structure is invalid.")
    return cast(dict[str, object], copied)


def _normalise_learned_parameters(payload: Mapping[str, object]) -> dict[str, object]:
    """Remove sub-machine-order variation from DuckDB aggregate accumulation."""

    try:
        copied = json.loads(_canonical_json(payload))
        for comparison in copied["comparisons"]:
            for level in comparison["comparison_levels"]:
                for key in ("m_probability", "u_probability"):
                    if key in level:
                        value = level[key]
                        if isinstance(value, bool) or not isinstance(value, (int, float)):
                            raise TypeError
                        level[key] = float(format(float(value), ".15g"))
    except (KeyError, TypeError):
        raise FellegiSunterError(
            "ML-FS-061", "The learned Splink parameter structure is invalid."
        ) from None
    if not isinstance(copied, dict):  # pragma: no cover - mapping input guarantees this.
        raise FellegiSunterError("ML-FS-061", "The learned Splink parameter structure is invalid.")
    return cast(dict[str, object], copied)


def _regularise_learned_parameters(
    payload: Mapping[str, object],
    *,
    probability_smoothing: float,
    m_effective_mass: int,
    u_effective_mass: int,
) -> dict[str, object]:
    """Apply the package-owned additive pseudo-count contract to Splink m/u values.

    The exact bounded training candidate count is the conservative aggregate m
    mass. The configured u-sampling limit is the conservative aggregate u mass;
    no training rows or realised sampled pairs enter the persisted contract.
    """

    if (
        isinstance(probability_smoothing, bool)
        or not isinstance(probability_smoothing, (int, float))
        or not math.isfinite(float(probability_smoothing))
        or probability_smoothing <= 0.0
        or isinstance(m_effective_mass, bool)
        or isinstance(u_effective_mass, bool)
        or not isinstance(m_effective_mass, int)
        or not isinstance(u_effective_mass, int)
        or m_effective_mass < 1
        or u_effective_mass < 1
    ):
        raise FellegiSunterError("ML-FS-061", "The learned Splink parameter structure is invalid.")
    try:
        copied = json.loads(_canonical_json(payload))
        comparisons = copied["comparisons"]
        if not isinstance(comparisons, list) or not comparisons:
            raise TypeError
        for comparison in comparisons:
            if not isinstance(comparison, dict):
                raise TypeError
            levels = comparison["comparison_levels"]
            if not isinstance(levels, list) or any(not isinstance(level, dict) for level in levels):
                raise TypeError
            non_null_levels = [level for level in levels if level.get("is_null_level") is not True]
            if len(non_null_levels) < 2:
                raise TypeError
            for probability_key, effective_mass in (
                ("m_probability", m_effective_mass),
                ("u_probability", u_effective_mass),
            ):
                probabilities: list[float] = []
                for level in non_null_levels:
                    raw_value = level.get(probability_key, 0.0)
                    if (
                        isinstance(raw_value, bool)
                        or not isinstance(raw_value, (int, float))
                        or not math.isfinite(float(raw_value))
                        or not 0.0 <= float(raw_value) <= 1.0
                    ):
                        raise TypeError
                    probabilities.append(float(raw_value))
                if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
                    raise TypeError
                denominator = effective_mass + probability_smoothing * len(non_null_levels)
                for level, probability in zip(non_null_levels, probabilities, strict=True):
                    level[probability_key] = (
                        probability * effective_mass + probability_smoothing
                    ) / denominator
    except (KeyError, TypeError, ValueError):
        raise FellegiSunterError(
            "ML-FS-061", "The learned Splink parameter structure is invalid."
        ) from None
    if not isinstance(copied, dict):  # pragma: no cover - mapping input guarantees this.
        raise FellegiSunterError("ML-FS-061", "The learned Splink parameter structure is invalid.")
    return cast(dict[str, object], copied)


def _validate_model_parameters(
    payload: Mapping[str, object], settings_plan: SplinkSettingsPlan
) -> str:
    expected = _expected_untrained_payload(settings_plan)
    if _without_learned_parameters(payload) != expected:
        raise FellegiSunterError(
            "ML-FS-062", "The learned Splink model drifted from package-owned settings."
        )
    learned: list[dict[str, object]] = []
    try:
        comparisons = payload["comparisons"]
        if not isinstance(comparisons, list) or len(comparisons) != settings_plan.comparison_count:
            raise TypeError
        for comparison in comparisons:
            if not isinstance(comparison, dict):
                raise TypeError
            output = comparison["output_column_name"]
            levels = comparison["comparison_levels"]
            if not isinstance(output, str) or not isinstance(levels, list):
                raise TypeError
            learned_levels: list[dict[str, float | str]] = []
            m_total = 0.0
            u_total = 0.0
            non_null_count = 0
            for level in levels:
                if not isinstance(level, dict):
                    raise TypeError
                if level.get("is_null_level") is True:
                    if "m_probability" in level or "u_probability" in level:
                        raise TypeError
                    continue
                m_value = level.get("m_probability")
                u_value = level.get("u_probability")
                if (
                    isinstance(m_value, bool)
                    or isinstance(u_value, bool)
                    or not isinstance(m_value, (int, float))
                    or not isinstance(u_value, (int, float))
                ):
                    raise TypeError
                m_float = float(m_value)
                u_float = float(u_value)
                if (
                    not math.isfinite(m_float)
                    or not math.isfinite(u_float)
                    or not 0.0 < m_float < 1.0
                    or not 0.0 < u_float < 1.0
                ):
                    raise TypeError
                m_total += m_float
                u_total += u_float
                non_null_count += 1
                learned_levels.append(
                    {
                        "sql_condition_digest": hashlib.sha256(
                            str(level["sql_condition"]).encode("utf-8")
                        ).hexdigest(),
                        "m_probability": m_float,
                        "u_probability": u_float,
                    }
                )
            if (
                non_null_count < 2
                or not math.isclose(m_total, 1.0, abs_tol=1e-9)
                or not math.isclose(u_total, 1.0, abs_tol=1e-9)
            ):
                raise TypeError
            learned.append({"output": output, "levels": learned_levels})
    except (KeyError, TypeError, ValueError):
        raise FellegiSunterError(
            "ML-FS-061", "The learned Splink parameter structure is invalid."
        ) from None
    return _digest(
        {
            "probability_two_random_records_match": payload.get(
                "probability_two_random_records_match"
            ),
            "comparisons": learned,
        }
    )


def _artifact_digest_payload(artifact: SplinkNativeModelArtifact) -> dict[str, object]:
    return {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "model_id": artifact.model_id,
        "model_version": artifact.model_version,
        "engine_version": artifact.engine_version,
        "splink_version": artifact.splink_version,
        "random_seed": artifact.random_seed,
        "configuration_digest": artifact.configuration_digest,
        "feature_schema_digest": artifact.feature_schema_digest,
        "settings_digest": artifact.settings_digest,
        "training_candidate_pair_set_digest": artifact.training_candidate_pair_set_digest,
        "candidate_pair_count": artifact.candidate_pair_count,
        "u_sample_pair_limit": artifact.u_sample_pair_limit,
        "probability_smoothing": artifact.probability_smoothing,
        "smoothing_method": artifact.smoothing_method,
        "comparison_count": artifact.comparison_count,
        "blocking_rule_count": artifact.blocking_rule_count,
        "parameter_digest": artifact.parameter_digest,
        "model_digest": artifact.model_digest,
        "probability_status": artifact.probability_status,
        "decision_authority": artifact.decision_authority,
        "relationship_authority": artifact.relationship_authority,
        "assignment_authority": artifact.assignment_authority,
        "merge_authority": artifact.merge_authority,
        "operational_validation": artifact.operational_validation,
    }


@dataclass(frozen=True, slots=True, repr=False)
class SplinkNativeModelArtifact:
    """Immutable aggregate contract for a value-hidden native Splink model."""

    model_id: str
    model_version: str
    engine_version: str
    splink_version: str
    random_seed: int
    configuration_digest: str
    feature_schema_digest: str
    settings_digest: str
    training_candidate_pair_set_digest: str
    candidate_pair_count: int
    u_sample_pair_limit: int
    probability_smoothing: float
    smoothing_method: Literal["additive_pseudocount_v1"]
    comparison_count: int
    blocking_rule_count: int
    parameter_digest: str
    model_digest: str
    artifact_digest: str
    model_json: str = field(repr=False)
    probability_status: Literal["model_posterior_uncalibrated"] = "model_posterior_uncalibrated"
    decision_authority: Literal["evidence_only"] = "evidence_only"
    relationship_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    operational_validation: Literal["not_established"] = "not_established"

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or (
            _IDENTIFIER_PATTERN.fullmatch(self.model_id) is None
            or self.model_version != _MODEL_VERSION
        ):
            raise FellegiSunterError("ML-FS-063", "The native Splink model identity is invalid.")
        for value in (
            self.configuration_digest,
            self.feature_schema_digest,
            self.settings_digest,
            self.training_candidate_pair_set_digest,
            self.parameter_digest,
            self.model_digest,
            self.artifact_digest,
        ):
            _require_digest(value)
        if (
            self.engine_version != __version__
            or self.splink_version != SUPPORTED_SPLINK_VERSION
            or isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (
                    self.candidate_pair_count,
                    self.u_sample_pair_limit,
                    self.comparison_count,
                    self.blocking_rule_count,
                )
            )
            or self.candidate_pair_count < 1
            or self.u_sample_pair_limit < 1
            or isinstance(self.probability_smoothing, bool)
            or not isinstance(self.probability_smoothing, (int, float))
            or not math.isfinite(float(self.probability_smoothing))
            or not 0.0 < self.probability_smoothing <= 100.0
            or self.smoothing_method != _SMOOTHING_METHOD
            or self.comparison_count < 1
            or self.blocking_rule_count < 1
            or self.probability_status != "model_posterior_uncalibrated"
            or self.decision_authority != "evidence_only"
            or self.relationship_authority != "none"
            or self.assignment_authority != "none"
            or self.merge_authority != "none"
            or self.operational_validation != "not_established"
        ):
            raise FellegiSunterError(
                "ML-FS-064", "The native Splink model metadata is incompatible."
            )

    def __repr__(self) -> str:
        return "<SplinkNativeModelArtifact aggregate-only>"

    def safe_summary(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "splink_version": self.splink_version,
            "candidate_pair_count": self.candidate_pair_count,
            "u_sample_pair_limit": self.u_sample_pair_limit,
            "probability_smoothing": self.probability_smoothing,
            "smoothing_method": self.smoothing_method,
            "comparison_count": self.comparison_count,
            "blocking_rule_count": self.blocking_rule_count,
            "parameter_digest": self.parameter_digest,
            "artifact_digest": self.artifact_digest,
            "probability_status": self.probability_status,
            "decision_authority": self.decision_authority,
            "relationship_authority": self.relationship_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "operational_validation": self.operational_validation,
        }


@dataclass(frozen=True, slots=True)
class SplinkNativeScoreResult:
    """Structural reference to native Splink pair evidence in local DuckDB."""

    table: TableRef
    pair_count: int
    model_id: str
    model_version: str
    artifact_digest: str
    scoring_candidate_pair_set_digest: str
    score_digest: str
    probability_status: str = "model_posterior_uncalibrated"
    decision_authority: str = "evidence_only"
    relationship_authority: str = "none"
    assignment_authority: str = "none"
    merge_authority: str = "none"

    def safe_summary(self) -> dict[str, object]:
        return {
            "pair_count": self.pair_count,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "artifact_digest": self.artifact_digest,
            "scoring_candidate_pair_set_digest": self.scoring_candidate_pair_set_digest,
            "score_digest": self.score_digest,
            "probability_status": self.probability_status,
            "decision_authority": self.decision_authority,
            "relationship_authority": self.relationship_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "schema_digest": self.table.schema_digest,
        }


class _RecipeBinding(Protocol):
    @property
    def champion_model_id(self) -> str: ...

    @property
    def champion_model_version(self) -> str: ...

    @property
    def champion_artifact_digest(self) -> str: ...

    @property
    def configuration_digest(self) -> str: ...

    @property
    def feature_schema_digest(self) -> str: ...


def assert_splink_native_recipe_binding(
    *,
    recipe: _RecipeBinding,
    artifact: SplinkNativeModelArtifact,
) -> None:
    """Reject a recipe unless it binds this exact native evidence artifact."""

    _assert_artifact_self_integrity(artifact)
    if (
        recipe.champion_model_id != artifact.model_id
        or recipe.champion_model_version != artifact.model_version
        or recipe.champion_artifact_digest != artifact.artifact_digest
        or recipe.configuration_digest != artifact.configuration_digest
        or recipe.feature_schema_digest != artifact.feature_schema_digest
    ):
        raise FellegiSunterError(
            "ML-FS-077", "The pipeline recipe does not bind this native Splink artifact."
        )


def _artifact_payload(artifact: SplinkNativeModelArtifact) -> dict[str, object]:
    try:
        model_parameters = json.loads(artifact.model_json, object_pairs_hook=_reject_duplicate_keys)
    except (FellegiSunterError, TypeError, ValueError):
        raise FellegiSunterError(
            "ML-FS-061", "The learned Splink parameter structure is invalid."
        ) from None
    return {
        **_artifact_digest_payload(artifact),
        "artifact_digest": artifact.artifact_digest,
        "model_parameters": model_parameters,
    }


def _assert_artifact_self_integrity(
    artifact: SplinkNativeModelArtifact,
) -> dict[str, object]:
    """Verify plan-independent canonical model and aggregate artifact integrity."""

    try:
        if len(artifact.model_json.encode("utf-8")) > _MAX_ARTIFACT_BYTES:
            raise TypeError
        parameters = json.loads(
            artifact.model_json,
            object_pairs_hook=_reject_duplicate_keys,
        )
        if not isinstance(parameters, dict):
            raise TypeError
        canonical_model = _canonical_json(parameters)
    except (FellegiSunterError, TypeError, ValueError):
        raise FellegiSunterError(
            "ML-FS-067", "The native Splink artifact failed integrity checks."
        ) from None
    if (
        artifact.model_json != canonical_model
        or artifact.model_digest != hashlib.sha256(canonical_model.encode("utf-8")).hexdigest()
        or artifact.artifact_digest != _digest(_artifact_digest_payload(artifact))
    ):
        raise FellegiSunterError("ML-FS-067", "The native Splink artifact failed integrity checks.")
    return parameters


def serialize_splink_native_model(artifact: SplinkNativeModelArtifact) -> str:
    """Return deterministic canonical JSON without rows, record IDs, or candidate pairs."""

    _assert_artifact_self_integrity(artifact)
    return _canonical_json(_artifact_payload(artifact)) + "\n"


def _assert_artifact_integrity(
    artifact: SplinkNativeModelArtifact,
    settings_plan: SplinkSettingsPlan,
) -> dict[str, object]:
    parameters = _assert_artifact_self_integrity(artifact)
    if parameters != _normalise_learned_parameters(parameters):
        raise FellegiSunterError("ML-FS-067", "The native Splink artifact failed integrity checks.")
    parameter_digest = _validate_model_parameters(parameters, settings_plan)
    if artifact.parameter_digest != parameter_digest:
        raise FellegiSunterError("ML-FS-067", "The native Splink artifact failed integrity checks.")
    return parameters


def deserialize_splink_native_model(
    payload: str,
    *,
    settings_plan: SplinkSettingsPlan,
    model: FellegiSunterModelConfig,
    configuration_digest: str,
    feature_schema_digest: str,
    random_seed: int,
) -> SplinkNativeModelArtifact:
    """Load only an exact package-owned, digest-checked native Splink model."""

    _require_supported_runtime()
    raw = _load_canonical_object(payload)
    if set(raw) != _EXPECTED_ARTIFACT_KEYS or raw.get("schema_version") != _ARTIFACT_SCHEMA_VERSION:
        raise FellegiSunterError("ML-FS-065", "The native Splink artifact schema is invalid.")
    model_parameters = raw.get("model_parameters")
    if not isinstance(model_parameters, dict):
        raise FellegiSunterError("ML-FS-065", "The native Splink artifact schema is invalid.")
    try:
        if model_parameters != _normalise_learned_parameters(model_parameters):
            raise TypeError
        model_json = _canonical_json(model_parameters)
        parameter_digest = _validate_model_parameters(model_parameters, settings_plan)
        model_digest = hashlib.sha256(model_json.encode("utf-8")).hexdigest()
        artifact = SplinkNativeModelArtifact(
            model_id=cast(str, raw["model_id"]),
            model_version=cast(str, raw["model_version"]),
            engine_version=cast(str, raw["engine_version"]),
            splink_version=cast(str, raw["splink_version"]),
            random_seed=cast(int, raw["random_seed"]),
            configuration_digest=cast(str, raw["configuration_digest"]),
            feature_schema_digest=cast(str, raw["feature_schema_digest"]),
            settings_digest=cast(str, raw["settings_digest"]),
            training_candidate_pair_set_digest=cast(str, raw["training_candidate_pair_set_digest"]),
            candidate_pair_count=cast(int, raw["candidate_pair_count"]),
            u_sample_pair_limit=cast(int, raw["u_sample_pair_limit"]),
            probability_smoothing=cast(float, raw["probability_smoothing"]),
            smoothing_method=cast(Any, raw["smoothing_method"]),
            comparison_count=cast(int, raw["comparison_count"]),
            blocking_rule_count=cast(int, raw["blocking_rule_count"]),
            parameter_digest=cast(str, raw["parameter_digest"]),
            model_digest=cast(str, raw["model_digest"]),
            artifact_digest=cast(str, raw["artifact_digest"]),
            model_json=model_json,
            probability_status=cast(Any, raw["probability_status"]),
            decision_authority=cast(Any, raw["decision_authority"]),
            relationship_authority=cast(Any, raw["relationship_authority"]),
            assignment_authority=cast(Any, raw["assignment_authority"]),
            merge_authority=cast(Any, raw["merge_authority"]),
            operational_validation=cast(Any, raw["operational_validation"]),
        )
    except (KeyError, TypeError, ValueError, FellegiSunterError):
        raise FellegiSunterError(
            "ML-FS-065", "The native Splink artifact schema is invalid."
        ) from None
    if (
        artifact.model_id != model.model_id
        or artifact.configuration_digest != configuration_digest
        or artifact.feature_schema_digest != feature_schema_digest
        or artifact.settings_digest != settings_plan.settings_digest
        or artifact.random_seed != random_seed
        or artifact.u_sample_pair_limit != model.u_max_pairs
        or artifact.probability_smoothing != model.probability_smoothing
        or artifact.smoothing_method != _SMOOTHING_METHOD
        or artifact.comparison_count != settings_plan.comparison_count
        or artifact.blocking_rule_count != settings_plan.blocking_rule_count
    ):
        raise FellegiSunterError("ML-FS-066", "The native Splink artifact contract has drifted.")
    if artifact.parameter_digest != parameter_digest or artifact.model_digest != model_digest:
        raise FellegiSunterError("ML-FS-067", "The native Splink artifact failed integrity checks.")
    if artifact.artifact_digest != _digest(_artifact_digest_payload(artifact)):
        raise FellegiSunterError("ML-FS-067", "The native Splink artifact failed integrity checks.")
    return artifact


class SplinkNativeDuckDBMatcher:
    """Fit and reload native Splink evidence under exact package candidate authority."""

    __slots__ = ("_store",)

    def __init__(self, store: DuckDBStore) -> None:
        self._store = store

    def fit(
        self,
        *,
        left: PreparedDataset,
        right: PreparedDataset,
        settings_plan: SplinkSettingsPlan,
        model: FellegiSunterModelConfig,
        configuration_digest: str,
        expected_pairs: Sequence[tuple[str, str]],
        maximum_candidate_pairs: int,
        random_seed: int,
    ) -> SplinkNativeModelArtifact:
        if maximum_candidate_pairs < 1 or len(expected_pairs) > maximum_candidate_pairs:
            raise FellegiSunterBudgetExceeded(
                "ML-FS-068", "Native Splink fitting exceeds the candidate-pair budget."
            )
        parity = SplinkCandidateParityChecker.check(
            store=self._store,
            left=left,
            right=right,
            settings_plan=settings_plan,
            expected_pairs=expected_pairs,
        )
        linker_type, settings_type, duckdb_api_type, runtime_version = _require_supported_runtime()
        raw_rules = settings_plan.settings.get("blocking_rules_to_generate_predictions")
        if (
            not isinstance(raw_rules, Sequence)
            or isinstance(raw_rules, (str, bytes))
            or not raw_rules
        ):
            raise FellegiSunterError("ML-FS-069", "Native Splink rejects Cartesian EM training.")
        try:
            records = [
                _prepared_records(self._store, left, safe_alias=_INPUT_ALIASES[0]),
                _prepared_records(self._store, right, safe_alias=_INPUT_ALIASES[1]),
            ]
            with _quiet_splink():
                linker = linker_type(
                    records,
                    settings_type(**_runtime_settings(settings_plan)),
                    db_api=duckdb_api_type(),
                    input_table_aliases=list(_INPUT_ALIASES),
                    set_up_basic_logging=False,
                )
                linker.training.estimate_u_using_random_sampling(
                    max_pairs=model.u_max_pairs,
                    seed=random_seed,
                )
                for rule in raw_rules:
                    linker.training.estimate_parameters_using_expectation_maximisation(
                        str(rule),
                        estimate_without_term_frequencies=True,
                        fix_probability_two_random_records_match=True,
                        fix_m_probabilities=False,
                        fix_u_probabilities=True,
                        populate_probability_two_random_records_match_from_trained_values=False,
                    )
                parameters = linker.misc.save_model_to_json()
        except FellegiSunterError:
            raise
        except Exception:
            raise FellegiSunterError(
                "ML-FS-070", "Native Splink parameter estimation failed safely."
            ) from None
        if not isinstance(parameters, dict):
            raise FellegiSunterError(
                "ML-FS-061", "The learned Splink parameter structure is invalid."
            )
        parameters = _regularise_learned_parameters(
            parameters,
            probability_smoothing=model.probability_smoothing,
            m_effective_mass=parity.expected_pair_count,
            u_effective_mass=model.u_max_pairs,
        )
        parameters = _normalise_learned_parameters(parameters)
        parameter_digest = _validate_model_parameters(parameters, settings_plan)
        model_json = _canonical_json(parameters)
        model_digest = hashlib.sha256(model_json.encode("utf-8")).hexdigest()
        feature_schema_digest = splink_native_feature_schema_digest(
            left=left,
            right=right,
            settings_plan=settings_plan,
        )
        provisional = SplinkNativeModelArtifact(
            model_id=model.model_id,
            model_version=_MODEL_VERSION,
            engine_version=__version__,
            splink_version=runtime_version,
            random_seed=random_seed,
            configuration_digest=_require_digest(configuration_digest),
            feature_schema_digest=feature_schema_digest,
            settings_digest=settings_plan.settings_digest,
            training_candidate_pair_set_digest=parity.pair_set_digest,
            candidate_pair_count=parity.expected_pair_count,
            u_sample_pair_limit=model.u_max_pairs,
            probability_smoothing=model.probability_smoothing,
            smoothing_method=_SMOOTHING_METHOD,
            comparison_count=settings_plan.comparison_count,
            blocking_rule_count=settings_plan.blocking_rule_count,
            parameter_digest=parameter_digest,
            model_digest=model_digest,
            artifact_digest="0" * 64,
            model_json=model_json,
        )
        return SplinkNativeModelArtifact(
            **{
                **{
                    field_name: getattr(provisional, field_name)
                    for field_name in provisional.__slots__
                },
                "artifact_digest": _digest(_artifact_digest_payload(provisional)),
            }
        )

    def score(
        self,
        *,
        left: PreparedDataset,
        right: PreparedDataset,
        settings_plan: SplinkSettingsPlan,
        artifact: SplinkNativeModelArtifact,
        expected_pairs: Sequence[tuple[str, str]],
        maximum_candidate_pairs: int,
    ) -> SplinkNativeScoreResult:
        if maximum_candidate_pairs < 1 or len(expected_pairs) > maximum_candidate_pairs:
            raise FellegiSunterBudgetExceeded(
                "ML-FS-071", "Native Splink scoring exceeds the candidate-pair budget."
            )
        current_pair_set_digest = _pair_set_digest(expected_pairs)
        if (
            len(expected_pairs) != len(set(expected_pairs))
            or not expected_pairs
            or artifact.settings_digest != settings_plan.settings_digest
        ):
            raise FellegiSunterError(
                "ML-FS-072", "Native Splink scoring candidate or settings parity drifted."
            )
        current_feature_schema_digest = splink_native_feature_schema_digest(
            left=left,
            right=right,
            settings_plan=settings_plan,
        )
        if current_feature_schema_digest != artifact.feature_schema_digest:
            raise FellegiSunterError("ML-FS-076", "Native Splink scoring feature schema drifted.")
        parameters = _assert_artifact_integrity(artifact, settings_plan)
        linker_type, _, duckdb_api_type, runtime_version = _require_supported_runtime()
        if runtime_version != artifact.splink_version:
            raise FellegiSunterError(
                "ML-FS-066", "The native Splink artifact contract has drifted."
            )
        try:
            records = [
                _prepared_records(self._store, left, safe_alias=_INPUT_ALIASES[0]),
                _prepared_records(self._store, right, safe_alias=_INPUT_ALIASES[1]),
            ]
            with _quiet_splink():
                linker = linker_type(
                    records,
                    parameters,
                    db_api=duckdb_api_type(),
                    input_table_aliases=list(_INPUT_ALIASES),
                    set_up_basic_logging=False,
                )
                raw_scores = linker.inference.predict().as_record_dict()
        except FellegiSunterError:
            raise
        except Exception:
            raise FellegiSunterError("ML-FS-073", "Native Splink scoring failed safely.") from None
        if not isinstance(raw_scores, list):
            raise FellegiSunterError("ML-FS-074", "Native Splink returned invalid evidence.")
        observed: set[tuple[str, str]] = set()
        rows: list[tuple[object, ...]] = []
        digest_rows: list[dict[str, object]] = []
        for raw in raw_scores:
            if not isinstance(raw, Mapping):
                raise FellegiSunterError("ML-FS-074", "Native Splink returned invalid evidence.")
            try:
                left_key = raw["__ml_record_key_l"]
                right_key = raw["__ml_record_key_r"]
                probability = raw["match_probability"]
                match_weight = raw["match_weight"]
            except KeyError:
                raise FellegiSunterError(
                    "ML-FS-074", "Native Splink returned invalid evidence."
                ) from None
            pair = (left_key, right_key)
            if (
                not isinstance(left_key, str)
                or not isinstance(right_key, str)
                or not left_key
                or not right_key
                or pair in observed
                or isinstance(probability, bool)
                or isinstance(match_weight, bool)
                or not isinstance(probability, (int, float))
                or not isinstance(match_weight, (int, float))
                or not math.isfinite(float(probability))
                or not math.isfinite(float(match_weight))
                or not 0.0 <= float(probability) <= 1.0
            ):
                raise FellegiSunterError("ML-FS-074", "Native Splink returned invalid evidence.")
            observed.add(cast(tuple[str, str], pair))
            rows.append(
                (
                    left_key,
                    right_key,
                    float(match_weight),
                    float(probability),
                    artifact.model_id,
                    artifact.model_version,
                    artifact.parameter_digest,
                    artifact.probability_status,
                    artifact.decision_authority,
                )
            )
            digest_rows.append(
                {
                    "pair_digest": _pair_digest(left_key, right_key),
                    "match_weight": float(match_weight),
                    "probability": float(probability),
                }
            )
        if observed != set(expected_pairs):
            raise FellegiSunterError("ML-FS-075", "Native Splink score pair parity drifted.")
        score_digest = _digest(sorted(digest_rows, key=lambda row: cast(str, row["pair_digest"])))
        table_name = (
            f"__ml_splink_scores_{artifact.artifact_digest[:8]}_"
            f"{current_pair_set_digest[:12]}_{score_digest[:12]}"
        )
        table = self._store.create_table_from_rows(
            table_name,
            (
                ColumnSpec("left_record_key", "VARCHAR"),
                ColumnSpec("right_record_key", "VARCHAR"),
                ColumnSpec("__ml_fs_match_weight", "DOUBLE"),
                ColumnSpec("__ml_fs_model_probability", "DOUBLE"),
                ColumnSpec("__ml_fs_model_id", "VARCHAR"),
                ColumnSpec("__ml_fs_model_version", "VARCHAR"),
                ColumnSpec("__ml_fs_parameter_digest", "VARCHAR"),
                ColumnSpec("__ml_fs_probability_status", "VARCHAR"),
                ColumnSpec("__ml_fs_decision_authority", "VARCHAR"),
            ),
            sorted(rows, key=lambda row: (cast(str, row[0]), cast(str, row[1]))),
        )
        return SplinkNativeScoreResult(
            table=table,
            pair_count=len(rows),
            model_id=artifact.model_id,
            model_version=artifact.model_version,
            artifact_digest=artifact.artifact_digest,
            scoring_candidate_pair_set_digest=current_pair_set_digest,
            score_digest=score_digest,
        )


__all__ = [
    "SplinkNativeDuckDBMatcher",
    "SplinkNativeModelArtifact",
    "SplinkNativeScoreResult",
    "assert_splink_native_recipe_binding",
    "deserialize_splink_native_model",
    "serialize_splink_native_model",
    "splink_native_feature_schema_digest",
]
