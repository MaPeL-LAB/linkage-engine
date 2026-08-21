"""Deterministic Fellegi-Sunter reference estimation over canonical features.

This module is deliberately evidence-only.  Its posterior is the probability
implied by the fitted Fellegi-Sunter mixture and configured prior; it is not an
independently calibrated operational match probability.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from mapel_linkage.comparisons import ComparisonFeatureResult
from mapel_linkage.configuration.models import (
    ComparisonConfig,
    FellegiSunterModelConfig,
    MissingLevel,
)
from mapel_linkage.domain.errors import (
    DataPlaneError,
    FellegiSunterBudgetExceeded,
    FellegiSunterError,
)
from mapel_linkage.domain.sql_identifiers import quote_identifier
from mapel_linkage.domain.table_refs import TableRef
from mapel_linkage.io.duckdb_store import DuckDBStore
from mapel_linkage.preprocessing import PreparedDataset

_PAIR_COLUMNS: Final[tuple[str, ...]] = (
    "left_record_key",
    "right_record_key",
    "retrieval_rule_ids",
    "retrieval_rule_count",
)
_MODEL_VERSION: Final[str] = "m2d-reference-v2"
_PROBABILITY_STATUS: Final[str] = "model_posterior_uncalibrated"
_DECISION_AUTHORITY: Final[str] = "evidence_only"


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sql_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_double(value: float) -> str:
    if not math.isfinite(value):
        raise FellegiSunterError("ML-FS-015", "Model evidence weights must remain finite.")
    return f"CAST({value!r} AS DOUBLE)"


def _stable_base2_logistic_sql(match_weight: str) -> str:
    """Return a base-2 logistic expression that never evaluates a positive exponent."""

    positive_tail = f"POWER(2.0, -({match_weight}))"
    negative_tail = f"POWER(2.0, ({match_weight}))"
    return (
        f"CASE WHEN ({match_weight}) >= 0.0 "
        f"THEN (1.0 / (1.0 + {positive_tail})) "
        f"ELSE ({negative_tail} / (1.0 + {negative_tail})) END"
    )


def _logistic_from_log_odds(log_odds: float) -> float:
    if log_odds >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    exp_value = math.exp(log_odds)
    return exp_value / (1.0 + exp_value)


def _aggregate_int(value: object, *, code: str) -> int:
    """Narrow a package-owned DuckDB aggregate to an integer safely."""

    if isinstance(value, bool):
        raise FellegiSunterError(code, "An invalid aggregate value was returned.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise FellegiSunterError(code, "An invalid aggregate value was returned.")


def _aggregate_float(value: object, *, code: str) -> float:
    """Narrow a package-owned DuckDB aggregate to a finite float safely."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FellegiSunterError(code, "An invalid aggregate value was returned.")
    result = float(value)
    if not math.isfinite(result):
        raise FellegiSunterError(code, "An invalid aggregate value was returned.")
    return result


@dataclass(frozen=True, slots=True)
class RandomPairSampleResult:
    """Structural result for a deterministic random-pair sample."""

    table: TableRef
    possible_pair_count: int
    sampled_pair_count: int
    maximum_pairs: int
    random_seed: int

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "possible_pair_count": self.possible_pair_count,
            "sampled_pair_count": self.sampled_pair_count,
            "maximum_pairs": self.maximum_pairs,
            "random_seed": self.random_seed,
            "schema_digest": self.table.schema_digest,
        }


class DuckDBRandomPairSampler:
    """Create a bounded deterministic cross-source sample for estimating u."""

    __slots__ = ("_store",)

    def __init__(self, store: DuckDBStore) -> None:
        self._store = store

    def sample(
        self,
        *,
        left: PreparedDataset,
        right: PreparedDataset,
        maximum_pairs: int,
        random_seed: int,
        record_key_column: str = "__ml_record_key",
    ) -> RandomPairSampleResult:
        if maximum_pairs <= 0:
            raise FellegiSunterError("ML-FS-001", "The random-pair sample limit must be positive.")
        if random_seed < 0:
            raise FellegiSunterError("ML-FS-002", "The random-pair seed must be non-negative.")
        possible_pairs = left.table.row_count * right.table.row_count
        sampled_pairs = min(possible_pairs, maximum_pairs)
        if sampled_pairs == 0:
            raise FellegiSunterError(
                "ML-FS-003", "Random-pair sampling requires non-empty prepared datasets."
            )

        left_table = quote_identifier(left.table.table_name)
        right_table = quote_identifier(right.table.table_name)
        key = quote_identifier(record_key_column)
        digest = _canonical_digest(
            {
                "left_schema": left.table.schema_digest,
                "right_schema": right.table.schema_digest,
                "maximum_pairs": maximum_pairs,
                "random_seed": random_seed,
            }
        )
        table_name = f"__ml_fs_u_sample_{digest[:12]}"
        order_expression = (
            f"md5(CAST(l.{key} AS VARCHAR) || ':' || CAST(r.{key} AS VARCHAR) "
            f"|| ':' || CAST({random_seed} AS VARCHAR))"
        )
        select = (
            f"SELECT l.{key} AS {quote_identifier('left_record_key')}, "
            f"r.{key} AS {quote_identifier('right_record_key')}, "
            f"{_sql_text('__ml_fs_u_random_sample')} AS "
            f"{quote_identifier('retrieval_rule_ids')}, "
            f"1 AS {quote_identifier('retrieval_rule_count')} "
            f"FROM {left_table} AS l CROSS JOIN {right_table} AS r "
            f"ORDER BY {order_expression}, l.{key}, r.{key} LIMIT {sampled_pairs}"
        )
        try:
            table = self._store._create_temp_table_as(table_name, select)
        except DataPlaneError:
            raise FellegiSunterError(
                "ML-FS-004", "The random-pair sample could not be materialised."
            ) from None
        if table.row_count != sampled_pairs:
            raise FellegiSunterError(
                "ML-FS-005", "The random-pair sample did not satisfy its bounded contract."
            )
        return RandomPairSampleResult(
            table=table,
            possible_pair_count=possible_pairs,
            sampled_pair_count=sampled_pairs,
            maximum_pairs=maximum_pairs,
            random_seed=random_seed,
        )


@dataclass(frozen=True, slots=True)
class FellegiSunterLevelParameters:
    """One comparison-level probability and evidence weight."""

    level: int
    m_probability: float
    u_probability: float
    log2_bayes_factor: float

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("level must be non-negative")
        for value in (self.m_probability, self.u_probability):
            if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
                raise ValueError(
                    "level probabilities must be finite and strictly between zero and one"
                )
        if not math.isfinite(self.log2_bayes_factor):
            raise ValueError("level evidence weight must be finite")


@dataclass(frozen=True, slots=True)
class FellegiSunterComparisonParameters:
    """Probability parameters for one configured comparison."""

    comparison_id: str = field(repr=False)
    levels: tuple[FellegiSunterLevelParameters, ...]
    missing_level: int | None = None

    def __post_init__(self) -> None:
        level_ids = [level.level for level in self.levels]
        if level_ids != list(range(len(self.levels))):
            raise ValueError("comparison levels must be contiguous and ordered")
        if self.missing_level is not None and self.missing_level not in level_ids:
            raise ValueError("missing_level must identify an available level")
        if not math.isclose(sum(level.m_probability for level in self.levels), 1.0, abs_tol=1e-9):
            raise ValueError("m probabilities must sum to one")
        if not math.isclose(sum(level.u_probability for level in self.levels), 1.0, abs_tol=1e-9):
            raise ValueError("u probabilities must sum to one")

    def by_level(self) -> Mapping[int, FellegiSunterLevelParameters]:
        return MappingProxyType({level.level: level for level in self.levels})


@dataclass(frozen=True, slots=True)
class FellegiSunterModelArtifact:
    """Immutable aggregate model artifact; no pair or field values are retained."""

    model_id: str
    model_version: str
    prior_probability: float
    random_seed: int
    u_sample_pair_count: int
    em_candidate_pair_count: int
    em_iterations: int
    converged: bool
    smoothing: float
    convergence_tolerance: float
    feature_schema_digest: str
    parameter_digest: str
    comparisons: Mapping[str, FellegiSunterComparisonParameters] = field(repr=False)
    probability_status: str = _PROBABILITY_STATUS
    decision_authority: str = _DECISION_AUTHORITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparisons", MappingProxyType(dict(self.comparisons)))
        if not 0.0 < self.prior_probability < 1.0:
            raise ValueError("prior_probability must be strictly between zero and one")
        if self.em_iterations <= 0:
            raise ValueError("em_iterations must be positive")

    def safe_summary(self) -> dict[str, int | float | str | bool]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "comparison_count": len(self.comparisons),
            "u_sample_pair_count": self.u_sample_pair_count,
            "em_candidate_pair_count": self.em_candidate_pair_count,
            "em_iterations": self.em_iterations,
            "converged": self.converged,
            "parameter_digest": self.parameter_digest,
            "probability_status": self.probability_status,
            "decision_authority": self.decision_authority,
        }


@dataclass(frozen=True, slots=True)
class FellegiSunterScoreResult:
    """Structural reference to local Fellegi-Sunter pair evidence."""

    table: TableRef
    pair_count: int
    model_id: str
    model_version: str
    parameter_digest: str
    probability_status: str = _PROBABILITY_STATUS
    decision_authority: str = _DECISION_AUTHORITY

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "pair_count": self.pair_count,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "parameter_digest": self.parameter_digest,
            "probability_status": self.probability_status,
            "decision_authority": self.decision_authority,
            "schema_digest": self.table.schema_digest,
        }


class DuckDBFellegiSunterMatcher:
    """Estimate and score a bounded Fellegi-Sunter reference model."""

    __slots__ = ("_store",)

    def __init__(self, store: DuckDBStore) -> None:
        self._store = store

    def fit(
        self,
        *,
        u_features: ComparisonFeatureResult,
        em_features: ComparisonFeatureResult,
        comparisons: Sequence[ComparisonConfig],
        model: FellegiSunterModelConfig,
        random_seed: int,
    ) -> FellegiSunterModelArtifact:
        if u_features.candidate_pair_count > model.u_max_pairs:
            raise FellegiSunterBudgetExceeded(
                "ML-FS-006", "The u-estimation feature sample exceeds its configured pair budget."
            )
        if em_features.candidate_pair_count == 0:
            raise FellegiSunterError(
                "ML-FS-007", "Expectation-maximisation requires candidate feature rows."
            )
        comparison_ids = [comparison.id for comparison in comparisons]
        if not comparison_ids or len(comparison_ids) != len(set(comparison_ids)):
            raise FellegiSunterError(
                "ML-FS-008", "Fellegi-Sunter fitting requires unique configured comparisons."
            )
        if set(comparison_ids) != set(u_features.columns) or set(comparison_ids) != set(
            em_features.columns
        ):
            raise FellegiSunterError(
                "ML-FS-009", "Feature tables do not match the configured comparison contract."
            )

        u_probabilities = self._estimate_u_probabilities(
            u_features,
            comparisons,
            smoothing=model.probability_smoothing,
        )
        patterns = self._comparison_patterns(em_features, comparisons)
        initial_m = self._initial_m_probabilities(comparisons, u_probabilities)
        m_probabilities, iterations, converged = self._estimate_m_probabilities(
            patterns=patterns,
            comparisons=comparisons,
            u_probabilities=u_probabilities,
            initial_m_probabilities=initial_m,
            prior_probability=model.probability_two_random_records_match,
            smoothing=model.probability_smoothing,
            maximum_iterations=model.em_max_iterations,
            convergence_tolerance=model.em_convergence,
        )
        comparison_parameters: dict[str, FellegiSunterComparisonParameters] = {}
        digest_payload: dict[str, object] = {
            "model_id": model.model_id,
            "model_version": _MODEL_VERSION,
            "prior_probability": model.probability_two_random_records_match,
            "random_seed": random_seed,
            "smoothing": model.probability_smoothing,
            "em_max_iterations": model.em_max_iterations,
            "em_convergence": model.em_convergence,
            "comparisons": {},
        }
        digest_comparisons: dict[str, object] = {}
        for comparison in comparisons:
            missing_level = self._missing_level(comparison)
            levels: list[FellegiSunterLevelParameters] = []
            digest_levels: list[dict[str, float | int]] = []
            for level in range(len(comparison.levels)):
                m_value = m_probabilities[comparison.id][level]
                u_value = u_probabilities[comparison.id][level]
                weight = 0.0 if level == missing_level else math.log2(m_value / u_value)
                levels.append(
                    FellegiSunterLevelParameters(
                        level=level,
                        m_probability=m_value,
                        u_probability=u_value,
                        log2_bayes_factor=weight,
                    )
                )
                digest_levels.append(
                    {
                        "level": level,
                        "m": m_value,
                        "u": u_value,
                        "log2_bf": weight,
                    }
                )
            comparison_parameters[comparison.id] = FellegiSunterComparisonParameters(
                comparison_id=comparison.id,
                levels=tuple(levels),
                missing_level=missing_level,
            )
            digest_comparisons[comparison.id] = digest_levels
        digest_payload["comparisons"] = digest_comparisons
        parameter_digest = _canonical_digest(digest_payload)
        feature_schema_digest = _canonical_digest(
            {
                "u": u_features.table.schema_digest,
                "em": em_features.table.schema_digest,
                "comparisons": comparison_ids,
            }
        )
        return FellegiSunterModelArtifact(
            model_id=model.model_id,
            model_version=_MODEL_VERSION,
            prior_probability=model.probability_two_random_records_match,
            random_seed=random_seed,
            u_sample_pair_count=u_features.candidate_pair_count,
            em_candidate_pair_count=em_features.candidate_pair_count,
            em_iterations=iterations,
            converged=converged,
            smoothing=model.probability_smoothing,
            convergence_tolerance=model.em_convergence,
            feature_schema_digest=feature_schema_digest,
            parameter_digest=parameter_digest,
            comparisons=comparison_parameters,
        )

    def score(
        self,
        *,
        features: ComparisonFeatureResult,
        model: FellegiSunterModelArtifact,
    ) -> FellegiSunterScoreResult:
        if set(features.columns) != set(model.comparisons):
            raise FellegiSunterError(
                "ML-FS-010", "The score feature contract does not match the model artifact."
            )
        table = quote_identifier(features.table.table_name)
        pair_projection = ", ".join(quote_identifier(column) for column in _PAIR_COLUMNS)
        component_expressions: list[str] = []
        for comparison_id, parameters in model.comparisons.items():
            level_column = features.columns[comparison_id].level
            cases = " ".join(
                f"WHEN {level.level} THEN {_sql_double(level.log2_bayes_factor)}"
                for level in parameters.levels
            )
            component_expressions.append(
                f"CASE {quote_identifier(level_column)} {cases} ELSE {_sql_double(0.0)} END"
            )
        log2_bayes_factor = " + ".join(component_expressions) or "0.0"
        prior_log2_odds = math.log2(model.prior_probability / (1.0 - model.prior_probability))
        match_weight = f"({_sql_double(prior_log2_odds)} + ({log2_bayes_factor}))"
        probability = _stable_base2_logistic_sql(match_weight)
        suffix = _canonical_digest(
            {
                "features": features.table.schema_digest,
                "parameters": model.parameter_digest,
            }
        )[:12]
        output_table = f"__ml_fs_scores_{suffix}"
        select = (
            f"SELECT {pair_projection}, "
            f"CAST(({log2_bayes_factor}) AS DOUBLE) AS "
            f"{quote_identifier('__ml_fs_log2_bayes_factor')}, "
            f"CAST({match_weight} AS DOUBLE) AS {quote_identifier('__ml_fs_match_weight')}, "
            f"CAST({probability} AS DOUBLE) AS "
            f"{quote_identifier('__ml_fs_model_probability')}, "
            f"{_sql_text(model.model_id)} AS {quote_identifier('__ml_fs_model_id')}, "
            f"{_sql_text(model.model_version)} AS {quote_identifier('__ml_fs_model_version')}, "
            f"{_sql_text(model.parameter_digest)} AS "
            f"{quote_identifier('__ml_fs_parameter_digest')}, "
            f"{_sql_text(model.probability_status)} AS "
            f"{quote_identifier('__ml_fs_probability_status')}, "
            f"{_sql_text(model.decision_authority)} AS "
            f"{quote_identifier('__ml_fs_decision_authority')} "
            f"FROM {table} ORDER BY {quote_identifier('left_record_key')}, "
            f"{quote_identifier('right_record_key')}"
        )
        try:
            score_table = self._store._create_temp_table_as(output_table, select)
        except DataPlaneError:
            raise FellegiSunterError(
                "ML-FS-011", "Fellegi-Sunter evidence scores could not be materialised."
            ) from None
        if score_table.row_count != features.candidate_pair_count:
            raise FellegiSunterError(
                "ML-FS-012", "Fellegi-Sunter scoring did not preserve candidate coverage."
            )
        quoted_output = quote_identifier(score_table.table_name)
        try:
            invalid_count = self._store._scalar_int(
                "SELECT COUNT(*) FROM "
                f"{quoted_output} WHERE "
                f"NOT isfinite({quote_identifier('__ml_fs_log2_bayes_factor')}) OR "
                f"NOT isfinite({quote_identifier('__ml_fs_match_weight')}) OR "
                f"NOT isfinite({quote_identifier('__ml_fs_model_probability')}) OR "
                f"{quote_identifier('__ml_fs_model_probability')} < 0.0 OR "
                f"{quote_identifier('__ml_fs_model_probability')} > 1.0"
            )
        except DataPlaneError:
            raise FellegiSunterError(
                "ML-FS-013", "Fellegi-Sunter score integrity could not be verified."
            ) from None
        if invalid_count:
            raise FellegiSunterError(
                "ML-FS-014", "Fellegi-Sunter scoring produced invalid aggregate evidence."
            )
        return FellegiSunterScoreResult(
            table=score_table,
            pair_count=score_table.row_count,
            model_id=model.model_id,
            model_version=model.model_version,
            parameter_digest=model.parameter_digest,
        )

    def _estimate_u_probabilities(
        self,
        features: ComparisonFeatureResult,
        comparisons: Sequence[ComparisonConfig],
        *,
        smoothing: float,
    ) -> dict[str, tuple[float, ...]]:
        table = quote_identifier(features.table.table_name)
        probabilities: dict[str, tuple[float, ...]] = {}
        for comparison in comparisons:
            level_column = quote_identifier(features.columns[comparison.id].level)
            sql = (
                f"SELECT {level_column}, COUNT(*) FROM {table} "
                f"WHERE {level_column} IS NOT NULL GROUP BY {level_column}"
            )
            rows = self._safe_internal_rows(sql, "ML-FS-013")
            counts = [0.0] * len(comparison.levels)
            for level, count in rows:
                level_index = _aggregate_int(level, code="ML-FS-014")
                if level_index < 0 or level_index >= len(counts):
                    raise FellegiSunterError(
                        "ML-FS-014", "An invalid comparison level was found during estimation."
                    )
                counts[level_index] = _aggregate_float(count, code="ML-FS-014")
            denominator = sum(counts) + smoothing * len(counts)
            probabilities[comparison.id] = tuple(
                (count + smoothing) / denominator for count in counts
            )
        return probabilities

    def _comparison_patterns(
        self,
        features: ComparisonFeatureResult,
        comparisons: Sequence[ComparisonConfig],
    ) -> tuple[tuple[tuple[int | None, ...], int], ...]:
        table = quote_identifier(features.table.table_name)
        level_columns = [quote_identifier(features.columns[item.id].level) for item in comparisons]
        projection = ", ".join(level_columns)
        sql = f"SELECT {projection}, COUNT(*) FROM {table} GROUP BY {projection}"
        rows = self._safe_internal_rows(sql, "ML-FS-015")
        patterns: list[tuple[tuple[int | None, ...], int]] = []
        for row in rows:
            levels = tuple(
                None if value is None else _aggregate_int(value, code="ML-FS-016")
                for value in row[:-1]
            )
            patterns.append((levels, _aggregate_int(row[-1], code="ML-FS-016")))
        if sum(count for _, count in patterns) != features.candidate_pair_count:
            raise FellegiSunterError(
                "ML-FS-016", "Comparison-vector aggregation did not preserve candidate coverage."
            )
        return tuple(patterns)

    @classmethod
    def _initial_m_probabilities(
        cls,
        comparisons: Sequence[ComparisonConfig],
        u_probabilities: Mapping[str, tuple[float, ...]],
    ) -> dict[str, tuple[float, ...]]:
        initial: dict[str, tuple[float, ...]] = {}
        for comparison in comparisons:
            missing = cls._missing_level(comparison)
            nonmissing = [level for level in range(len(comparison.levels)) if level != missing]
            missing_mass = 0.0 if missing is None else u_probabilities[comparison.id][missing]
            raw = [math.exp(-float(rank)) for rank in range(len(nonmissing))]
            raw_total = sum(raw)
            remaining_mass = 1.0 - missing_mass
            values = [0.0] * len(comparison.levels)
            for level, value in zip(nonmissing, raw, strict=True):
                values[level] = remaining_mass * value / raw_total
            if missing is not None:
                values[missing] = missing_mass
            initial[comparison.id] = tuple(values)
        return initial

    @classmethod
    def _estimate_m_probabilities(
        cls,
        *,
        patterns: Sequence[tuple[tuple[int | None, ...], int]],
        comparisons: Sequence[ComparisonConfig],
        u_probabilities: Mapping[str, tuple[float, ...]],
        initial_m_probabilities: Mapping[str, tuple[float, ...]],
        prior_probability: float,
        smoothing: float,
        maximum_iterations: int,
        convergence_tolerance: float,
    ) -> tuple[dict[str, tuple[float, ...]], int, bool]:
        current = dict(initial_m_probabilities)
        prior_log_odds = math.log(prior_probability / (1.0 - prior_probability))
        converged = False
        for iteration in range(1, maximum_iterations + 1):
            weighted_counts = {
                comparison.id: [0.0] * len(comparison.levels) for comparison in comparisons
            }
            observed_totals = {comparison.id: 0.0 for comparison in comparisons}
            for vector, count in patterns:
                log_odds = prior_log_odds
                for index, comparison in enumerate(comparisons):
                    level = vector[index]
                    if level is None or level == cls._missing_level(comparison):
                        continue
                    log_odds += math.log(
                        current[comparison.id][level] / u_probabilities[comparison.id][level]
                    )
                posterior = _logistic_from_log_odds(log_odds)
                weighted = posterior * count
                for index, comparison in enumerate(comparisons):
                    level = vector[index]
                    if level is None or level == cls._missing_level(comparison):
                        continue
                    weighted_counts[comparison.id][level] += weighted
                    observed_totals[comparison.id] += weighted

            updated: dict[str, tuple[float, ...]] = {}
            maximum_change = 0.0
            for comparison in comparisons:
                missing = cls._missing_level(comparison)
                nonmissing = [level for level in range(len(comparison.levels)) if level != missing]
                missing_mass = 0.0 if missing is None else u_probabilities[comparison.id][missing]
                remaining_mass = 1.0 - missing_mass
                denominator = observed_totals[comparison.id] + smoothing * len(nonmissing)
                values = [0.0] * len(comparison.levels)
                if denominator <= 0.0:
                    for level in nonmissing:
                        values[level] = current[comparison.id][level]
                else:
                    for level in nonmissing:
                        values[level] = (
                            remaining_mass
                            * (weighted_counts[comparison.id][level] + smoothing)
                            / denominator
                        )
                if missing is not None:
                    values[missing] = missing_mass
                updated[comparison.id] = tuple(values)
                maximum_change = max(
                    maximum_change,
                    max(
                        abs(new - old)
                        for new, old in zip(values, current[comparison.id], strict=True)
                    ),
                )
            current = updated
            if maximum_change <= convergence_tolerance:
                converged = True
                return current, iteration, converged
        return current, maximum_iterations, converged

    @staticmethod
    def _missing_level(comparison: ComparisonConfig) -> int | None:
        for index, level in enumerate(comparison.levels):
            if isinstance(level, MissingLevel):
                return index
        return None

    def _safe_internal_rows(self, sql: str, code: str) -> list[tuple[object, ...]]:
        try:
            return self._store._fetch_internal_rows(sql)
        except DataPlaneError:
            raise FellegiSunterError(
                code, "An internal Fellegi-Sunter aggregate could not be calculated."
            ) from None
