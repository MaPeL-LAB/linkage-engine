"""Package-owned compilation boundary for Splink 4 settings.

Project configuration never supplies Splink SQL or Python objects.  This
adapter emits only package-generated expressions over canonical internal
columns.  The reference matcher remains the deterministic test oracle while
Splink is the designated production Fellegi-Sunter adapter target.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from mapel_linkage.configuration.models import (
    AllPredicate,
    AnyPredicate,
    BlockPredicate,
    CategoricalComparison,
    ComparisonConfig,
    DamerauLevenshteinComparison,
    DateDifferenceComparison,
    DateWindowPredicate,
    ElseLevel,
    ExactComparison,
    ExactLevel,
    ExactPredicate,
    FellegiSunterModelConfig,
    JaroWinklerComparison,
    LevenshteinComparison,
    MaximumDifferenceLevel,
    MissingLevel,
    NumericDifferenceComparison,
    PrefixEqualPredicate,
    QGramComparison,
    ThresholdLevel,
)
from mapel_linkage.domain.errors import FellegiSunterError
from mapel_linkage.domain.sql_identifiers import quote_identifier
from mapel_linkage.preprocessing import PreparedDataset


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _column_pair(column: str) -> tuple[str, str]:
    return (quote_identifier(f"{column}_l"), quote_identifier(f"{column}_r"))


@dataclass(frozen=True, slots=True)
class SplinkSettingsPlan:
    """Immutable non-row-bearing settings plan for the optional Splink runtime."""

    settings_digest: str
    comparison_count: int
    blocking_rule_count: int
    settings: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "settings_digest": self.settings_digest,
            "comparison_count": self.comparison_count,
            "blocking_rule_count": self.blocking_rule_count,
        }

    def build_settings_creator(self) -> object:
        """Build a Splink SettingsCreator only when the optional runtime is installed."""

        try:
            from splink import SettingsCreator
        except ImportError:
            raise FellegiSunterError(
                "ML-FS-020", "The optional Splink runtime is unavailable."
            ) from None
        try:
            return SettingsCreator(**dict(self.settings))
        except Exception:
            raise FellegiSunterError(
                "ML-FS-021", "The package-owned Splink settings plan was rejected."
            ) from None


class SplinkSettingsPlanCompiler:
    """Compile validated configuration to package-owned Splink settings."""

    def compile(
        self,
        *,
        left: PreparedDataset,
        right: PreparedDataset,
        comparisons: Sequence[ComparisonConfig],
        blocking_rules: Sequence[BlockPredicate],
        model: FellegiSunterModelConfig,
    ) -> SplinkSettingsPlan:
        if not comparisons:
            raise FellegiSunterError(
                "ML-FS-022", "Splink settings require at least one configured comparison."
            )
        comparison_payload = [
            self._comparison(comparison, left, right) for comparison in comparisons
        ]
        blocking_payload = [self._predicate(predicate, left, right) for predicate in blocking_rules]
        settings: dict[str, Any] = {
            "link_type": "link_only",
            "comparisons": comparison_payload,
            "blocking_rules_to_generate_predictions": blocking_payload,
            "probability_two_random_records_match": model.probability_two_random_records_match,
            "em_convergence": model.em_convergence,
            "max_iterations": model.em_max_iterations,
            "retain_matching_columns": False,
            "retain_intermediate_calculation_columns": True,
            "unique_id_column_name": "__ml_record_key",
            "source_dataset_column_name": "__ml_dataset_id",
        }
        return SplinkSettingsPlan(
            settings_digest=_digest(settings),
            comparison_count=len(comparison_payload),
            blocking_rule_count=len(blocking_payload),
            settings=settings,
        )

    def _comparison(
        self,
        comparison: ComparisonConfig,
        left: PreparedDataset,
        right: PreparedDataset,
    ) -> dict[str, Any]:
        try:
            left_column = left.variable_columns[comparison.variable]
            right_column = right.variable_columns[comparison.variable]
        except KeyError:
            raise FellegiSunterError(
                "ML-FS-023", "A Splink comparison references an unavailable canonical variable."
            ) from None
        if left_column != right_column:
            raise FellegiSunterError(
                "ML-FS-024", "Canonical comparison columns are inconsistent between datasets."
            )
        left_value, right_value = _column_pair(left_column)
        levels: list[dict[str, Any]] = []
        for level in comparison.levels:
            if isinstance(level, MissingLevel):
                levels.append(
                    {
                        "sql_condition": f"{left_value} IS NULL OR {right_value} IS NULL",
                        "label_for_charts": "Missing",
                        "is_null_level": True,
                    }
                )
            elif isinstance(level, ExactLevel):
                levels.append(
                    {
                        "sql_condition": f"{left_value} = {right_value}",
                        "label_for_charts": "Exact",
                    }
                )
            elif isinstance(level, ThresholdLevel):
                levels.append(
                    {
                        "sql_condition": self._threshold_condition(
                            comparison, left_value, right_value, level.minimum
                        ),
                        "label_for_charts": "Similarity threshold",
                    }
                )
            elif isinstance(level, MaximumDifferenceLevel):
                levels.append(
                    {
                        "sql_condition": self._difference_condition(
                            comparison, left_value, right_value, float(level.value)
                        ),
                        "label_for_charts": "Difference threshold",
                    }
                )
            elif isinstance(level, ElseLevel):
                levels.append({"sql_condition": "ELSE", "label_for_charts": "Else"})
        return {
            "output_column_name": f"cmp_{hashlib.sha256(comparison.id.encode()).hexdigest()[:16]}",
            "comparison_description": "Package-owned canonical comparison",
            "comparison_levels": levels,
        }

    @staticmethod
    def _threshold_condition(
        comparison: ComparisonConfig,
        left_value: str,
        right_value: str,
        threshold: float,
    ) -> str:
        function = comparison.function
        if isinstance(function, (ExactComparison, CategoricalComparison)):
            metric = f"CASE WHEN {left_value} = {right_value} THEN 1.0 ELSE 0.0 END"
        elif isinstance(function, JaroWinklerComparison):
            metric = f"jaro_winkler_similarity({left_value}, {right_value})"
        elif isinstance(function, LevenshteinComparison):
            distance = f"levenshtein({left_value}, {right_value})"
            length = f"GREATEST(LENGTH({left_value}), LENGTH({right_value}))"
            metric = (
                f"CASE WHEN {length} = 0 THEN 1.0 ELSE "
                f"GREATEST(0.0, 1.0 - CAST({distance} AS DOUBLE) / {length}) END"
            )
        elif isinstance(function, DamerauLevenshteinComparison):
            distance = f"damerau_levenshtein({left_value}, {right_value})"
            length = f"GREATEST(LENGTH({left_value}), LENGTH({right_value}))"
            metric = (
                f"CASE WHEN {length} = 0 THEN 1.0 ELSE "
                f"GREATEST(0.0, 1.0 - CAST({distance} AS DOUBLE) / {length}) END"
            )
        elif isinstance(function, QGramComparison):
            q = function.q

            def grams(value: str) -> str:
                indices = f"range(1, GREATEST(LENGTH({value}) - {q} + 2, 2))"
                return f"list_distinct(list_transform({indices}, i -> SUBSTR({value}, i, {q})))"

            left_grams = grams(left_value)
            right_grams = grams(right_value)
            intersection = f"list_count(list_intersect({left_grams}, {right_grams}))"
            denominator = f"list_count({left_grams}) + list_count({right_grams})"
            metric = f"COALESCE(2.0 * {intersection} / NULLIF({denominator}, 0), 1.0)"
        else:
            raise FellegiSunterError(
                "ML-FS-026", "A non-similarity comparison used a similarity threshold."
            )
        return f"{metric} >= {threshold!r}"

    @staticmethod
    def _difference_condition(
        comparison: ComparisonConfig,
        left_value: str,
        right_value: str,
        maximum: float,
    ) -> str:
        function = comparison.function
        if isinstance(function, DateDifferenceComparison):
            metric = f"ABS(date_diff('{function.unit}', {left_value}, {right_value}))"
        elif isinstance(function, NumericDifferenceComparison):
            metric = (
                f"ABS(CAST({left_value} AS DOUBLE) - CAST({right_value} AS DOUBLE)) "
                f"/ {function.scale!r}"
            )
        else:
            raise FellegiSunterError(
                "ML-FS-027", "A non-difference comparison used a difference threshold."
            )
        return f"{metric} <= {maximum!r}"

    def _predicate(
        self,
        predicate: BlockPredicate,
        left: PreparedDataset,
        right: PreparedDataset,
    ) -> str:
        if isinstance(predicate, AllPredicate):
            return " AND ".join(
                f"({self._predicate(term, left, right)})" for term in predicate.terms
            )
        if isinstance(predicate, AnyPredicate):
            return " OR ".join(
                f"({self._predicate(term, left, right)})" for term in predicate.terms
            )
        try:
            left_column = left.variable_columns[predicate.variable]
            right_column = right.variable_columns[predicate.variable]
        except KeyError:
            raise FellegiSunterError(
                "ML-FS-028", "A Splink blocking rule references an unavailable variable."
            ) from None
        if left_column != right_column:
            raise FellegiSunterError(
                "ML-FS-029", "Canonical blocking columns are inconsistent between datasets."
            )
        left_value = f"l.{quote_identifier(left_column)}"
        right_value = f"r.{quote_identifier(right_column)}"
        if isinstance(predicate, ExactPredicate):
            return f"{left_value} = {right_value}"
        if isinstance(predicate, PrefixEqualPredicate):
            return (
                f"SUBSTR(CAST({left_value} AS VARCHAR), 1, {predicate.length}) = "
                f"SUBSTR(CAST({right_value} AS VARCHAR), 1, {predicate.length})"
            )
        if isinstance(predicate, DateWindowPredicate):
            return f"ABS(date_diff('day', {left_value}, {right_value})) <= {predicate.maximum_days}"
        raise FellegiSunterError(
            "ML-FS-030", "An unsupported Splink blocking predicate was rejected."
        )
