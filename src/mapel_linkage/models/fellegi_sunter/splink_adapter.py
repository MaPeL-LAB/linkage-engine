"""Package-owned compilation boundary for Splink 4 settings.

Project configuration never supplies Splink SQL or Python objects.  This
adapter emits only package-generated expressions over canonical internal
columns.  The reference matcher remains the deterministic test oracle while
Splink is the designated production Fellegi-Sunter adapter target.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import io
import json
import logging
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
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
from mapel_linkage.io import DuckDBStore
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


@dataclass(frozen=True, slots=True)
class SplinkCandidateParityReport:
    """Aggregate-only evidence that Splink and engine blocking produce the same pairs."""

    expected_pair_count: int
    observed_pair_count: int
    pair_set_digest: str
    settings_digest: str
    splink_version: str
    parity: bool = True
    decision_authority: str = "none"
    relationship_authority: str = "none"
    merge_authority: str = "none"

    def safe_summary(self) -> dict[str, bool | int | str]:
        return {
            "expected_pair_count": self.expected_pair_count,
            "observed_pair_count": self.observed_pair_count,
            "pair_set_digest": self.pair_set_digest,
            "settings_digest": self.settings_digest,
            "splink_version": self.splink_version,
            "parity": self.parity,
            "decision_authority": self.decision_authority,
            "relationship_authority": self.relationship_authority,
            "merge_authority": self.merge_authority,
        }


def _pair_digest(left_record_key: str, right_record_key: str) -> str:
    return hashlib.sha256(f"{left_record_key}\x1f{right_record_key}".encode()).hexdigest()


def _prepared_records(
    store: DuckDBStore,
    dataset: PreparedDataset,
) -> list[dict[str, object]]:
    columns = (
        "__ml_record_key",
        "__ml_dataset_id",
        *tuple(sorted(set(dataset.variable_columns.values()))),
    )
    select_list = ", ".join(quote_identifier(column) for column in columns)
    rows = store._fetch_model_rows(
        f"SELECT {select_list} FROM {quote_identifier(dataset.table.table_name)} "
        f"ORDER BY {quote_identifier('__ml_record_key')}"
    )
    return [dict(zip(columns, row, strict=True)) for row in rows]


class SplinkCandidateParityChecker:
    """Run Splink deterministic blocking internally and compare only pair-key sets."""

    @staticmethod
    def check(
        *,
        store: DuckDBStore,
        left: PreparedDataset,
        right: PreparedDataset,
        settings_plan: SplinkSettingsPlan,
        expected_pairs: Sequence[tuple[str, str]],
    ) -> SplinkCandidateParityReport:
        expected = frozenset(expected_pairs)
        if not expected:
            raise FellegiSunterError(
                "ML-FS-031", "Splink candidate parity requires a non-empty bounded pair set."
            )
        if len(expected) != len(expected_pairs) or any(
            not left_key or not right_key for left_key, right_key in expected_pairs
        ):
            raise FellegiSunterError(
                "ML-FS-038", "Splink candidate parity requires unique valid expected pairs."
            )
        try:
            splink = importlib.import_module("splink")
            linker_type = splink.Linker
            settings_type = splink.SettingsCreator
            duckdb_api_type = splink.DuckDBAPI
        except (ImportError, AttributeError):
            raise FellegiSunterError(
                "ML-FS-032", "The optional Splink runtime is unavailable."
            ) from None

        raw_rules = settings_plan.settings.get("blocking_rules_to_generate_predictions")
        if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes)):
            raise FellegiSunterError(
                "ML-FS-033", "The package-owned Splink blocking plan is invalid."
            )
        rules = tuple(str(rule) for rule in raw_rules)
        if not rules:
            raise FellegiSunterError(
                "ML-FS-034", "Splink candidate parity rejects an unblocked Cartesian join."
            )

        settings = settings_type(
            link_type="link_only",
            blocking_rules_to_generate_predictions=list(rules),
            unique_id_column_name="__ml_record_key",
            retain_matching_columns=False,
            retain_intermediate_calculation_columns=False,
        )
        logger = logging.getLogger("splink")
        previous_level = logger.level
        previous_disable = logging.root.manager.disable
        try:
            logger.setLevel(logging.CRITICAL)
            logging.disable(logging.CRITICAL)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                linker = linker_type(
                    [
                        _prepared_records(store, left),
                        _prepared_records(store, right),
                    ],
                    settings,
                    db_api=duckdb_api_type(),
                    input_table_aliases=[left.dataset_id, right.dataset_id],
                )
                records = linker.inference.deterministic_link().as_record_dict()
        except Exception:
            raise FellegiSunterError(
                "ML-FS-035", "Splink candidate parity could not be evaluated safely."
            ) from None
        finally:
            logging.disable(previous_disable)
            logger.setLevel(previous_level)

        if not isinstance(records, list):
            raise FellegiSunterError(
                "ML-FS-036", "Splink candidate parity returned an invalid result."
            )
        observed_pairs: set[tuple[str, str]] = set()
        for record in records:
            if not isinstance(record, Mapping):
                raise FellegiSunterError(
                    "ML-FS-036", "Splink candidate parity returned an invalid result."
                )
            try:
                left_key = record["__ml_record_key_l"]
                right_key = record["__ml_record_key_r"]
            except KeyError:
                raise FellegiSunterError(
                    "ML-FS-036", "Splink candidate parity returned an invalid result."
                ) from None
            if (
                not isinstance(left_key, str)
                or not isinstance(right_key, str)
                or not left_key
                or not right_key
                or (left_key, right_key) in observed_pairs
            ):
                raise FellegiSunterError(
                    "ML-FS-036", "Splink candidate parity returned an invalid result."
                )
            observed_pairs.add((left_key, right_key))
        observed = frozenset(observed_pairs)
        if observed != expected:
            raise FellegiSunterError(
                "ML-FS-037", "DuckDB and Splink candidate pair sets are inconsistent."
            )
        pair_set_digest = _digest(sorted(_pair_digest(*pair) for pair in observed))
        try:
            runtime_version = importlib.metadata.version("splink")
        except importlib.metadata.PackageNotFoundError:
            runtime_version = "unavailable"
        return SplinkCandidateParityReport(
            expected_pair_count=len(expected),
            observed_pair_count=len(observed),
            pair_set_digest=pair_set_digest,
            settings_digest=settings_plan.settings_digest,
            splink_version=runtime_version,
        )
