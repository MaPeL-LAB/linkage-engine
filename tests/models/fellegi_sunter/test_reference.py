from __future__ import annotations

import math
from dataclasses import replace
from datetime import date

import pytest

from mapel_linkage.comparisons import (
    ComparisonFeatureResult,
    DuckDBComparisonFeatureBuilder,
)
from mapel_linkage.configuration.models import ComparisonConfig, FellegiSunterModelConfig
from mapel_linkage.domain import FellegiSunterBudgetExceeded, FellegiSunterError, TableRef
from mapel_linkage.io import ColumnSpec, DuckDBStore
from mapel_linkage.models import (
    DuckDBFellegiSunterMatcher,
    DuckDBRandomPairSampler,
    FellegiSunterModelArtifact,
)
from mapel_linkage.preprocessing import PreparedDataset

_PREPARED_COLUMNS = (
    ColumnSpec("__ml_record_key", "VARCHAR"),
    ColumnSpec("v_text", "VARCHAR"),
    ColumnSpec("m_text", "BOOLEAN"),
    ColumnSpec("v_date", "DATE"),
    ColumnSpec("m_date", "BOOLEAN"),
)
_CANDIDATE_COLUMNS = (
    ColumnSpec("left_record_key", "VARCHAR"),
    ColumnSpec("right_record_key", "VARCHAR"),
    ColumnSpec("retrieval_rule_ids", "VARCHAR"),
    ColumnSpec("retrieval_rule_count", "INTEGER"),
)


def _prepared(store: DuckDBStore) -> tuple[PreparedDataset, PreparedDataset]:
    left_table = store.create_table_from_rows(
        "fs_left",
        _PREPARED_COLUMNS,
        (
            ("left-1", "alpha", False, date(2020, 1, 1), False),
            ("left-2", "bravo", False, date(2020, 2, 2), False),
            ("left-3", "charlie", False, date(2020, 3, 3), False),
            ("left-4", None, True, None, True),
        ),
    )
    right_table = store.create_table_from_rows(
        "fs_right",
        _PREPARED_COLUMNS,
        (
            ("right-1", "alpha", False, date(2020, 1, 1), False),
            ("right-2", "bravo", False, date(2020, 2, 3), False),
            ("right-3", "delta", False, date(2020, 4, 4), False),
            ("right-4", None, True, None, True),
        ),
    )
    value_columns = {"text_value": "v_text", "date_value": "v_date"}
    missing_columns = {"text_value": "m_text", "date_value": "m_date"}
    return (
        PreparedDataset("left", left_table, value_columns, missing_columns),
        PreparedDataset("right", right_table, value_columns, missing_columns),
    )


def _comparisons() -> tuple[ComparisonConfig, ...]:
    return tuple(
        ComparisonConfig.model_validate(payload)
        for payload in (
            {
                "id": "text_similarity",
                "variable": "text_value",
                "function": {"kind": "jaro_winkler"},
                "levels": [
                    {"kind": "missing"},
                    {"kind": "exact"},
                    {"kind": "threshold", "minimum": 0.8},
                    {"kind": "else"},
                ],
            },
            {
                "id": "date_distance",
                "variable": "date_value",
                "function": {"kind": "date_difference", "unit": "day"},
                "levels": [
                    {"kind": "missing"},
                    {"kind": "exact"},
                    {"kind": "maximum_difference", "value": 2.0},
                    {"kind": "else"},
                ],
            },
        )
    )


def _candidate_table(store: DuckDBStore) -> TableRef:
    return store.create_table_from_rows(
        "fs_candidates",
        _CANDIDATE_COLUMNS,
        (
            ("left-1", "right-1", "rule_a", 1),
            ("left-2", "right-2", "rule_a", 1),
            ("left-3", "right-3", "rule_b", 1),
            ("left-4", "right-4", "rule_b", 1),
            ("left-1", "right-3", "rule_c", 1),
            ("left-3", "right-1", "rule_c", 1),
        ),
    )


def _model_config(**overrides: object) -> FellegiSunterModelConfig:
    payload: dict[str, object] = {
        "enabled": True,
        "implementation": "splink_duckdb",
        "model_id": "fs_baseline",
        "probability_two_random_records_match": 0.1,
        "u_max_pairs": 16,
        "em_max_iterations": 50,
        "em_convergence": 0.000001,
        "probability_smoothing": 0.5,
    }
    payload.update(overrides)
    return FellegiSunterModelConfig.model_validate(payload)


def _fit_fixture(
    store: DuckDBStore,
) -> tuple[
    DuckDBFellegiSunterMatcher,
    FellegiSunterModelArtifact,
    ComparisonFeatureResult,
    tuple[ComparisonConfig, ...],
]:
    left, right = _prepared(store)
    comparisons = _comparisons()
    sampler = DuckDBRandomPairSampler(store).sample(
        left=left,
        right=right,
        maximum_pairs=16,
        random_seed=20260817,
    )
    builder = DuckDBComparisonFeatureBuilder(store)
    u_features = builder.build(
        candidates=sampler.table,
        left=left,
        right=right,
        comparisons=comparisons,
    )
    em_features = builder.build(
        candidates=_candidate_table(store),
        left=left,
        right=right,
        comparisons=comparisons,
    )
    matcher = DuckDBFellegiSunterMatcher(store)
    artifact = matcher.fit(
        u_features=u_features,
        em_features=em_features,
        comparisons=comparisons,
        model=_model_config(),
        random_seed=20260817,
    )
    return matcher, artifact, em_features, comparisons


def test_random_pair_sample_is_bounded_and_seed_deterministic() -> None:
    with DuckDBStore() as store:
        left, right = _prepared(store)
        sampler = DuckDBRandomPairSampler(store)
        first = sampler.sample(left=left, right=right, maximum_pairs=5, random_seed=73)
        first_pairs = store._connection.execute(
            f'SELECT left_record_key, right_record_key FROM "{first.table.table_name}" '
            "ORDER BY left_record_key, right_record_key"
        ).fetchall()
        second = sampler.sample(left=left, right=right, maximum_pairs=5, random_seed=73)
        second_pairs = store._connection.execute(
            f'SELECT left_record_key, right_record_key FROM "{second.table.table_name}" '
            "ORDER BY left_record_key, right_record_key"
        ).fetchall()

    assert first.sampled_pair_count == 5
    assert first.possible_pair_count == 16
    assert first_pairs == second_pairs
    assert len(first_pairs) == 5


def test_reference_model_is_deterministic_normalised_and_evidence_only() -> None:
    with DuckDBStore() as store:
        matcher, artifact, _, comparisons = _fit_fixture(store)
        second_matcher, second_artifact, _, _ = _fit_fixture(store)

    assert isinstance(matcher, DuckDBFellegiSunterMatcher)
    assert isinstance(second_matcher, DuckDBFellegiSunterMatcher)
    assert artifact.model_version == "m2d-reference-v2"
    assert artifact.parameter_digest == second_artifact.parameter_digest
    assert artifact.probability_status == "model_posterior_uncalibrated"
    assert artifact.decision_authority == "evidence_only"
    assert 1 <= artifact.em_iterations <= 50
    for comparison in comparisons:
        parameters = artifact.comparisons[comparison.id]
        assert sum(level.m_probability for level in parameters.levels) == pytest.approx(1.0)
        assert sum(level.u_probability for level in parameters.levels) == pytest.approx(1.0)
        assert parameters.levels[0].log2_bayes_factor == 0.0
        assert parameters.levels[1].log2_bayes_factor > parameters.levels[-1].log2_bayes_factor


def test_reference_scoring_preserves_pairs_and_orders_stronger_evidence_higher() -> None:
    with DuckDBStore() as store:
        matcher, artifact, features, _ = _fit_fixture(store)
        result = matcher.score(features=features, model=artifact)
        exact_probability = store._connection.execute(
            f'SELECT __ml_fs_model_probability FROM "{result.table.table_name}" '
            "WHERE left_record_key = 'left-1' AND right_record_key = 'right-1'"
        ).fetchone()[0]
        mismatch_probability = store._connection.execute(
            f'SELECT __ml_fs_model_probability FROM "{result.table.table_name}" '
            "WHERE left_record_key = 'left-1' AND right_record_key = 'right-3'"
        ).fetchone()[0]
        status_rows = store._connection.execute(
            f"SELECT DISTINCT __ml_fs_probability_status, __ml_fs_decision_authority "
            f'FROM "{result.table.table_name}"'
        ).fetchall()

    assert result.pair_count == features.candidate_pair_count
    assert 0.0 < mismatch_probability < exact_probability < 1.0
    assert status_rows == [("model_posterior_uncalibrated", "evidence_only")]


def test_reference_scoring_is_stable_for_extreme_finite_evidence_weights() -> None:
    with DuckDBStore() as store:
        matcher, artifact, features, _ = _fit_fixture(store)
        comparisons = {}
        for comparison_id, parameters in artifact.comparisons.items():
            levels = tuple(
                replace(
                    level,
                    log2_bayes_factor=(
                        0.0
                        if level.level == parameters.missing_level
                        else 2048.0
                        if level.level == 1
                        else -2048.0
                    ),
                )
                for level in parameters.levels
            )
            comparisons[comparison_id] = replace(parameters, levels=levels)
        extreme = replace(
            artifact,
            prior_probability=0.5,
            parameter_digest="f" * 64,
            comparisons=comparisons,
        )

        result = matcher.score(features=features, model=extreme)
        evidence = store._connection.execute(
            f"SELECT __ml_fs_match_weight, __ml_fs_model_probability "
            f'FROM "{result.table.table_name}"'
        ).fetchall()

    assert len(evidence) == features.candidate_pair_count
    assert all(math.isfinite(float(weight)) for weight, _ in evidence)
    assert all(math.isfinite(float(probability)) for _, probability in evidence)
    assert all(0.0 <= float(probability) <= 1.0 for _, probability in evidence)
    assert any(float(probability) == 0.0 for _, probability in evidence)
    assert any(float(probability) == 1.0 for _, probability in evidence)


def test_fit_rejects_u_sample_beyond_model_budget() -> None:
    with DuckDBStore() as store:
        left, right = _prepared(store)
        comparisons = _comparisons()
        builder = DuckDBComparisonFeatureBuilder(store)
        sample = DuckDBRandomPairSampler(store).sample(
            left=left,
            right=right,
            maximum_pairs=8,
            random_seed=1,
        )
        features = builder.build(
            candidates=sample.table,
            left=left,
            right=right,
            comparisons=comparisons,
        )
        matcher = DuckDBFellegiSunterMatcher(store)
        with pytest.raises(FellegiSunterBudgetExceeded) as captured:
            matcher.fit(
                u_features=features,
                em_features=features,
                comparisons=comparisons,
                model=_model_config(u_max_pairs=4),
                random_seed=1,
            )

    assert str(captured.value) == (
        "ML-FS-006: The u-estimation feature sample exceeds its configured pair budget."
    )


def test_score_rejects_incompatible_feature_contract_without_values() -> None:
    sentinel = "SYNTHETIC-SENSITIVE-COMPARISON"
    with DuckDBStore() as store:
        matcher, artifact, features, _ = _fit_fixture(store)
        incompatible = type(features)(
            table=features.table,
            candidate_pair_count=features.candidate_pair_count,
            configured_comparison_count=1,
            columns={sentinel: next(iter(features.columns.values()))},
        )
        with pytest.raises(FellegiSunterError) as captured:
            matcher.score(features=incompatible, model=artifact)

    assert sentinel not in str(captured.value)
