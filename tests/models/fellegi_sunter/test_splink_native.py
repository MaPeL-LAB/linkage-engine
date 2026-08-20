from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mapel_linkage.candidate_generation import (
    BlockingRule,
    DuckDBCandidateGenerator,
    Exact,
    PrefixEqual,
)
from mapel_linkage.configuration.models import (
    ComparisonConfig,
    ExactPredicate,
    FellegiSunterModelConfig,
    PrefixEqualPredicate,
)
from mapel_linkage.domain.errors import FellegiSunterBudgetExceeded, FellegiSunterError
from mapel_linkage.io import ColumnSpec, DuckDBStore
from mapel_linkage.models.fellegi_sunter.splink_adapter import (
    SplinkSettingsPlan,
    SplinkSettingsPlanCompiler,
)
from mapel_linkage.models.fellegi_sunter.splink_native import (
    SUPPORTED_SPLINK_VERSION,
    SplinkNativeDuckDBMatcher,
    SplinkNativeModelArtifact,
    _regularise_learned_parameters,
    assert_splink_native_recipe_binding,
    deserialize_splink_native_model,
    serialize_splink_native_model,
    splink_native_feature_schema_digest,
)
from mapel_linkage.preprocessing import PreparedDataset

_SEED = 20260816
_CONFIGURATION_DIGEST = "a" * 64
_COLUMNS = (
    ColumnSpec("__ml_record_key", "VARCHAR"),
    ColumnSpec("__ml_dataset_id", "VARCHAR"),
    ColumnSpec("canonical_text", "VARCHAR"),
    ColumnSpec("canonical_group", "VARCHAR"),
)


def _comparisons() -> tuple[ComparisonConfig, ...]:
    return tuple(
        ComparisonConfig.model_validate(payload)
        for payload in (
            {
                "id": "text_similarity",
                "variable": "text",
                "function": {"kind": "jaro_winkler"},
                "levels": [
                    {"kind": "missing"},
                    {"kind": "exact"},
                    {"kind": "threshold", "minimum": 0.85},
                    {"kind": "else"},
                ],
            },
            {
                "id": "group_agreement",
                "variable": "group",
                "function": {"kind": "categorical"},
                "levels": [
                    {"kind": "missing"},
                    {"kind": "exact"},
                    {"kind": "else"},
                ],
            },
        )
    )


def _model() -> FellegiSunterModelConfig:
    return FellegiSunterModelConfig.model_validate(
        {
            "implementation": "splink_duckdb",
            "model_id": "fs_native_baseline",
            "probability_two_random_records_match": 0.05,
            "u_max_pairs": 256,
            "em_max_iterations": 25,
            "em_convergence": 0.0001,
        }
    )


def test_additive_smoothing_regularises_boundaries_and_nonboundary_distributions() -> None:
    raw: dict[str, Any] = {
        "comparisons": [
            {
                "comparison_levels": [
                    {"is_null_level": True},
                    {"m_probability": 1.0, "u_probability": 0.0},
                    {"m_probability": 0.0, "u_probability": 0.25},
                    {"u_probability": 0.75},
                ]
            },
            {
                "comparison_levels": [
                    {"m_probability": 0.6, "u_probability": 0.3},
                    {"m_probability": 0.4, "u_probability": 0.7},
                ]
            },
        ]
    }

    regularised = _regularise_learned_parameters(
        raw,
        probability_smoothing=0.5,
        m_effective_mass=10,
        u_effective_mass=20,
    )
    comparisons = cast(list[dict[str, Any]], regularised["comparisons"])
    first_levels = cast(list[dict[str, float]], comparisons[0]["comparison_levels"])[1:]
    second_levels = cast(list[dict[str, float]], comparisons[1]["comparison_levels"])

    assert "m_probability" not in raw["comparisons"][0]["comparison_levels"][3]
    assert [level["m_probability"] for level in first_levels] == pytest.approx(
        [(1.0 * 10 + 0.5) / 11.5, 0.5 / 11.5, 0.5 / 11.5]
    )
    assert [level["u_probability"] for level in first_levels] == pytest.approx(
        [0.5 / 21.5, (0.25 * 20 + 0.5) / 21.5, (0.75 * 20 + 0.5) / 21.5]
    )
    assert [level["m_probability"] for level in second_levels] == pytest.approx(
        [(0.6 * 10 + 0.5) / 11.0, (0.4 * 10 + 0.5) / 11.0]
    )
    assert [level["u_probability"] for level in second_levels] == pytest.approx(
        [(0.3 * 20 + 0.5) / 21.0, (0.7 * 20 + 0.5) / 21.0]
    )
    for comparison in comparisons:
        non_null = [
            level
            for level in comparison["comparison_levels"]
            if level.get("is_null_level") is not True
        ]
        for probability_key in ("m_probability", "u_probability"):
            probabilities = [level[probability_key] for level in non_null]
            assert sum(probabilities) == pytest.approx(1.0)
            assert all(0.0 < value < 1.0 for value in probabilities)


def _prepared(store: DuckDBStore) -> tuple[PreparedDataset, PreparedDataset]:
    left_rows = (
        ("left-01", "source_a", "alpha", "group_1"),
        ("left-02", "source_a", "alpho", "group_2"),
        ("left-03", "source_a", "bravo", "group_1"),
        ("left-04", "source_a", "brava", "group_2"),
        ("left-05", "source_a", "charlie", "group_3"),
        ("left-06", "source_a", "charly", "group_4"),
        ("left-07", "source_a", "delta", "group_3"),
        ("left-08", "source_a", "echo", "group_4"),
    )
    right_rows = (
        ("right-01", "source_b", "alpha", "group_1"),
        ("right-02", "source_b", "alphi", "group_2"),
        ("right-03", "source_b", "bravo", "group_1"),
        ("right-04", "source_b", "breva", "group_2"),
        ("right-05", "source_b", "charlie", "group_3"),
        ("right-06", "source_b", "charli", "group_4"),
        ("right-07", "source_b", "delta", "group_3"),
        ("right-08", "source_b", "foxtrot", "group_4"),
    )
    columns = {"text": "canonical_text", "group": "canonical_group"}
    missing = {"text": "missing_text", "group": "missing_group"}
    return (
        PreparedDataset(
            "source_a",
            store.create_table_from_rows("native_source_a", _COLUMNS, left_rows),
            columns,
            missing,
        ),
        PreparedDataset(
            "source_b",
            store.create_table_from_rows("native_source_b", _COLUMNS, right_rows),
            columns,
            missing,
        ),
    )


def _held_prepared(store: DuckDBStore) -> tuple[PreparedDataset, PreparedDataset]:
    left_rows = (
        ("held-left-01", "source_a", "alpine", "group_1"),
        ("held-left-02", "source_a", "amber", "group_2"),
        ("held-left-03", "source_a", "birch", "group_1"),
        ("held-left-04", "source_a", "cedar", "group_3"),
    )
    right_rows = (
        ("held-right-01", "source_b", "alpine", "group_1"),
        ("held-right-02", "source_b", "amper", "group_2"),
        ("held-right-03", "source_b", "beech", "group_1"),
        ("held-right-04", "source_b", "cedar", "group_4"),
    )
    columns = {"text": "canonical_text", "group": "canonical_group"}
    missing = {"text": "missing_text", "group": "missing_group"}
    return (
        PreparedDataset(
            "source_a",
            store.create_table_from_rows("held_source_a", _COLUMNS, left_rows),
            columns,
            missing,
        ),
        PreparedDataset(
            "source_b",
            store.create_table_from_rows("held_source_b", _COLUMNS, right_rows),
            columns,
            missing,
        ),
    )


def _candidate_pairs(
    store: DuckDBStore,
    left: PreparedDataset,
    right: PreparedDataset,
) -> tuple[tuple[str, str], ...]:
    candidates = DuckDBCandidateGenerator(store).generate(
        left=left.table,
        right=right.table,
        variable_columns=left.variable_columns,
        rules=(
            BlockingRule("group_exact", Exact("group")),
            BlockingRule("text_prefix", PrefixEqual("text", 1)),
        ),
        maximum_candidate_pairs=256,
    )
    raw_pairs = store._fetch_model_rows(
        f'SELECT left_record_key, right_record_key FROM "{candidates.table.table_name}" '
        "ORDER BY left_record_key, right_record_key"
    )
    return tuple((str(left_key), str(right_key)) for left_key, right_key in raw_pairs)


def _fit_once(
    store: DuckDBStore,
) -> tuple[SplinkNativeModelArtifact, SplinkSettingsPlan, tuple[tuple[str, str], ...]]:
    left, right = _prepared(store)
    comparisons = _comparisons()
    predicates = (
        ExactPredicate.model_validate({"kind": "exact", "variable": "group"}),
        PrefixEqualPredicate.model_validate(
            {"kind": "prefix_equal", "variable": "text", "length": 1}
        ),
    )
    settings_plan = SplinkSettingsPlanCompiler().compile(
        left=left,
        right=right,
        comparisons=comparisons,
        blocking_rules=predicates,
        model=_model(),
    )
    pairs = _candidate_pairs(store, left, right)
    artifact = SplinkNativeDuckDBMatcher(store).fit(
        left=left,
        right=right,
        settings_plan=settings_plan,
        model=_model(),
        configuration_digest=_CONFIGURATION_DIGEST,
        expected_pairs=pairs,
        maximum_candidate_pairs=256,
        random_seed=_SEED,
    )
    return (artifact, settings_plan, pairs)


def test_real_splink_fit_canonical_reload_and_score_are_deterministic() -> None:
    with DuckDBStore() as first_store:
        first, first_plan, first_pairs = _fit_once(first_store)
        first_json = serialize_splink_native_model(first)
        reloaded = deserialize_splink_native_model(
            first_json,
            settings_plan=first_plan,
            model=_model(),
            configuration_digest=_CONFIGURATION_DIGEST,
            feature_schema_digest=first.feature_schema_digest,
            random_seed=_SEED,
        )
        first_left, first_right = _prepared(first_store)
        first_scores = SplinkNativeDuckDBMatcher(first_store).score(
            left=first_left,
            right=first_right,
            settings_plan=first_plan,
            artifact=reloaded,
            expected_pairs=first_pairs,
            maximum_candidate_pairs=256,
        )

    with DuckDBStore() as second_store:
        second, _, second_pairs = _fit_once(second_store)
        second_json = serialize_splink_native_model(second)

    assert first.splink_version == SUPPORTED_SPLINK_VERSION == "4.0.16"
    assert first.probability_smoothing == _model().probability_smoothing == 0.5
    assert first.smoothing_method == "additive_pseudocount_v1"
    assert first_json == second_json
    assert first == reloaded
    assert first_pairs == second_pairs
    assert first_scores.pair_count == len(first_pairs)
    assert (
        first_scores.scoring_candidate_pair_set_digest == first.training_candidate_pair_set_digest
    )
    assert first.decision_authority == "evidence_only"
    assert first.relationship_authority == "none"
    assert first.assignment_authority == "none"
    assert first.merge_authority == "none"
    assert first.operational_validation == "not_established"

    protected_fragments = (
        "left-01",
        "right-01",
        "alpha",
        "group_1",
    )
    rendered = repr(first) + repr(first_scores)
    for fragment in protected_fragments:
        assert fragment not in first_json
        assert fragment not in rendered


def test_serialization_and_recipe_binding_reject_forged_self_integrity() -> None:
    with DuckDBStore() as store:
        artifact, _, _ = _fit_once(store)
    recipe_binding = SimpleNamespace(
        champion_model_id=artifact.model_id,
        champion_model_version=artifact.model_version,
        champion_artifact_digest=artifact.artifact_digest,
        configuration_digest=artifact.configuration_digest,
        feature_schema_digest=artifact.feature_schema_digest,
    )
    forged_artifacts = (
        replace(artifact, model_digest="7" * 64),
        replace(artifact, model_json=artifact.model_json + " "),
        replace(artifact, artifact_digest="8" * 64),
    )

    for forged in forged_artifacts:
        with pytest.raises(FellegiSunterError, match="ML-FS-067"):
            serialize_splink_native_model(forged)
        with pytest.raises(FellegiSunterError, match="ML-FS-067"):
            assert_splink_native_recipe_binding(
                recipe=recipe_binding,
                artifact=forged,
            )


def test_reloaded_model_scores_distinct_held_data_deterministically_with_exact_parity() -> None:
    with DuckDBStore() as store:
        artifact, settings_plan, training_pairs = _fit_once(store)
        payload = serialize_splink_native_model(artifact)
        training_left, training_right = _prepared(store)
        training_scores = SplinkNativeDuckDBMatcher(store).score(
            left=training_left,
            right=training_right,
            settings_plan=settings_plan,
            artifact=artifact,
            expected_pairs=training_pairs,
            maximum_candidate_pairs=256,
        )
        held_left, held_right = _held_prepared(store)
        held_pairs = _candidate_pairs(store, held_left, held_right)
        held_feature_digest = splink_native_feature_schema_digest(
            left=held_left,
            right=held_right,
            settings_plan=settings_plan,
        )
        reloaded = deserialize_splink_native_model(
            payload,
            settings_plan=settings_plan,
            model=_model(),
            configuration_digest=_CONFIGURATION_DIGEST,
            feature_schema_digest=held_feature_digest,
            random_seed=_SEED,
        )
        first_held_scores = SplinkNativeDuckDBMatcher(store).score(
            left=held_left,
            right=held_right,
            settings_plan=settings_plan,
            artifact=reloaded,
            expected_pairs=held_pairs,
            maximum_candidate_pairs=256,
        )
        observed_pairs = store._fetch_model_rows(
            f"SELECT left_record_key, right_record_key FROM "
            f'"{first_held_scores.table.table_name}" ORDER BY left_record_key, right_record_key'
        )
        second_held_scores = SplinkNativeDuckDBMatcher(store).score(
            left=held_left,
            right=held_right,
            settings_plan=settings_plan,
            artifact=reloaded,
            expected_pairs=held_pairs,
            maximum_candidate_pairs=256,
        )

    assert held_feature_digest == artifact.feature_schema_digest
    assert held_pairs != training_pairs
    assert (
        first_held_scores.scoring_candidate_pair_set_digest
        != artifact.training_candidate_pair_set_digest
    )
    assert first_held_scores.score_digest == second_held_scores.score_digest
    assert first_held_scores.table.table_name == second_held_scores.table.table_name
    assert first_held_scores.table.table_name != training_scores.table.table_name
    assert tuple((str(left), str(right)) for left, right in observed_pairs) == held_pairs


def test_loader_rejects_noncanonical_duplicate_tampered_and_drifted_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with DuckDBStore() as store:
        artifact, settings_plan, _ = _fit_once(store)
    payload = serialize_splink_native_model(artifact)

    with pytest.raises(FellegiSunterError, match="ML-FS-056"):
        deserialize_splink_native_model(
            json.dumps(json.loads(payload), indent=2),
            settings_plan=settings_plan,
            model=_model(),
            configuration_digest=_CONFIGURATION_DIGEST,
            feature_schema_digest=artifact.feature_schema_digest,
            random_seed=_SEED,
        )

    with pytest.raises(FellegiSunterError, match="ML-FS-052"):
        deserialize_splink_native_model(
            payload.replace('"schema_version":"1"', '"schema_version":"1","schema_version":"1"'),
            settings_plan=settings_plan,
            model=_model(),
            configuration_digest=_CONFIGURATION_DIGEST,
            feature_schema_digest=artifact.feature_schema_digest,
            random_seed=_SEED,
        )

    tampered = json.loads(payload)
    tampered["model_digest"] = "9" * 64
    with pytest.raises(FellegiSunterError, match="ML-FS-067"):
        deserialize_splink_native_model(
            json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n",
            settings_plan=settings_plan,
            model=_model(),
            configuration_digest=_CONFIGURATION_DIGEST,
            feature_schema_digest=artifact.feature_schema_digest,
            random_seed=_SEED,
        )

    authority_drift = json.loads(payload)
    authority_drift["decision_authority"] = "relationship_decision"
    with pytest.raises(FellegiSunterError, match="ML-FS-065"):
        deserialize_splink_native_model(
            json.dumps(authority_drift, sort_keys=True, separators=(",", ":")) + "\n",
            settings_plan=settings_plan,
            model=_model(),
            configuration_digest=_CONFIGURATION_DIGEST,
            feature_schema_digest=artifact.feature_schema_digest,
            random_seed=_SEED,
        )

    with pytest.raises(FellegiSunterError, match="ML-FS-066"):
        deserialize_splink_native_model(
            payload,
            settings_plan=settings_plan,
            model=_model(),
            configuration_digest="b" * 64,
            feature_schema_digest=artifact.feature_schema_digest,
            random_seed=_SEED,
        )

    with pytest.raises(FellegiSunterError, match="ML-FS-066"):
        deserialize_splink_native_model(
            payload,
            settings_plan=settings_plan,
            model=_model().model_copy(update={"probability_smoothing": 0.75}),
            configuration_digest=_CONFIGURATION_DIGEST,
            feature_schema_digest=artifact.feature_schema_digest,
            random_seed=_SEED,
        )

    smoothing_method_drift = json.loads(payload)
    smoothing_method_drift["smoothing_method"] = "undeclared_method"
    with pytest.raises(FellegiSunterError, match="ML-FS-065"):
        deserialize_splink_native_model(
            json.dumps(smoothing_method_drift, sort_keys=True, separators=(",", ":")) + "\n",
            settings_plan=settings_plan,
            model=_model(),
            configuration_digest=_CONFIGURATION_DIGEST,
            feature_schema_digest=artifact.feature_schema_digest,
            random_seed=_SEED,
        )

    monkeypatch.setattr(
        "mapel_linkage.models.fellegi_sunter.splink_native.importlib.metadata.version",
        lambda _: "4.0.15",
    )
    with pytest.raises(FellegiSunterError, match="ML-FS-058"):
        deserialize_splink_native_model(
            payload,
            settings_plan=settings_plan,
            model=_model(),
            configuration_digest=_CONFIGURATION_DIGEST,
            feature_schema_digest=artifact.feature_schema_digest,
            random_seed=_SEED,
        )


def test_native_fit_and_score_fail_closed_on_budget_candidate_and_schema_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with DuckDBStore() as store:
        artifact, settings_plan, pairs = _fit_once(store)
        left, right = _prepared(store)
        matcher = SplinkNativeDuckDBMatcher(store)

        with pytest.raises(FellegiSunterBudgetExceeded, match="ML-FS-068"):
            matcher.fit(
                left=left,
                right=right,
                settings_plan=settings_plan,
                model=_model(),
                configuration_digest=_CONFIGURATION_DIGEST,
                expected_pairs=pairs,
                maximum_candidate_pairs=len(pairs) - 1,
                random_seed=_SEED,
            )

        with pytest.raises(FellegiSunterError, match="ML-FS-075"):
            matcher.score(
                left=left,
                right=right,
                settings_plan=settings_plan,
                artifact=artifact,
                expected_pairs=pairs[:-1],
                maximum_candidate_pairs=256,
            )

        with pytest.raises(FellegiSunterError, match="ML-FS-067"):
            matcher.score(
                left=left,
                right=right,
                settings_plan=settings_plan,
                artifact=replace(artifact, model_digest="9" * 64),
                expected_pairs=pairs,
                maximum_candidate_pairs=256,
            )

        drifted_left = replace(
            left,
            table=replace(left.table, schema_digest="f" * 64),
        )
        monkeypatch.setattr(
            "mapel_linkage.models.fellegi_sunter.splink_native._require_supported_runtime",
            lambda: (_ for _ in ()).throw(AssertionError("runtime must not execute")),
        )
        with pytest.raises(FellegiSunterError, match="ML-FS-076"):
            matcher.score(
                left=drifted_left,
                right=right,
                settings_plan=settings_plan,
                artifact=artifact,
                expected_pairs=pairs,
                maximum_candidate_pairs=256,
            )
