from __future__ import annotations

import pytest

from mapel_linkage.domain.errors import BoostedTreeError
from mapel_linkage.io.duckdb_store import DuckDBStore
from mapel_linkage.models.boosted import DuckDBVerifiedMatrixBuilder
from tests.models.boosted.helpers import (
    feature_result,
    label,
    label_batch,
    model_config,
    training_labels,
    training_rows,
)


def test_training_matrix_retains_all_positives_and_selects_hard_negatives_deterministically() -> (
    None
):
    with DuckDBStore() as store:
        features = feature_result(store, "boost_train_features", training_rows())
        builder = DuckDBVerifiedMatrixBuilder(store)
        plan = model_config(maximum_training_pairs=6, hard_negative_fraction=1.0)
        first = builder.build_labelled(
            features=features,
            labels=training_labels(),
            model=plan,
            random_seed=20260817,
            apply_training_selection=True,
        )
        second = builder.build_labelled(
            features=features,
            labels=training_labels(),
            model=plan,
            random_seed=20260817,
            apply_training_selection=True,
        )

    assert first.pair_count == 6
    assert first.positive_count == 4
    assert first.negative_count == 2
    assert first.hard_negative_count == 2
    assert first.selection_digest == second.selection_digest
    assert first.features.flags.writeable is False
    assert first.labels.flags.writeable is False


def test_scoring_matrix_exposes_only_aggregate_summary() -> None:
    sentinel = "SYNTHETIC-PRIVATE-PAIR-REFERENCE"
    rows = (training_rows()[0], (sentinel, *training_rows()[1][1:]))
    with DuckDBStore() as store:
        features = feature_result(store, "boost_score_features", rows)
        matrix = DuckDBVerifiedMatrixBuilder(store).build_scoring(features=features)

    assert matrix.pair_count == 2
    assert sentinel not in repr(matrix)
    assert sentinel not in str(matrix.safe_summary())
    assert "pair_references" not in str(matrix.safe_summary())


def test_label_without_candidate_feature_row_is_rejected_without_value() -> None:
    sentinel = "SYNTHETIC-MISSING-CANDIDATE-PAIR"
    labels = training_labels()
    extended = label_batch(
        "training",
        (
            *labels.labels,
            label(sentinel, "missing-right", 0, "missing-entity"),
        ),
    )
    with DuckDBStore() as store:
        features = feature_result(store, "boost_missing_label_features", training_rows())
        builder = DuckDBVerifiedMatrixBuilder(store)
        with pytest.raises(BoostedTreeError) as captured:
            builder.build_labelled(features=features, labels=extended)

    assert captured.value.code == "ML-BOOST-010"
    assert sentinel not in str(captured.value)


def test_training_selection_cannot_use_validation_partition() -> None:
    labels = label_batch(
        "validation",
        (
            label("train-l1", "train-r1", 1, "valid-entity-1"),
            label("train-l1", "train-r2", 0, "valid-entity-2"),
        ),
    )
    with DuckDBStore() as store:
        features = feature_result(store, "boost_partition_features", training_rows())
        builder = DuckDBVerifiedMatrixBuilder(store)
        with pytest.raises(BoostedTreeError) as captured:
            builder.build_labelled(
                features=features,
                labels=labels,
                model=model_config(),
                apply_training_selection=True,
            )

    assert captured.value.code == "ML-BOOST-002"
