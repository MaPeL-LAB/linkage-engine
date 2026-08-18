from __future__ import annotations

import importlib
import json
import types
from pathlib import Path
from unittest import mock

import pytest

from mapel_linkage.domain.errors import BoostedTreeBudgetExceeded, BoostedTreeError
from mapel_linkage.governance.labels import assert_disjoint_label_partitions
from mapel_linkage.governance.paths import PathPolicy
from mapel_linkage.io.duckdb_store import DuckDBStore
from mapel_linkage.models.boosted import (
    BoostedFeatureMatrix,
    DuckDBVerifiedMatrixBuilder,
    LightGBMModelArtifact,
    LightGBMPairClassifier,
    read_lightgbm_artifact,
    write_lightgbm_artifact,
)
from tests.models.boosted.helpers import (
    feature_result,
    model_config,
    training_labels,
    training_rows,
    validation_labels,
    validation_rows,
)

_lgb_installed: types.ModuleType | None
try:
    _lgb_installed = importlib.import_module("lightgbm")
except ModuleNotFoundError:
    _lgb_installed = None

_requires_lightgbm = pytest.mark.skipif(
    _lgb_installed is None, reason="LightGBM is not installed in the current environment"
)


def _lgb_config(**overrides: object) -> object:
    return model_config(
        implementation="lightgbm_classifier", model_id="lgb_pair_classifier", **overrides
    )


def _fit(
    store: DuckDBStore,
) -> tuple[LightGBMPairClassifier, LightGBMModelArtifact, DuckDBVerifiedMatrixBuilder]:
    builder = DuckDBVerifiedMatrixBuilder(store)
    features = feature_result(store, "lgb_training_features", training_rows())
    cfg = _lgb_config()
    matrix = builder.build_labelled(
        features=features,
        labels=training_labels(),
        model=cfg,  # type: ignore[arg-type]
        random_seed=20260817,
        apply_training_selection=True,
    )
    classifier = LightGBMPairClassifier(store)
    artifact = classifier.fit(
        matrix=matrix,
        model=cfg,  # type: ignore[arg-type]
        random_seed=20260817,
        configuration_digest="d" * 64,
    )
    return classifier, artifact, builder


@_requires_lightgbm
def test_lightgbm_artifact_is_deterministic_uncalibrated_and_evidence_only() -> None:
    with DuckDBStore() as store:
        classifier, first, _ = _fit(store)
        _, second, _ = _fit(store)

    assert isinstance(classifier, LightGBMPairClassifier)
    assert first.model_digest == second.model_digest
    assert first.parameter_digest == second.parameter_digest
    assert first.probability_status == "model_score_uncalibrated"
    assert first.calibration_status == "not_calibrated"
    assert first.decision_authority == "evidence_only"
    assert first.real_data_validation_status == "not_established"
    assert "train-l1" not in repr(first)


@_requires_lightgbm
def test_lightgbm_scoring_preserves_pairs_and_stronger_evidence_scores_higher() -> None:
    with DuckDBStore() as store:
        classifier, artifact, builder = _fit(store)
        features = feature_result(store, "lgb_score_features", training_rows())
        matrix = builder.build_scoring(features=features)
        result = classifier.score(matrix=matrix, model=artifact)
        exact = store._connection.execute(
            f'SELECT __ml_bt_model_score FROM "{result.table.table_name}" '
            "WHERE left_record_key = 'train-l1' AND right_record_key = 'train-r1'"
        ).fetchone()[0]
        mismatch = store._connection.execute(
            f'SELECT __ml_bt_model_score FROM "{result.table.table_name}" '
            "WHERE left_record_key = 'train-l1' AND right_record_key = 'train-r4'"
        ).fetchone()[0]
        statuses = store._connection.execute(
            f"SELECT DISTINCT __ml_bt_probability_status, __ml_bt_calibration_status, "
            f'__ml_bt_decision_authority FROM "{result.table.table_name}"'
        ).fetchall()

    assert result.pair_count == len(training_rows())
    assert 0.0 <= mismatch <= exact <= 1.0
    assert statuses == [("model_score_uncalibrated", "not_calibrated", "evidence_only")]


@_requires_lightgbm
def test_lightgbm_validation_uses_nontraining_partition() -> None:
    with DuckDBStore() as store:
        classifier, artifact, builder = _fit(store)
        validation_features = feature_result(store, "lgb_validation_features", validation_rows())
        validation_matrix = builder.build_labelled(
            features=validation_features,
            labels=validation_labels(),
        )
        disjointness = assert_disjoint_label_partitions((training_labels(), validation_labels()))
        report = classifier.evaluate(
            matrix=validation_matrix,
            model=artifact,
            disjointness=disjointness,
        )

    assert report.pair_count == len(validation_rows())
    assert report.threshold_authority == "diagnostic_only"
    assert report.calibration_status == "not_calibrated"
    assert report.evaluation_scope == "synthetic_mechanical_evaluation"
    assert report.real_data_validation_status == "not_established"


@_requires_lightgbm
def test_lightgbm_native_model_write_and_read(tmp_path: Path) -> None:
    with DuckDBStore() as store:
        _, artifact, _ = _fit(store)
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    written = write_lightgbm_artifact(
        artifact=artifact,
        model_path="artifacts/models/lgb.txt",
        manifest_path="artifacts/models/lgb.manifest.json",
        policy=policy,
    )

    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    assert written.model_path.read_text(encoding="utf-8") == artifact.model_str
    assert manifest["model_digest"] == artifact.model_digest
    assert "train-l1" not in written.manifest_path.read_text(encoding="utf-8")

    reloaded = read_lightgbm_artifact(
        model_path="artifacts/models/lgb.txt",
        manifest_path="artifacts/models/lgb.manifest.json",
        policy=policy,
    )
    assert reloaded.model_digest == artifact.model_digest
    assert reloaded.feature_names == artifact.feature_names
    assert reloaded.random_seed == artifact.random_seed


@_requires_lightgbm
def test_lightgbm_scoring_rejects_feature_schema_mismatch() -> None:
    with DuckDBStore() as store:
        classifier, artifact, builder = _fit(store)
        features = feature_result(store, "lgb_schema_features", training_rows())
        matrix = builder.build_scoring(features=features)
        incompatible = BoostedFeatureMatrix(
            features=matrix.features,
            pair_references=matrix.pair_references,
            pair_digests=matrix.pair_digests,
            feature_names=matrix.feature_names,
            feature_schema_digest="f" * 64,
        )
        with pytest.raises(BoostedTreeError, match="ML-BOOST-028"):
            classifier.score(matrix=incompatible, model=artifact)


def test_lightgbm_fit_rejects_nontraining_partition() -> None:
    with DuckDBStore() as store:
        features = feature_result(store, "lgb_nontraining_features", validation_rows())
        matrix = DuckDBVerifiedMatrixBuilder(store).build_labelled(
            features=features,
            labels=validation_labels(),
        )
        with pytest.raises(BoostedTreeError, match="ML-BOOST-022"):
            LightGBMPairClassifier(store).fit(
                matrix=matrix,
                model=_lgb_config(),  # type: ignore[arg-type]
                random_seed=20260817,
                configuration_digest="d" * 64,
            )


def test_lightgbm_fit_rejects_matrix_over_budget() -> None:
    with DuckDBStore() as store:
        builder = DuckDBVerifiedMatrixBuilder(store)
        features = feature_result(store, "lgb_budget_features", training_rows())
        matrix = builder.build_labelled(
            features=features,
            labels=training_labels(),
        )
        with pytest.raises(BoostedTreeBudgetExceeded, match="ML-BOOST-023"):
            LightGBMPairClassifier(store).fit(
                matrix=matrix,
                model=_lgb_config(maximum_training_pairs=6),  # type: ignore[arg-type]
                random_seed=20260817,
                configuration_digest="d" * 64,
            )


@_requires_lightgbm
def test_lightgbm_reader_rejects_tampered_model(tmp_path: Path) -> None:
    with DuckDBStore() as store:
        _, artifact, _ = _fit(store)
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    written = write_lightgbm_artifact(
        artifact=artifact,
        model_path="artifacts/models/lgb.txt",
        manifest_path="artifacts/models/lgb.manifest.json",
        policy=policy,
    )
    written.model_path.write_text(artifact.model_str + " \n", encoding="utf-8")

    with pytest.raises(BoostedTreeError, match="ML-BOOST-048"):
        read_lightgbm_artifact(
            model_path="artifacts/models/lgb.txt",
            manifest_path="artifacts/models/lgb.manifest.json",
            policy=policy,
        )


def test_lightgbm_missing_dependency() -> None:
    with (
        mock.patch("mapel_linkage.models.boosted.lightgbm_classifier._lightgbm", None),
        DuckDBStore() as store,
    ):
        features = feature_result(store, "lgb_mock_features", training_rows())
        matrix = DuckDBVerifiedMatrixBuilder(store).build_labelled(
            features=features,
            labels=training_labels(),
        )
        with pytest.raises(BoostedTreeError, match="ML-BOOST-020"):
            LightGBMPairClassifier(store).fit(
                matrix=matrix,
                model=_lgb_config(),  # type: ignore[arg-type]
                random_seed=1,
                configuration_digest="d" * 64,
            )
