from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mapel_linkage.domain.errors import BoostedTreeBudgetExceeded, BoostedTreeError
from mapel_linkage.governance.labels import assert_disjoint_label_partitions
from mapel_linkage.governance.paths import PathPolicy
from mapel_linkage.io.duckdb_store import DuckDBStore
from mapel_linkage.models.boosted import (
    BoostedFeatureMatrix,
    DuckDBVerifiedMatrixBuilder,
    XGBoostModelArtifact,
    XGBoostPairClassifier,
    read_xgboost_artifact,
    write_xgboost_artifact,
)
from tests.models.boosted.helpers import (
    feature_result,
    label,
    label_batch,
    model_config,
    training_labels,
    training_rows,
    validation_labels,
    validation_rows,
)


def _fit(
    store: DuckDBStore,
) -> tuple[XGBoostPairClassifier, XGBoostModelArtifact, DuckDBVerifiedMatrixBuilder]:
    builder = DuckDBVerifiedMatrixBuilder(store)
    features = feature_result(store, "xgb_training_features", training_rows())
    matrix = builder.build_labelled(
        features=features,
        labels=training_labels(),
        model=model_config(),
        random_seed=20260817,
        apply_training_selection=True,
    )
    classifier = XGBoostPairClassifier(store)
    artifact = classifier.fit(
        matrix=matrix,
        model=model_config(),
        random_seed=20260817,
        configuration_digest="d" * 64,
    )
    return classifier, artifact, builder


def test_xgboost_artifact_is_deterministic_uncalibrated_and_evidence_only() -> None:
    with DuckDBStore() as store:
        classifier, first, _ = _fit(store)
        _, second, _ = _fit(store)

    assert isinstance(classifier, XGBoostPairClassifier)
    assert first.model_digest == second.model_digest
    assert first.parameter_digest == second.parameter_digest
    assert first.probability_status == "model_score_uncalibrated"
    assert first.calibration_status == "not_calibrated"
    assert first.decision_authority == "evidence_only"
    assert first.real_data_validation_status == "not_established"
    assert "train-l1" not in repr(first)


def test_xgboost_scoring_preserves_pairs_and_stronger_evidence_scores_higher() -> None:
    with DuckDBStore() as store:
        classifier, artifact, builder = _fit(store)
        features = feature_result(store, "xgb_score_features", training_rows())
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
    assert 0.0 <= mismatch < exact <= 1.0
    assert statuses == [("model_score_uncalibrated", "not_calibrated", "evidence_only")]


def test_validation_uses_nontraining_partition_and_reports_diagnostic_metrics() -> None:
    with DuckDBStore() as store:
        classifier, artifact, builder = _fit(store)
        validation_features = feature_result(store, "xgb_validation_features", validation_rows())
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
    assert report.average_precision >= 0.8
    assert report.threshold_authority == "diagnostic_only"
    assert report.calibration_status == "not_calibrated"
    assert report.evaluation_scope == "synthetic_mechanical_evaluation"
    assert report.real_data_validation_status == "not_established"


def test_native_model_and_safe_manifest_write_only_under_approved_output_root(
    tmp_path: Path,
) -> None:
    with DuckDBStore() as store:
        _, artifact, _ = _fit(store)
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    written = write_xgboost_artifact(
        artifact=artifact,
        model_path="artifacts/models/xgb.json",
        manifest_path="artifacts/models/xgb.manifest.json",
        policy=policy,
    )

    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    assert written.model_path.read_bytes() == artifact.model_json
    assert manifest["model_digest"] == artifact.model_digest
    assert "model_json" not in manifest
    assert "pair_references" not in manifest
    assert "train-l1" not in written.manifest_path.read_text(encoding="utf-8")
    assert "train-l1" not in written.model_path.read_text(encoding="utf-8")

    reloaded = read_xgboost_artifact(
        model_path="artifacts/models/xgb.json",
        manifest_path="artifacts/models/xgb.manifest.json",
        policy=policy,
    )
    assert reloaded.model_digest == artifact.model_digest
    assert reloaded.feature_names == artifact.feature_names
    assert reloaded.random_seed == artifact.random_seed


def test_scoring_rejects_feature_schema_mismatch_without_pair_values() -> None:
    sentinel = "SYNTHETIC-PRIVATE-SCHEMA-PAIR"
    with DuckDBStore() as store:
        classifier, artifact, builder = _fit(store)
        features = feature_result(store, "xgb_schema_features", training_rows())
        matrix = builder.build_scoring(features=features)
        incompatible = BoostedFeatureMatrix(
            features=matrix.features,
            pair_references=((sentinel, "right"), *matrix.pair_references[1:]),
            pair_digests=matrix.pair_digests,
            feature_names=matrix.feature_names,
            feature_schema_digest="f" * 64,
        )
        with pytest.raises(BoostedTreeError) as captured:
            classifier.score(matrix=incompatible, model=artifact)

    assert captured.value.code == "ML-BOOST-028"
    assert sentinel not in str(captured.value)


def test_validation_rejects_partition_proof_for_different_label_snapshot() -> None:
    with DuckDBStore() as store:
        classifier, artifact, builder = _fit(store)
        validation_features = feature_result(store, "xgb_proof_features", validation_rows())
        validation_matrix = builder.build_labelled(
            features=validation_features,
            labels=validation_labels(),
        )
        unrelated = assert_disjoint_label_partitions(
            (
                training_labels(),
                label_batch(
                    "test",
                    (
                        label("other-left", "other-right", 1, "other-entity-1"),
                        label("other-left-2", "other-right-2", 0, "other-entity-2"),
                    ),
                ),
            )
        )
        with pytest.raises(BoostedTreeError) as captured:
            classifier.evaluate(
                matrix=validation_matrix,
                model=artifact,
                disjointness=unrelated,
            )

    assert captured.value.code == "ML-BOOST-041"


def test_xgboost_fit_rejects_nontraining_partition() -> None:
    with DuckDBStore() as store:
        features = feature_result(store, "xgb_nontraining_features", validation_rows())
        matrix = DuckDBVerifiedMatrixBuilder(store).build_labelled(
            features=features,
            labels=validation_labels(),
        )
        with pytest.raises(BoostedTreeError) as captured:
            XGBoostPairClassifier(store).fit(
                matrix=matrix,
                model=model_config(),
                random_seed=20260817,
                configuration_digest="d" * 64,
            )

    assert captured.value.code == "ML-BOOST-022"


def test_xgboost_fit_rejects_matrix_over_model_budget() -> None:
    with DuckDBStore() as store:
        builder = DuckDBVerifiedMatrixBuilder(store)
        features = feature_result(store, "xgb_budget_features", training_rows())
        matrix = builder.build_labelled(
            features=features,
            labels=training_labels(),
        )
        with pytest.raises(BoostedTreeBudgetExceeded) as captured:
            XGBoostPairClassifier(store).fit(
                matrix=matrix,
                model=model_config(maximum_training_pairs=6),
                random_seed=20260817,
                configuration_digest="d" * 64,
            )

    assert captured.value.code == "ML-BOOST-023"


def test_tampered_native_model_payload_is_rejected() -> None:
    with DuckDBStore() as store:
        _, artifact, _ = _fit(store)

    with pytest.raises(ValueError, match="model digest"):
        replace(artifact, model_json=artifact.model_json + b" ")


def test_model_and_manifest_paths_must_be_distinct(tmp_path: Path) -> None:
    with DuckDBStore() as store:
        _, artifact, _ = _fit(store)
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    with pytest.raises(BoostedTreeError) as captured:
        write_xgboost_artifact(
            artifact=artifact,
            model_path="artifacts/models/xgb.json",
            manifest_path="artifacts/models/xgb.json",
            policy=policy,
        )

    assert captured.value.code == "ML-BOOST-040"


def test_reader_rejects_tampered_model_without_exposing_path(tmp_path: Path) -> None:
    with DuckDBStore() as store:
        _, artifact, _ = _fit(store)
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    written = write_xgboost_artifact(
        artifact=artifact,
        model_path="artifacts/models/xgb.json",
        manifest_path="artifacts/models/xgb.manifest.json",
        policy=policy,
    )
    written.model_path.write_bytes(artifact.model_json + b" ")

    with pytest.raises(BoostedTreeError) as captured:
        read_xgboost_artifact(
            model_path="artifacts/models/xgb.json",
            manifest_path="artifacts/models/xgb.manifest.json",
            policy=policy,
        )

    assert captured.value.code == "ML-BOOST-048"
    assert str(tmp_path) not in str(captured.value)
