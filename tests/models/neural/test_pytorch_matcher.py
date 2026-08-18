from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from mapel_linkage.domain.errors import NeuralModelError
from mapel_linkage.governance.labels import assert_disjoint_label_partitions
from mapel_linkage.governance.paths import PathPolicy
from mapel_linkage.io.duckdb_store import DuckDBStore
from mapel_linkage.models.boosted import BoostedFeatureMatrix, DuckDBVerifiedMatrixBuilder
from mapel_linkage.models.neural import (
    PyTorchModelArtifact,
    PyTorchPairMatcher,
    read_pytorch_artifact,
    write_pytorch_artifact,
)
from tests.models.boosted.helpers import (
    feature_result,
    training_labels,
    training_rows,
    validation_labels,
    validation_rows,
)

try:
    import torch as _torch_installed
except ModuleNotFoundError:
    _torch_installed = None  # type: ignore[assignment]

_requires_torch = pytest.mark.skipif(
    _torch_installed is None, reason="PyTorch is not installed in the current environment"
)


def _fit(
    store: DuckDBStore,
) -> tuple[PyTorchPairMatcher, PyTorchModelArtifact, DuckDBVerifiedMatrixBuilder]:
    builder = DuckDBVerifiedMatrixBuilder(store)
    features = feature_result(store, "pt_training_features", training_rows())
    matrix = builder.build_labelled(
        features=features,
        labels=training_labels(),
    )
    matcher = PyTorchPairMatcher(store)
    artifact = matcher.fit(
        matrix=matrix,
        random_seed=42,
        configuration_digest="d" * 64,
        epochs=30,
    )
    return matcher, artifact, builder


@_requires_torch
def test_pytorch_matcher_is_deterministic_and_evidence_only() -> None:
    with DuckDBStore() as store:
        matcher, first, _ = _fit(store)
        _, second, _ = _fit(store)

    assert isinstance(matcher, PyTorchPairMatcher)
    assert first.model_digest == second.model_digest
    assert first.probability_status == "model_score_uncalibrated"
    assert first.calibration_status == "not_calibrated"
    assert first.decision_authority == "evidence_only"
    assert first.real_data_validation_status == "not_established"
    assert "train-l1" not in repr(first)


@_requires_torch
def test_pytorch_scoring_and_materialization() -> None:
    with DuckDBStore() as store:
        matcher, artifact, builder = _fit(store)
        features = feature_result(store, "pt_score_features", training_rows())
        matrix = builder.build_scoring(features=features)
        result = matcher.score(matrix=matrix, model=artifact)
        exact = store._connection.execute(
            f'SELECT __ml_bt_model_score FROM "{result.table.table_name}" '
            "WHERE left_record_key = 'train-l1' AND right_record_key = 'train-r1'"
        ).fetchone()[0]
        mismatch = store._connection.execute(
            f'SELECT __ml_bt_model_score FROM "{result.table.table_name}" '
            "WHERE left_record_key = 'train-l1' AND right_record_key = 'train-r4'"
        ).fetchone()[0]

    assert result.pair_count == len(training_rows())
    assert 0.0 <= mismatch <= exact <= 1.0


@_requires_torch
def test_pytorch_validation_uses_nontraining_partition() -> None:
    with DuckDBStore() as store:
        matcher, artifact, builder = _fit(store)
        validation_features = feature_result(store, "pt_validation_features", validation_rows())
        validation_matrix = builder.build_labelled(
            features=validation_features,
            labels=validation_labels(),
        )
        disjointness = assert_disjoint_label_partitions((training_labels(), validation_labels()))
        report = matcher.evaluate(
            matrix=validation_matrix,
            model=artifact,
            disjointness=disjointness,
        )

    assert report.pair_count == len(validation_rows())
    assert report.threshold_authority == "diagnostic_only"
    assert report.calibration_status == "not_calibrated"


@_requires_torch
def test_pytorch_artifact_write_and_read(tmp_path: Path) -> None:
    with DuckDBStore() as store:
        _, artifact, _ = _fit(store)
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    written = write_pytorch_artifact(
        artifact=artifact,
        model_path="artifacts/models/pt.json",
        manifest_path="artifacts/models/pt.manifest.json",
        policy=policy,
    )

    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_digest"] == artifact.model_digest

    reloaded = read_pytorch_artifact(
        model_path="artifacts/models/pt.json",
        manifest_path="artifacts/models/pt.manifest.json",
        policy=policy,
    )
    assert reloaded.model_digest == artifact.model_digest

    # Tamper payload
    written.model_path.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(NeuralModelError, match="ML-NEUR-018"):
        read_pytorch_artifact(
            model_path="artifacts/models/pt.json",
            manifest_path="artifacts/models/pt.manifest.json",
            policy=policy,
        )


@_requires_torch
def test_pytorch_scoring_rejects_feature_schema_mismatch() -> None:
    with DuckDBStore() as store:
        matcher, artifact, builder = _fit(store)
        features = feature_result(store, "pt_schema_features", training_rows())
        matrix = builder.build_scoring(features=features)
        incompatible = BoostedFeatureMatrix(
            features=matrix.features,
            pair_references=matrix.pair_references,
            pair_digests=matrix.pair_digests,
            feature_names=matrix.feature_names,
            feature_schema_digest="f" * 64,
        )
        with pytest.raises(NeuralModelError, match="ML-NEUR-009"):
            matcher.score(matrix=incompatible, model=artifact)


def test_pytorch_missing_dependency() -> None:
    with (
        mock.patch("mapel_linkage.models.neural.pytorch_matcher._torch", None),
        DuckDBStore() as store,
    ):
        features = feature_result(store, "pt_mock_features", training_rows())
        matrix = DuckDBVerifiedMatrixBuilder(store).build_labelled(
            features=features,
            labels=training_labels(),
        )
        with pytest.raises(NeuralModelError, match="ML-NEUR-001"):
            PyTorchPairMatcher(store).fit(
                matrix=matrix,
                random_seed=1,
                configuration_digest="d" * 64,
            )
