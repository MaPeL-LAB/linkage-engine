from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mapel_linkage.domain.errors import EnsembleError
from mapel_linkage.governance.labels import assert_disjoint_label_partitions
from mapel_linkage.governance.paths import PathPolicy
from mapel_linkage.io.duckdb_store import DuckDBStore
from mapel_linkage.models.ensembles import (
    StackingPairClassifier,
    read_stacking_artifact,
    write_stacking_artifact,
)
from tests.models.boosted.helpers import label, label_batch


def _sample_data() -> tuple[dict[str, np.ndarray], np.ndarray]:
    # 8 pairs, first 4 matches, next 4 non-matches
    labels = np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8)
    fs_scores = np.asarray([0.95, 0.90, 0.85, 0.80, 0.20, 0.15, 0.10, 0.05], dtype=np.float64)
    xgb_scores = np.asarray([0.98, 0.92, 0.88, 0.75, 0.30, 0.25, 0.12, 0.08], dtype=np.float64)
    lgb_scores = np.asarray([0.96, 0.94, 0.82, 0.78, 0.28, 0.18, 0.15, 0.04], dtype=np.float64)
    base_scores = {
        "fs_baseline": fs_scores,
        "xgb_pair_classifier": xgb_scores,
        "lgb_pair_classifier": lgb_scores,
    }
    return base_scores, labels


def test_stacking_classifier_fit_and_predict() -> None:
    base_scores, labels = _sample_data()
    classifier = StackingPairClassifier()
    artifact = classifier.fit(
        base_scores=base_scores,
        labels=labels,
        base_model_ids=("fs_baseline", "xgb_pair_classifier", "lgb_pair_classifier"),
        random_seed=42,
    )
    assert artifact.model_id == "stacking_ensemble"
    assert artifact.decision_authority == "evidence_only"
    assert len(artifact.base_model_weights) == 3
    assert all(w >= 0.0 for w in artifact.base_model_weights)

    predictions = classifier.predict(base_scores=base_scores, model=artifact)
    assert len(predictions) == len(labels)
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)
    # Matches should score higher than non-matches
    assert np.mean(predictions[:4]) > np.mean(predictions[4:])


def test_stacking_score_materialization() -> None:
    base_scores, labels = _sample_data()
    pair_refs = tuple((f"left_{i}", f"right_{i}") for i in range(len(labels)))
    with DuckDBStore() as store:
        classifier = StackingPairClassifier(store)
        artifact = classifier.fit(
            base_scores=base_scores,
            labels=labels,
            base_model_ids=("fs_baseline", "xgb_pair_classifier", "lgb_pair_classifier"),
        )
        result = classifier.score(
            base_scores=base_scores,
            pair_references=pair_refs,
            model=artifact,
        )
        assert result.pair_count == len(labels)
        scores = store._connection.execute(
            f'SELECT __ml_bt_model_score FROM "{result.table.table_name}"'
        ).fetchall()
        assert len(scores) == len(labels)


def test_stacking_evaluate() -> None:
    base_scores, labels = _sample_data()
    classifier = StackingPairClassifier()
    artifact = classifier.fit(
        base_scores=base_scores,
        labels=labels,
        base_model_ids=("fs_baseline", "xgb_pair_classifier", "lgb_pair_classifier"),
    )
    lbls = tuple(
        label(f"left_{i}", f"right_{i}", int(labels[i]), f"ent_{i}") for i in range(len(labels))
    )
    train_batch = label_batch("training", lbls[:4])
    valid_batch = label_batch("validation", lbls[4:])
    disjointness = assert_disjoint_label_partitions((train_batch, valid_batch))

    report = classifier.evaluate(
        base_scores=base_scores,
        labels=labels,
        model=artifact,
        disjointness=disjointness,
        partition="validation",
    )
    assert report.pair_count == len(labels)
    assert report.threshold_authority == "diagnostic_only"


def test_stacking_artifact_round_trip(tmp_path: Path) -> None:
    base_scores, labels = _sample_data()
    classifier = StackingPairClassifier()
    artifact = classifier.fit(
        base_scores=base_scores,
        labels=labels,
        base_model_ids=("fs_baseline", "xgb_pair_classifier", "lgb_pair_classifier"),
    )
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    written = write_stacking_artifact(
        artifact=artifact,
        model_path="artifacts/models/stacking.json",
        manifest_path="artifacts/models/stacking.manifest.json",
        policy=policy,
    )
    restored = read_stacking_artifact(
        model_path="artifacts/models/stacking.json",
        manifest_path="artifacts/models/stacking.manifest.json",
        policy=policy,
    )
    assert restored.model_digest == artifact.model_digest
    assert restored.base_model_weights == artifact.base_model_weights

    # Tamper payload
    written.model_path.write_text(
        '{"base_model_ids":["a","b"],"base_model_weights":[1.0,2.0],"intercept":0.0}',
        encoding="utf-8",
    )
    with pytest.raises(EnsembleError, match="ML-ENS-018"):
        read_stacking_artifact(
            model_path="artifacts/models/stacking.json",
            manifest_path="artifacts/models/stacking.manifest.json",
            policy=policy,
        )


def test_stacking_rejects_invalid_contracts() -> None:
    base_scores, labels = _sample_data()
    classifier = StackingPairClassifier()
    with pytest.raises(EnsembleError, match="ML-ENS-001"):
        classifier.fit(
            base_scores=base_scores,
            labels=labels,
            base_model_ids=("fs_baseline", "xgb_pair_classifier"),
            partition="validation",  # type: ignore[arg-type]
        )

    with pytest.raises(EnsembleError, match="ML-ENS-002"):
        classifier.fit(
            base_scores=base_scores,
            labels=labels,
            base_model_ids=("fs_baseline",),
        )

    with pytest.raises(EnsembleError, match="ML-ENS-010"):
        classifier.predict(
            base_scores={"only_one": np.array([0.5, 0.5])},
            model=classifier.fit(
                base_scores=base_scores,
                labels=labels,
                base_model_ids=("fs_baseline", "xgb_pair_classifier"),
            ),
        )
