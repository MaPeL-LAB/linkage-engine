from __future__ import annotations

import hashlib

import numpy as np

from mapel_linkage.governance.labels import VerifiedLabelBatch, VerifiedPairLabel
from mapel_linkage.models.boosted import (
    BoostedFeatureMatrix,
    XGBoostModelArtifact,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_verified_labels_matrices_and_model_artifacts_hide_row_bearing_values() -> None:
    left = "SYNTHETIC-PRIVATE-LEFT-REFERENCE"
    right = "SYNTHETIC-PRIVATE-RIGHT-REFERENCE"
    label = VerifiedPairLabel(
        left_record_key=left,
        right_record_key=right,
        label=1,
        entity_component_digests=(_digest("entity-component"),),
    )
    batch = VerifiedLabelBatch(
        source_kind="synthetic_truth",
        verification_protocol="synthetic_v1",
        source_digest=_digest("source"),
        partition="training",
        labels=(label,),
    )
    matrix = BoostedFeatureMatrix(
        features=np.asarray([[1.0, 0.0]], dtype=np.float64),
        pair_references=((left, right),),
        pair_digests=(label.pair_digest(),),
        feature_names=("private_feature_a", "private_feature_b"),
        feature_schema_digest=_digest("schema"),
    )
    model_json = b"{}"
    artifact = XGBoostModelArtifact(
        model_id="xgb_pair_classifier",
        model_version="m2e-xgboost-v1",
        engine_version="0.1.0.dev6",
        configuration_digest=_digest("configuration"),
        random_seed=7,
        training_pair_count=2,
        positive_count=1,
        negative_count=1,
        hard_negative_count=1,
        feature_schema_digest=_digest("schema"),
        label_authority_digest=batch.label_authority_digest,
        training_selection_digest=_digest("selection"),
        parameter_digest=_digest("parameters"),
        model_digest=hashlib.sha256(model_json).hexdigest(),
        xgboost_version="synthetic-version",
        label_source_kind="synthetic_truth",
        feature_names=("private_feature_a", "private_feature_b"),
        model_json=model_json,
    )

    rendered = "\n".join((repr(label), repr(batch), repr(matrix), repr(artifact)))
    for sentinel in (left, right, "private_feature_a", "private_feature_b"):
        assert sentinel not in rendered
        assert sentinel not in str(artifact.manifest())
    assert "pair_references" not in str(matrix.safe_summary())
    assert "model_json" not in str(artifact.manifest())
