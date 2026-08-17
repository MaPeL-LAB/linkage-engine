"""Verified-label boosted pair-classifier contracts."""

from mapel_linkage.models.boosted.training import (
    BoostedFeatureMatrix,
    BoostedLabelledMatrix,
    DuckDBVerifiedMatrixBuilder,
)
from mapel_linkage.models.boosted.xgboost_classifier import (
    BoostedTreeScoreResult,
    WrittenXGBoostArtifact,
    XGBoostModelArtifact,
    XGBoostPairClassifier,
    read_xgboost_artifact,
    write_xgboost_artifact,
)

__all__ = [
    "BoostedFeatureMatrix",
    "BoostedLabelledMatrix",
    "BoostedTreeScoreResult",
    "DuckDBVerifiedMatrixBuilder",
    "WrittenXGBoostArtifact",
    "XGBoostModelArtifact",
    "XGBoostPairClassifier",
    "read_xgboost_artifact",
    "write_xgboost_artifact",
]
