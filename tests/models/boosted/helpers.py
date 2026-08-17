from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import cast

from mapel_linkage.comparisons import ComparisonFeatureColumns, ComparisonFeatureResult
from mapel_linkage.configuration.models import BoostedTreeModelConfig
from mapel_linkage.governance.labels import (
    LabelPartition,
    LabelSourceKind,
    PairLabel,
    VerifiedLabelBatch,
    VerifiedPairLabel,
)
from mapel_linkage.io.duckdb_store import ColumnSpec, DuckDBStore

_COLUMNS = (
    ColumnSpec("left_record_key", "VARCHAR"),
    ColumnSpec("right_record_key", "VARCHAR"),
    ColumnSpec("retrieval_rule_ids", "VARCHAR"),
    ColumnSpec("retrieval_rule_count", "INTEGER"),
    ColumnSpec("cmp_value", "DOUBLE"),
    ColumnSpec("cmp_level", "INTEGER"),
    ColumnSpec("cmp_exact", "BOOLEAN"),
    ColumnSpec("cmp_missing_left", "BOOLEAN"),
    ColumnSpec("cmp_missing_right", "BOOLEAN"),
    ColumnSpec("cmp_missing_both", "BOOLEAN"),
    ColumnSpec("cmp_missing_any", "BOOLEAN"),
)
_FEATURE_COLUMNS = ComparisonFeatureColumns(
    value="cmp_value",
    level="cmp_level",
    exact="cmp_exact",
    missing_left="cmp_missing_left",
    missing_right="cmp_missing_right",
    missing_both="cmp_missing_both",
    missing_any="cmp_missing_any",
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def feature_result(
    store: DuckDBStore,
    table_name: str,
    rows: Sequence[tuple[object, ...]],
) -> ComparisonFeatureResult:
    table = store.create_table_from_rows(table_name, _COLUMNS, rows)
    return ComparisonFeatureResult(
        table=table,
        candidate_pair_count=table.row_count,
        configured_comparison_count=1,
        columns={"synthetic_similarity": _FEATURE_COLUMNS},
    )


def feature_row(
    left: str,
    right: str,
    value: float,
    level: int,
    exact: bool,
) -> tuple[object, ...]:
    return (left, right, "synthetic_rule", 1, value, level, exact, False, False, False, False)


def label(
    left: str,
    right: str,
    target: int,
    entity_tag: str,
    household_tag: str | None = None,
) -> VerifiedPairLabel:
    return VerifiedPairLabel(
        left_record_key=left,
        right_record_key=right,
        label=cast(PairLabel, target),
        entity_component_digests=(digest(entity_tag),),
        household_component_digests=(() if household_tag is None else (digest(household_tag),)),
    )


def label_batch(
    partition: str,
    labels: Sequence[VerifiedPairLabel],
    *,
    source_kind: str = "synthetic_truth",
) -> VerifiedLabelBatch:
    return VerifiedLabelBatch(
        source_kind=cast(LabelSourceKind, source_kind),
        verification_protocol="synthetic_v1",
        source_digest=digest(f"source-{partition}"),
        partition=cast(LabelPartition, partition),
        labels=tuple(labels),
    )


def model_config(**overrides: object) -> BoostedTreeModelConfig:
    payload: dict[str, object] = {
        "enabled": True,
        "implementation": "xgboost_classifier",
        "model_id": "xgb_pair_classifier",
        "require_verified_labels": True,
        "n_estimators": 30,
        "max_depth": 2,
        "learning_rate": 0.2,
        "subsample": 1.0,
        "column_sample": 1.0,
        "maximum_training_pairs": 100,
        "hard_negative_fraction": 0.75,
        "n_jobs": 1,
        "deterministic_mode": True,
    }
    payload.update(overrides)
    return BoostedTreeModelConfig.model_validate(payload)


def training_rows() -> tuple[tuple[object, ...], ...]:
    return (
        feature_row("train-l1", "train-r1", 1.00, 1, True),
        feature_row("train-l2", "train-r2", 0.98, 1, True),
        feature_row("train-l3", "train-r3", 0.95, 1, True),
        feature_row("train-l4", "train-r4", 0.92, 1, True),
        feature_row("train-l1", "train-r2", 0.79, 2, False),
        feature_row("train-l2", "train-r1", 0.77, 2, False),
        feature_row("train-l3", "train-r4", 0.65, 3, False),
        feature_row("train-l4", "train-r3", 0.62, 3, False),
        feature_row("train-l1", "train-r4", 0.25, 4, False),
        feature_row("train-l2", "train-r3", 0.20, 4, False),
        feature_row("train-l3", "train-r1", 0.15, 4, False),
        feature_row("train-l4", "train-r2", 0.10, 4, False),
    )


def training_labels() -> VerifiedLabelBatch:
    rows = training_rows()
    labels: list[VerifiedPairLabel] = []
    for index, row in enumerate(rows):
        left = str(row[0])
        right = str(row[1])
        target = 1 if index < 4 else 0
        labels.append(label(left, right, target, f"train-entity-{index}"))
    return label_batch("training", labels)


def validation_rows() -> tuple[tuple[object, ...], ...]:
    return (
        feature_row("valid-l1", "valid-r1", 0.99, 1, True),
        feature_row("valid-l2", "valid-r2", 0.93, 1, True),
        feature_row("valid-l3", "valid-r3", 0.88, 2, False),
        feature_row("valid-l1", "valid-r2", 0.72, 2, False),
        feature_row("valid-l2", "valid-r3", 0.35, 4, False),
        feature_row("valid-l3", "valid-r1", 0.12, 4, False),
    )


def validation_labels() -> VerifiedLabelBatch:
    rows = validation_rows()
    labels: list[VerifiedPairLabel] = []
    for index, row in enumerate(rows):
        left = str(row[0])
        right = str(row[1])
        target = 1 if index < 3 else 0
        labels.append(label(left, right, target, f"valid-entity-{index}"))
    return label_batch("validation", labels)
