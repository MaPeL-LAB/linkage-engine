"""Verified-label feature matrices and deterministic hard-negative selection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mapel_linkage.comparisons import ComparisonFeatureResult
from mapel_linkage.configuration.models import BoostedTreeModelConfig
from mapel_linkage.domain.errors import (
    BoostedTreeBudgetExceeded,
    BoostedTreeError,
    DataPlaneError,
)
from mapel_linkage.domain.sql_identifiers import quote_identifier
from mapel_linkage.governance.labels import (
    LabelPartition,
    LabelSourceKind,
    VerifiedLabelBatch,
)
from mapel_linkage.io.duckdb_store import ColumnSpec, DuckDBStore

_PAIR_COLUMNS: tuple[str, str] = ("left_record_key", "right_record_key")


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _feature_names(features: ComparisonFeatureResult) -> tuple[str, ...]:
    names: list[str] = []
    for columns in features.columns.values():
        names.extend(
            (
                columns.value,
                columns.level,
                columns.exact,
                columns.missing_left,
                columns.missing_right,
                columns.missing_both,
                columns.missing_any,
            )
        )
    return tuple(names)


def _as_float(value: object) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        converted = float(value)
        if math.isfinite(converted):
            return converted
    raise BoostedTreeError(
        "ML-BOOST-001", "A comparison feature could not be converted to a model value."
    )


def _immutable_float_matrix(
    values: NDArray[np.float64] | Sequence[Sequence[float]],
) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64).copy()
    matrix.setflags(write=False)
    return matrix


def _immutable_label_vector(
    values: NDArray[np.int8] | Sequence[int],
) -> NDArray[np.int8]:
    labels = np.asarray(values, dtype=np.int8).copy()
    labels.setflags(write=False)
    return labels


@dataclass(frozen=True, slots=True)
class BoostedFeatureMatrix:
    """Private pair references plus immutable numeric comparison features."""

    features: NDArray[np.float64] = field(repr=False)
    pair_references: tuple[tuple[str, str], ...] = field(repr=False)
    pair_digests: tuple[str, ...] = field(repr=False)
    feature_names: tuple[str, ...] = field(repr=False)
    feature_schema_digest: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.features, dtype=np.float64).copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "features", matrix)
        if matrix.ndim != 2:
            raise ValueError("features must be a two-dimensional matrix")
        if matrix.shape[0] != len(self.pair_references):
            raise ValueError("pair references must align with feature rows")
        if len(self.pair_digests) != len(self.pair_references):
            raise ValueError("pair digests must align with feature rows")
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError("feature names must align with feature columns")

    @property
    def pair_count(self) -> int:
        return len(self.pair_references)

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "pair_count": self.pair_count,
            "feature_count": len(self.feature_names),
            "feature_schema_digest": self.feature_schema_digest,
        }


@dataclass(frozen=True, slots=True)
class BoostedLabelledMatrix(BoostedFeatureMatrix):
    """A verified labelled matrix with private labels and protected partition metadata."""

    labels: NDArray[np.int8] = field(repr=False)
    partition: LabelPartition
    label_source_kind: LabelSourceKind
    label_authority_digest: str
    selection_digest: str
    positive_count: int
    negative_count: int
    hard_negative_count: int = 0

    def __post_init__(self) -> None:
        BoostedFeatureMatrix.__post_init__(self)
        labels = np.asarray(self.labels, dtype=np.int8).copy()
        labels.setflags(write=False)
        object.__setattr__(self, "labels", labels)
        if labels.ndim != 1 or labels.shape[0] != self.pair_count:
            raise ValueError("labels must align with feature rows")
        if not np.all(np.isin(labels, (0, 1))):
            raise ValueError("labels must be binary")
        if self.positive_count != int(labels.sum()):
            raise ValueError("positive_count does not match the label vector")
        if self.negative_count != self.pair_count - self.positive_count:
            raise ValueError("negative_count does not match the label vector")
        if not 0 <= self.hard_negative_count <= self.negative_count:
            raise ValueError("hard_negative_count is outside the selected negative count")

    def safe_summary(self) -> dict[str, int | str]:
        return {
            **super().safe_summary(),
            "partition": self.partition,
            "label_source_kind": self.label_source_kind,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "hard_negative_count": self.hard_negative_count,
            "label_authority_digest": self.label_authority_digest,
            "selection_digest": self.selection_digest,
        }


class DuckDBVerifiedMatrixBuilder:
    """Build scoring and verified labelled matrices from canonical comparison features."""

    __slots__ = ("_store",)

    def __init__(self, store: DuckDBStore) -> None:
        self._store = store

    def build_scoring(self, *, features: ComparisonFeatureResult) -> BoostedFeatureMatrix:
        names = _feature_names(features)
        rows = self._feature_rows(features=features, feature_names=names)
        matrix, pair_references, pair_digests = self._convert_rows(rows, feature_count=len(names))
        return BoostedFeatureMatrix(
            features=matrix,
            pair_references=pair_references,
            pair_digests=pair_digests,
            feature_names=names,
            feature_schema_digest=self._feature_schema_digest(features, names),
        )

    def build_labelled(
        self,
        *,
        features: ComparisonFeatureResult,
        labels: VerifiedLabelBatch,
        model: BoostedTreeModelConfig | None = None,
        random_seed: int = 0,
        apply_training_selection: bool = False,
    ) -> BoostedLabelledMatrix:
        if apply_training_selection and labels.partition != "training":
            raise BoostedTreeError(
                "ML-BOOST-002", "Hard-negative selection is restricted to the training partition."
            )
        if apply_training_selection and model is None:
            raise BoostedTreeError(
                "ML-BOOST-003", "Training selection requires a boosted-tree model plan."
            )
        if random_seed < 0:
            raise BoostedTreeError(
                "ML-BOOST-004", "The boosted-tree selection seed must be non-negative."
            )

        names = _feature_names(features)
        rows = self._labelled_rows(features=features, labels=labels, feature_names=names)
        matrix, pair_references, pair_digests, target = self._convert_labelled_rows(
            rows,
            feature_count=len(names),
        )
        if not np.any(target == 1) or not np.any(target == 0):
            raise BoostedTreeError(
                "ML-BOOST-005", "Verified labelled matrices require matches and nonmatches."
            )

        selected_indices = tuple(range(len(pair_references)))
        hard_negative_count = 0
        if apply_training_selection:
            assert model is not None
            selected_indices, hard_negative_count = self._select_training_rows(
                matrix=matrix,
                labels=target,
                pair_digests=pair_digests,
                feature_names=names,
                model=model,
                random_seed=random_seed,
            )

        selected_features = _immutable_float_matrix(matrix[list(selected_indices), :])
        selected_labels = _immutable_label_vector(target[list(selected_indices)])
        selected_pairs = tuple(pair_references[index] for index in selected_indices)
        selected_digests = tuple(pair_digests[index] for index in selected_indices)
        feature_schema_digest = self._feature_schema_digest(features, names)
        selection_digest = _canonical_digest(
            {
                "feature_schema_digest": feature_schema_digest,
                "label_authority_digest": labels.label_authority_digest,
                "partition": labels.partition,
                "random_seed": random_seed,
                "selected_pairs": [
                    {"pair_digest": digest, "label": int(label)}
                    for digest, label in zip(selected_digests, selected_labels, strict=True)
                ],
            }
        )
        positive_count = int(selected_labels.sum())
        return BoostedLabelledMatrix(
            features=selected_features,
            pair_references=selected_pairs,
            pair_digests=selected_digests,
            feature_names=names,
            feature_schema_digest=feature_schema_digest,
            labels=selected_labels,
            partition=labels.partition,
            label_source_kind=labels.source_kind,
            label_authority_digest=labels.label_authority_digest,
            selection_digest=selection_digest,
            positive_count=positive_count,
            negative_count=len(selected_labels) - positive_count,
            hard_negative_count=hard_negative_count,
        )

    def _feature_rows(
        self,
        *,
        features: ComparisonFeatureResult,
        feature_names: tuple[str, ...],
    ) -> list[tuple[object, ...]]:
        feature_table = quote_identifier(features.table.table_name)
        projection = ", ".join(
            (
                quote_identifier(_PAIR_COLUMNS[0]),
                quote_identifier(_PAIR_COLUMNS[1]),
                *(quote_identifier(name) for name in feature_names),
            )
        )
        sql = (
            f"SELECT {projection} FROM {feature_table} "
            f"ORDER BY {quote_identifier(_PAIR_COLUMNS[0])}, "
            f"{quote_identifier(_PAIR_COLUMNS[1])}"
        )
        try:
            rows = self._store._fetch_model_rows(sql)
        except DataPlaneError:
            raise BoostedTreeError(
                "ML-BOOST-006", "The comparison feature matrix could not be read safely."
            ) from None
        if len(rows) != features.candidate_pair_count:
            raise BoostedTreeError(
                "ML-BOOST-007", "The comparison feature matrix violates its pair contract."
            )
        return rows

    def _labelled_rows(
        self,
        *,
        features: ComparisonFeatureResult,
        labels: VerifiedLabelBatch,
        feature_names: tuple[str, ...],
    ) -> list[tuple[object, ...]]:
        label_table_name = f"__ml_verified_labels_{labels.label_authority_digest[:12]}"
        label_rows = tuple(
            (
                item.left_record_key,
                item.right_record_key,
                int(item.label),
                item.pair_digest(),
            )
            for item in labels.labels
        )
        try:
            label_table = self._store.create_table_from_rows(
                label_table_name,
                (
                    ColumnSpec("left_record_key", "VARCHAR"),
                    ColumnSpec("right_record_key", "VARCHAR"),
                    ColumnSpec("label", "INTEGER"),
                    ColumnSpec("pair_digest", "VARCHAR"),
                ),
                label_rows,
            )
        except DataPlaneError:
            raise BoostedTreeError(
                "ML-BOOST-008", "The verified label snapshot could not be materialised safely."
            ) from None

        feature_table = quote_identifier(features.table.table_name)
        label_table_sql = quote_identifier(label_table.table_name)
        feature_projection = ", ".join(f"f.{quote_identifier(name)}" for name in feature_names)
        sql = (
            f"SELECT l.{quote_identifier('left_record_key')}, "
            f"l.{quote_identifier('right_record_key')}, "
            f"l.{quote_identifier('pair_digest')}, l.{quote_identifier('label')}, "
            f"{feature_projection} "
            f"FROM {label_table_sql} AS l INNER JOIN {feature_table} AS f "
            f"ON l.{quote_identifier('left_record_key')} = "
            f"f.{quote_identifier('left_record_key')} "
            f"AND l.{quote_identifier('right_record_key')} = "
            f"f.{quote_identifier('right_record_key')} "
            f"ORDER BY l.{quote_identifier('pair_digest')}"
        )
        try:
            rows = self._store._fetch_model_rows(sql)
        except DataPlaneError:
            raise BoostedTreeError(
                "ML-BOOST-009", "Verified labels could not be joined to comparison features."
            ) from None
        if len(rows) != len(labels.labels):
            raise BoostedTreeError(
                "ML-BOOST-010", "A verified label does not resolve to a candidate feature row."
            )
        return rows

    @staticmethod
    def _convert_rows(
        rows: Sequence[tuple[object, ...]],
        *,
        feature_count: int,
    ) -> tuple[NDArray[np.float64], tuple[tuple[str, str], ...], tuple[str, ...]]:
        values: list[list[float]] = []
        pairs: list[tuple[str, str]] = []
        digests: list[str] = []
        for row in rows:
            if len(row) != feature_count + 2:
                raise BoostedTreeError(
                    "ML-BOOST-011", "A model feature row violates the internal schema."
                )
            left, right = row[:2]
            if not isinstance(left, str) or not isinstance(right, str):
                raise BoostedTreeError(
                    "ML-BOOST-012", "A model feature row has an invalid private pair reference."
                )
            pairs.append((left, right))
            digests.append(hashlib.sha256((left + "\x00" + right).encode("utf-8")).hexdigest())
            values.append([_as_float(value) for value in row[2:]])
        return (_immutable_float_matrix(values), tuple(pairs), tuple(digests))

    @staticmethod
    def _convert_labelled_rows(
        rows: Sequence[tuple[object, ...]],
        *,
        feature_count: int,
    ) -> tuple[
        NDArray[np.float64],
        tuple[tuple[str, str], ...],
        tuple[str, ...],
        NDArray[np.int8],
    ]:
        values: list[list[float]] = []
        pairs: list[tuple[str, str]] = []
        digests: list[str] = []
        targets: list[int] = []
        for row in rows:
            if len(row) != feature_count + 4:
                raise BoostedTreeError(
                    "ML-BOOST-013", "A labelled model row violates the internal schema."
                )
            left, right, pair_digest, label = row[:4]
            if (
                not isinstance(left, str)
                or not isinstance(right, str)
                or not isinstance(pair_digest, str)
                or label not in (0, 1)
            ):
                raise BoostedTreeError("ML-BOOST-014", "A labelled model row is invalid.")
            pairs.append((left, right))
            digests.append(pair_digest)
            targets.append(int(label))
            values.append([_as_float(value) for value in row[4:]])
        return (
            _immutable_float_matrix(values),
            tuple(pairs),
            tuple(digests),
            _immutable_label_vector(targets),
        )

    @staticmethod
    def _feature_schema_digest(
        features: ComparisonFeatureResult,
        feature_names: tuple[str, ...],
    ) -> str:
        return _canonical_digest(
            {
                "table_schema_digest": features.table.schema_digest,
                "feature_names": feature_names,
            }
        )

    @classmethod
    def _select_training_rows(
        cls,
        *,
        matrix: NDArray[np.float64],
        labels: NDArray[np.int8],
        pair_digests: tuple[str, ...],
        feature_names: tuple[str, ...],
        model: BoostedTreeModelConfig,
        random_seed: int,
    ) -> tuple[tuple[int, ...], int]:
        positives = [index for index, label in enumerate(labels) if int(label) == 1]
        negatives = [index for index, label in enumerate(labels) if int(label) == 0]
        budget = model.maximum_training_pairs
        if len(positives) > budget:
            raise BoostedTreeBudgetExceeded(
                "ML-BOOST-015", "Verified match labels exceed the training-pair budget."
            )
        remaining_capacity = budget - len(positives)
        if remaining_capacity <= 0 or not negatives:
            raise BoostedTreeError(
                "ML-BOOST-016", "The training selection has no capacity for verified nonmatches."
            )

        if len(negatives) <= remaining_capacity:
            selected_negatives = negatives
            hard_count = 0
        else:
            requested_hard = math.floor(remaining_capacity * model.hard_negative_fraction + 0.5)
            requested_hard = min(max(requested_hard, 0), remaining_capacity)
            ranked = sorted(
                negatives,
                key=lambda index: (
                    -cls._hardness(matrix[index], feature_names),
                    cls._seeded_tie(pair_digests[index], random_seed),
                ),
            )
            hard = ranked[:requested_hard]
            hard_set = set(hard)
            remaining = sorted(
                (index for index in negatives if index not in hard_set),
                key=lambda index: cls._seeded_tie(pair_digests[index], random_seed),
            )
            random_needed = remaining_capacity - len(hard)
            selected_negatives = [*hard, *remaining[:random_needed]]
            hard_count = len(hard)

        selected = [*positives, *selected_negatives]
        selected.sort(key=lambda index: pair_digests[index])
        return (tuple(selected), hard_count)

    @staticmethod
    def _hardness(row: NDArray[np.float64], feature_names: tuple[str, ...]) -> float:
        score = 0.0
        for value, name in zip(row, feature_names, strict=True):
            if not math.isfinite(float(value)):
                continue
            if name.endswith("_exact") and value >= 0.5:
                score += 4.0
            elif name.endswith("_level") and value > 0.0:
                score += 1.0 / float(value)
            elif name.endswith("_missing_any") and value >= 0.5:
                score -= 0.5
        return score

    @staticmethod
    def _seeded_tie(pair_digest: str, random_seed: int) -> str:
        return hashlib.sha256(f"{pair_digest}:{random_seed}".encode()).hexdigest()
