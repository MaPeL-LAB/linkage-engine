"""Build leakage-controlled candidate-ranking matrices."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, cast

import numpy as np

from mapel_linkage.domain.errors import RankingError
from mapel_linkage.models.boosted.training import BoostedFeatureMatrix, BoostedLabelledMatrix
from mapel_linkage.models.ranking.contracts import (
    RankingFeatureMatrix,
    RankingMatrix,
    canonical_digest,
)


def build_ranking_matrix(
    matrix: BoostedLabelledMatrix,
    *,
    query_side: Literal["source", "target"],
    exclude_uninformative_queries: bool = True,
) -> RankingMatrix:
    """Group verified pair features by one source/target record for learned ranking."""

    side_index = 0 if query_side == "source" else 1
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(matrix.pair_references):
        grouped[pair[side_index]].append(index)

    selected_groups: list[tuple[str, list[int]]] = []
    excluded_singleton = 0
    excluded_uninformative = 0
    for query_key, indices in grouped.items():
        if len(indices) < 2:
            excluded_singleton += 1
            continue
        labels = {int(matrix.labels[index]) for index in indices}
        if exclude_uninformative_queries and labels != {0, 1}:
            excluded_uninformative += 1
            continue
        indices.sort(key=lambda index: matrix.pair_digests[index])
        selected_groups.append((query_key, indices))
    selected_groups.sort(key=lambda item: item[0])
    selected_indices = [index for _, indices in selected_groups for index in indices]
    if not selected_indices:
        raise RankingError("ML-RANK-016", "No informative candidate-ranking groups are available.")
    if matrix.partition not in {"training", "validation", "test"}:
        raise RankingError(
            "ML-RANK-029",
            "Calibration and decision labels cannot be used as ranking-model evidence.",
        )

    pair_references = tuple(matrix.pair_references[index] for index in selected_indices)
    pair_digests = tuple(matrix.pair_digests[index] for index in selected_indices)
    query_keys = tuple(query for query, indices in selected_groups for _ in indices)
    group_sizes = tuple(len(indices) for _, indices in selected_groups)
    features = np.asarray(matrix.features[selected_indices, :], dtype=np.float64)
    relevance = np.asarray(matrix.labels[selected_indices], dtype=np.float64)
    selection_digest = canonical_digest(
        {
            "source_selection_digest": matrix.selection_digest,
            "query_side": query_side,
            "selected_pair_digests": pair_digests,
            "group_sizes": group_sizes,
            "exclude_uninformative_queries": exclude_uninformative_queries,
        }
    )
    return RankingMatrix(
        features=features,
        relevance=relevance,
        pair_references=pair_references,
        pair_digests=pair_digests,
        query_keys=query_keys,
        group_sizes=group_sizes,
        feature_names=matrix.feature_names,
        feature_schema_digest=matrix.feature_schema_digest,
        partition=cast(Literal["training", "validation", "test"], matrix.partition),
        label_source_kind=matrix.label_source_kind,
        label_authority_digest=matrix.label_authority_digest,
        selection_digest=selection_digest,
        query_side=query_side,
        excluded_singleton_query_count=excluded_singleton,
        excluded_uninformative_query_count=excluded_uninformative,
    )


def build_ranking_scoring_matrix(
    matrix: BoostedFeatureMatrix,
    *,
    query_side: Literal["source", "target"],
) -> RankingFeatureMatrix:
    """Group every candidate for ranking prediction without requiring labels."""

    if not isinstance(matrix, BoostedFeatureMatrix):
        raise RankingError("ML-RANK-023", "A ranking scoring matrix is invalid.")
    side_index = 0 if query_side == "source" else 1
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(matrix.pair_references):
        grouped[pair[side_index]].append(index)
    groups: list[tuple[str, list[int]]] = []
    for query, indices in grouped.items():
        indices.sort(key=lambda index: matrix.pair_digests[index])
        groups.append((query, indices))
    groups.sort(key=lambda item: item[0])
    selected = [index for _, indices in groups for index in indices]
    return RankingFeatureMatrix(
        features=np.asarray(matrix.features[selected, :], dtype=np.float64),
        pair_references=tuple(matrix.pair_references[index] for index in selected),
        pair_digests=tuple(matrix.pair_digests[index] for index in selected),
        query_keys=tuple(query for query, indices in groups for _ in indices),
        group_sizes=tuple(len(indices) for _, indices in groups),
        feature_names=matrix.feature_names,
        feature_schema_digest=matrix.feature_schema_digest,
        query_side=query_side,
    )
