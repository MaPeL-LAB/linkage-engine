from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from mapel_linkage.configuration.models import RankingModelConfig
from mapel_linkage.domain.errors import RankingError
from mapel_linkage.governance.labels import LabelPartition
from mapel_linkage.governance.paths import PathPolicy
from mapel_linkage.models.boosted.training import BoostedLabelledMatrix
from mapel_linkage.models.ranking import (
    LightGBMRanker,
    build_ranking_matrix,
    read_lightgbm_ranker_artifact,
    write_lightgbm_ranker_artifact,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def labelled_matrix(partition: LabelPartition = "training") -> BoostedLabelledMatrix:
    pairs: list[tuple[str, str]] = []
    features: list[list[float]] = []
    labels: list[int] = []
    for query in range(4):
        for candidate in range(3):
            pairs.append((f"left-{query}", f"right-{query}-{candidate}"))
            if candidate == 0:
                features.append([0.98 - query * 0.01, 1.0, 0.0])
                labels.append(1)
            elif candidate == 1:
                features.append([0.65 - query * 0.01, 0.0, 0.0])
                labels.append(0)
            else:
                features.append([0.12 + query * 0.01, 0.0, 1.0])
                labels.append(0)
    pair_refs = tuple(pairs)
    pair_digests = tuple(digest(f"{left}\x00{right}") for left, right in pair_refs)
    label_array = np.asarray(labels, dtype=np.int8)
    return BoostedLabelledMatrix(
        features=np.asarray(features, dtype=np.float64),
        pair_references=pair_refs,
        pair_digests=pair_digests,
        feature_names=("similarity", "exact", "missing_any"),
        feature_schema_digest=digest("ranking-features"),
        labels=label_array,
        partition=partition,
        label_source_kind="synthetic_truth",
        label_authority_digest=digest(f"{partition}-labels"),
        selection_digest=digest(f"{partition}-selection"),
        positive_count=int(label_array.sum()),
        negative_count=len(label_array) - int(label_array.sum()),
    )


def config() -> RankingModelConfig:
    return RankingModelConfig.model_validate(
        {
            "enabled": True,
            "implementation": "lightgbm_ranker",
            "model_id": "lgb_candidate_ranker",
            "query_side": "source",
            "top_k": 2,
            "require_verified_labels": True,
            "n_estimators": 30,
            "max_depth": 2,
            "learning_rate": 0.2,
            "maximum_training_pairs": 100,
            "n_jobs": 1,
            "deterministic_mode": True,
        }
    )


def test_lightgbm_ranker_is_deterministic_and_ranking_only() -> None:
    matrix = build_ranking_matrix(labelled_matrix(), query_side="source")
    first = LightGBMRanker.fit(
        matrix=matrix,
        model=config(),
        random_seed=20260817,
        configuration_digest=digest("configuration"),
    )
    second = LightGBMRanker.fit(
        matrix=matrix,
        model=config(),
        random_seed=20260817,
        configuration_digest=digest("configuration"),
    )
    result = LightGBMRanker.score(matrix=matrix, model=first)
    assert first.model_digest == second.model_digest
    assert result.query_count == 4
    assert result.top_k_membership.sum() == 8
    assert result.decision_authority == "ranking_only"
    assert result.relationship_authority == "none"
    assert "left-0" not in repr(result)


def test_true_candidates_rank_first_on_fixture() -> None:
    matrix = build_ranking_matrix(labelled_matrix(), query_side="source")
    artifact = LightGBMRanker.fit(
        matrix=matrix,
        model=config(),
        random_seed=7,
        configuration_digest=digest("configuration"),
    )
    result = LightGBMRanker.score(matrix=matrix, model=artifact)
    for index, relevance in enumerate(matrix.relevance):
        if relevance == 1.0:
            assert result.ranks[index] == 1


def test_lightgbm_ranker_rejects_nontraining_fit() -> None:
    validation = build_ranking_matrix(labelled_matrix("validation"), query_side="source")
    with pytest.raises(RankingError, match="ML-RANK-018"):
        LightGBMRanker.fit(
            matrix=validation,
            model=config(),
            random_seed=1,
            configuration_digest=digest("configuration"),
        )


def test_lightgbm_ranking_artifact_round_trip_and_tamper(tmp_path: Path) -> None:
    matrix = build_ranking_matrix(labelled_matrix(), query_side="source")
    artifact = LightGBMRanker.fit(
        matrix=matrix,
        model=config(),
        random_seed=11,
        configuration_digest=digest("configuration"),
    )
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    written = write_lightgbm_ranker_artifact(
        artifact=artifact,
        model_path="artifacts/models/lgb_ranker.txt",
        manifest_path="artifacts/models/lgb_ranker.manifest.json",
        policy=policy,
    )
    restored = read_lightgbm_ranker_artifact(
        model_path="artifacts/models/lgb_ranker.txt",
        manifest_path="artifacts/models/lgb_ranker.manifest.json",
        policy=policy,
    )
    assert restored.artifact_digest == artifact.artifact_digest

    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    manifest["top_k"] = int(manifest["top_k"]) + 1
    written.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RankingError, match="ML-RANK-028"):
        read_lightgbm_ranker_artifact(
            model_path="artifacts/models/lgb_ranker.txt",
            manifest_path="artifacts/models/lgb_ranker.manifest.json",
            policy=policy,
        )


def test_lightgbm_ranker_missing_dependency() -> None:
    with mock.patch("mapel_linkage.models.ranking.lightgbm_ranker._lightgbm", None):
        matrix = build_ranking_matrix(labelled_matrix(), query_side="source")
        with pytest.raises(RankingError, match="ML-RANK-017"):
            LightGBMRanker.fit(
                matrix=matrix,
                model=config(),
                random_seed=1,
                configuration_digest=digest("configuration"),
            )
