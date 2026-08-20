from __future__ import annotations

from typing import cast

import pytest

from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.domain.table_refs import TableRef
from mapel_linkage.io import DuckDBStore
from mapel_linkage.models.fellegi_sunter import (
    SplinkNativeModelArtifact,
    SplinkNativeScoreResult,
    SplinkSettingsPlan,
)
from mapel_linkage.pipeline.inference_runner import NativeSplinkInferenceReplay
from mapel_linkage.pipeline.score_evidence import issue_native_splink_score_evidence
from mapel_linkage.preprocessing import PreparedDataset, surrogate_record_key


def _prepared(dataset_id: str, table_name: str) -> PreparedDataset:
    return PreparedDataset(
        dataset_id=dataset_id,
        table=TableRef(table_name, "a" * 64, 2),
        variable_columns={"value": "value"},
        missingness_columns={"value": "value_missing"},
    )


@pytest.mark.parametrize("mutation", ["reorder", "substitute"])
def test_native_replay_rejects_public_to_prepared_pair_mapping_drift(mutation: str) -> None:
    public_pairs = (("left-1", "right-1"), ("left-2", "right-2"))
    prepared_pairs = tuple(
        (
            surrogate_record_key("source_a", left),
            surrogate_record_key("source_b", right),
        )
        for left, right in public_pairs
    )
    selected = (
        tuple(reversed(prepared_pairs))
        if mutation == "reorder"
        else (
            prepared_pairs[0],
            (
                surrogate_record_key("source_a", "left-substituted"),
                prepared_pairs[1][1],
            ),
        )
    )
    expected = tuple(dict.fromkeys((*prepared_pairs, *selected)))
    settings = SplinkSettingsPlan(
        settings_digest="b" * 64,
        comparison_count=1,
        blocking_rule_count=1,
        settings={
            "link_type": "link_only",
            "unique_id_column_name": "__ml_record_key",
            "source_dataset_column_name": "__ml_dataset_id",
        },
    )

    with DuckDBStore() as store, pytest.raises(PipelineError, match="ML-PIPE-088"):
        NativeSplinkInferenceReplay(
            store=store,
            left=_prepared("source_a", "prepared_left"),
            right=_prepared("source_b", "prepared_right"),
            settings_plan=settings,
            model_artifact=cast(SplinkNativeModelArtifact, object()),
            expected_prepared_pairs=expected,
            selected_prepared_pairs=selected,
            public_pair_references=public_pairs,
            maximum_candidate_pairs=10,
        )


def test_native_score_issuer_rejects_substituted_pair_digest() -> None:
    with DuckDBStore() as store, pytest.raises(PipelineError, match="ML-PIPE-080"):
        issue_native_splink_score_evidence(
            store=store,
            score_result=cast(SplinkNativeScoreResult, object()),
            model_artifact=cast(SplinkNativeModelArtifact, object()),
            pair_references=(("left-1", "right-1"),),
            pair_digests=("0" * 64,),
        )
