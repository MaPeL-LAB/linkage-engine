from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mapel_linkage.assignment.contracts import pair_digest
from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.pipeline.inference_runner import infer_with_approved_recipe
from mapel_linkage.pipeline.recipes import (
    OperationalValidationStatus,
    PipelineRecipeArtifact,
    RecipeApprovalStatus,
    RecipeExecutionMode,
)
from mapel_linkage.pipeline.score_evidence import PairScoreEvidenceBatch
from tests.pipeline.test_inference_runner import _make_dummy_calibrator


def _recipe(*, champion_artifact_digest: str = "a" * 64) -> PipelineRecipeArtifact:
    calibrator = _make_dummy_calibrator()
    return PipelineRecipeArtifact(
        recipe_id="score_evidence_recipe",
        recipe_version="v1",
        linkage_mode="link_only",
        assignment_constraint="one_to_one",
        configuration_digest="c" * 64,
        candidate_plan_digest="d" * 64,
        feature_schema_digest="f" * 64,
        champion_model_id="xgb_champion",
        champion_model_version="v1",
        champion_artifact_digest=champion_artifact_digest,
        calibrator_digest=calibrator.calibrator_digest,
        ranking_artifact_digest=None,
        decision_policy_digest="e" * 64,
        validation_evidence_digest="1" * 64,
        approval_status=RecipeApprovalStatus.APPROVED_FOR_INFERENCE,
        operational_validation=OperationalValidationStatus.LOCALLY_ESTABLISHED,
    )


def _evidence(
    *,
    pair_digests: tuple[str, ...] | None = None,
    scores: tuple[float, ...] = (0.9, 0.1),
) -> PairScoreEvidenceBatch:
    digests = pair_digests or (
        pair_digest("source_1", "target_1"),
        pair_digest("source_2", "target_2"),
    )
    return PairScoreEvidenceBatch._issue(
        pair_digests=digests,
        scores=np.asarray(scores, dtype=np.float64),
        champion_model_id="xgb_champion",
        champion_model_version="v1",
        champion_artifact_digest="a" * 64,
        configuration_digest="c" * 64,
        feature_schema_digest="f" * 64,
        probability_status="model_score_uncalibrated",
    )


def _with_artifact(
    evidence: PairScoreEvidenceBatch,
    artifact: object,
) -> PairScoreEvidenceBatch:
    forged = object.__new__(PairScoreEvidenceBatch)
    object.__setattr__(forged, "artifact", artifact)
    object.__setattr__(forged, "pair_digests", evidence.pair_digests)
    object.__setattr__(forged, "scores", evidence.scores)
    object.__setattr__(forged, "_issuer", evidence._issuer)
    return forged


def test_score_evidence_is_value_hidden_and_detects_metadata_tampering() -> None:
    evidence = _evidence()

    evidence.assert_valid_contract()
    assert "source_1" not in repr(evidence)
    assert "0.9" not in repr(evidence)
    assert "scores" not in evidence.safe_summary()
    assert "pair_digests" not in evidence.safe_summary()

    forged = _with_artifact(
        evidence,
        replace(evidence.artifact, score_digest="0" * 64),
    )
    with pytest.raises(PipelineError, match="ML-PIPE-067"):
        forged.assert_valid_contract()


def test_score_evidence_detects_pair_order_score_and_recipe_drift() -> None:
    evidence = _evidence()
    expected_pairs = evidence.pair_digests

    evidence.assert_matches(recipe=_recipe(), pair_digests=expected_pairs)
    evidence.assert_scores(np.asarray((0.9, 0.1), dtype=np.float64))

    with pytest.raises(PipelineError, match="ML-PIPE-068"):
        evidence.assert_matches(recipe=_recipe(), pair_digests=tuple(reversed(expected_pairs)))
    with pytest.raises(PipelineError, match="ML-PIPE-068"):
        evidence.assert_matches(
            recipe=_recipe(champion_artifact_digest="b" * 64),
            pair_digests=expected_pairs,
        )

    substituted = _evidence(scores=(0.8, 0.2))
    substituted.assert_matches(recipe=_recipe(), pair_digests=expected_pairs)
    with pytest.raises(PipelineError, match="ML-PIPE-068"):
        substituted.assert_scores(np.asarray((0.9, 0.1), dtype=np.float64))


def test_score_evidence_cannot_authorize_inference_without_typed_artifact_replay() -> None:
    calibrator = _make_dummy_calibrator()
    evidence = _evidence()

    with pytest.raises(PipelineError, match="ML-PIPE-069"):
        infer_with_approved_recipe(
            recipe=_recipe(),
            source_record_keys=("source_1", "source_2"),
            pair_references=(("source_1", "target_1"), ("source_2", "target_2")),
            score_evidence=evidence,
            calibrator_artifact=calibrator,
            execution_mode=RecipeExecutionMode.INFERENCE,
        )
