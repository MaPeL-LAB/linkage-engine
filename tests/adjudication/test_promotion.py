from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from mapel_linkage.adjudication.disagreement import resolve_pair_consensus
from mapel_linkage.adjudication.promotion import (
    PromotionConfig,
    evaluate_promotion_eligibility,
    promote_to_verified_batch,
)
from mapel_linkage.adjudication.review_import import AdjudicationOutcome, AdjudicationRecord
from mapel_linkage.domain.errors import AdjudicationError
from mapel_linkage.governance.labels import assert_disjoint_label_partitions


def _digest(val: str) -> str:
    return hashlib.sha256(val.encode("utf-8")).hexdigest()


def _make_rec(
    event_id: str,
    left: str,
    right: str,
    decision: AdjudicationOutcome,
    confidence: float,
    reviewer_id: str = "rev_1",
    protocol: str = "proto_v1",
    entity: str | None = None,
) -> AdjudicationRecord:
    e_digests = (_digest(entity or f"entity_{left}"),)
    return AdjudicationRecord(
        event_id=event_id,
        left_record_key=left,
        right_record_key=right,
        decision=decision,
        confidence=confidence,
        reviewer_id=reviewer_id,
        timestamp=datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
        protocol_version=protocol,
        entity_component_digests=e_digests,
    )


def test_promote_clean_match_and_nonmatch() -> None:
    rec_match = _make_rec("e1", "l1", "r1", "match", 0.95)
    rec_nonmatch = _make_rec("e2", "l2", "r2", "nonmatch", 0.90)

    config = PromotionConfig(target_partition="training", min_confidence=0.85)

    eval_match = evaluate_promotion_eligibility(rec_match, config)
    assert eval_match.is_eligible is True
    assert eval_match.promoted_label == 1
    assert eval_match.target_partition == "training"
    assert len(eval_match.rejection_reasons) == 0

    eval_nonmatch = evaluate_promotion_eligibility(rec_nonmatch, config)
    assert eval_nonmatch.is_eligible is True
    assert eval_nonmatch.promoted_label == 0
    assert eval_nonmatch.target_partition == "training"


def test_rejection_of_non_binary_outcomes() -> None:
    config = PromotionConfig(target_partition="training", min_confidence=0.80)

    for non_binary in ("uncertain", "insufficient_information", "duplicate_review"):
        rec = _make_rec(f"e_{non_binary}", "l", "r", non_binary, 0.90)
        evaluation = evaluate_promotion_eligibility(rec, config)
        assert evaluation.is_eligible is False
        assert evaluation.promoted_label is None
        assert evaluation.is_audit_only is True
        assert "non_binary_outcome_audit_only" in evaluation.rejection_reasons


def test_rejection_of_insufficient_confidence() -> None:
    rec = _make_rec("e1", "l1", "r1", "match", 0.70)
    config = PromotionConfig(target_partition="training", min_confidence=0.85)

    evaluation = evaluate_promotion_eligibility(rec, config)
    assert evaluation.is_eligible is False
    assert "insufficient_confidence" in evaluation.rejection_reasons


def test_rejection_of_insufficient_reviewers() -> None:
    rec = _make_rec("e1", "l1", "r1", "match", 0.95)
    consensus = resolve_pair_consensus((rec,))

    config = PromotionConfig(
        target_partition="training",
        min_confidence=0.85,
        require_double_review=True,
    )

    evaluation = evaluate_promotion_eligibility(consensus, config)
    assert evaluation.is_eligible is False
    assert "requires_double_review" in evaluation.rejection_reasons


def test_rejection_of_unapproved_protocol() -> None:
    rec = _make_rec("e1", "l1", "r1", "match", 0.95, protocol="unapproved_proto")
    config = PromotionConfig(
        target_partition="training",
        min_confidence=0.85,
        allowed_protocols=frozenset({"proto_v1", "proto_v2"}),
    )

    evaluation = evaluate_promotion_eligibility(rec, config)
    assert evaluation.is_eligible is False
    assert "unapproved_protocol_version" in evaluation.rejection_reasons


def test_locked_test_partition_protection() -> None:
    rec = _make_rec("e1", "l_test", "r_test", "match", 0.98)
    test_pair_digest = rec.pair_digest()

    # Attempting to promote locked test pair to training
    train_config = PromotionConfig(target_partition="training", min_confidence=0.80)
    eval_train = evaluate_promotion_eligibility(
        rec, train_config, locked_test_pairs=frozenset({test_pair_digest})
    )
    assert eval_train.is_eligible is False
    assert "locked_test_partition_violation" in eval_train.rejection_reasons
    assert eval_train.is_audit_only is True

    # Promoting to test partition is permitted
    test_config = PromotionConfig(target_partition="test", min_confidence=0.80)
    eval_test = evaluate_promotion_eligibility(
        rec, test_config, locked_test_pairs=frozenset({test_pair_digest})
    )
    assert eval_test.is_eligible is True
    assert eval_test.promoted_label == 1
    assert eval_test.target_partition == "test"


def test_promote_to_verified_batch_end_to_end() -> None:
    rec1 = _make_rec("e1", "l1", "r1", "match", 0.95, entity="ent_1")
    rec2 = _make_rec("e2", "l2", "r2", "nonmatch", 0.90, entity="ent_2")
    rec_bad = _make_rec("e3", "l3", "r3", "uncertain", 0.50, entity="ent_3")

    config = PromotionConfig(target_partition="training", min_confidence=0.80)
    batch, summary = promote_to_verified_batch(
        [rec1, rec2, rec_bad],
        config,
        verification_protocol="proto_v1",
        source_digest=_digest("source_digest_1"),
    )

    assert batch.source_kind == "verified_human_adjudication"
    assert batch.partition == "training"
    assert batch.verification_protocol == "proto_v1"
    assert len(batch.labels) == 2
    assert summary.total_evaluated == 3
    assert summary.eligible_count == 2
    assert summary.promoted_positive_count == 1
    assert summary.promoted_negative_count == 1
    assert summary.audit_only_count == 1


def test_promote_to_verified_batch_empty_eligible_rejection() -> None:
    rec_bad = _make_rec("e1", "l1", "r1", "uncertain", 0.50)
    config = PromotionConfig(target_partition="training", min_confidence=0.80)

    with pytest.raises(AdjudicationError) as captured:
        promote_to_verified_batch(
            [rec_bad],
            config,
            verification_protocol="proto_v1",
            source_digest=_digest("source"),
        )
    assert captured.value.code == "ML-ADJ-017"


def test_partition_disjointness_with_promoted_batches() -> None:
    train_rec = _make_rec("e1", "l_tr", "r_tr", "match", 0.95, entity="ent_train")
    val_rec = _make_rec("e2", "l_val", "r_val", "nonmatch", 0.90, entity="ent_val")

    train_batch, _ = promote_to_verified_batch(
        [train_rec],
        PromotionConfig(target_partition="training"),
        verification_protocol="proto_v1",
        source_digest=_digest("source_train"),
    )

    val_batch, _ = promote_to_verified_batch(
        [val_rec],
        PromotionConfig(target_partition="validation"),
        verification_protocol="proto_v1",
        source_digest=_digest("source_val"),
    )

    report = assert_disjoint_label_partitions((train_batch, val_batch))
    assert report.partition_count == 2
    assert report.entity_component_count == 2


def test_privacy_repr_hides_private_record_keys() -> None:
    sentinel_left = "SYNTHETIC-PRIVATE-LEFT-PROMO"
    sentinel_right = "SYNTHETIC-PRIVATE-RIGHT-PROMO"
    rec = _make_rec("e1", sentinel_left, sentinel_right, "match", 0.95)
    evaluation = evaluate_promotion_eligibility(rec, PromotionConfig())

    rendered = repr(evaluation)
    assert sentinel_left not in rendered
    assert sentinel_right not in rendered

    summary = evaluation.safe_summary()
    assert summary["is_eligible"] is True
    assert summary["promoted_label"] == 1
