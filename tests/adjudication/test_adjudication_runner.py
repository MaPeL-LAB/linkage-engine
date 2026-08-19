"""Unit tests for adjudication workflow runner, multi-reviewer consensus, and audit ledger."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mapel_linkage.adjudication.adjudication_runner import (
    GENESIS_PREV_DIGEST,
    AdjudicationAuditLedger,
    AdjudicationImportResult,
    AdjudicationLedgerEntry,
    AdjudicationWorkflowRunner,
    ConsensusReport,
    LabelPromotionResult,
)
from mapel_linkage.adjudication.review_import import (
    AdjudicationOutcome,
    AdjudicationRecord,
)
from mapel_linkage.domain.errors import AdjudicationError, LabelProvenanceError


def _digest(val: str) -> str:
    return hashlib.sha256(val.encode("utf-8")).hexdigest()


def _make_rec(
    event_id: str,
    left: str,
    right: str,
    decision: AdjudicationOutcome,
    confidence: float,
    reviewer_id: str = "rev_alice",
    protocol: str = "proto_v1",
    entity: str | None = None,
    superseded_event_id: str | None = None,
) -> AdjudicationRecord:
    e_digests = (_digest(entity or f"entity_{left}"),)
    return AdjudicationRecord(
        event_id=event_id,
        left_record_key=left,
        right_record_key=right,
        decision=decision,
        confidence=confidence,
        reviewer_id=reviewer_id,
        timestamp=datetime(2026, 8, 19, 10, 0, tzinfo=UTC),
        protocol_version=protocol,
        entity_component_digests=e_digests,
        superseded_event_id=superseded_event_id,
        notes="Reviewer note for auditing only.",
    )


# ---------------------------------------------------------------------------
# 1. Review Import & Append-Only Ledger Writing Tests
# ---------------------------------------------------------------------------


def test_import_reviews_from_jsonl(tmp_path: Path) -> None:
    records_data = [
        {
            "event_id": "evt_001",
            "left_record_key": "left_1",
            "right_record_key": "right_1",
            "decision": "match",
            "confidence": 0.95,
            "reviewer_id": "rev_alice",
            "timestamp": "2026-08-19T10:00:00Z",
            "protocol_version": "proto_v1",
            "entity_component_digests": [_digest("ent_1")],
        },
        {
            "event_id": "evt_002",
            "left_record_key": "left_2",
            "right_record_key": "right_2",
            "decision": "nonmatch",
            "confidence": 0.88,
            "reviewer_id": "rev_bob",
            "timestamp": "2026-08-19T10:05:00Z",
            "protocol_version": "proto_v1",
            "entity_component_digests": [_digest("ent_2")],
        },
    ]
    jsonl_path = tmp_path / "reviews.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in records_data) + "\n", encoding="utf-8")

    ledger_file = tmp_path / "ledger.json"
    result = AdjudicationWorkflowRunner.import_reviews(
        jsonl_path,
        ledger_path=ledger_file,
    )

    assert isinstance(result, AdjudicationImportResult)
    assert result.total_imported == 2
    assert result.active_record_count == 2
    assert result.superseded_record_count == 0
    assert result.ledger_entry_count == 2
    assert ledger_file.is_file()

    # Verify ledger entries and hash chain
    ledger = result.ledger
    assert len(ledger.entries) == 2
    assert ledger.entries[0].entry_index == 0
    assert ledger.entries[0].prev_entry_digest == GENESIS_PREV_DIGEST
    assert ledger.entries[1].entry_index == 1
    assert ledger.entries[1].prev_entry_digest == ledger.entries[0].entry_digest
    ledger.verify_integrity()
    assert ledger.is_valid() is True


def test_import_reviews_with_candidate_pair_validation() -> None:
    rec1 = _make_rec("evt_valid", "left_a", "right_a", "match", 0.92)
    rec2 = _make_rec("evt_unknown", "left_unregistered", "right_unregistered", "match", 0.90)

    candidate_pairs = [("left_a", "right_a"), ("left_b", "right_b")]

    # 1. Importing valid record against candidates succeeds
    result = AdjudicationWorkflowRunner.import_reviews(
        [rec1],
        candidate_pair_references=candidate_pairs,
    )
    assert result.total_imported == 1

    # 2. Importing record with unregistered pair fails with ML-ADJ-020 under strict mode
    with pytest.raises(AdjudicationError) as exc_info:
        AdjudicationWorkflowRunner.import_reviews(
            [rec2],
            candidate_pair_references=candidate_pairs,
            strict_candidate_check=True,
        )
    assert exc_info.value.code == "ML-ADJ-020"


def test_append_only_ledger_extension(tmp_path: Path) -> None:
    rec1 = _make_rec("evt_1", "l1", "r1", "match", 0.95, reviewer_id="rev_1")
    rec2 = _make_rec("evt_2", "l2", "r2", "nonmatch", 0.90, reviewer_id="rev_2")
    rec3 = _make_rec("evt_3", "l3", "r3", "match", 0.85, reviewer_id="rev_3")

    ledger_path = tmp_path / "chain_ledger.json"

    # Batch 1
    res1 = AdjudicationWorkflowRunner.import_reviews(
        [rec1, rec2],
        ledger_path=ledger_path,
    )
    assert res1.ledger_entry_count == 2
    assert res1.ledger.entries[-1].entry_index == 1

    # Batch 2 appending to existing ledger
    res2 = AdjudicationWorkflowRunner.import_reviews(
        [rec3],
        ledger_path=ledger_path,
    )
    assert res2.ledger_entry_count == 3
    assert res2.ledger.entries[2].entry_index == 2
    assert res2.ledger.entries[2].prev_entry_digest == res1.ledger.entries[1].entry_digest
    res2.ledger.verify_integrity()


def test_rejection_of_duplicate_event_in_ledger() -> None:
    rec1 = _make_rec("evt_dup", "l1", "r1", "match", 0.95)
    ledger = AdjudicationAuditLedger.create_empty()
    updated = ledger.append_records([rec1])

    # Appending record with same event ID is rejected
    with pytest.raises(AdjudicationError) as exc_info:
        updated.append_records([rec1])
    assert exc_info.value.code == "ML-ADJ-009"


# ---------------------------------------------------------------------------
# 2. Multi-Reviewer Consensus Resolution Tests
# ---------------------------------------------------------------------------


def test_consensus_unanimous_and_majority() -> None:
    # Pair 1: Unanimous match
    p1_r1 = _make_rec("p1_1", "l1", "r1", "match", 0.90, "rev1")
    p1_r2 = _make_rec("p1_2", "l1", "r1", "match", 0.80, "rev2")

    # Pair 2: Majority match (2 match vs 1 nonmatch)
    p2_r1 = _make_rec("p2_1", "l2", "r2", "match", 0.95, "rev1")
    p2_r2 = _make_rec("p2_2", "l2", "r2", "match", 0.85, "rev2")
    p2_r3 = _make_rec("p2_3", "l2", "r2", "nonmatch", 0.70, "rev3")

    # Pair 3: Tied dispute (1 match vs 1 nonmatch)
    p3_r1 = _make_rec("p3_1", "l3", "r3", "match", 0.80, "rev1")
    p3_r2 = _make_rec("p3_2", "l3", "r3", "nonmatch", 0.80, "rev2")

    report = AdjudicationWorkflowRunner.resolve_consensus(
        [p1_r1, p1_r2, p2_r1, p2_r2, p2_r3, p3_r1, p3_r2],
        policy="majority_vote",
    )

    assert isinstance(report, ConsensusReport)
    assert report.disagreement_report.total_pairs == 3
    assert report.disagreement_report.resolved_pairs == 2
    assert report.disagreement_report.unresolved_pairs == 1
    assert report.disagreement_report.conflict_count == 2
    assert report.disagreement_report.unanimous_count == 1
    assert report.disagreement_report.majority_count == 1

    assert len(report.resolved_decisions) == 2
    assert len(report.unresolved_decisions) == 1
    assert len(report.conflicts) == 2


def test_consensus_senior_reviewer_override() -> None:
    rec_j1 = _make_rec("e1", "l1", "r1", "match", 0.60, "rev_junior1")
    rec_j2 = _make_rec("e2", "l1", "r1", "nonmatch", 0.60, "rev_junior2")
    rec_senior = _make_rec("e3", "l1", "r1", "match", 0.95, "rev_senior")

    report = AdjudicationWorkflowRunner.resolve_consensus(
        [rec_j1, rec_j2, rec_senior],
        policy="senior_reviewer_override",
        senior_reviewers={"rev_senior"},
    )

    assert report.disagreement_report.resolved_pairs == 1
    assert report.disagreement_report.senior_override_count == 1
    resolved = report.resolved_decisions[0]
    assert resolved.consensus_outcome == "match"
    assert resolved.resolution_method == "senior_override"
    assert resolved.senior_reviewer_id == "rev_senior"


def test_consensus_agreement_threshold_filtering() -> None:
    # 2 match vs 1 nonmatch = 2/3 (66.7%) agreement
    p1 = _make_rec("p1_1", "l1", "r1", "match", 0.90, "rev1")
    p2 = _make_rec("p1_2", "l1", "r1", "match", 0.90, "rev2")
    p3 = _make_rec("p1_3", "l1", "r1", "nonmatch", 0.80, "rev3")

    # With threshold 0.60, agreement ratio 0.67 is accepted
    report_pass = AdjudicationWorkflowRunner.resolve_consensus(
        [p1, p2, p3],
        policy="majority_vote",
        agreement_threshold=0.60,
    )
    assert report_pass.disagreement_report.resolved_pairs == 1

    # With threshold 0.80, agreement ratio 0.67 is rejected as unresolved
    report_fail = AdjudicationWorkflowRunner.resolve_consensus(
        [p1, p2, p3],
        policy="majority_vote",
        agreement_threshold=0.80,
    )
    assert report_fail.disagreement_report.resolved_pairs == 0
    assert report_fail.disagreement_report.unresolved_pairs == 1
    assert "below_threshold" in str(report_fail.unresolved_decisions[0].dispute_reason)


# ---------------------------------------------------------------------------
# 3. Label Promotion & Partition Disjointness Validation Tests
# ---------------------------------------------------------------------------


def test_promote_to_verified_labels_strict_governance(tmp_path: Path) -> None:
    rec1 = _make_rec("e1", "l1", "r1", "match", 0.95, entity="ent_1")
    rec2 = _make_rec("e2", "l2", "r2", "nonmatch", 0.90, entity="ent_2")
    rec_unresolved = _make_rec("e3", "l3", "r3", "uncertain", 0.50, entity="ent_3")

    report = AdjudicationWorkflowRunner.resolve_consensus([rec1, rec2, rec_unresolved])
    manifest_out = tmp_path / "promotion_manifest.json"

    promo_result = AdjudicationWorkflowRunner.promote_to_verified_labels(
        report,
        target_partition="training",
        min_confidence=0.80,
        output_manifest_path=manifest_out,
    )

    assert isinstance(promo_result, LabelPromotionResult)
    assert promo_result.target_partition == "training"
    assert promo_result.promotion_summary.eligible_count == 2
    assert promo_result.promotion_summary.promoted_positive_count == 1
    assert promo_result.promotion_summary.promoted_negative_count == 1
    assert promo_result.promotion_summary.audit_only_count == 1

    # Strict governance: automatic retraining is NEVER triggered
    assert promo_result.retraining_triggered is False
    summary = promo_result.safe_summary()
    assert summary["retraining_triggered"] is False
    assert manifest_out.is_file()


def test_partition_disjointness_across_promoted_batches() -> None:
    train_rec = _make_rec("e1", "l_tr", "r_tr", "match", 0.95, entity="ent_train")
    val_rec = _make_rec("e2", "l_val", "r_val", "nonmatch", 0.90, entity="ent_val")

    # Promote training batch
    train_result = AdjudicationWorkflowRunner.promote_to_verified_labels(
        [train_rec],
        target_partition="training",
        min_confidence=0.80,
    )

    # Promote validation batch with existing training batch disjointness verification
    val_result = AdjudicationWorkflowRunner.promote_to_verified_labels(
        [val_rec],
        target_partition="validation",
        min_confidence=0.80,
        existing_partition_batches=(train_result.verified_batch,),
    )

    assert val_result.disjointness_report is not None
    assert val_result.disjointness_report.partition_count == 2
    assert val_result.disjointness_report.entity_component_count == 2


def test_partition_disjointness_violation_rejection() -> None:
    shared_entity = "ent_shared"
    train_rec = _make_rec("e1", "l_tr", "r_tr", "match", 0.95, entity=shared_entity)
    val_rec = _make_rec("e2", "l_val", "r_val", "nonmatch", 0.90, entity=shared_entity)

    train_result = AdjudicationWorkflowRunner.promote_to_verified_labels(
        [train_rec],
        target_partition="training",
        min_confidence=0.80,
    )

    # Promoting validation batch sharing an entity with training partition
    # raises LabelProvenanceError due to partition disjointness violation.
    with pytest.raises(LabelProvenanceError) as exc_info:
        AdjudicationWorkflowRunner.promote_to_verified_labels(
            [val_rec],
            target_partition="validation",
            min_confidence=0.80,
            existing_partition_batches=(train_result.verified_batch,),
        )
    assert exc_info.value.code == "ML-LABEL-013"


def test_locked_test_partition_protection_violation() -> None:
    test_rec = _make_rec("e_test", "l_test", "r_test", "match", 0.99)
    locked_pairs = frozenset({test_rec.pair_digest()})

    # Attempting to promote locked test pair to training raises AdjudicationError
    with pytest.raises(AdjudicationError) as exc_info:
        AdjudicationWorkflowRunner.promote_to_verified_labels(
            [test_rec],
            target_partition="training",
            locked_test_pairs=locked_pairs,
        )
    assert exc_info.value.code in ("ML-ADJ-017", "ML-ADJ-022")


# ---------------------------------------------------------------------------
# 4. Audit Trail Tamper Detection Tests
# ---------------------------------------------------------------------------


def test_audit_trail_tamper_detection() -> None:
    rec1 = _make_rec("e1", "l1", "r1", "match", 0.95, reviewer_id="rev_alice")
    rec2 = _make_rec("e2", "l2", "r2", "nonmatch", 0.88, reviewer_id="rev_bob")

    import_res = AdjudicationWorkflowRunner.import_reviews([rec1, rec2])
    ledger = import_res.ledger

    # Valid ledger passes integrity check
    ledger.verify_integrity()
    assert ledger.is_valid() is True

    # 1. Tamper with decision in an entry
    tampered_entry = AdjudicationLedgerEntry(
        entry_index=0,
        event_id=ledger.entries[0].event_id,
        pair_digest=ledger.entries[0].pair_digest,
        reviewer_id=ledger.entries[0].reviewer_id,
        decision="nonmatch",  # Tampered from "match"
        confidence=ledger.entries[0].confidence,
        timestamp=ledger.entries[0].timestamp,
        protocol_version=ledger.entries[0].protocol_version,
        canonical_event_digest=ledger.entries[0].canonical_event_digest,
        prev_entry_digest=ledger.entries[0].prev_entry_digest,
        entry_digest=ledger.entries[0].entry_digest,
    )
    tampered_ledger = AdjudicationAuditLedger(
        entries=(tampered_entry, ledger.entries[1]),
        ledger_id=ledger.ledger_id,
        ledger_digest=ledger.ledger_digest,
        created_at=ledger.created_at,
    )
    assert tampered_ledger.is_valid() is False
    with pytest.raises(AdjudicationError) as exc_info:
        tampered_ledger.verify_integrity()
    assert exc_info.value.code == "ML-ADJ-021"

    # 2. Tamper with hash chain (prev_entry_digest)
    tampered_prev_entry = AdjudicationLedgerEntry(
        entry_index=1,
        event_id=ledger.entries[1].event_id,
        pair_digest=ledger.entries[1].pair_digest,
        reviewer_id=ledger.entries[1].reviewer_id,
        decision=ledger.entries[1].decision,
        confidence=ledger.entries[1].confidence,
        timestamp=ledger.entries[1].timestamp,
        protocol_version=ledger.entries[1].protocol_version,
        canonical_event_digest=ledger.entries[1].canonical_event_digest,
        prev_entry_digest="f" * 64,  # Corrupted previous digest
        entry_digest=ledger.entries[1].entry_digest,
    )
    tampered_chain_ledger = AdjudicationAuditLedger(
        entries=(ledger.entries[0], tampered_prev_entry),
        ledger_id=ledger.ledger_id,
        ledger_digest=ledger.ledger_digest,
        created_at=ledger.created_at,
    )
    assert tampered_chain_ledger.is_valid() is False
    with pytest.raises(AdjudicationError) as exc_info:
        tampered_chain_ledger.verify_integrity()
    assert exc_info.value.code == "ML-ADJ-021"


def test_audit_trail_json_roundtrip_and_tamper(tmp_path: Path) -> None:
    rec1 = _make_rec("e1", "l1", "r1", "match", 0.95, reviewer_id="rev_alice")
    import_res = AdjudicationWorkflowRunner.import_reviews([rec1])

    ledger_path = tmp_path / "audit.json"
    import_res.ledger.write_to_file(ledger_path)

    # Valid load
    loaded = AdjudicationAuditLedger.from_json(ledger_path.read_text(encoding="utf-8"))
    assert loaded.is_valid() is True

    # Tamper with file content on disk
    raw_data = json.loads(ledger_path.read_text(encoding="utf-8"))
    raw_data["entries"][0]["confidence"] = 0.50  # Tampered confidence
    tampered_json = json.dumps(raw_data)

    with pytest.raises(AdjudicationError) as exc_info:
        AdjudicationAuditLedger.from_json(tampered_json)
    assert exc_info.value.code == "ML-ADJ-021"


# ---------------------------------------------------------------------------
# 5. Privacy Repr & Safe Summaries Tests
# ---------------------------------------------------------------------------


def test_privacy_and_safe_summaries() -> None:
    sentinel_left = "PRIVATE-RECORD-KEY-LEFT-12345"
    sentinel_right = "PRIVATE-RECORD-KEY-RIGHT-67890"
    rec = _make_rec("e1", sentinel_left, sentinel_right, "match", 0.95)

    import_res = AdjudicationWorkflowRunner.import_reviews([rec])
    consensus_report = AdjudicationWorkflowRunner.resolve_consensus(import_res.imported_batch)
    promo_result = AdjudicationWorkflowRunner.promote_to_verified_labels(consensus_report)

    # Repr checks
    for obj in (
        import_res,
        import_res.ledger,
        import_res.ledger.entries[0],
        consensus_report,
        promo_result,
    ):
        rendered = repr(obj)
        assert sentinel_left not in rendered
        assert sentinel_right not in rendered
        assert "Reviewer note for auditing only." not in rendered

    # Safe summary checks
    import_summary = import_res.safe_summary()
    assert "input_digest" in import_summary
    assert sentinel_left not in str(import_summary)

    ledger_summary = import_res.ledger.safe_summary()
    assert ledger_summary["entry_count"] == 1
    assert "unique_pair_count" in ledger_summary

    consensus_summary = consensus_report.safe_summary()
    assert consensus_summary["resolved_pairs"] == 1

    promo_summary = promo_result.safe_summary()
    assert promo_summary["eligible_count"] == 1
    assert promo_summary["retraining_triggered"] is False
