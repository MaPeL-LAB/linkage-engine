from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mapel_linkage.adjudication.review_import import (
    AdjudicationRecord,
    import_adjudication_records,
    import_adjudications_from_csv,
    import_adjudications_from_jsonl,
)
from mapel_linkage.domain.errors import AdjudicationError


def _digest(val: str) -> str:
    return hashlib.sha256(val.encode("utf-8")).hexdigest()


def test_import_jsonl_valid_records_and_digest(tmp_path: Path) -> None:
    sentinel_left = "SYNTHETIC-PRIVATE-LEFT-001"
    sentinel_right = "SYNTHETIC-PRIVATE-RIGHT-001"

    records_data = [
        {
            "event_id": "adj_evt_001",
            "left_record_key": sentinel_left,
            "right_record_key": sentinel_right,
            "decision": "match",
            "confidence": 0.95,
            "reviewer_id": "rev_alice",
            "timestamp": "2026-08-18T10:00:00Z",
            "protocol_version": "v1.0",
            "entity_component_digests": [_digest("entity_001")],
            "household_component_digests": [_digest("hh_001")],
            "notes": "Verified against synthetic truth.",
        },
        {
            "event_id": "adj_evt_002",
            "left_record_key": "left_002",
            "right_record_key": "right_002",
            "decision": "nonmatch",
            "confidence": 0.85,
            "reviewer_id": "rev_bob",
            "timestamp": "2026-08-18T10:05:00+00:00",
            "protocol_version": "v1.0",
        },
    ]

    jsonl_file = tmp_path / "adjudications.jsonl"
    jsonl_content = "\n".join(json.dumps(r) for r in records_data) + "\n"
    jsonl_file.write_text(jsonl_content, encoding="utf-8")

    batch = import_adjudications_from_jsonl(jsonl_file)

    assert batch.raw_record_count == 2
    assert batch.active_record_count == 2
    assert batch.superseded_event_count == 0
    assert batch.input_digest == hashlib.sha256(jsonl_file.read_bytes()).hexdigest()

    active = batch.active_records()
    assert len(active) == 2
    assert active[0].event_id == "adj_evt_001"
    assert active[0].decision == "match"
    assert active[0].confidence == 0.95
    assert active[0].timestamp == datetime(2026, 8, 18, 10, 0, tzinfo=UTC)

    # Privacy checks: record keys and notes must be hidden in repr
    rendered_record = repr(active[0])
    rendered_batch = repr(batch)
    assert sentinel_left not in rendered_record
    assert sentinel_right not in rendered_record
    assert sentinel_left not in rendered_batch
    assert "Verified against synthetic truth." not in rendered_record

    summary = active[0].safe_summary()
    assert summary["event_id"] == "adj_evt_001"
    assert summary["decision"] == "match"
    assert "has_entity_provenance" in summary


def test_import_csv_valid_records(tmp_path: Path) -> None:
    d1 = _digest("e1")
    d2 = _digest("e2")
    csv_lines = [
        "event_id,source_record_ref,target_record_ref,outcome,confidence"
        ",reviewer,timestamp,protocol,entity_components",
        f"evt_csv_1,src_01,tgt_01,match,0.92,rev_carol,2026-08-18T11:00:00Z,proto_v1,{d1}",
        f"evt_csv_2,src_02,tgt_02,uncertain,0.50,rev_dan,2026-08-18T11:15:00Z,proto_v1,{d2}",
    ]
    csv_content = "\n".join(csv_lines) + "\n"

    csv_file = tmp_path / "adjudications.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    batch = import_adjudications_from_csv(csv_file)
    assert batch.raw_record_count == 2
    assert batch.active_record_count == 2
    records = batch.active_records()
    assert records[0].decision == "match"
    assert records[0].confidence == 0.92
    assert records[1].decision == "uncertain"
    assert records[0].entity_component_digests == (_digest("e1"),)


def test_import_adjudication_records_dict_ingestion() -> None:
    records = [
        {
            "event_id": "evt_dict_1",
            "left_record_key": "left_a",
            "right_record_key": "right_a",
            "decision": "match",
            "confidence": 0.88,
            "reviewer_id": "rev_1",
            "timestamp": datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
            "protocol_version": "v1.0",
        }
    ]
    batch = import_adjudication_records(records)
    assert batch.raw_record_count == 1
    assert len(batch.active_records()) == 1


def test_supersession_single_and_chain() -> None:
    records = [
        {
            "event_id": "evt_v1",
            "left_record_key": "left_x",
            "right_record_key": "right_x",
            "decision": "uncertain",
            "confidence": 0.50,
            "reviewer_id": "rev_junior",
            "timestamp": "2026-08-18T08:00:00Z",
            "protocol_version": "v1",
        },
        {
            "event_id": "evt_v2",
            "left_record_key": "left_x",
            "right_record_key": "right_x",
            "decision": "nonmatch",
            "confidence": 0.70,
            "reviewer_id": "rev_senior",
            "timestamp": "2026-08-18T09:00:00Z",
            "protocol_version": "v1",
            "superseded_event_id": "evt_v1",
        },
        {
            "event_id": "evt_v3",
            "left_record_key": "left_x",
            "right_record_key": "right_x",
            "decision": "match",
            "confidence": 0.95,
            "reviewer_id": "rev_lead",
            "timestamp": "2026-08-18T10:00:00Z",
            "protocol_version": "v1",
            "superseded_event_id": "evt_v2",
        },
        {
            "event_id": "evt_unrelated",
            "left_record_key": "left_y",
            "right_record_key": "right_y",
            "decision": "match",
            "confidence": 0.90,
            "reviewer_id": "rev_junior",
            "timestamp": "2026-08-18T08:30:00Z",
            "protocol_version": "v1",
        },
    ]

    batch = import_adjudication_records(records)
    assert batch.raw_record_count == 4
    assert batch.active_record_count == 2
    assert batch.superseded_event_count == 2

    active = batch.active_records()
    active_ids = {r.event_id for r in active}
    assert active_ids == {"evt_v3", "evt_unrelated"}

    by_pair = batch.by_pair()
    assert len(by_pair) == 2


def test_supersession_cycle_rejection() -> None:
    records = [
        {
            "event_id": "evt_cycle_1",
            "left_record_key": "left_1",
            "right_record_key": "right_1",
            "decision": "match",
            "confidence": 0.8,
            "reviewer_id": "rev_1",
            "timestamp": "2026-08-18T08:00:00Z",
            "protocol_version": "v1",
            "superseded_event_id": "evt_cycle_2",
        },
        {
            "event_id": "evt_cycle_2",
            "left_record_key": "left_1",
            "right_record_key": "right_1",
            "decision": "match",
            "confidence": 0.9,
            "reviewer_id": "rev_2",
            "timestamp": "2026-08-18T09:00:00Z",
            "protocol_version": "v1",
            "superseded_event_id": "evt_cycle_1",
        },
    ]

    batch = import_adjudication_records(records)
    with pytest.raises(AdjudicationError) as captured:
        batch.active_records()
    assert captured.value.code == "ML-ADJ-014"


def test_self_supersession_rejection() -> None:
    records = [
        {
            "event_id": "evt_self",
            "left_record_key": "left_1",
            "right_record_key": "right_1",
            "decision": "match",
            "confidence": 0.8,
            "reviewer_id": "rev_1",
            "timestamp": "2026-08-18T08:00:00Z",
            "protocol_version": "v1",
            "superseded_event_id": "evt_self",
        }
    ]
    with pytest.raises(AdjudicationError) as captured:
        import_adjudication_records(records)
    assert captured.value.code == "ML-ADJ-014"


def test_invalid_confidence_rejection() -> None:
    base = {
        "event_id": "evt_err",
        "left_record_key": "left_1",
        "right_record_key": "right_1",
        "decision": "match",
        "reviewer_id": "rev_1",
        "timestamp": "2026-08-18T08:00:00Z",
        "protocol_version": "v1",
    }

    with pytest.raises(AdjudicationError) as cap1:
        import_adjudication_records([{**base, "confidence": 1.5}])
    assert cap1.value.code == "ML-ADJ-011"

    with pytest.raises(AdjudicationError) as cap2:
        import_adjudication_records([{**base, "confidence": -0.1}])
    assert cap2.value.code == "ML-ADJ-011"

    with pytest.raises(AdjudicationError) as cap3:
        import_adjudication_records([{**base, "confidence": "high"}])
    assert cap3.value.code == "ML-ADJ-011"


def test_invalid_decision_outcome_rejection() -> None:
    with pytest.raises(AdjudicationError) as captured:
        import_adjudication_records(
            [
                {
                    "event_id": "evt_err",
                    "left_record_key": "l",
                    "right_record_key": "r",
                    "decision": "invalid_status",
                    "confidence": 0.9,
                    "reviewer_id": "rev_1",
                    "timestamp": "2026-08-18T08:00:00Z",
                    "protocol_version": "v1",
                }
            ]
        )
    assert captured.value.code == "ML-ADJ-010"


def test_invalid_protocol_version_rejection() -> None:
    with pytest.raises(AdjudicationError) as captured:
        import_adjudication_records(
            [
                {
                    "event_id": "evt_err",
                    "left_record_key": "l",
                    "right_record_key": "r",
                    "decision": "match",
                    "confidence": 0.9,
                    "reviewer_id": "rev_1",
                    "timestamp": "2026-08-18T08:00:00Z",
                    "protocol_version": "123-invalid-lead-digit!",
                }
            ]
        )
    assert captured.value.code == "ML-ADJ-013"


def test_missing_required_fields_rejection() -> None:
    with pytest.raises(AdjudicationError) as captured:
        import_adjudication_records(
            [
                {
                    "left_record_key": "l",
                    "right_record_key": "r",
                    "decision": "match",
                    "confidence": 0.9,
                    "reviewer_id": "rev_1",
                    "timestamp": "2026-08-18T08:00:00Z",
                    "protocol_version": "v1",
                }
            ]
        )
    assert captured.value.code == "ML-ADJ-009"


def test_pair_references_with_null_bytes_rejection() -> None:
    sentinel = "SYNTHETIC\x00PRIVATE"
    with pytest.raises(AdjudicationError) as captured:
        AdjudicationRecord(
            event_id="evt_null",
            left_record_key=sentinel,
            right_record_key="right_normal",
            decision="match",
            confidence=0.9,
            reviewer_id="rev_1",
            timestamp=datetime(2026, 8, 18, tzinfo=UTC),
            protocol_version="v1",
        )
    assert captured.value.code == "ML-ADJ-009"
    assert "SYNTHETIC" not in str(captured.value)


def test_duplicate_event_id_rejection() -> None:
    rec = {
        "event_id": "evt_dup",
        "left_record_key": "l",
        "right_record_key": "r",
        "decision": "match",
        "confidence": 0.9,
        "reviewer_id": "rev_1",
        "timestamp": "2026-08-18T08:00:00Z",
        "protocol_version": "v1",
    }
    with pytest.raises(AdjudicationError) as captured:
        import_adjudication_records([rec, rec])
    assert captured.value.code == "ML-ADJ-009"
