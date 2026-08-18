from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mapel_linkage.adjudication import build_review_queue, write_review_queue
from mapel_linkage.configuration.models import OutputConfig
from mapel_linkage.decisions import RelationshipDecision, RelationshipStatus
from mapel_linkage.domain.errors import AdjudicationError
from mapel_linkage.governance.paths import PathPolicy


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def decision(source: str, status: RelationshipStatus) -> RelationshipDecision:
    target = None if status in {"unresolved", "no_match"} else f"target-{source}"
    rule = f"rule_{status}"
    relationship_id = hashlib.sha256(
        json.dumps(
            {
                "run_id": "a" * 32,
                "source": source,
                "target": target or "NO_MATCH",
                "status": status,
                "decision_rule_id": rule,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return RelationshipDecision(
        relationship_id=relationship_id,
        source_dataset_id="source_a",
        target_dataset_id="source_b",
        source_record_ref=source,
        target_record_ref=target,
        relationship_status=status,
        model_family="xgboost",
        model_version="v1",
        calibrated_probability=(None if status in {"unresolved", "no_match"} else 0.72),
        candidate_rank=(None if status in {"unresolved", "no_match"} else 1),
        probability_margin=0.04,
        decision_rule_id=rule,
        assignment_method="ortools_min_cost_flow",
        assignment_constraint="one_to_one",
        anchor_rule_ids=(),
        candidate_rule_ids=("block_a",),
        run_id="a" * 32,
        configuration_digest=digest("configuration"),
        feature_schema_digest=digest("features"),
        non_sensitive_provenance=(
            ("candidate_search_complete", "true"),
            ("calibration_valid", "true"),
            ("assignment_changed_top1", "false"),
        ),
        created_at=datetime(2026, 8, 17, tzinfo=UTC),
        review_reason_codes=(
            ("review_probability_region",)
            if status == "review_required"
            else (("unresolved_insufficient_probability",) if status == "unresolved" else ())
        ),
    )


def test_review_queue_contains_only_uncertain_decisions_and_hides_refs(tmp_path: Path) -> None:
    sentinel = "SYNTHETIC-PRIVATE-SOURCE"
    queue = build_review_queue(
        (
            decision("confirmed", "confirmed"),
            decision(sentinel, "review_required"),
            decision("unresolved", "unresolved"),
            decision("no-match", "no_match"),
        )
    )
    assert queue.relationship_count == 2
    assert sentinel not in repr(queue)
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    output = OutputConfig.model_validate(
        {
            "restricted_directory": "private/outputs",
            "permitted_fields": [
                "relationship_id",
                "source_record_ref",
                "target_record_ref",
                "relationship_status",
                "calibrated_probability",
                "candidate_rank",
                "probability_margin",
                "decision_rule_id",
                "assignment_method",
                "run_id",
                "model_version",
                "review_reason_codes",
            ],
            "permitted_variable_values": [],
        }
    )
    written = write_review_queue(
        queue=queue,
        output=output,
        queue_path="private/outputs/review.jsonl",
        manifest_path="artifacts/runs/review.manifest.json",
        policy=policy,
    )
    rows = [json.loads(line) for line in written.queue_path.read_text().splitlines()]
    assert len(rows) == 2
    assert all("review_reason_codes" in row for row in rows)
    assert all("model_family" not in row for row in rows)
    manifest = json.loads(written.manifest_path.read_text())
    assert manifest["relationship_count"] == 2
    assert (
        manifest["restricted_payload_digest"]
        == hashlib.sha256(written.queue_path.read_bytes()).hexdigest()
    )


def test_review_queue_rejects_missing_reason_code_permission(tmp_path: Path) -> None:
    queue = build_review_queue((decision("review", "review_required"),))
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    output = OutputConfig.model_validate(
        {
            "restricted_directory": "private/outputs",
            "permitted_fields": ["relationship_id", "relationship_status"],
            "permitted_variable_values": [],
        }
    )
    with pytest.raises(AdjudicationError) as captured:
        write_review_queue(
            queue=queue,
            output=output,
            queue_path="private/outputs/review.jsonl",
            manifest_path="artifacts/runs/review.manifest.json",
            policy=policy,
        )
    assert captured.value.code == "ML-ADJ-004"


def test_review_queue_rejects_colliding_output_paths(tmp_path: Path) -> None:
    queue = build_review_queue((decision("review", "review_required"),))
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    output = OutputConfig.model_validate(
        {
            "restricted_directory": "private/outputs",
            "permitted_fields": ["relationship_id", "review_reason_codes"],
            "permitted_variable_values": [],
        }
    )
    with pytest.raises(AdjudicationError, match="ML-ADJ-007"):
        write_review_queue(
            queue=queue,
            output=output,
            queue_path="private/outputs/review.json",
            manifest_path="private/outputs/review.json",
            policy=policy,
        )
