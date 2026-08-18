"""Restricted local review queue for uncertain relationship decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from mapel_linkage.configuration.models import OutputConfig
from mapel_linkage.decisions import RelationshipDecision
from mapel_linkage.domain.errors import AdjudicationError
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.paths import PathPolicy


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class ReviewQueueEntry:
    relationship_id: str
    source_record_ref: str = field(repr=False)
    target_record_ref: str | None = field(repr=False)
    relationship_status: str
    calibrated_probability: float | None
    candidate_rank: int | None
    probability_margin: float
    review_reason_codes: tuple[str, ...]
    model_version: str
    decision_rule_id: str
    assignment_method: str
    run_id: str

    def restricted_digest_payload(self) -> dict[str, object]:
        return {
            "relationship_id": self.relationship_id,
            "source_record_ref": self.source_record_ref,
            "target_record_ref": self.target_record_ref,
            "relationship_status": self.relationship_status,
            "calibrated_probability": self.calibrated_probability,
            "candidate_rank": self.candidate_rank,
            "probability_margin": self.probability_margin,
            "review_reason_codes": self.review_reason_codes,
            "model_version": self.model_version,
            "decision_rule_id": self.decision_rule_id,
            "assignment_method": self.assignment_method,
            "run_id": self.run_id,
        }

    def safe_summary(self) -> dict[str, object]:
        return {
            "relationship_id": self.relationship_id,
            "relationship_status": self.relationship_status,
            "review_reason_count": len(self.review_reason_codes),
            "model_version": self.model_version,
            "decision_rule_id": self.decision_rule_id,
            "assignment_method": self.assignment_method,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True, repr=False)
class ReviewQueue:
    entries: tuple[ReviewQueueEntry, ...] = field(repr=False)
    run_id: str
    queue_digest: str
    relationship_count: int
    review_required_count: int
    unresolved_count: int
    export_authority: str = "restricted_local_only"

    def __post_init__(self) -> None:
        expected_digest = _digest(
            {
                "run_id": self.run_id,
                "entries": [entry.restricted_digest_payload() for entry in self.entries],
            }
        )
        if (
            self.relationship_count != len(self.entries)
            or self.review_required_count
            != sum(entry.relationship_status == "review_required" for entry in self.entries)
            or self.unresolved_count
            != sum(entry.relationship_status == "unresolved" for entry in self.entries)
            or any(entry.run_id != self.run_id for entry in self.entries)
            or any(
                entry.relationship_status not in {"review_required", "unresolved"}
                or not entry.review_reason_codes
                for entry in self.entries
            )
            or self.queue_digest != expected_digest
        ):
            raise AdjudicationError("ML-ADJ-006", "Review queue integrity validation failed.")

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "run_id": self.run_id,
            "relationship_count": self.relationship_count,
            "review_required_count": self.review_required_count,
            "unresolved_count": self.unresolved_count,
            "queue_digest": self.queue_digest,
            "export_authority": self.export_authority,
        }


@dataclass(frozen=True, slots=True)
class WrittenReviewQueue:
    queue_path: Path = field(repr=False)
    manifest_path: Path = field(repr=False)
    queue_digest: str
    relationship_count: int

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "queue_digest": self.queue_digest,
            "relationship_count": self.relationship_count,
            "artifact_format": "restricted_jsonl",
        }


def build_review_queue(decisions: tuple[RelationshipDecision, ...]) -> ReviewQueue:
    if not decisions:
        raise AdjudicationError("ML-ADJ-001", "A review queue requires relationship decisions.")
    run_ids = {decision.run_id for decision in decisions}
    if len(run_ids) != 1:
        raise AdjudicationError("ML-ADJ-002", "Review decisions must belong to one run.")
    entries = tuple(
        ReviewQueueEntry(
            relationship_id=decision.relationship_id,
            source_record_ref=decision.source_record_ref,
            target_record_ref=decision.target_record_ref,
            relationship_status=decision.relationship_status,
            calibrated_probability=decision.calibrated_probability,
            candidate_rank=decision.candidate_rank,
            probability_margin=decision.probability_margin,
            review_reason_codes=decision.review_reason_codes,
            model_version=decision.model_version,
            decision_rule_id=decision.decision_rule_id,
            assignment_method=decision.assignment_method,
            run_id=decision.run_id,
        )
        for decision in decisions
        if decision.relationship_status in {"review_required", "unresolved"}
    )
    if any(not entry.review_reason_codes for entry in entries):
        raise AdjudicationError("ML-ADJ-005", "Every review entry requires reason codes.")
    payload = [entry.restricted_digest_payload() for entry in entries]
    queue_digest = _digest({"run_id": next(iter(run_ids)), "entries": payload})
    return ReviewQueue(
        entries=entries,
        run_id=next(iter(run_ids)),
        queue_digest=queue_digest,
        relationship_count=len(entries),
        review_required_count=sum(
            entry.relationship_status == "review_required" for entry in entries
        ),
        unresolved_count=sum(entry.relationship_status == "unresolved" for entry in entries),
    )


def _entry_mapping(entry: ReviewQueueEntry, output: OutputConfig) -> dict[str, object]:
    available: dict[str, object] = {
        "relationship_id": entry.relationship_id,
        "source_record_ref": entry.source_record_ref,
        "target_record_ref": entry.target_record_ref,
        "relationship_status": entry.relationship_status,
        "calibrated_probability": entry.calibrated_probability,
        "candidate_rank": entry.candidate_rank,
        "probability_margin": entry.probability_margin,
        "decision_rule_id": entry.decision_rule_id,
        "assignment_method": entry.assignment_method,
        "run_id": entry.run_id,
        "model_version": entry.model_version,
        "review_reason_codes": list(entry.review_reason_codes),
    }
    allowed = set(output.permitted_fields)
    if "review_reason_codes" not in allowed:
        raise AdjudicationError(
            "ML-ADJ-004",
            "Restricted review export must permit package-generated reason codes.",
        )
    return {name: value for name, value in available.items() if name in allowed}


def write_review_queue(
    *,
    queue: ReviewQueue,
    output: OutputConfig,
    queue_path: str,
    manifest_path: str,
    policy: PathPolicy,
) -> WrittenReviewQueue:
    destination = policy.resolve_output(queue_path)
    manifest_destination = policy.resolve_output(manifest_path)
    if destination == manifest_destination:
        raise AdjudicationError("ML-ADJ-007", "Review queue output paths must differ.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        queue_text = "".join(
            json.dumps(_entry_mapping(entry, output), sort_keys=True) + "\n"
            for entry in queue.entries
        )
        restricted_payload_digest = hashlib.sha256(queue_text.encode("utf-8")).hexdigest()
        atomic_write_text(destination, queue_text)
        atomic_write_text(
            manifest_destination,
            json.dumps(
                {
                    **queue.safe_summary(),
                    "restricted_payload_digest": restricted_payload_digest,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    except (OSError, TypeError, ValueError):
        raise AdjudicationError(
            "ML-ADJ-003", "A restricted review queue could not be written."
        ) from None
    return WrittenReviewQueue(
        destination, manifest_destination, queue.queue_digest, queue.relationship_count
    )
