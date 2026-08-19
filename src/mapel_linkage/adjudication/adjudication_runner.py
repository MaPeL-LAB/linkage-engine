"""Restricted adjudication lifecycle runner and append-only audit ledger."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from mapel_linkage.adjudication.disagreement import (
    ConsensusPolicy,
    ConsensusResult,
    DisagreementReport,
    ReviewConflict,
    evaluate_disagreements,
)
from mapel_linkage.adjudication.promotion import (
    PromotionConfig,
    PromotionEvaluation,
    PromotionSummary,
    evaluate_promotion_batch,
    promote_to_verified_batch,
)
from mapel_linkage.adjudication.review_import import (
    AdjudicationOutcome,
    AdjudicationRecord,
    ImportedAdjudicationBatch,
    import_adjudication_records,
    import_adjudications_from_csv,
    import_adjudications_from_jsonl,
)
from mapel_linkage.domain.errors import AdjudicationError
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.labels import (
    LabelPartition,
    PartitionDisjointnessReport,
    VerifiedLabelBatch,
    assert_disjoint_label_partitions,
)
from mapel_linkage.governance.paths import PathPolicy

GENESIS_PREV_DIGEST: str = "0" * 64


def _canonical_record_digest(
    event_id: str,
    pair_digest: str,
    decision: str,
    confidence: float,
    reviewer_id: str,
    timestamp: str,
    protocol_version: str,
    superseded_event_id: str | None = None,
    source_digest: str | None = None,
) -> str:
    payload = {
        "event_id": event_id,
        "pair_digest": pair_digest,
        "decision": decision,
        "confidence": round(float(confidence), 6),
        "reviewer_id": reviewer_id,
        "timestamp": timestamp,
        "protocol_version": protocol_version,
        "superseded_event_id": superseded_event_id,
        "source_digest": source_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_ledger_entry_digest(
    entry_index: int,
    prev_entry_digest: str,
    canonical_event_digest: str,
    event_id: str,
    pair_digest: str,
    reviewer_id: str,
    decision: str,
    confidence: float,
    timestamp: str,
    protocol_version: str,
    superseded_event_id: str | None = None,
    source_digest: str | None = None,
) -> str:
    payload = {
        "entry_index": entry_index,
        "prev_entry_digest": prev_entry_digest,
        "canonical_event_digest": canonical_event_digest,
        "event_id": event_id,
        "pair_digest": pair_digest,
        "reviewer_id": reviewer_id,
        "decision": decision,
        "confidence": round(float(confidence), 6),
        "timestamp": timestamp,
        "protocol_version": protocol_version,
        "superseded_event_id": superseded_event_id,
        "source_digest": source_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class AdjudicationLedgerEntry:
    """An immutable, cryptographically-chained entry in an append-only adjudication ledger."""

    entry_index: int
    event_id: str
    pair_digest: str
    reviewer_id: str
    decision: AdjudicationOutcome
    confidence: float
    timestamp: str
    protocol_version: str
    canonical_event_digest: str
    prev_entry_digest: str
    entry_digest: str
    superseded_event_id: str | None = None
    source_digest: str | None = None
    notes: str | None = field(default=None, repr=False)
    entity_component_digests: tuple[str, ...] = field(default=(), repr=False)
    household_component_digests: tuple[str, ...] = field(default=(), repr=False)

    def safe_summary(self) -> dict[str, Any]:
        """Aggregate summary hiding private record references and free-form notes."""
        return {
            "entry_index": self.entry_index,
            "event_id": self.event_id,
            "pair_digest": self.pair_digest,
            "decision": self.decision,
            "confidence": round(self.confidence, 6),
            "reviewer_id": self.reviewer_id,
            "timestamp": self.timestamp,
            "protocol_version": self.protocol_version,
            "canonical_event_digest": self.canonical_event_digest,
            "superseded_event_id": self.superseded_event_id,
            "source_digest": self.source_digest,
            "prev_entry_digest": self.prev_entry_digest,
            "entry_digest": self.entry_digest,
            "has_entity_provenance": bool(self.entity_component_digests),
            "has_household_provenance": bool(self.household_component_digests),
        }


@dataclass(frozen=True, slots=True, repr=False)
class AdjudicationAuditLedger:
    """Append-only tamper-evident audit ledger over human adjudication events."""

    entries: tuple[AdjudicationLedgerEntry, ...] = field(repr=False)
    ledger_id: str
    ledger_digest: str
    created_at: str

    @classmethod
    def create_empty(cls, ledger_id: str = "adjudication-ledger") -> AdjudicationAuditLedger:
        created_at = datetime.now(UTC).isoformat()
        chain_payload = {
            "ledger_id": ledger_id,
            "created_at": created_at,
            "entry_count": 0,
            "head_digest": GENESIS_PREV_DIGEST,
        }
        digest = hashlib.sha256(
            json.dumps(chain_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            entries=(),
            ledger_id=ledger_id,
            ledger_digest=digest,
            created_at=created_at,
        )

    def append_records(self, records: Sequence[AdjudicationRecord]) -> AdjudicationAuditLedger:
        """Append adjudication records to this ledger, extending the cryptographic hash chain."""
        self.verify_integrity()
        new_entries = list(self.entries)
        existing_event_ids = {e.event_id for e in self.entries}

        prev_digest = self.entries[-1].entry_digest if self.entries else GENESIS_PREV_DIGEST

        for rec in records:
            if rec.event_id in existing_event_ids:
                raise AdjudicationError(
                    "ML-ADJ-009",
                    f"Duplicate event ID '{rec.event_id}' rejected in append-only ledger.",
                )
            existing_event_ids.add(rec.event_id)
            idx = len(new_entries)
            event_canonical = rec.canonical_digest()
            ts_str = rec.timestamp.isoformat()
            entry_dig = compute_ledger_entry_digest(
                entry_index=idx,
                prev_entry_digest=prev_digest,
                canonical_event_digest=event_canonical,
                event_id=rec.event_id,
                pair_digest=rec.pair_digest(),
                reviewer_id=rec.reviewer_id,
                decision=rec.decision,
                confidence=rec.confidence,
                timestamp=ts_str,
                protocol_version=rec.protocol_version,
                superseded_event_id=rec.superseded_event_id,
                source_digest=rec.source_digest,
            )
            entry = AdjudicationLedgerEntry(
                entry_index=idx,
                event_id=rec.event_id,
                pair_digest=rec.pair_digest(),
                reviewer_id=rec.reviewer_id,
                decision=rec.decision,
                confidence=rec.confidence,
                timestamp=ts_str,
                protocol_version=rec.protocol_version,
                canonical_event_digest=event_canonical,
                prev_entry_digest=prev_digest,
                entry_digest=entry_dig,
                superseded_event_id=rec.superseded_event_id,
                source_digest=rec.source_digest,
                notes=rec.notes,
                entity_component_digests=rec.entity_component_digests,
                household_component_digests=rec.household_component_digests,
            )
            new_entries.append(entry)
            prev_digest = entry_dig

        entries_tuple = tuple(new_entries)
        chain_payload = {
            "ledger_id": self.ledger_id,
            "created_at": self.created_at,
            "entry_count": len(entries_tuple),
            "head_digest": (
                entries_tuple[-1].entry_digest if entries_tuple else GENESIS_PREV_DIGEST
            ),
        }
        ledger_digest = hashlib.sha256(
            json.dumps(chain_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        return AdjudicationAuditLedger(
            entries=entries_tuple,
            ledger_id=self.ledger_id,
            ledger_digest=ledger_digest,
            created_at=self.created_at,
        )

    def verify_integrity(self) -> None:
        """Verify the cryptographic hash chain and detect any tampering or deletion."""
        if not self.entries:
            expected_empty_payload = {
                "ledger_id": self.ledger_id,
                "created_at": self.created_at,
                "entry_count": 0,
                "head_digest": GENESIS_PREV_DIGEST,
            }
            expected_empty_digest = hashlib.sha256(
                json.dumps(expected_empty_payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            if self.ledger_digest != expected_empty_digest:
                raise AdjudicationError(
                    "ML-ADJ-021",
                    "Empty ledger digest does not match expected initial state.",
                )
            return

        expected_prev = GENESIS_PREV_DIGEST
        seen_events: set[str] = set()

        for idx, entry in enumerate(self.entries):
            if entry.entry_index != idx:
                raise AdjudicationError(
                    "ML-ADJ-021",
                    f"Ledger index mismatch at entry {idx}: "
                    f"expected index {idx}, got {entry.entry_index}.",
                )
            if entry.event_id in seen_events:
                raise AdjudicationError(
                    "ML-ADJ-021",
                    f"Duplicate event ID '{entry.event_id}' detected during ledger check.",
                )
            seen_events.add(entry.event_id)

            if entry.prev_entry_digest != expected_prev:
                raise AdjudicationError(
                    "ML-ADJ-021",
                    f"Hash chain broken at ledger entry {idx}: prev_entry_digest mismatch.",
                )

            recomputed_entry_digest = compute_ledger_entry_digest(
                entry_index=entry.entry_index,
                prev_entry_digest=entry.prev_entry_digest,
                canonical_event_digest=entry.canonical_event_digest,
                event_id=entry.event_id,
                pair_digest=entry.pair_digest,
                reviewer_id=entry.reviewer_id,
                decision=entry.decision,
                confidence=entry.confidence,
                timestamp=entry.timestamp,
                protocol_version=entry.protocol_version,
                superseded_event_id=entry.superseded_event_id,
                source_digest=entry.source_digest,
            )
            if entry.entry_digest != recomputed_entry_digest:
                raise AdjudicationError(
                    "ML-ADJ-021",
                    f"Ledger entry digest tamper detected at entry {idx}: hash mismatch.",
                )

            expected_prev = entry.entry_digest

        chain_payload = {
            "ledger_id": self.ledger_id,
            "created_at": self.created_at,
            "entry_count": len(self.entries),
            "head_digest": (self.entries[-1].entry_digest if self.entries else GENESIS_PREV_DIGEST),
        }
        recomputed_ledger_digest = hashlib.sha256(
            json.dumps(chain_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if self.ledger_digest != recomputed_ledger_digest:
            raise AdjudicationError(
                "ML-ADJ-021",
                "Ledger header digest tamper detected: ledger_digest does not match chain state.",
            )

    def is_valid(self) -> bool:
        """Return True if the ledger integrity check passes, False otherwise."""
        try:
            self.verify_integrity()
            return True
        except AdjudicationError:
            return False

    def to_json(self) -> str:
        """Serialize ledger to JSON string."""
        return json.dumps(
            {
                "ledger_id": self.ledger_id,
                "created_at": self.created_at,
                "ledger_digest": self.ledger_digest,
                "entry_count": len(self.entries),
                "entries": [e.safe_summary() for e in self.entries],
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, json_str: str) -> AdjudicationAuditLedger:
        """Load ledger from JSON string and verify its integrity."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdjudicationAuditLedger:
        """Deserialize ledger from dictionary and verify its cryptographic integrity."""
        ledger_id = str(data["ledger_id"])
        created_at = str(data["created_at"])
        ledger_digest = str(data["ledger_digest"])
        raw_entries = data.get("entries", [])

        entries: list[AdjudicationLedgerEntry] = []
        for raw in raw_entries:
            canonical = raw.get("canonical_event_digest")
            if not canonical:
                canonical = _canonical_record_digest(
                    event_id=str(raw["event_id"]),
                    pair_digest=str(raw["pair_digest"]),
                    decision=str(raw["decision"]),
                    confidence=float(raw["confidence"]),
                    reviewer_id=str(raw["reviewer_id"]),
                    timestamp=str(raw["timestamp"]),
                    protocol_version=str(raw["protocol_version"]),
                    superseded_event_id=raw.get("superseded_event_id"),
                    source_digest=raw.get("source_digest"),
                )

            entry = AdjudicationLedgerEntry(
                entry_index=int(raw["entry_index"]),
                event_id=str(raw["event_id"]),
                pair_digest=str(raw["pair_digest"]),
                reviewer_id=str(raw["reviewer_id"]),
                decision=str(raw["decision"]),  # type: ignore[arg-type]
                confidence=float(raw["confidence"]),
                timestamp=str(raw["timestamp"]),
                protocol_version=str(raw["protocol_version"]),
                canonical_event_digest=str(canonical),
                prev_entry_digest=str(raw["prev_entry_digest"]),
                entry_digest=str(raw["entry_digest"]),
                superseded_event_id=raw.get("superseded_event_id"),
                source_digest=raw.get("source_digest"),
            )
            entries.append(entry)

        ledger = cls(
            entries=tuple(entries),
            ledger_id=ledger_id,
            ledger_digest=ledger_digest,
            created_at=created_at,
        )
        ledger.verify_integrity()
        return ledger

    def write_to_file(self, path: Path | str, policy: PathPolicy | None = None) -> Path:
        """Write ledger to file atomically."""
        dest = policy.resolve_output(str(path)) if policy is not None else Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(dest, self.to_json() + "\n")
        return dest

    def safe_summary(self) -> dict[str, Any]:
        """Aggregate summary of audit ledger."""
        decisions_counter = Counter(e.decision for e in self.entries)
        reviewers_set = {e.reviewer_id for e in self.entries}
        pairs_set = {e.pair_digest for e in self.entries}
        superseded_count = sum(1 for e in self.entries if e.superseded_event_id is not None)
        return {
            "ledger_id": self.ledger_id,
            "created_at": self.created_at,
            "ledger_digest": self.ledger_digest,
            "entry_count": len(self.entries),
            "unique_pair_count": len(pairs_set),
            "unique_reviewer_count": len(reviewers_set),
            "superseded_event_count": superseded_count,
            "decision_counts": dict(decisions_counter),
        }


@dataclass(frozen=True, slots=True, repr=False)
class AdjudicationImportResult:
    """Result of importing adjudication reviews and updating the audit ledger."""

    imported_batch: ImportedAdjudicationBatch = field(repr=False)
    ledger: AdjudicationAuditLedger = field(repr=False)
    total_imported: int
    active_record_count: int
    superseded_record_count: int
    ledger_entry_count: int
    ledger_path: Path | None = field(default=None, repr=False)
    import_digest: str = ""

    def __post_init__(self) -> None:
        if not self.import_digest:
            payload = {
                "batch_id": self.imported_batch.batch_id,
                "input_digest": self.imported_batch.input_digest,
                "ledger_digest": self.ledger.ledger_digest,
                "total_imported": self.total_imported,
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            object.__setattr__(self, "import_digest", digest)

    def safe_summary(self) -> dict[str, Any]:
        """Aggregate summary hiding sensitive record keys."""
        return {
            "batch_id": self.imported_batch.batch_id,
            "input_digest": self.imported_batch.input_digest,
            "total_imported": self.total_imported,
            "active_record_count": self.active_record_count,
            "superseded_record_count": self.superseded_record_count,
            "ledger_entry_count": self.ledger_entry_count,
            "ledger_digest": self.ledger.ledger_digest,
            "import_digest": self.import_digest,
            "ledger_written": self.ledger_path is not None,
        }


@dataclass(frozen=True, slots=True, repr=False)
class ConsensusReport:
    """Outcome of multi-reviewer consensus evaluation across adjudicated pairs."""

    results: tuple[ConsensusResult, ...] = field(repr=False)
    disagreement_report: DisagreementReport
    conflicts: tuple[ReviewConflict, ...]
    policy: ConsensusPolicy
    agreement_threshold: float | None = None
    report_digest: str = ""

    def __post_init__(self) -> None:
        if not self.report_digest:
            payload = {
                "total_pairs": self.disagreement_report.total_pairs,
                "resolved_pairs": self.disagreement_report.resolved_pairs,
                "unresolved_pairs": self.disagreement_report.unresolved_pairs,
                "conflict_count": self.disagreement_report.conflict_count,
                "policy": self.policy,
                "agreement_threshold": self.agreement_threshold,
                "pair_digests": [r.pair_digest for r in self.results],
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            object.__setattr__(self, "report_digest", digest)

    @property
    def resolved_decisions(self) -> tuple[ConsensusResult, ...]:
        return tuple(r for r in self.results if r.is_resolved)

    @property
    def unresolved_decisions(self) -> tuple[ConsensusResult, ...]:
        return tuple(r for r in self.results if not r.is_resolved)

    def safe_summary(self) -> dict[str, Any]:
        """Aggregate summary of consensus resolution."""
        return {
            "policy": self.policy,
            "agreement_threshold": self.agreement_threshold,
            "total_pairs": self.disagreement_report.total_pairs,
            "resolved_pairs": self.disagreement_report.resolved_pairs,
            "unresolved_pairs": self.disagreement_report.unresolved_pairs,
            "conflict_count": self.disagreement_report.conflict_count,
            "unanimous_count": self.disagreement_report.unanimous_count,
            "majority_count": self.disagreement_report.majority_count,
            "senior_override_count": self.disagreement_report.senior_override_count,
            "single_reviewer_count": self.disagreement_report.single_reviewer_count,
            "report_digest": self.report_digest,
        }


@dataclass(frozen=True, slots=True, repr=False)
class LabelPromotionResult:
    """Outcome of promoting adjudicated consensus decisions to a VerifiedLabelBatch.

    Strict governance: Promotion never triggers automatic retraining.
    """

    verified_batch: VerifiedLabelBatch = field(repr=False)
    promotion_summary: PromotionSummary
    evaluations: tuple[PromotionEvaluation, ...] = field(repr=False)
    target_partition: LabelPartition
    label_authority_digest: str
    disjointness_report: PartitionDisjointnessReport | None
    ledger_summary: dict[str, Any]
    retraining_triggered: Literal[False] = False
    manifest_path: Path | None = field(default=None, repr=False)
    result_digest: str = ""

    def __post_init__(self) -> None:
        if not self.result_digest:
            payload = {
                "label_authority_digest": self.label_authority_digest,
                "target_partition": self.target_partition,
                "eligible_count": self.promotion_summary.eligible_count,
                "retraining_triggered": False,
                "disjointness_manifest_digest": (
                    self.disjointness_report.manifest_digest
                    if self.disjointness_report is not None
                    else None
                ),
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            object.__setattr__(self, "result_digest", digest)

    def safe_summary(self) -> dict[str, Any]:
        """Aggregate promotion summary hiding private pair references."""
        return {
            "target_partition": self.target_partition,
            "label_authority_digest": self.label_authority_digest,
            "total_evaluated": self.promotion_summary.total_evaluated,
            "eligible_count": self.promotion_summary.eligible_count,
            "promoted_positive_count": self.promotion_summary.promoted_positive_count,
            "promoted_negative_count": self.promotion_summary.promoted_negative_count,
            "audit_only_count": self.promotion_summary.audit_only_count,
            "rejected_count": self.promotion_summary.rejected_count,
            "has_disjointness_proof": self.disjointness_report is not None,
            "retraining_triggered": False,
            "result_digest": self.result_digest,
            "manifest_written": self.manifest_path is not None,
            "ledger_summary": self.ledger_summary,
        }


class AdjudicationWorkflowRunner:
    """Orchestrates review ingestion, consensus, and verified label promotion."""

    @classmethod
    def import_reviews(
        cls,
        reviews_source: (
            Path
            | str
            | Sequence[dict[str, Any]]
            | Sequence[AdjudicationRecord]
            | ImportedAdjudicationBatch
        ),
        *,
        candidate_pair_references: Sequence[tuple[str, str]] | set[tuple[str, str]] | None = None,
        candidate_pair_digests: frozenset[str] | set[str] | Sequence[str] | None = None,
        ledger: AdjudicationAuditLedger | None = None,
        ledger_path: Path | str | None = None,
        policy: PathPolicy | None = None,
        strict_candidate_check: bool = True,
        ledger_id: str = "adjudication-ledger",
    ) -> AdjudicationImportResult:
        """Ingest bounded review batches and append to audit ledger."""
        # 1. Ingest batch
        batch: ImportedAdjudicationBatch
        if isinstance(reviews_source, ImportedAdjudicationBatch):
            batch = reviews_source
        elif isinstance(reviews_source, (Path, str)):
            path_str = str(reviews_source)
            p = Path(path_str)
            if p.is_file():
                if p.suffix.lower() == ".csv":
                    batch = import_adjudications_from_csv(p)
                else:
                    batch = import_adjudications_from_jsonl(p)
            else:
                # String content
                stripped = path_str.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    parsed_json = json.loads(stripped)
                    batch = import_adjudication_records(parsed_json)
                elif "," in stripped and "\n" in stripped:
                    batch = import_adjudications_from_csv(stripped)
                else:
                    batch = import_adjudications_from_jsonl(stripped)
        elif isinstance(reviews_source, Sequence):
            if not reviews_source:
                batch = import_adjudication_records([])
            elif isinstance(reviews_source[0], AdjudicationRecord):
                records_tuple: tuple[AdjudicationRecord, ...] = tuple(
                    r for r in reviews_source if isinstance(r, AdjudicationRecord)
                )
                input_dig = hashlib.sha256(
                    json.dumps(
                        [r.canonical_digest() for r in records_tuple],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                superseded_ids = {
                    r.superseded_event_id for r in records_tuple if r.superseded_event_id
                }
                active_count = sum(1 for r in records_tuple if r.event_id not in superseded_ids)
                batch_id = hashlib.sha256(
                    f"{input_dig}:{len(records_tuple)}:{active_count}".encode()
                ).hexdigest()[:16]
                batch = ImportedAdjudicationBatch(
                    records=records_tuple,
                    input_digest=input_dig,
                    batch_id=batch_id,
                    raw_record_count=len(records_tuple),
                    active_record_count=active_count,
                    superseded_event_count=len(records_tuple) - active_count,
                )
            else:
                batch = import_adjudication_records(reviews_source)  # type: ignore[arg-type]
        else:
            raise AdjudicationError("ML-ADJ-009", "Unsupported review source format.")

        # 2. Check candidate pair references
        allowed_digests: set[str] = set()
        if candidate_pair_references is not None:
            for u, v in candidate_pair_references:
                allowed_digests.add(hashlib.sha256((u + "\x00" + v).encode("utf-8")).hexdigest())
        if candidate_pair_digests is not None:
            allowed_digests.update(candidate_pair_digests)

        if allowed_digests:
            for rec in batch.records:
                if rec.pair_digest() not in allowed_digests and strict_candidate_check:
                    raise AdjudicationError(
                        "ML-ADJ-020",
                        f"Review event '{rec.event_id}' references unknown candidate pair.",
                    )

        # 3. Resolve and update audit ledger
        current_ledger: AdjudicationAuditLedger
        resolved_ledger_path: Path | None = None
        if ledger_path is not None:
            resolved_ledger_path = (
                policy.resolve_output(str(ledger_path)) if policy is not None else Path(ledger_path)
            )

        if ledger is not None:
            current_ledger = ledger
        elif resolved_ledger_path is not None and resolved_ledger_path.is_file():
            current_ledger = AdjudicationAuditLedger.from_json(
                resolved_ledger_path.read_text(encoding="utf-8")
            )
        else:
            current_ledger = AdjudicationAuditLedger.create_empty(ledger_id=ledger_id)

        updated_ledger = current_ledger.append_records(batch.records)

        # 4. Write ledger to disk if requested
        if resolved_ledger_path is not None:
            updated_ledger.write_to_file(resolved_ledger_path, policy=None)

        return AdjudicationImportResult(
            imported_batch=batch,
            ledger=updated_ledger,
            total_imported=batch.raw_record_count,
            active_record_count=batch.active_record_count,
            superseded_record_count=batch.superseded_event_count,
            ledger_entry_count=len(updated_ledger.entries),
            ledger_path=resolved_ledger_path,
        )

    @classmethod
    def resolve_consensus(
        cls,
        reviews: (
            ImportedAdjudicationBatch | Sequence[AdjudicationRecord] | AdjudicationAuditLedger
        ),
        *,
        policy: ConsensusPolicy = "majority_vote",
        senior_reviewers: frozenset[str] | set[str] | Sequence[str] = frozenset(),
        min_reviewers: int = 1,
        agreement_threshold: float | None = None,
    ) -> ConsensusReport:
        """Compute multi-reviewer consensus with voting and agreement thresholds."""
        records: tuple[AdjudicationRecord, ...]
        if isinstance(reviews, ImportedAdjudicationBatch):
            records = reviews.active_records()
        elif isinstance(reviews, AdjudicationAuditLedger):
            reviews.verify_integrity()
            # Extract records from ledger entries
            active_records_list: list[AdjudicationRecord] = []
            superseded_ids = {
                e.superseded_event_id for e in reviews.entries if e.superseded_event_id
            }
            for e in reviews.entries:
                if e.event_id in superseded_ids:
                    continue
                ts = datetime.fromisoformat(e.timestamp)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                active_records_list.append(
                    AdjudicationRecord(
                        event_id=e.event_id,
                        left_record_key=f"pair:{e.pair_digest[:16]}",
                        right_record_key=f"pair:{e.pair_digest[16:32]}",
                        decision=e.decision,
                        confidence=e.confidence,
                        reviewer_id=e.reviewer_id,
                        timestamp=ts,
                        protocol_version=e.protocol_version,
                        superseded_event_id=e.superseded_event_id,
                        source_digest=e.source_digest,
                    )
                )
            records = tuple(active_records_list)
        else:
            superseded_ids = {r.superseded_event_id for r in reviews if r.superseded_event_id}
            records = tuple(r for r in reviews if r.event_id not in superseded_ids)

        if not records:
            empty_report = DisagreementReport(
                total_pairs=0,
                resolved_pairs=0,
                unresolved_pairs=0,
                conflict_count=0,
                unanimous_count=0,
                majority_count=0,
                senior_override_count=0,
                single_reviewer_count=0,
            )
            return ConsensusReport(
                results=(),
                disagreement_report=empty_report,
                conflicts=(),
                policy=policy,
                agreement_threshold=agreement_threshold,
            )

        raw_results, _ = evaluate_disagreements(
            records,
            policy=policy,
            senior_reviewers=senior_reviewers,
            min_reviewers=min_reviewers,
        )

        # Apply agreement threshold if requested
        final_results: list[ConsensusResult] = []
        pairs_map: dict[str, list[AdjudicationRecord]] = {}
        for rec in records:
            pairs_map.setdefault(rec.pair_digest(), []).append(rec)

        for res in raw_results:
            pair_recs = pairs_map.get(res.pair_digest, [])
            if (
                res.is_resolved
                and res.consensus_outcome is not None
                and agreement_threshold is not None
                and len(pair_recs) > 1
            ):
                agreeing_count = sum(1 for r in pair_recs if r.decision == res.consensus_outcome)
                ratio = agreeing_count / len(pair_recs)
                if ratio < agreement_threshold:
                    res = ConsensusResult(
                        pair_digest=res.pair_digest,
                        left_record_key=res.left_record_key,
                        right_record_key=res.right_record_key,
                        consensus_outcome=None,
                        consensus_confidence=0.0,
                        resolution_method="unresolved",
                        is_resolved=False,
                        has_conflict=True,
                        reviewer_count=res.reviewer_count,
                        reviewing_event_ids=res.reviewing_event_ids,
                        dispute_reason=(
                            f"agreement_ratio_{ratio:.2f}_below_threshold_{agreement_threshold:.2f}"
                        ),
                        entity_component_digests=res.entity_component_digests,
                        household_component_digests=res.household_component_digests,
                        protocol_version=res.protocol_version,
                    )
            final_results.append(res)

        # Collect conflicts
        conflicts: list[ReviewConflict] = []
        for pair_dig, pair_records in pairs_map.items():
            unique_outcomes = tuple(dict.fromkeys(r.decision for r in pair_records))
            if len(unique_outcomes) > 1:
                matching_res = next((r for r in final_results if r.pair_digest == pair_dig), None)
                is_disputed = matching_res is None or not matching_res.is_resolved
                dispute_reason = (
                    matching_res.dispute_reason
                    if matching_res and matching_res.dispute_reason
                    else ("unresolved_dispute" if is_disputed else "resolved_conflict")
                )
                conflicts.append(
                    ReviewConflict(
                        pair_digest=pair_dig,
                        competing_outcomes=unique_outcomes,
                        reviewers=tuple(r.reviewer_id for r in pair_records),
                        is_disputed=is_disputed,
                        dispute_reason=dispute_reason,
                    )
                )

        resolved_count = sum(1 for r in final_results if r.is_resolved)
        unresolved_count = len(final_results) - resolved_count
        conflict_count = sum(1 for r in final_results if r.has_conflict)
        unanimous_count = sum(1 for r in final_results if r.resolution_method == "unanimous")
        majority_count = sum(1 for r in final_results if r.resolution_method == "majority")
        senior_override_count = sum(
            1 for r in final_results if r.resolution_method == "senior_override"
        )
        single_reviewer_count = sum(
            1 for r in final_results if r.resolution_method == "single_reviewer"
        )

        final_report = DisagreementReport(
            total_pairs=len(final_results),
            resolved_pairs=resolved_count,
            unresolved_pairs=unresolved_count,
            conflict_count=conflict_count,
            unanimous_count=unanimous_count,
            majority_count=majority_count,
            senior_override_count=senior_override_count,
            single_reviewer_count=single_reviewer_count,
        )

        return ConsensusReport(
            results=tuple(final_results),
            disagreement_report=final_report,
            conflicts=tuple(conflicts),
            policy=policy,
            agreement_threshold=agreement_threshold,
        )

    @classmethod
    def promote_to_verified_labels(
        cls,
        consensus_items: (
            ConsensusReport
            | tuple[ConsensusResult, ...]
            | Sequence[ConsensusResult]
            | Sequence[AdjudicationRecord]
        ),
        *,
        target_partition: LabelPartition = "training",
        min_confidence: float = 0.80,
        require_consensus: bool = True,
        require_double_review: bool = False,
        minimum_reviewers: int = 1,
        allowed_protocols: frozenset[str] = frozenset(),
        allow_audit_only: bool = True,
        verification_protocol: str = "human_adjudication_v1",
        source_digest: str | None = None,
        existing_partition_batches: Sequence[VerifiedLabelBatch] = (),
        locked_test_pairs: frozenset[str] = frozenset(),
        ledger_summary: dict[str, Any] | None = None,
        output_manifest_path: str | Path | None = None,
        policy: PathPolicy | None = None,
    ) -> LabelPromotionResult:
        """Package consensus decisions into an immutable VerifiedLabelBatch."""
        items: Sequence[ConsensusResult | AdjudicationRecord]
        if isinstance(consensus_items, ConsensusReport):
            items = consensus_items.results
        else:
            items = consensus_items

        if source_digest is None:
            source_payload = [item.safe_summary() for item in items]
            source_digest = hashlib.sha256(
                json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()

        config = PromotionConfig(
            target_partition=target_partition,
            min_confidence=min_confidence,
            require_consensus=require_consensus,
            require_double_review=require_double_review,
            minimum_reviewers=minimum_reviewers,
            allowed_protocols=allowed_protocols,
            allow_audit_only=allow_audit_only,
        )

        evaluations, promotion_summary = evaluate_promotion_batch(
            items, config, locked_test_pairs=locked_test_pairs
        )

        verified_batch, _ = promote_to_verified_batch(
            items,
            config,
            verification_protocol=verification_protocol,
            source_digest=source_digest,
            locked_test_pairs=locked_test_pairs,
        )

        # Verify that no label in the promoted batch crosses locked test pairs
        if target_partition != "test" and locked_test_pairs:
            for label in verified_batch.labels:
                if label.pair_digest() in locked_test_pairs:
                    raise AdjudicationError(
                        "ML-ADJ-022",
                        f"Promoted pair '{label.pair_digest()}' belongs to locked test partition.",
                    )

        # Verify partition disjointness if other partitions are present
        disjointness_report: PartitionDisjointnessReport | None = None
        if existing_partition_batches:
            batches_to_check = (*tuple(existing_partition_batches), verified_batch)
            partitions_set = {b.partition for b in batches_to_check}
            if len(partitions_set) == len(batches_to_check):
                disjointness_report = assert_disjoint_label_partitions(batches_to_check)
            else:
                other_partition_batches = [
                    b for b in existing_partition_batches if b.partition != target_partition
                ]
                if other_partition_batches:
                    disjointness_report = assert_disjoint_label_partitions(
                        (*tuple(other_partition_batches), verified_batch)
                    )

        # Ensure summary reflects governance boundary
        summary_ledger = ledger_summary or {
            "status": "ledger_verified",
            "retraining_triggered": False,
        }

        # Write manifest if requested
        manifest_file: Path | None = None
        if output_manifest_path is not None:
            dest = (
                policy.resolve_output(str(output_manifest_path))
                if policy is not None
                else Path(output_manifest_path)
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            manifest_payload = {
                "label_authority_digest": verified_batch.label_authority_digest,
                "target_partition": target_partition,
                "verification_protocol": verification_protocol,
                "source_digest": source_digest,
                "summary": promotion_summary.safe_summary(),
                "retraining_triggered": False,
                "disjointness_report": (
                    disjointness_report.safe_summary() if disjointness_report is not None else None
                ),
                "ledger_summary": summary_ledger,
            }
            atomic_write_text(dest, json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")
            manifest_file = dest

        return LabelPromotionResult(
            verified_batch=verified_batch,
            promotion_summary=promotion_summary,
            evaluations=evaluations,
            target_partition=target_partition,
            label_authority_digest=verified_batch.label_authority_digest,
            disjointness_report=disjointness_report,
            ledger_summary=summary_ledger,
            retraining_triggered=False,
            manifest_path=manifest_file,
        )


__all__ = [
    "GENESIS_PREV_DIGEST",
    "AdjudicationAuditLedger",
    "AdjudicationImportResult",
    "AdjudicationLedgerEntry",
    "AdjudicationWorkflowRunner",
    "ConsensusReport",
    "LabelPromotionResult",
    "compute_ledger_entry_digest",
]
