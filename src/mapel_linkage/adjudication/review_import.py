"""Append-only adjudication decision ingestion and schema validation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from mapel_linkage.domain.errors import AdjudicationError

AdjudicationOutcome = Literal[
    "match",
    "nonmatch",
    "uncertain",
    "insufficient_information",
    "duplicate_review",
]

_ALLOWED_OUTCOMES: frozenset[str] = frozenset(
    {"match", "nonmatch", "uncertain", "insufficient_information", "duplicate_review"}
)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_digest(content: bytes | str) -> str:
    raw_bytes = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(raw_bytes).hexdigest()


def _parse_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise AdjudicationError("ML-ADJ-012", "Adjudication timestamp is empty.")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError as exc:
            raise AdjudicationError(
                "ML-ADJ-012", "Adjudication timestamp is not a valid ISO-8601 string."
            ) from exc
    raise AdjudicationError("ML-ADJ-012", "Adjudication timestamp must be a string or datetime.")


def _require_digest(value: str, *, code: str, message: str) -> None:
    if _DIGEST_PATTERN.fullmatch(value) is None:
        raise AdjudicationError(code, message)


def _require_component_digests(values: tuple[str, ...], *, code: str, message: str) -> None:
    if len(values) > 8 or len(values) != len(set(values)):
        raise AdjudicationError(code, message)
    for value in values:
        if _DIGEST_PATTERN.fullmatch(value) is None:
            raise AdjudicationError(code, message)


@dataclass(frozen=True, slots=True, repr=False)
class AdjudicationRecord:
    """An append-only human adjudication event whose pair references stay private."""

    event_id: str
    left_record_key: str = field(repr=False)
    right_record_key: str = field(repr=False)
    decision: AdjudicationOutcome
    confidence: float
    reviewer_id: str
    timestamp: datetime
    protocol_version: str
    superseded_event_id: str | None = None
    source_digest: str | None = None
    notes: str | None = field(default=None, repr=False)
    entity_component_digests: tuple[str, ...] = field(default=(), repr=False)
    household_component_digests: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not self.event_id or len(self.event_id) > 256 or "\x00" in self.event_id:
            raise AdjudicationError("ML-ADJ-009", "Adjudication event ID is invalid.")
        if (
            not self.left_record_key
            or not self.right_record_key
            or len(self.left_record_key) > 1024
            or len(self.right_record_key) > 1024
            or "\x00" in self.left_record_key
            or "\x00" in self.right_record_key
        ):
            raise AdjudicationError("ML-ADJ-009", "Adjudication pair reference is invalid.")
        if self.decision not in _ALLOWED_OUTCOMES:
            raise AdjudicationError("ML-ADJ-010", "Adjudication decision outcome is invalid.")
        if (
            not isinstance(self.confidence, (int, float))
            or math.isnan(self.confidence)
            or math.isinf(self.confidence)
            or self.confidence < 0.0
            or self.confidence > 1.0
        ):
            raise AdjudicationError(
                "ML-ADJ-011", "Adjudication confidence must be a float between 0.0 and 1.0."
            )
        if not self.reviewer_id or len(self.reviewer_id) > 256 or "\x00" in self.reviewer_id:
            raise AdjudicationError("ML-ADJ-012", "Adjudication reviewer ID is invalid.")
        if _IDENTIFIER_PATTERN.fullmatch(self.protocol_version) is None:
            raise AdjudicationError(
                "ML-ADJ-013", "Adjudication protocol version identifier is invalid."
            )
        if self.source_digest is not None:
            _require_digest(
                self.source_digest,
                code="ML-ADJ-009",
                message="Adjudication source digest is invalid.",
            )
        if self.superseded_event_id is not None and (
            not self.superseded_event_id
            or len(self.superseded_event_id) > 256
            or "\x00" in self.superseded_event_id
            or self.superseded_event_id == self.event_id
        ):
            raise AdjudicationError(
                "ML-ADJ-014", "Adjudication superseded event reference is invalid."
            )
        _require_component_digests(
            self.entity_component_digests,
            code="ML-ADJ-009",
            message="Adjudication entity component digest is invalid.",
        )
        _require_component_digests(
            self.household_component_digests,
            code="ML-ADJ-009",
            message="Adjudication household component digest is invalid.",
        )

    def pair_digest(self) -> str:
        """Deterministic pair digest hiding private record identifiers."""
        return hashlib.sha256(
            (self.left_record_key + "\x00" + self.right_record_key).encode("utf-8")
        ).hexdigest()

    def canonical_digest(self) -> str:
        """Deterministic cryptographic digest over the event payload."""
        return _digest(
            {
                "event_id": self.event_id,
                "pair_digest": self.pair_digest(),
                "decision": self.decision,
                "confidence": round(float(self.confidence), 6),
                "reviewer_id": self.reviewer_id,
                "timestamp": self.timestamp.isoformat(),
                "protocol_version": self.protocol_version,
                "superseded_event_id": self.superseded_event_id,
                "source_digest": self.source_digest,
            }
        )

    def safe_summary(self) -> dict[str, object]:
        """Aggregate summary hiding private record references and free-form notes."""
        return {
            "event_id": self.event_id,
            "pair_digest": self.pair_digest(),
            "decision": self.decision,
            "confidence": round(float(self.confidence), 6),
            "reviewer_id": self.reviewer_id,
            "timestamp": self.timestamp.isoformat(),
            "protocol_version": self.protocol_version,
            "superseded_event_id": self.superseded_event_id,
            "has_entity_provenance": bool(self.entity_component_digests),
            "has_household_provenance": bool(self.household_component_digests),
        }


@dataclass(frozen=True, slots=True, repr=False)
class ImportedAdjudicationBatch:
    """An imported batch of append-only human adjudication events with provenance."""

    records: tuple[AdjudicationRecord, ...] = field(repr=False)
    input_digest: str
    batch_id: str
    raw_record_count: int
    active_record_count: int
    superseded_event_count: int

    def __post_init__(self) -> None:
        _require_digest(
            self.input_digest,
            code="ML-ADJ-009",
            message="Imported batch input digest is invalid.",
        )
        if self.raw_record_count != len(self.records):
            raise AdjudicationError("ML-ADJ-009", "Imported batch record count mismatch.")

    def active_records(self) -> tuple[AdjudicationRecord, ...]:
        """Return non-superseded records after resolving append-only supersession chains."""
        superseded_ids: set[str] = set()
        supersedes_map: dict[str, str] = {}

        for rec in self.records:
            if rec.superseded_event_id:
                superseded_ids.add(rec.superseded_event_id)
                supersedes_map[rec.event_id] = rec.superseded_event_id

        # Verify no cycles exist in supersession chains
        for start_id in supersedes_map:
            visited = {start_id}
            curr = supersedes_map.get(start_id)
            while curr:
                if curr in visited:
                    raise AdjudicationError(
                        "ML-ADJ-014", "Circular supersession detected in adjudication records."
                    )
                visited.add(curr)
                curr = supersedes_map.get(curr)

        return tuple(rec for rec in self.records if rec.event_id not in superseded_ids)

    def by_pair(self) -> dict[str, tuple[AdjudicationRecord, ...]]:
        """Group active (non-superseded) records by pair digest."""
        groups: dict[str, list[AdjudicationRecord]] = {}
        for rec in self.active_records():
            groups.setdefault(rec.pair_digest(), []).append(rec)
        return {k: tuple(v) for k, v in groups.items()}

    def safe_summary(self) -> dict[str, object]:
        """Summary of imported batch without sensitive records."""
        return {
            "batch_id": self.batch_id,
            "input_digest": self.input_digest,
            "raw_record_count": self.raw_record_count,
            "active_record_count": self.active_record_count,
            "superseded_event_count": self.superseded_event_count,
        }


def _extract_string_tuple(val: Any) -> tuple[str, ...]:
    if not val:
        return ()
    if isinstance(val, (list, tuple)):
        return tuple(str(x).strip() for x in val if str(x).strip())
    if isinstance(val, str):
        text = val.strip()
        if not text:
            return ()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return tuple(str(x).strip() for x in parsed if str(x).strip())
            except json.JSONDecodeError:
                pass
        if ";" in text:
            return tuple(x.strip() for x in text.split(";") if x.strip())
        if "," in text:
            return tuple(x.strip() for x in text.split(",") if x.strip())
        return (text,)
    return ()


def _parse_record_dict(raw: dict[str, Any], row_idx: int) -> AdjudicationRecord:
    event_id = raw.get("event_id") or raw.get("adjudication_event_id") or raw.get("id")
    if not event_id or not isinstance(event_id, str):
        raise AdjudicationError(
            "ML-ADJ-009", f"Row {row_idx} is missing required 'event_id' field."
        )

    left_key = (
        raw.get("left_record_key")
        or raw.get("source_record_ref")
        or raw.get("left_id")
        or raw.get("source_ref")
        or raw.get("source_key")
    )
    right_key = (
        raw.get("right_record_key")
        or raw.get("target_record_ref")
        or raw.get("right_id")
        or raw.get("target_ref")
        or raw.get("target_key")
    )
    if not left_key or not right_key:
        raise AdjudicationError(
            "ML-ADJ-009", f"Row {row_idx} is missing required pair reference fields."
        )

    decision_raw = raw.get("decision") or raw.get("outcome") or raw.get("status")
    if not decision_raw or str(decision_raw) not in _ALLOWED_OUTCOMES:
        raise AdjudicationError(
            "ML-ADJ-010", f"Row {row_idx} has an invalid adjudication decision outcome."
        )
    decision: AdjudicationOutcome = str(decision_raw)  # type: ignore[assignment]

    confidence_raw = raw.get("confidence")
    if confidence_raw is None:
        raise AdjudicationError("ML-ADJ-011", f"Row {row_idx} is missing required confidence.")
    try:
        confidence = float(confidence_raw)
    except (ValueError, TypeError) as exc:
        raise AdjudicationError(
            "ML-ADJ-011", f"Row {row_idx} has non-numeric confidence value."
        ) from exc

    reviewer_id = raw.get("reviewer_id") or raw.get("reviewer") or raw.get("annotator_id")
    if not reviewer_id or not isinstance(reviewer_id, str):
        raise AdjudicationError(
            "ML-ADJ-012", f"Row {row_idx} is missing required 'reviewer_id' field."
        )

    timestamp_raw = raw.get("timestamp") or raw.get("created_at") or raw.get("adjudicated_at")
    if not timestamp_raw:
        raise AdjudicationError(
            "ML-ADJ-012", f"Row {row_idx} is missing required 'timestamp' field."
        )
    timestamp = _parse_timestamp(timestamp_raw)

    protocol_version = (
        raw.get("protocol_version")
        or raw.get("verification_protocol")
        or raw.get("protocol")
        or raw.get("version")
    )
    if not protocol_version or not isinstance(protocol_version, str):
        raise AdjudicationError(
            "ML-ADJ-013", f"Row {row_idx} is missing required 'protocol_version' field."
        )

    superseded_raw = (
        raw.get("superseded_event_id")
        or raw.get("supersedes_event_id")
        or raw.get("supersedes_label_id")
    )
    superseded_event_id = str(superseded_raw).strip() if superseded_raw else None

    source_digest_raw = raw.get("source_digest") or raw.get("source_artifact_digest")
    source_digest = str(source_digest_raw).strip() if source_digest_raw else None

    notes_raw = raw.get("notes") or raw.get("comment") or raw.get("annotation_notes")
    notes = str(notes_raw).strip() if notes_raw else None

    entity_components = _extract_string_tuple(
        raw.get("entity_component_digests") or raw.get("entity_components")
    )
    household_components = _extract_string_tuple(
        raw.get("household_component_digests") or raw.get("household_components")
    )

    return AdjudicationRecord(
        event_id=str(event_id).strip(),
        left_record_key=str(left_key).strip(),
        right_record_key=str(right_key).strip(),
        decision=decision,
        confidence=confidence,
        reviewer_id=str(reviewer_id).strip(),
        timestamp=timestamp,
        protocol_version=str(protocol_version).strip(),
        superseded_event_id=superseded_event_id,
        source_digest=source_digest,
        notes=notes,
        entity_component_digests=entity_components,
        household_component_digests=household_components,
    )


def import_adjudication_records(
    records: Iterable[dict[str, Any]],
    *,
    input_digest: str | None = None,
) -> ImportedAdjudicationBatch:
    """Build an ImportedAdjudicationBatch from an iterable of record dictionaries."""
    parsed: list[AdjudicationRecord] = []
    seen_event_ids: set[str] = set()

    for idx, raw in enumerate(records, start=1):
        if not isinstance(raw, dict):
            raise AdjudicationError("ML-ADJ-009", f"Record at index {idx} is not a valid object.")
        rec = _parse_record_dict(raw, idx)
        if rec.event_id in seen_event_ids:
            raise AdjudicationError(
                "ML-ADJ-009", f"Duplicate adjudication event ID '{rec.event_id}' in batch."
            )
        seen_event_ids.add(rec.event_id)
        parsed.append(rec)

    records_tuple = tuple(parsed)
    if input_digest is None:
        canonical_payload = [rec.canonical_digest() for rec in records_tuple]
        input_digest = _digest({"records": canonical_payload})

    superseded_ids = {rec.superseded_event_id for rec in records_tuple if rec.superseded_event_id}
    active_count = sum(1 for rec in records_tuple if rec.event_id not in superseded_ids)
    superseded_count = len(records_tuple) - active_count
    batch_id = hashlib.sha256(
        f"{input_digest}:{len(records_tuple)}:{active_count}".encode()
    ).hexdigest()[:16]

    return ImportedAdjudicationBatch(
        records=records_tuple,
        input_digest=input_digest,
        batch_id=batch_id,
        raw_record_count=len(records_tuple),
        active_record_count=active_count,
        superseded_event_count=superseded_count,
    )


def import_adjudications_from_jsonl(path_or_content: Path | str) -> ImportedAdjudicationBatch:
    """Ingest append-only human adjudication events from a JSONL file or string."""
    raw_content: str
    computed_digest: str

    if isinstance(path_or_content, Path) or (
        isinstance(path_or_content, str)
        and "\n" not in path_or_content
        and Path(path_or_content).exists()
    ):
        file_path = Path(path_or_content)
        try:
            raw_bytes = file_path.read_bytes()
            computed_digest = hashlib.sha256(raw_bytes).hexdigest()
            raw_content = raw_bytes.decode("utf-8")
        except OSError as exc:
            raise AdjudicationError(
                "ML-ADJ-008", f"Could not read adjudication JSONL file: {exc}"
            ) from exc
    else:
        raw_content = str(path_or_content)
        computed_digest = _content_digest(raw_content)

    dict_records: list[dict[str, Any]] = []
    for line_idx, line in enumerate(raw_content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise AdjudicationError(
                "ML-ADJ-009", f"Line {line_idx} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise AdjudicationError(
                "ML-ADJ-009", f"Line {line_idx} does not contain a JSON object."
            )
        dict_records.append(parsed)

    return import_adjudication_records(dict_records, input_digest=computed_digest)


def import_adjudications_from_csv(path_or_content: Path | str) -> ImportedAdjudicationBatch:
    """Ingest append-only human adjudication events from a CSV file or string."""
    raw_content: str
    computed_digest: str

    if isinstance(path_or_content, Path) or (
        isinstance(path_or_content, str)
        and "\n" not in path_or_content
        and Path(path_or_content).exists()
    ):
        file_path = Path(path_or_content)
        try:
            raw_bytes = file_path.read_bytes()
            computed_digest = hashlib.sha256(raw_bytes).hexdigest()
            raw_content = raw_bytes.decode("utf-8")
        except OSError as exc:
            raise AdjudicationError(
                "ML-ADJ-008", f"Could not read adjudication CSV file: {exc}"
            ) from exc
    else:
        raw_content = str(path_or_content)
        computed_digest = _content_digest(raw_content)

    reader = csv.DictReader(io.StringIO(raw_content))
    dict_records: list[dict[str, Any]] = []
    for row in reader:
        dict_records.append(dict(row))

    return import_adjudication_records(dict_records, input_digest=computed_digest)
