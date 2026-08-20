"""Deterministic synthetic linkage inputs with separately held truth."""

from __future__ import annotations

import json
import random
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from mapel_linkage.governance.errors import SafeError, SafeErrorCode

_GENERATOR_VERSION = "0.1"


class SyntheticGenerationConfig(BaseModel):
    """Safe parameters for repository and CI synthetic data generation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    seed: Annotated[StrictInt, Field(ge=0, le=4294967295)] = 20260816
    entity_count: Annotated[StrictInt, Field(ge=8, le=100_000)] = 24
    left_only_count: Annotated[StrictInt, Field(ge=1, le=10_000)] = 2
    right_only_count: Annotated[StrictInt, Field(ge=1, le=10_000)] = 2
    duplicate_count: Annotated[StrictInt, Field(ge=1, le=10_000)] = 2
    right_duplicate_count: Annotated[StrictInt, Field(ge=0, le=10_000)] = 0
    competing_candidate_count: Annotated[StrictInt, Field(ge=1, le=10_000)] = 2
    source_a_missing_rate: Annotated[StrictFloat, Field(ge=0.0, le=0.5)] = 0.05
    source_b_missing_rate: Annotated[StrictFloat, Field(ge=0.0, le=0.8)] = 0.20
    source_b_typo_rate: Annotated[StrictFloat, Field(ge=0.0, le=0.8)] = 0.35
    source_b_date_shift_rate: Annotated[StrictFloat, Field(ge=0.0, le=0.8)] = 0.20


@dataclass(frozen=True, slots=True, repr=False)
class SyntheticRecord:
    """One generated input row; its representation deliberately hides values."""

    record_key: str = field(repr=False)
    label_value: str | None = field(repr=False)
    date_value: str | None = field(repr=False)
    group_value: str | None = field(repr=False)

    def __repr__(self) -> str:
        return "<SyntheticRecord restricted>"

    def as_mapping(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True, repr=False)
class SyntheticTruthRecord:
    """Test-only entity truth kept outside linkage model inputs."""

    dataset_id: str = field(repr=False)
    record_key: str = field(repr=False)
    entity_key: str = field(repr=False)
    household_key: str = field(repr=False)

    def __repr__(self) -> str:
        return "<SyntheticTruthRecord restricted>"

    def as_mapping(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SyntheticProvenance:
    generator_version: str
    seed: int
    entity_count: int
    source_a_count: int
    source_b_count: int
    truth_record_count: int
    left_only_count: int
    right_only_count: int
    duplicate_count: int
    right_duplicate_count: int
    competing_candidate_count: int
    source_a_missing_rate: float
    source_b_missing_rate: float
    source_b_typo_rate: float
    source_b_date_shift_rate: float


@dataclass(frozen=True, slots=True, repr=False)
class SyntheticBundle:
    source_a: tuple[SyntheticRecord, ...] = field(repr=False)
    source_b: tuple[SyntheticRecord, ...] = field(repr=False)
    truth: tuple[SyntheticTruthRecord, ...] = field(repr=False)
    provenance: SyntheticProvenance

    def __repr__(self) -> str:
        return (
            "SyntheticBundle("
            f"source_a_count={len(self.source_a)}, "
            f"source_b_count={len(self.source_b)}, "
            f"truth_record_count={len(self.truth)}, "
            f"seed={self.provenance.seed})"
        )


def matches_synthetic_fixture_layout(*, source_format: str, record_id_column: str) -> bool:
    """Validate the package-owned generated-fixture IO mapping contract."""

    return source_format == "jsonl" and record_id_column == "record_key"


_SYLLABLES = (
    "beka",
    "daro",
    "feni",
    "guma",
    "hilo",
    "jari",
    "keto",
    "luma",
    "mavi",
    "noro",
    "peta",
    "riso",
    "suna",
    "tavi",
    "welo",
    "zari",
)


def _mutate_label(value: str, rng: random.Random) -> str:
    if len(value) < 2:
        return value + "x"
    position = rng.randrange(len(value) - 1)
    characters = list(value)
    characters[position], characters[position + 1] = (
        characters[position + 1],
        characters[position],
    )
    return "".join(characters)


def _maybe_missing(value: str, rate: float, rng: random.Random) -> str | None:
    return None if rng.random() < rate else value


def generate_synthetic_bundle(
    config: SyntheticGenerationConfig | None = None,
) -> SyntheticBundle:
    """Generate two source tables and a distinct truth table."""

    spec = config or SyntheticGenerationConfig()
    rng = random.Random(spec.seed)
    source_a: list[SyntheticRecord] = []
    source_b: list[SyntheticRecord] = []
    truth: list[SyntheticTruthRecord] = []
    bases: list[tuple[str, str, str, str]] = []

    for index in range(spec.entity_count):
        entity_key = f"E{index:06d}"
        household_key = f"H{index // 3:05d}"
        label = f"{rng.choice(_SYLLABLES)}-{index:05d}"
        base_date = date(1980 + (index % 30), 1 + (index % 12), 1 + (index % 27))
        date_text = base_date.isoformat()
        group = f"G{index % 7:02d}"
        bases.append((entity_key, household_key, label, date_text))

        key_a = f"A{index:06d}"
        record_a = SyntheticRecord(
            record_key=key_a,
            label_value=_maybe_missing(label, spec.source_a_missing_rate, rng),
            date_value=date_text,
            group_value=group,
        )
        source_a.append(record_a)
        truth.append(SyntheticTruthRecord("source_a", key_a, entity_key, household_key))

        label_b = label
        if rng.random() < spec.source_b_typo_rate:
            label_b = _mutate_label(label_b, rng)
        date_b = base_date
        if rng.random() < spec.source_b_date_shift_rate:
            date_b += timedelta(days=rng.choice((-2, -1, 1, 2)))
        key_b = f"B{index:06d}"
        record_b = SyntheticRecord(
            record_key=key_b,
            label_value=_maybe_missing(label_b, spec.source_b_missing_rate, rng),
            date_value=_maybe_missing(date_b.isoformat(), spec.source_b_missing_rate / 2, rng),
            group_value=group,
        )
        source_b.append(record_b)
        truth.append(SyntheticTruthRecord("source_b", key_b, entity_key, household_key))

    for index in range(min(spec.duplicate_count, len(bases))):
        entity_key, household_key, label, date_text = bases[index]
        key = f"AD{index:05d}"
        source_a.append(
            SyntheticRecord(
                record_key=key,
                label_value=_mutate_label(label, rng),
                date_value=date_text,
                group_value=f"G{index % 7:02d}",
            )
        )
        truth.append(SyntheticTruthRecord("source_a", key, entity_key, household_key))

    for index in range(min(spec.right_duplicate_count, len(bases))):
        entity_key, household_key, label, date_text = bases[index]
        key = f"BD{index:05d}"
        source_b.append(
            SyntheticRecord(
                record_key=key,
                label_value=_mutate_label(label, rng),
                date_value=date_text,
                group_value=f"G{index % 7:02d}",
            )
        )
        truth.append(SyntheticTruthRecord("source_b", key, entity_key, household_key))

    for index in range(spec.left_only_count):
        entity_key = f"LA{index:05d}"
        household_key = f"LH{index:04d}"
        key = f"AL{index:05d}"
        source_a.append(
            SyntheticRecord(
                record_key=key,
                label_value=f"leftonly-{index:05d}",
                date_value=date(1970 + index % 30, 6, 1 + index % 20).isoformat(),
                group_value=f"LG{index % 3}",
            )
        )
        truth.append(SyntheticTruthRecord("source_a", key, entity_key, household_key))

    for index in range(spec.right_only_count):
        entity_key = f"RB{index:05d}"
        household_key = f"RH{index:04d}"
        key = f"BR{index:05d}"
        source_b.append(
            SyntheticRecord(
                record_key=key,
                label_value=f"rightonly-{index:05d}",
                date_value=date(1990 + index % 20, 7, 1 + index % 20).isoformat(),
                group_value=f"RG{index % 3}",
            )
        )
        truth.append(SyntheticTruthRecord("source_b", key, entity_key, household_key))

    for index in range(min(spec.competing_candidate_count, len(bases))):
        _, _, label, date_text = bases[index]
        entity_key = f"CB{index:05d}"
        household_key = f"CH{index:04d}"
        key = f"BC{index:05d}"
        source_b.append(
            SyntheticRecord(
                record_key=key,
                label_value=label,
                date_value=date_text,
                group_value=f"CG{index % 2}",
            )
        )
        truth.append(SyntheticTruthRecord("source_b", key, entity_key, household_key))

    source_a.sort(key=lambda record: record.record_key)
    source_b.sort(key=lambda record: record.record_key)
    truth.sort(key=lambda record: (record.dataset_id, record.record_key))
    provenance = SyntheticProvenance(
        generator_version=_GENERATOR_VERSION,
        seed=spec.seed,
        entity_count=spec.entity_count,
        source_a_count=len(source_a),
        source_b_count=len(source_b),
        truth_record_count=len(truth),
        left_only_count=spec.left_only_count,
        right_only_count=spec.right_only_count,
        duplicate_count=min(spec.duplicate_count, len(bases)),
        right_duplicate_count=min(spec.right_duplicate_count, len(bases)),
        competing_candidate_count=min(spec.competing_candidate_count, len(bases)),
        source_a_missing_rate=spec.source_a_missing_rate,
        source_b_missing_rate=spec.source_b_missing_rate,
        source_b_typo_rate=spec.source_b_typo_rate,
        source_b_date_shift_rate=spec.source_b_date_shift_rate,
    )
    return SyntheticBundle(tuple(source_a), tuple(source_b), tuple(truth), provenance)


def _json_lines(records: Iterable[Mapping[str, object]]) -> str:
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def write_synthetic_bundle(directory: Path, bundle: SyntheticBundle) -> tuple[Path, ...]:
    """Write local generated fixtures while translating filesystem errors safely."""

    source_a_path = directory / "source_a.jsonl"
    source_b_path = directory / "source_b.jsonl"
    truth_path = directory / "truth.jsonl"
    provenance_path = directory / "provenance.json"
    temporary_paths: list[Path] = []
    try:
        payloads = (
            (source_a_path, _json_lines(record.as_mapping() for record in bundle.source_a)),
            (source_b_path, _json_lines(record.as_mapping() for record in bundle.source_b)),
            (truth_path, _json_lines(record.as_mapping() for record in bundle.truth)),
            (
                provenance_path,
                json.dumps(asdict(bundle.provenance), indent=2, sort_keys=True) + "\n",
            ),
        )
        directory.mkdir(parents=True, exist_ok=True)
        staged: list[tuple[Path, Path]] = []
        for destination, text in payloads:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=directory,
                delete=False,
            ) as handle:
                handle.write(text)
                temporary = Path(handle.name)
            temporary_paths.append(temporary)
            staged.append((temporary, destination))
        for temporary, destination in staged:
            temporary.replace(destination)
            temporary_paths.remove(temporary)
    except (OSError, TypeError, ValueError):
        for temporary in temporary_paths:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise SafeError(
            SafeErrorCode.SYNTHETIC_GENERATION,
            "Synthetic fixtures could not be written.",
        ) from None
    return source_a_path, source_b_path, truth_path, provenance_path
