from __future__ import annotations

import json
from pathlib import Path

import pytest

from mapel_linkage.governance.errors import SafeError, SafeErrorCode
from mapel_linkage.synthetic import (
    SyntheticGenerationConfig,
    generate_synthetic_bundle,
    write_synthetic_bundle,
)


def test_generator_is_seed_reproducible() -> None:
    config = SyntheticGenerationConfig(seed=42, entity_count=12)
    first = generate_synthetic_bundle(config)
    second = generate_synthetic_bundle(config)
    assert first == second
    assert repr(first.source_a[0]) == "<SyntheticRecord restricted>"
    assert first.source_a[0].record_key not in repr(first)


def test_truth_is_separate_from_model_input() -> None:
    bundle = generate_synthetic_bundle(SyntheticGenerationConfig(entity_count=12))
    assert not hasattr(bundle.source_a[0], "entity_key")
    assert not hasattr(bundle.source_b[0], "household_key")
    assert len(bundle.truth) == len(bundle.source_a) + len(bundle.source_b)


def test_generator_includes_required_edge_cases() -> None:
    bundle = generate_synthetic_bundle(
        SyntheticGenerationConfig(
            seed=7,
            entity_count=12,
            duplicate_count=2,
            left_only_count=2,
            right_only_count=2,
            competing_candidate_count=2,
            source_b_missing_rate=0.5,
        )
    )
    entity_counts: dict[str, int] = {}
    for truth in bundle.truth:
        if truth.dataset_id == "source_a":
            entity_counts[truth.entity_key] = entity_counts.get(truth.entity_key, 0) + 1
    assert any(count > 1 for count in entity_counts.values())
    assert any(
        record.label_value is None or record.date_value is None for record in bundle.source_b
    )
    assert any(record.record_key.startswith("AL") for record in bundle.source_a)
    assert any(record.record_key.startswith("BR") for record in bundle.source_b)
    assert any(record.record_key.startswith("BC") for record in bundle.source_b)


def test_bundle_writer_keeps_truth_in_a_distinct_file(tmp_path: Path) -> None:
    bundle = generate_synthetic_bundle(SyntheticGenerationConfig(seed=9, entity_count=10))
    paths = write_synthetic_bundle(tmp_path, bundle)
    assert {path.name for path in paths} == {
        "source_a.jsonl",
        "source_b.jsonl",
        "truth.jsonl",
        "provenance.json",
    }
    source_row = json.loads((tmp_path / "source_a.jsonl").read_text().splitlines()[0])
    assert "entity_key" not in source_row
    truth_row = json.loads((tmp_path / "truth.jsonl").read_text().splitlines()[0])
    assert "entity_key" in truth_row


def test_source_specific_corruption_changes_matched_records() -> None:
    bundle = generate_synthetic_bundle(
        SyntheticGenerationConfig(
            seed=12,
            entity_count=20,
            source_a_missing_rate=0.0,
            source_b_missing_rate=0.0,
            source_b_typo_rate=0.8,
            source_b_date_shift_rate=0.8,
        )
    )
    source_a_by_entity = {
        truth.entity_key: next(
            record for record in bundle.source_a if record.record_key == truth.record_key
        )
        for truth in bundle.truth
        if truth.dataset_id == "source_a" and truth.record_key.startswith("A0")
    }
    source_b_by_entity = {
        truth.entity_key: next(
            record for record in bundle.source_b if record.record_key == truth.record_key
        )
        for truth in bundle.truth
        if truth.dataset_id == "source_b" and truth.record_key.startswith("B0")
    }
    common = set(source_a_by_entity) & set(source_b_by_entity)
    assert any(
        (
            source_a_by_entity[entity].label_value != source_b_by_entity[entity].label_value
            or source_a_by_entity[entity].date_value != source_b_by_entity[entity].date_value
        )
        for entity in common
    )


def test_bundle_writer_translates_filesystem_error_without_path(tmp_path: Path) -> None:
    bundle = generate_synthetic_bundle(SyntheticGenerationConfig(entity_count=10))
    destination = tmp_path / "SYNTHETIC-PRIVATE-DESTINATION"
    destination.write_text("not a directory", encoding="utf-8")
    with pytest.raises(SafeError) as caught:
        write_synthetic_bundle(destination, bundle)
    assert caught.value.code == SafeErrorCode.SYNTHETIC_GENERATION
    assert str(destination) not in caught.value.render()


def test_bundle_writer_preserves_existing_fixtures_when_staging_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "source_a.jsonl"
    existing.write_text("existing-synthetic-fixture\n", encoding="utf-8")
    bundle = generate_synthetic_bundle(SyntheticGenerationConfig(entity_count=10))

    def fail_replace(self: Path, target: Path) -> Path:
        del self, target
        raise OSError

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(SafeError):
        write_synthetic_bundle(tmp_path, bundle)

    assert existing.read_text(encoding="utf-8") == "existing-synthetic-fixture\n"
