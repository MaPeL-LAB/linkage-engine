from __future__ import annotations

import json
from pathlib import Path

import pytest

from mapel_linkage.cli.main import main
from mapel_linkage.domain.errors import ArtifactMigrationError
from mapel_linkage.governance.artifact_migration import (
    build_artifact_migration_policy,
    deserialize_artifact_migration_plan,
    execute_artifact_migration,
    plan_artifact_migration,
    serialize_artifact_migration_plan,
)
from mapel_linkage.governance.errors import SafeError
from mapel_linkage.governance.manifests import RunManifest
from mapel_linkage.governance.paths import PathPolicy


def _legacy_payload() -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "run_id": "a" * 32,
        "created_at": "2026-08-22T12:00:00Z",
        "status": "synthetic_validated",
        "engine_version": "0.2.0.dev3",
        "configuration_digest": "b" * 64,
        "registry_digest": "c" * 64,
        "random_seed": 20260816,
        "python_version": "3.12.11",
        "platform": "Darwin-arm64",
        "process_hash_seed": "0",
        "package_versions": {
            "mapel-linkage-engine": "0.2.0.dev3",
            "pydantic": "2.11.9",
        },
        "dataset_count": 2,
        "variable_count": 5,
    }


def _canonical(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _prepare(tmp_path: Path) -> tuple[bytes, PathPolicy]:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (tmp_path / "private").mkdir()
    source = artifacts / "legacy-run-manifest.json"
    source_bytes = _canonical(_legacy_payload()).encode("utf-8")
    source.write_bytes(source_bytes)
    return source_bytes, build_artifact_migration_policy(tmp_path)


def test_plan_execute_reload_rollback_and_idempotence_are_digest_bound(tmp_path: Path) -> None:
    source_bytes, policy = _prepare(tmp_path)
    target = tmp_path / "artifacts" / "migrated-run-manifest.json"

    plan = plan_artifact_migration(
        source_path="artifacts/legacy-run-manifest.json",
        target_path="artifacts/migrated-run-manifest.json",
        artifact_kind="run_manifest",
        target_schema_version="1",
        policy=policy,
    )

    assert not target.exists()
    assert plan.source_schema_version == "0.1"
    assert plan.target_schema_version == "1"
    assert plan.migration_authority == "none"
    assert plan.release_authority == "none"
    assert plan.decision_authority == "none"
    assert plan.assignment_authority == "none"
    assert plan.merge_authority == "none"
    assert plan.operational_validity == "not_established"
    assert deserialize_artifact_migration_plan(serialize_artifact_migration_plan(plan)) == plan

    first = execute_artifact_migration(
        source_path="artifacts/legacy-run-manifest.json",
        target_path="artifacts/migrated-run-manifest.json",
        plan=plan,
        policy=policy,
    )
    assert first.status == "written"
    assert first.source_preserved is True
    assert first.source_reloaded_for_rollback is True
    assert first.target_reloaded is True
    assert (tmp_path / "artifacts" / "legacy-run-manifest.json").read_bytes() == source_bytes

    target_text = target.read_text(encoding="utf-8")
    migrated = RunManifest.model_validate_json(target_text)
    assert migrated.schema_version == "1"
    assert target_text == _canonical(migrated.model_dump(mode="json"))
    source_meaning = _legacy_payload()
    source_meaning.pop("schema_version")
    target_meaning = migrated.model_dump(mode="json")
    target_meaning.pop("schema_version")
    assert target_meaning == source_meaning

    second = execute_artifact_migration(
        source_path="artifacts/legacy-run-manifest.json",
        target_path="artifacts/migrated-run-manifest.json",
        plan=plan,
        policy=policy,
    )
    assert second.status == "already_present"
    assert second.target_digest == first.target_digest


def test_stale_plan_and_conflicting_target_fail_without_overwrite(tmp_path: Path) -> None:
    _, policy = _prepare(tmp_path)
    plan = plan_artifact_migration(
        source_path="artifacts/legacy-run-manifest.json",
        target_path="artifacts/migrated-run-manifest.json",
        artifact_kind="run_manifest",
        target_schema_version="1",
        policy=policy,
    )
    source = tmp_path / "artifacts" / "legacy-run-manifest.json"
    drifted = _legacy_payload()
    drifted["dataset_count"] = 3
    source.write_text(_canonical(drifted), encoding="utf-8")

    with pytest.raises(ArtifactMigrationError, match="ML-MIGRATE-007"):
        execute_artifact_migration(
            source_path="artifacts/legacy-run-manifest.json",
            target_path="artifacts/migrated-run-manifest.json",
            plan=plan,
            policy=policy,
        )
    assert not (tmp_path / "artifacts" / "migrated-run-manifest.json").exists()

    source.write_text(_canonical(_legacy_payload()), encoding="utf-8")
    conflict = tmp_path / "artifacts" / "migrated-run-manifest.json"
    conflict.write_text('{"conflict":true}\n', encoding="utf-8")
    with pytest.raises(ArtifactMigrationError, match="ML-MIGRATE-008"):
        execute_artifact_migration(
            source_path="artifacts/legacy-run-manifest.json",
            target_path="artifacts/migrated-run-manifest.json",
            plan=plan,
            policy=policy,
        )
    assert conflict.read_text(encoding="utf-8") == '{"conflict":true}\n'


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": "2"},
        {"unexpected": "SYNTHETIC-RESTRICTED-VALUE"},
        {"run_id": "SYNTHETIC-RESTRICTED-VALUE"},
    ],
)
def test_unsupported_or_invalid_source_fails_without_echoing_values(
    tmp_path: Path,
    mutation: dict[str, object],
) -> None:
    _, policy = _prepare(tmp_path)
    payload = _legacy_payload()
    payload.update(mutation)
    source = tmp_path / "artifacts" / "legacy-run-manifest.json"
    source.write_text(_canonical(payload), encoding="utf-8")

    with pytest.raises(ArtifactMigrationError) as caught:
        plan_artifact_migration(
            source_path="artifacts/legacy-run-manifest.json",
            target_path="artifacts/migrated-run-manifest.json",
            artifact_kind="run_manifest",
            target_schema_version="1",
            policy=policy,
        )
    rendered = str(caught.value)
    assert "SYNTHETIC-RESTRICTED-VALUE" not in rendered
    assert str(tmp_path) not in rendered
    assert not (tmp_path / "artifacts" / "migrated-run-manifest.json").exists()


def test_duplicate_keys_and_noncanonical_or_tampered_plans_fail_closed(tmp_path: Path) -> None:
    _, policy = _prepare(tmp_path)
    source = tmp_path / "artifacts" / "legacy-run-manifest.json"
    source.write_text('{"schema_version":"0.1","schema_version":"0.1"}\n', encoding="utf-8")
    with pytest.raises(ArtifactMigrationError, match="ML-MIGRATE-003"):
        plan_artifact_migration(
            source_path="artifacts/legacy-run-manifest.json",
            target_path="artifacts/migrated-run-manifest.json",
            artifact_kind="run_manifest",
            target_schema_version="1",
            policy=policy,
        )

    source.write_text(_canonical(_legacy_payload()), encoding="utf-8")
    plan = plan_artifact_migration(
        source_path="artifacts/legacy-run-manifest.json",
        target_path="artifacts/migrated-run-manifest.json",
        artifact_kind="run_manifest",
        target_schema_version="1",
        policy=policy,
    )
    canonical = serialize_artifact_migration_plan(plan)
    with pytest.raises(ArtifactMigrationError, match="ML-MIGRATE-007"):
        deserialize_artifact_migration_plan(canonical.rstrip())
    tampered = json.loads(canonical)
    tampered["source_digest"] = "f" * 64
    with pytest.raises(ArtifactMigrationError, match="ML-MIGRATE-007"):
        deserialize_artifact_migration_plan(_canonical(tampered))


def test_path_escape_symlinks_and_same_path_are_rejected(tmp_path: Path) -> None:
    _, policy = _prepare(tmp_path)
    external = tmp_path / "external.json"
    external.write_text("unchanged\n", encoding="utf-8")
    link = tmp_path / "artifacts" / "linked-source.json"
    link.symlink_to(external)

    with pytest.raises(SafeError):
        plan_artifact_migration(
            source_path="artifacts/linked-source.json",
            target_path="artifacts/migrated-run-manifest.json",
            artifact_kind="run_manifest",
            target_schema_version="1",
            policy=policy,
        )
    target_link = tmp_path / "artifacts" / "linked-target.json"
    target_link.symlink_to(external)
    with pytest.raises(SafeError):
        plan_artifact_migration(
            source_path="artifacts/legacy-run-manifest.json",
            target_path="artifacts/linked-target.json",
            artifact_kind="run_manifest",
            target_schema_version="1",
            policy=policy,
        )
    with pytest.raises(SafeError):
        plan_artifact_migration(
            source_path="../external.json",
            target_path="artifacts/migrated-run-manifest.json",
            artifact_kind="run_manifest",
            target_schema_version="1",
            policy=policy,
        )
    with pytest.raises(ArtifactMigrationError, match="ML-MIGRATE-006"):
        plan_artifact_migration(
            source_path="artifacts/legacy-run-manifest.json",
            target_path="artifacts/legacy-run-manifest.json",
            artifact_kind="run_manifest",
            target_schema_version="1",
            policy=policy,
        )
    assert external.read_text(encoding="utf-8") == "unchanged\n"


def test_source_size_boundary_fails_before_parsing_or_target_write(tmp_path: Path) -> None:
    _, policy = _prepare(tmp_path)
    source = tmp_path / "artifacts" / "legacy-run-manifest.json"
    source.write_bytes(b"{" + (b"x" * 1_048_576) + b"}")

    with pytest.raises(ArtifactMigrationError, match="ML-MIGRATE-002") as caught:
        plan_artifact_migration(
            source_path="artifacts/legacy-run-manifest.json",
            target_path="artifacts/migrated-run-manifest.json",
            artifact_kind="run_manifest",
            target_schema_version="1",
            policy=policy,
        )
    assert str(tmp_path) not in str(caught.value)
    assert not (tmp_path / "artifacts" / "migrated-run-manifest.json").exists()


def test_cli_requires_separate_dry_run_plan_and_emits_only_aggregate_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _prepare(tmp_path)
    common = [
        "migrate-artifact",
        "--project-root",
        str(tmp_path),
        "--source",
        "artifacts/legacy-run-manifest.json",
        "--target",
        "artifacts/migrated-run-manifest.json",
        "--artifact-kind",
        "run_manifest",
        "--target-version",
        "1",
    ]
    assert main([*common, "--dry-run"]) == 0
    plan_text = capsys.readouterr().out
    plan_payload = json.loads(plan_text)
    assert plan_payload["source_schema_version"] == "0.1"
    assert plan_payload["target_schema_version"] == "1"
    assert str(tmp_path) not in plan_text
    assert not (tmp_path / "artifacts" / "migrated-run-manifest.json").exists()

    plan_path = tmp_path / "artifacts" / "migration-plan.json"
    plan_path.write_text(plan_text, encoding="utf-8")
    assert main([*common, "--plan-file", "artifacts/migration-plan.json"]) == 0
    result_text = capsys.readouterr().out
    result = json.loads(result_text)
    assert result["status"] == "written"
    assert result["source_preserved"] is True
    assert result["migration_authority"] == "none"
    assert result["release_authority"] == "none"
    assert result["operational_validity"] == "not_established"
    assert str(tmp_path) not in result_text


def test_current_run_manifest_producer_uses_target_schema() -> None:
    assert RunManifest.model_fields["schema_version"].default == "1"
