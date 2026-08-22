"""Fail-closed, allow-listed migration for aggregate versioned artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from mapel_linkage.domain.errors import ArtifactMigrationError
from mapel_linkage.governance.atomic import atomic_create_bytes
from mapel_linkage.governance.manifests import RunManifest, _LegacyRunManifestV01
from mapel_linkage.governance.paths import PathPolicy

_MAX_ARTIFACT_BYTES = 1_048_576
_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_RUN_MANIFEST_KIND: Literal["run_manifest"] = "run_manifest"
_RUN_MANIFEST_SOURCE_VERSION: Literal["0.1"] = "0.1"
_RUN_MANIFEST_TARGET_VERSION: Literal["1"] = "1"
_RUN_MANIFEST_TRANSFORMATION: Literal["run_manifest_0_1_to_1"] = "run_manifest_0_1_to_1"


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ArtifactMigrationError(
                "ML-MIGRATE-003",
                "The source artifact schema is invalid.",
            )
        payload[key] = value
    return payload


def _load_json_object(payload: bytes) -> dict[str, object]:
    try:
        decoded = payload.decode("utf-8")
        raw = json.loads(decoded, object_pairs_hook=_reject_duplicate_keys)
    except ArtifactMigrationError:
        raise
    except (UnicodeError, TypeError, ValueError):
        raise ArtifactMigrationError(
            "ML-MIGRATE-002",
            "The source artifact is not valid bounded UTF-8 JSON.",
        ) from None
    if not isinstance(raw, dict):
        raise ArtifactMigrationError(
            "ML-MIGRATE-003",
            "The source artifact schema is invalid.",
        )
    return raw


def _read_bounded_file(path: Path, *, source: bool) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise ArtifactMigrationError(
                "ML-MIGRATE-001",
                "A migration artifact is unavailable or path-unsafe.",
            )
        size = path.stat().st_size
        if size < 2 or size > _MAX_ARTIFACT_BYTES:
            raise ArtifactMigrationError(
                "ML-MIGRATE-002" if source else "ML-MIGRATE-008",
                (
                    "The source artifact exceeds its aggregate size boundary."
                    if source
                    else "An existing migration target conflicts with the approved plan."
                ),
            )
        return path.read_bytes()
    except ArtifactMigrationError:
        raise
    except OSError:
        raise ArtifactMigrationError(
            "ML-MIGRATE-001",
            "A migration artifact is unavailable or path-unsafe.",
        ) from None


class ArtifactMigrationPlan(BaseModel):
    """Digest-bound dry-run plan with no filesystem paths or artifact values."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True
    )

    schema_version: Literal["1"] = "1"
    artifact_kind: Literal["run_manifest"]
    source_schema_version: Literal["0.1"]
    target_schema_version: Literal["1"]
    transformation: Literal["run_manifest_0_1_to_1"]
    source_digest: Annotated[str, Field(pattern=_DIGEST_PATTERN)]
    target_digest: Annotated[str, Field(pattern=_DIGEST_PATTERN)]
    source_size_bytes: Annotated[StrictInt, Field(ge=2, le=_MAX_ARTIFACT_BYTES)]
    target_size_bytes: Annotated[StrictInt, Field(ge=2, le=_MAX_ARTIFACT_BYTES)]
    migration_authority: Literal["none"] = "none"
    release_authority: Literal["none"] = "none"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"
    report_classification: Literal["aggregate_only"] = "aggregate_only"

    @property
    def plan_digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return _digest(payload)

    def safe_summary(self) -> dict[str, str | int | bool]:
        return {
            "status": "planned",
            "plan_digest": self.plan_digest,
            "artifact_kind": self.artifact_kind,
            "source_schema_version": self.source_schema_version,
            "target_schema_version": self.target_schema_version,
            "transformation": self.transformation,
            "source_digest": self.source_digest,
            "target_digest": self.target_digest,
            "source_size_bytes": self.source_size_bytes,
            "target_size_bytes": self.target_size_bytes,
            "source_write_performed": False,
            "target_write_performed": False,
            "migration_authority": self.migration_authority,
            "release_authority": self.release_authority,
            "decision_authority": self.decision_authority,
            "assignment_authority": self.assignment_authority,
            "merge_authority": self.merge_authority,
            "operational_validity": self.operational_validity,
            "report_classification": self.report_classification,
        }


class ArtifactMigrationResult(BaseModel):
    """Aggregate execution result that retains every authority boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True
    )

    status: Literal["written", "already_present"]
    plan_digest: Annotated[str, Field(pattern=_DIGEST_PATTERN)]
    artifact_kind: Literal["run_manifest"]
    source_schema_version: Literal["0.1"]
    target_schema_version: Literal["1"]
    source_digest: Annotated[str, Field(pattern=_DIGEST_PATTERN)]
    target_digest: Annotated[str, Field(pattern=_DIGEST_PATTERN)]
    source_preserved: Literal[True] = True
    source_reloaded_for_rollback: Literal[True] = True
    target_reloaded: Literal[True] = True
    migration_authority: Literal["none"] = "none"
    release_authority: Literal["none"] = "none"
    decision_authority: Literal["none"] = "none"
    assignment_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"
    operational_validity: Literal["not_established"] = "not_established"
    report_classification: Literal["aggregate_only"] = "aggregate_only"

    def safe_summary(self) -> dict[str, str | bool]:
        return self.model_dump(mode="json")


def _migrate_run_manifest_0_1_to_1(payload: bytes) -> bytes:
    raw = _load_json_object(payload)
    try:
        legacy = _LegacyRunManifestV01.model_validate(raw)
        target_payload = legacy.model_dump(mode="json")
        target_payload["schema_version"] = _RUN_MANIFEST_TARGET_VERSION
        target = RunManifest.model_validate(target_payload)
        serialized = _canonical_json(target.model_dump(mode="json")).encode("utf-8")
        reloaded = RunManifest.model_validate_json(serialized)
    except ValidationError:
        raise ArtifactMigrationError(
            "ML-MIGRATE-003",
            "The source artifact schema is invalid.",
        ) from None
    if reloaded != target:
        raise ArtifactMigrationError(
            "ML-MIGRATE-005",
            "The migrated target failed canonical reload verification.",
        )
    return serialized


type _MigrationKey = tuple[str, str, str]
type _MigrationTransform = Callable[[bytes], bytes]

_MIGRATIONS: MappingProxyType[_MigrationKey, _MigrationTransform] = MappingProxyType(
    {
        (
            _RUN_MANIFEST_KIND,
            _RUN_MANIFEST_SOURCE_VERSION,
            _RUN_MANIFEST_TARGET_VERSION,
        ): _migrate_run_manifest_0_1_to_1,
    }
)


def _materialize_target(
    *, payload: bytes, artifact_kind: str, target_schema_version: str
) -> tuple[Literal["0.1"], Literal["run_manifest_0_1_to_1"], bytes]:
    raw = _load_json_object(payload)
    source_version = raw.get("schema_version")
    if not isinstance(source_version, str):
        raise ArtifactMigrationError(
            "ML-MIGRATE-003",
            "The source artifact schema is invalid.",
        )
    key = (artifact_kind, source_version, target_schema_version)
    transform = _MIGRATIONS.get(key)
    if transform is None:
        raise ArtifactMigrationError(
            "ML-MIGRATE-004",
            "The requested source-to-target artifact migration is not allow-listed.",
        )
    return _RUN_MANIFEST_SOURCE_VERSION, _RUN_MANIFEST_TRANSFORMATION, transform(payload)


def build_artifact_migration_policy(project_root: Path) -> PathPolicy:
    """Build the fixed local artifact/private path envelope for migration."""

    try:
        root = project_root.resolve(strict=True)
    except OSError:
        raise ArtifactMigrationError(
            "ML-MIGRATE-001",
            "The migration project root is unavailable or path-unsafe.",
        ) from None
    if root in {Path(root.anchor).resolve(strict=False), Path.home().resolve(strict=False)}:
        raise ArtifactMigrationError(
            "ML-MIGRATE-001",
            "A broad migration project root is not permitted.",
        )
    return PathPolicy.build(
        project_root=root,
        configured_input_roots=("artifacts", "private"),
        configured_output_roots=("artifacts", "private"),
        host_input_roots=(root / "artifacts", root / "private"),
        host_output_roots=(root / "artifacts", root / "private"),
    )


def plan_artifact_migration(
    *,
    source_path: str,
    target_path: str,
    artifact_kind: Literal["run_manifest"],
    target_schema_version: Literal["1"],
    policy: PathPolicy,
) -> ArtifactMigrationPlan:
    """Perform a no-write dry run and return its immutable digest-bound plan."""

    source = policy.resolve_input(source_path)
    target = policy.resolve_output(target_path)
    if source == target:
        raise ArtifactMigrationError(
            "ML-MIGRATE-006",
            "Source and target artifacts must use distinct managed paths.",
        )
    source_payload = _read_bounded_file(source, source=True)
    source_version, transformation, target_payload = _materialize_target(
        payload=source_payload,
        artifact_kind=artifact_kind,
        target_schema_version=target_schema_version,
    )
    try:
        plan = ArtifactMigrationPlan(
            artifact_kind=artifact_kind,
            source_schema_version=source_version,
            target_schema_version=target_schema_version,
            transformation=transformation,
            source_digest=_digest(source_payload),
            target_digest=_digest(target_payload),
            source_size_bytes=len(source_payload),
            target_size_bytes=len(target_payload),
        )
    except ValidationError:
        raise ArtifactMigrationError(
            "ML-MIGRATE-004",
            "The requested source-to-target artifact migration is not allow-listed.",
        ) from None
    if target.exists():
        existing = _read_bounded_file(target, source=False)
        if existing != target_payload:
            raise ArtifactMigrationError(
                "ML-MIGRATE-008",
                "An existing migration target conflicts with the approved plan.",
            )
    return plan


def serialize_artifact_migration_plan(plan: ArtifactMigrationPlan) -> str:
    """Serialize a canonical plan for separate reviewed execution."""

    payload = plan.model_dump(mode="json")
    payload["plan_digest"] = plan.plan_digest
    return _canonical_json(payload)


def deserialize_artifact_migration_plan(payload: str) -> ArtifactMigrationPlan:
    """Strictly load a canonical plan without accepting drift or unknown fields."""

    if len(payload.encode("utf-8")) > 65_536:
        raise ArtifactMigrationError(
            "ML-MIGRATE-007",
            "The migration plan is invalid or does not match the dry run.",
        )
    try:
        raw = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
        if not isinstance(raw, dict) or payload != _canonical_json(raw):
            raise ValueError
        declared_digest = raw.pop("plan_digest")
        plan = ArtifactMigrationPlan.model_validate(raw)
    except (ArtifactMigrationError, KeyError, TypeError, ValueError, ValidationError):
        raise ArtifactMigrationError(
            "ML-MIGRATE-007",
            "The migration plan is invalid or does not match the dry run.",
        ) from None
    if declared_digest != plan.plan_digest:
        raise ArtifactMigrationError(
            "ML-MIGRATE-007",
            "The migration plan is invalid or does not match the dry run.",
        )
    return plan


def load_artifact_migration_plan(*, plan_path: str, policy: PathPolicy) -> ArtifactMigrationPlan:
    """Load a bounded plan from the managed local artifact envelope."""

    path = policy.resolve_input(plan_path)
    payload = _read_bounded_file(path, source=True)
    try:
        text = payload.decode("utf-8")
    except UnicodeError:
        raise ArtifactMigrationError(
            "ML-MIGRATE-007",
            "The migration plan is invalid or does not match the dry run.",
        ) from None
    return deserialize_artifact_migration_plan(text)


def execute_artifact_migration(
    *,
    source_path: str,
    target_path: str,
    plan: ArtifactMigrationPlan,
    policy: PathPolicy,
) -> ArtifactMigrationResult:
    """Execute exactly one prior dry-run plan without replacing any artifact."""

    recomputed = plan_artifact_migration(
        source_path=source_path,
        target_path=target_path,
        artifact_kind=plan.artifact_kind,
        target_schema_version=plan.target_schema_version,
        policy=policy,
    )
    if recomputed != plan or recomputed.plan_digest != plan.plan_digest:
        raise ArtifactMigrationError(
            "ML-MIGRATE-007",
            "The migration plan is invalid or does not match the dry run.",
        )

    source = policy.resolve_input(source_path)
    target = policy.resolve_output(target_path)
    source_payload = _read_bounded_file(source, source=True)
    _, _, target_payload = _materialize_target(
        payload=source_payload,
        artifact_kind=plan.artifact_kind,
        target_schema_version=plan.target_schema_version,
    )

    status: Literal["written", "already_present"]
    if target.exists():
        existing = _read_bounded_file(target, source=False)
        if existing != target_payload:
            raise ArtifactMigrationError(
                "ML-MIGRATE-008",
                "An existing migration target conflicts with the approved plan.",
            )
        status = "already_present"
    else:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target = policy.resolve_output(target_path)
            atomic_create_bytes(target, target_payload)
            status = "written"
        except FileExistsError:
            existing = _read_bounded_file(target, source=False)
            if existing != target_payload:
                raise ArtifactMigrationError(
                    "ML-MIGRATE-008",
                    "An existing migration target conflicts with the approved plan.",
                ) from None
            status = "already_present"
        except ArtifactMigrationError:
            raise
        except OSError:
            raise ArtifactMigrationError(
                "ML-MIGRATE-009",
                "The migration target could not be created safely.",
            ) from None

    written = _read_bounded_file(target, source=False)
    retained_source = _read_bounded_file(source, source=True)
    try:
        RunManifest.model_validate_json(written)
        _LegacyRunManifestV01.model_validate(_load_json_object(retained_source))
    except ValidationError:
        raise ArtifactMigrationError(
            "ML-MIGRATE-005",
            "The migrated target or retained source failed reload verification.",
        ) from None
    if (
        written != target_payload
        or _digest(written) != plan.target_digest
        or retained_source != source_payload
        or _digest(retained_source) != plan.source_digest
    ):
        raise ArtifactMigrationError(
            "ML-MIGRATE-005",
            "The migrated target or retained source failed digest verification.",
        )
    return ArtifactMigrationResult(
        status=status,
        plan_digest=plan.plan_digest,
        artifact_kind=plan.artifact_kind,
        source_schema_version=plan.source_schema_version,
        target_schema_version=plan.target_schema_version,
        source_digest=plan.source_digest,
        target_digest=plan.target_digest,
    )


__all__ = [
    "ArtifactMigrationPlan",
    "ArtifactMigrationResult",
    "build_artifact_migration_policy",
    "deserialize_artifact_migration_plan",
    "execute_artifact_migration",
    "load_artifact_migration_plan",
    "plan_artifact_migration",
    "serialize_artifact_migration_plan",
]
