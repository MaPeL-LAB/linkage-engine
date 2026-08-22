"""Privacy-safe run manifests."""

from __future__ import annotations

import json
import os
import platform
import sys
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from mapel_linkage import __version__
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.errors import SafeError, SafeErrorCode
from mapel_linkage.governance.paths import PathPolicy


class _RunManifestFields(BaseModel):
    """Shared strictly validated aggregate fields across manifest versions."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True
    )

    run_id: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{32}$")]
    created_at: datetime
    status: Annotated[StrictStr, Field(pattern=r"^[a-z_]+$")]
    engine_version: StrictStr
    configuration_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    registry_digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    random_seed: Annotated[StrictInt, Field(ge=0)]
    python_version: StrictStr
    platform: StrictStr
    process_hash_seed: StrictStr | None
    package_versions: dict[StrictStr, StrictStr]
    dataset_count: Annotated[StrictInt, Field(ge=0)]
    variable_count: Annotated[StrictInt, Field(ge=0)]


class RunManifest(_RunManifestFields):
    """Current aggregate structural metadata for a single execution."""

    schema_version: Literal["1"] = "1"


class _LegacyRunManifestV01(_RunManifestFields):
    """Strict legacy contract retained only for the allow-listed migration path."""

    schema_version: Literal["0.1"]


def _version_for(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def create_run_manifest(
    *,
    configuration_digest: str,
    registry_digest: str,
    random_seed: int,
    dataset_count: int,
    variable_count: int,
    status: str = "validated",
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> RunManifest:
    versions = {
        name: _version_for(name)
        for name in (
            "duckdb",
            "mapel-linkage-engine",
            "numpy",
            "pydantic",
            "PyYAML",
            "scikit-learn",
            "splink",
            "xgboost",
        )
    }
    return RunManifest(
        run_id=run_id or uuid.uuid4().hex,
        created_at=created_at or datetime.now(UTC),
        status=status,
        engine_version=__version__,
        configuration_digest=configuration_digest,
        registry_digest=registry_digest,
        random_seed=random_seed,
        python_version=platform.python_version(),
        platform=f"{platform.system()}-{platform.machine()}",
        process_hash_seed=os.environ.get("PYTHONHASHSEED"),
        package_versions=versions,
        dataset_count=dataset_count,
        variable_count=variable_count,
    )


def write_manifest(path: str, manifest: RunManifest, policy: PathPolicy) -> Path:
    destination = policy.resolve_output(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_text(
            destination,
            json.dumps(
                manifest.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    except OSError:
        raise SafeError(
            SafeErrorCode.MANIFEST_WRITE,
            "The run manifest could not be written.",
        ) from None
    return destination


def manifest_runtime_summary(manifest: RunManifest) -> dict[str, str | int]:
    """Return a row-free summary suitable for status reporting."""

    return {
        "run_id": manifest.run_id,
        "status": manifest.status,
        "dataset_count": manifest.dataset_count,
        "variable_count": manifest.variable_count,
        "engine_version": manifest.engine_version,
        "python_major_minor": ".".join(str(part) for part in sys.version_info[:2]),
    }
