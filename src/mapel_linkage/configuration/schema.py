"""Machine-readable configuration schema generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mapel_linkage.configuration.models import LinkageConfig


def configuration_json_schema() -> dict[str, Any]:
    return LinkageConfig.model_json_schema(
        mode="validation",
        ref_template="#/$defs/{model}",
    )


def write_configuration_json_schema(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(configuration_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
