from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs/examples/synthetic_link_only.yaml"


def valid_payload() -> dict[str, Any]:
    payload_raw: object = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload_raw, dict)
    payload = cast(dict[str, Any], payload_raw)
    return deepcopy(payload)


def yaml_text(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False)
