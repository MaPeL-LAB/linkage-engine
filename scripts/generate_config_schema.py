#!/usr/bin/env python3
"""Regenerate the committed Linkage Engine configuration JSON Schema."""

from __future__ import annotations

from pathlib import Path

from mapel_linkage.configuration.schema import write_configuration_json_schema

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    write_configuration_json_schema(ROOT / "schemas/linkage-config.schema.json")
    print("Configuration JSON Schema generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
