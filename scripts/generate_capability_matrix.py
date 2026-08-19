#!/usr/bin/env python3
"""Generate the capability matrix from the package-owned registry."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mapel_linkage.capabilities import capability_matrix_markdown  # noqa: E402

OUTPUT = ROOT / "docs" / "CAPABILITY_MATRIX.md"


def main() -> int:
    OUTPUT.write_text(capability_matrix_markdown(), encoding="utf-8")
    print("Capability matrix generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
