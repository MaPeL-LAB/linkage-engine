#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "ERROR: bootstrap failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT="${1:-$(pwd)}"
if [[ ! -d "$ROOT" || ! -f "$ROOT/pyproject.toml" ]]; then
  echo "ERROR: Project root must contain pyproject.toml." >&2
  exit 2
fi
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf '%s\n' "ERROR: Python 3.12 is required." >&2
  exit 2
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit("ERROR: Python 3.12 is required.")
PY

"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints/ci-py312.txt -e ".[core,dev]"
python scripts/generate_config_schema.py
python scripts/generate_repository_manifest.py
python scripts/verify_repository.py
mapel-linkage init-local-project --directory .
mapel-linkage doctor --project-root .
pytest -q tests/end_to_end/test_complete_synthetic_vertical_slice.py

printf '%s\n' "Local Linkage Engine workspace initialized and synthetic smoke test passed."
