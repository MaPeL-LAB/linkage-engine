#!/usr/bin/env bash
set -Eeuo pipefail
CURRENT_COMMAND="preflight validation"
trap 'echo "ERROR: lifecycle run failed at line ${LINENO}; command: ${CURRENT_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CANONICAL_PYTHON="${HOME:-}/.venvs/mapel-linkage-engine-m2-py312/bin/python"
PYTHON_BIN=""
OUTPUT_DIR="${REPO_ROOT}/artifacts/e2e_lifecycle"
DRY_RUN=false
FULL_RUN=false

usage() {
  cat <<'EOF'
Usage: scripts/run_e2e_lifecycle.sh [OPTIONS]

Run the deterministic synthetic lifecycle and its focused tests. Use --full
to add formatting, linting, typing, the complete test suite, generated-file
synchronisation, repository verification, and distribution verification.

Options:
  --python PATH       Python 3.12 interpreter to use.
  --output-dir DIR    Aggregate-only output directory.
  --full              Run the complete repository verification workflow.
  --dry-run           Print commands without executing or writing files.
  -h, --help          Show this help.
EOF
}

while (($# > 0)); do
  case "$1" in
    --python)
      if (($# < 2)); then
        echo "ERROR: --python requires a path." >&2
        exit 2
      fi
      PYTHON_BIN="$2"
      shift 2
      ;;
    --output-dir)
      if (($# < 2)); then
        echo "ERROR: --output-dir requires a directory." >&2
        exit 2
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --full)
      FULL_RUN=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${REPO_ROOT}/pyproject.toml" || ! -f "${REPO_ROOT}/AGENTS.md" ]]; then
  echo "ERROR: Could not resolve the Linkage Engine repository root." >&2
  exit 2
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${CANONICAL_PYTHON}" ]]; then
    PYTHON_BIN="${CANONICAL_PYTHON}"
  elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.12)"
  else
    echo "ERROR: Python 3.12 was not found; pass --python PATH." >&2
    exit 2
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: The selected Python path is not executable." >&2
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit("ERROR: Linkage Engine requires Python 3.12 for this workflow.")
PY

cd "${REPO_ROOT}"

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${REPO_ROOT}" <<'PY'
import os
import sys
from pathlib import Path

raw_output = Path(sys.argv[1])
repository_root = Path(sys.argv[2])
output = Path(os.path.abspath(raw_output if raw_output.is_absolute() else repository_root / raw_output))
if output in {Path("/"), repository_root}:
    raise SystemExit("ERROR: Refusing an unsafe lifecycle output directory.")
try:
    if any(candidate.is_symlink() for candidate in (output, *output.parents)):
        raise SystemExit("ERROR: The lifecycle output path cannot contain a symbolic link.")
    if output.exists() and not output.is_dir():
        raise SystemExit("ERROR: The lifecycle output path must be a directory.")
except OSError:
    raise SystemExit("ERROR: The lifecycle output path is invalid.") from None
PY

run() {
  local label="$1"
  shift
  echo "  ${label}"
  if [[ "${DRY_RUN}" == false ]]; then
    CURRENT_COMMAND="${label}"
    "$@"
    CURRENT_COMMAND="idle"
  fi
}

export MAPEL_TEST_DATA_POLICY="synthetic_only"
export MAPEL_RANDOM_SEED="20260816"

echo "[preflight] Repository: linkage-engine"
echo "[preflight] Python: verified Python 3.12 interpreter"
echo "[preflight] Data policy: ${MAPEL_TEST_DATA_POLICY}"
echo "[preflight] Seed: ${MAPEL_RANDOM_SEED}"
echo "[preflight] Output: aggregate-only directory configured"

if [[ "${DRY_RUN}" == false ]]; then
  mkdir -p "${OUTPUT_DIR}"
fi

echo "[focused] Running the aggregate synthetic lifecycle."
run "python examples/e2e_linkage_lifecycle.py --output-dir <aggregate-dir>" \
  "${PYTHON_BIN}" examples/e2e_linkage_lifecycle.py --output-dir "${OUTPUT_DIR}"

echo "[focused] Running lifecycle and recipe-boundary tests."
run "python -m pytest <focused lifecycle tests> -q" "${PYTHON_BIN}" -m pytest \
  tests/pipeline/test_recipes.py \
  tests/pipeline/test_inference_runner.py \
  tests/pipeline/test_portfolio_runner.py \
  tests/end_to_end/test_e2e_lifecycle.py \
  -q

if [[ "${FULL_RUN}" == true ]]; then
  echo "[full] Checking formatting and linting."
  run "python -m ruff format --check ." "${PYTHON_BIN}" -m ruff format --check .
  run "python -m ruff check ." "${PYTHON_BIN}" -m ruff check .

  echo "[full] Running strict typing and the complete test suite."
  run "python -m mypy src tests" "${PYTHON_BIN}" -m mypy src tests
  run "python -m pytest" "${PYTHON_BIN}" -m pytest

  echo "[full] Synchronising generated capability and repository records."
  run "python scripts/generate_capability_matrix.py" \
    "${PYTHON_BIN}" scripts/generate_capability_matrix.py
  run "python scripts/generate_repository_manifest.py" \
    "${PYTHON_BIN}" scripts/generate_repository_manifest.py

  echo "[full] Verifying repository and distributions."
  run "python scripts/verify_repository.py" "${PYTHON_BIN}" scripts/verify_repository.py
  run "python scripts/verify_distribution.py" "${PYTHON_BIN}" scripts/verify_distribution.py
fi

if [[ "${DRY_RUN}" == true ]]; then
  echo "Changed: nothing (dry run)."
  if [[ "${FULL_RUN}" == true ]]; then
    echo "Next command: scripts/run_e2e_lifecycle.sh --full"
  else
    echo "Next command: scripts/run_e2e_lifecycle.sh"
  fi
else
  echo "Changed: wrote deterministic aggregate lifecycle outputs to the configured directory."
  if [[ "${FULL_RUN}" == true ]]; then
    echo "Changed: regenerated docs/CAPABILITY_MATRIX.md and REPOSITORY_MANIFEST.txt."
  fi
  echo "Next command: git diff --check && git status --short"
fi
