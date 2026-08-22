#!/usr/bin/env bash
set -Eeuo pipefail

CURRENT_COMMAND="preflight validation"

on_error() {
  local exit_status="$1"
  local line_number="$2"
  echo "ERROR: M8 scale benchmark failed at line ${line_number}; command: ${CURRENT_COMMAND}" >&2
  exit "${exit_status}"
}

trap 'on_error "$?" "${LINENO}"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
USER_HOME_DIR="${HOME:-}"
CANONICAL_PYTHON="${USER_HOME_DIR%/}/.venvs/mapel-linkage-engine-m2-py312/bin/python"
PYTHON_BIN=""
DRY_RUN=false
FORWARD_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_m8_scale_benchmarks.sh [OPTIONS]

Plan, run, or resume the aggregate-only M8 generated-synthetic scale matrix.
The default matrix contains ten cases and uses ten workers.

Options:
  --python PATH          Python 3.12 interpreter.
  --entity-counts LIST  Increasing comma-separated counts (default: 100,250,500,1000,2000).
  --repetitions N       Repetitions per count from 1 to 5 (default: 2).
  --workers N           Concurrent workers from 1 to 10 (default: 10).
  --output-dir DIR      Project-relative ignored artifacts directory.
  --dry-run             Validate and print the deterministic plan without writing.
  -h, --help            Show this help.
EOF
}

while (($# > 0)); do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { echo "ERROR: --python requires a path." >&2; exit 2; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --entity-counts|--repetitions|--workers|--output-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: The selected option requires a value." >&2; exit 2; }
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      FORWARD_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unsupported option." >&2
      usage >&2
      exit 2
      ;;
  esac
done

for required_command in bash mkdir; do
  command -v "${required_command}" >/dev/null 2>&1 || {
    echo "ERROR: A required command is unavailable." >&2
    exit 2
  }
done

CURRENT_COMMAND="bash syntax check"
bash -n "${BASH_SOURCE[0]}"

[[ -f "${REPO_ROOT}/pyproject.toml" && -f "${REPO_ROOT}/AGENTS.md" ]] || {
  echo "ERROR: Could not resolve the Linkage Engine repository root." >&2
  exit 2
}

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

[[ -x "${PYTHON_BIN}" ]] || {
  echo "ERROR: The selected Python path is not executable." >&2
  exit 2
}

cd "${REPO_ROOT}"
export MAPEL_TEST_DATA_POLICY="synthetic_only"
export MAPEL_RANDOM_SEED="20260816"

echo "[preflight] Repository: linkage-engine"
echo "[preflight] Data policy: synthetic_only"
echo "[preflight] Seed: 20260816"
echo "[preflight] Default concurrency: 10 workers; maximum: 10"
echo "[plan] Validating the deterministic scale matrix and safe output boundary."

CURRENT_COMMAND="python scripts/run_m8_scale_benchmarks.py"
"${PYTHON_BIN}" scripts/run_m8_scale_benchmarks.py \
  --python "${PYTHON_BIN}" \
  "${FORWARD_ARGS[@]}"
CURRENT_COMMAND="complete"

if [[ "${DRY_RUN}" == true ]]; then
  echo "Changed: none (dry-run planning only)."
  echo "Next command: scripts/run_m8_scale_benchmarks.sh"
else
  echo "Changed: wrote or resumed aggregate synthetic scale evidence in the configured ignored artifact directory."
  echo "Next command: python scripts/verify_release_readiness.py --expect-blocked"
fi
