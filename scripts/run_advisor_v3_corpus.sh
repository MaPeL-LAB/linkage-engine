#!/usr/bin/env bash
set -Eeuo pipefail

CURRENT_COMMAND="preflight"
on_error() {
  local exit_status="$1"
  local line_number="$2"
  printf 'ERROR: advisor-v3 corpus run failed at line %s; command: %s\n' \
    "${line_number}" "${CURRENT_COMMAND}" >&2
  exit "${exit_status}"
}
trap 'on_error "$?" "${LINENO}"' ERR
worker_pids=()
on_interrupt() {
  trap - INT TERM
  echo "[interrupt] stopping active advisor-v3 workers; retained evidence remains resumable." >&2
  local worker_pid
  for worker_pid in "${worker_pids[@]}"; do
    kill "${worker_pid}" 2>/dev/null || true
  done
  for worker_pid in "${worker_pids[@]}"; do
    wait "${worker_pid}" 2>/dev/null || true
  done
  exit 130
}
trap 'on_interrupt' INT TERM

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_PARENT="${SCRIPT_PATH%/*}"
[[ "${SCRIPT_PARENT}" != "${SCRIPT_PATH}" ]] || SCRIPT_PARENT="."
SCRIPT_DIR="$(cd "${SCRIPT_PARENT}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

USER_HOME_DIR="${HOME:-}"
[[ -n "${USER_HOME_DIR}" && "${USER_HOME_DIR}" == /* ]] || {
  echo "ERROR: A valid user home is required to locate the canonical environment." >&2
  exit 2
}
CANONICAL_PYTHON="${USER_HOME_DIR%/}/.venvs/mapel-linkage-engine-m2-py312/bin/python"
PYTHON_BIN=""
REGISTRY_DIR="private/benchmark_registry/advisor_v3_execution_v1"
PREREGISTRATION="docs/evidence/advisor_v3_preregistration_20260821.json"
WORKERS=10
APPROVAL_REFERENCE=""
APPROVE_EXECUTION=false
FULL_RUN=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: scripts/run_advisor_v3_corpus.sh [OPTIONS]

Plan or resume the preregistered 84-family/336-instance advisor-v3 synthetic
corpus. Heavy execution uses ten workers by default and requires explicit approval.

Options:
  --python PATH                 Python 3.12 interpreter.
  --registry-dir RELATIVE_DIR  Registry under ignored private/benchmark_registry/.
  --workers N                  Concurrent workers from 1 to 10 (default: 10).
  --full                       Verify, prepare governance, execute, and audit.
  --approve-execution          Record explicit synthetic execution approval.
  --approval-reference REF     Non-identifying approval reference.
  --dry-run                    Validate and print the plan without writing.
  -h, --help                   Show this help.
EOF
}

while (($# > 0)); do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { echo "ERROR: --python requires a path." >&2; exit 2; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --registry-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --registry-dir requires a path." >&2; exit 2; }
      REGISTRY_DIR="$2"
      shift 2
      ;;
    --workers)
      [[ $# -ge 2 ]] || { echo "ERROR: --workers requires a number." >&2; exit 2; }
      WORKERS="$2"
      shift 2
      ;;
    --full)
      FULL_RUN=true
      shift
      ;;
    --approve-execution)
      APPROVE_EXECUTION=true
      shift
      ;;
    --approval-reference)
      [[ $# -ge 2 ]] || { echo "ERROR: --approval-reference requires a value." >&2; exit 2; }
      APPROVAL_REFERENCE="$2"
      shift 2
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
      echo "ERROR: Unsupported option." >&2
      usage >&2
      exit 2
      ;;
  esac
done

for required_command in bash git; do
  command -v "${required_command}" >/dev/null 2>&1 || {
    echo "ERROR: A required command is unavailable." >&2
    exit 2
  }
done
[[ -f "${REPO_ROOT}/pyproject.toml" && -f "${REPO_ROOT}/AGENTS.md" ]] || {
  echo "ERROR: Could not resolve the Linkage Engine repository root." >&2
  exit 2
}
[[ -f "${REPO_ROOT}/${PREREGISTRATION}" ]] || {
  echo "ERROR: The committed advisor-v3 preregistration is unavailable." >&2
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
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python is not executable." >&2; exit 2; }
CURRENT_COMMAND="<python> <identity-and-version-check>"
PYTHON_IDENTITY="$("${PYTHON_BIN}" -c 'import sys; print("mapel-linkage-py312" if sys.version_info[:2] == (3, 12) else "invalid")')"
[[ "${PYTHON_IDENTITY}" == "mapel-linkage-py312" ]] || {
  echo "ERROR: The selected interpreter is not Python 3.12." >&2
  exit 2
}

case "${WORKERS}" in
  ''|*[!0-9]*) echo "ERROR: --workers must be an integer from 1 to 10." >&2; exit 2 ;;
esac
((WORKERS >= 1 && WORKERS <= 10)) || {
  echo "ERROR: --workers must be an integer from 1 to 10." >&2
  exit 2
}

CURRENT_COMMAND="<python> <registry-path-check>"
"${PYTHON_BIN}" - "${REGISTRY_DIR}" "${REPO_ROOT}" <<'PY'
import os
import sys
from pathlib import Path

relative = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve(strict=True)
if relative.is_absolute() or not relative.parts or ".." in relative.parts:
    raise SystemExit("ERROR: --registry-dir must be project-relative without '..'.")
if relative.parts[:2] != ("private", "benchmark_registry"):
    raise SystemExit(
        "ERROR: --registry-dir must remain under ignored private/benchmark_registry/."
    )
candidate = root
for part in relative.parts:
    candidate /= part
    if candidate.is_symlink():
        raise SystemExit("ERROR: --registry-dir cannot traverse symbolic links.")
resolved = Path(os.path.abspath(candidate))
if not resolved.is_relative_to(root) or (resolved.exists() and not resolved.is_dir()):
    raise SystemExit("ERROR: --registry-dir is outside the approved project boundary.")
PY

if [[ "${FULL_RUN}" == true && "${DRY_RUN}" == false ]]; then
  [[ "${APPROVE_EXECUTION}" == true && -n "${APPROVAL_REFERENCE}" ]] || {
    echo "ERROR: --full requires --approve-execution and --approval-reference." >&2
    exit 2
  }
fi

run_cli() {
  "${PYTHON_BIN}" -c \
    'from mapel_linkage.cli.main import main; raise SystemExit(main())' "$@"
}

cd "${REPO_ROOT}"
export MAPEL_TEST_DATA_POLICY="synthetic_only"
export MAPEL_RANDOM_SEED="20260816"
export PYTHONDONTWRITEBYTECODE="1"
export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export VECLIB_MAXIMUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"

echo "[preflight] shell syntax"
CURRENT_COMMAND="bash -n scripts/run_advisor_v3_corpus.sh"
bash -n "${BASH_SOURCE[0]}"

echo "[preflight] outcome-free advisor-v3 plan and geometry"
CURRENT_COMMAND="mapel-linkage plan-advisor-v3-corpus"
run_cli plan-advisor-v3-corpus

if [[ "${DRY_RUN}" == true ]]; then
  echo "[dry-run] verification, governance writes, execution, and audit were not run."
  echo "Changed: none (dry-run planning only)."
  echo "Next command: scripts/run_advisor_v3_corpus.sh --full --approve-execution --approval-reference '<APPROVAL_REFERENCE>'"
  exit 0
fi

if [[ "${FULL_RUN}" == false ]]; then
  echo "Changed: none (planning only)."
  echo "Next command: scripts/run_advisor_v3_corpus.sh --full --approve-execution --approval-reference '<APPROVAL_REFERENCE>'"
  exit 0
fi

echo "[verify] focused advisor-v3 formatting, lint, typing, and tests"
CURRENT_COMMAND="<python> -m ruff format --check <advisor-v3 paths>"
"${PYTHON_BIN}" -m ruff format --check \
  src/mapel_linkage/benchmarking \
  src/mapel_linkage/recommendation \
  src/mapel_linkage/cli/main.py \
  tests/benchmarking/test_advisor_v3.py \
  tests/recommendation/test_advisor_v3_features.py
CURRENT_COMMAND="<python> -m ruff check <advisor-v3 paths>"
"${PYTHON_BIN}" -m ruff check --no-cache \
  src/mapel_linkage/benchmarking \
  src/mapel_linkage/recommendation \
  src/mapel_linkage/cli/main.py \
  tests/benchmarking/test_advisor_v3.py \
  tests/recommendation/test_advisor_v3_features.py
CURRENT_COMMAND="<python> -m mypy <advisor-v3 paths>"
"${PYTHON_BIN}" -m mypy --no-incremental \
  src/mapel_linkage/benchmarking \
  src/mapel_linkage/recommendation \
  src/mapel_linkage/cli/main.py
CURRENT_COMMAND="<python> -m pytest <advisor-v3 tests> -q"
"${PYTHON_BIN}" -m pytest -p no:cacheprovider \
  tests/benchmarking/test_advisor_v3.py \
  tests/recommendation/test_advisor_v3_features.py \
  -q

echo "[prepare] serial immutable governance"
CURRENT_COMMAND="mapel-linkage prepare-advisor-v3-corpus"
run_cli prepare-advisor-v3-corpus \
  --project-root . \
  --registry-dir "${REGISTRY_DIR}" \
  --preregistration "${PREREGISTRATION}" \
  --approve-execution \
  --approval-reference "${APPROVAL_REFERENCE}"

run_worker() {
  local worker_index="$1"
  local shard_index
  local child_pid=""
  trap 'if [[ -n "${child_pid}" ]]; then kill "${child_pid}" 2>/dev/null || true; wait "${child_pid}" 2>/dev/null || true; fi; exit 130' INT TERM
  for ((shard_index = worker_index; shard_index < 42; shard_index += WORKERS)); do
    printf '[execute] worker %d shard %d of 42\n' "$((worker_index + 1))" "$((shard_index + 1))"
    run_cli run-advisor-v3-corpus \
      --project-root . \
      --registry-dir "${REGISTRY_DIR}" \
      --shard-index "${shard_index}" \
      --approve-execution \
      --approval-reference "${APPROVAL_REFERENCE}" &
    child_pid="$!"
    wait "${child_pid}"
    child_pid=""
  done
  trap - INT TERM
}

echo "[execute] approved advisor-v3 corpus with ${WORKERS} workers"
CURRENT_COMMAND="parallel advisor-v3 whole-family shard workers"
for ((worker_index = 0; worker_index < WORKERS; worker_index += 1)); do
  run_worker "${worker_index}" &
  worker_pids+=("$!")
done
worker_failure=0
for worker_pid in "${worker_pids[@]}"; do
  if ! wait "${worker_pid}"; then
    worker_failure=1
  fi
done
if ((worker_failure != 0)); then
  echo "ERROR: At least one advisor-v3 worker failed; completed evidence remains resumable." >&2
  exit 1
fi

echo "[audit] exact retained grid and required-adapter coverage"
CURRENT_COMMAND="mapel-linkage audit-advisor-v3-corpus"
run_cli audit-advisor-v3-corpus \
  --project-root . \
  --registry-dir "${REGISTRY_DIR}" \
  --approval-reference "${APPROVAL_REFERENCE}"

echo "Changed: wrote or idempotently resumed preregistered aggregate synthetic advisor-v3 evidence in the configured ignored registry."
echo "Next command: return to Codex and report 'done' for aggregate review before any locked qualification approval."
