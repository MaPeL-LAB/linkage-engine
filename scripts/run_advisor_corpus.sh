#!/usr/bin/env bash
set -Eeuo pipefail

CURRENT_COMMAND="preflight"
on_error() {
  local exit_status="$1"
  local line_number="$2"
  printf 'ERROR: advisor corpus run failed at line %s; command: %s\n' \
    "${line_number}" "${CURRENT_COMMAND}" >&2
  exit "${exit_status}"
}
trap 'on_error "$?" "${LINENO}"' ERR

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
REGISTRY_DIR="private/benchmark_registry/advisor_v2_execution_v2"
SHARD_COUNT=32
REPLICATES=5
APPROVAL_REFERENCE=""
APPROVE_EXECUTION=false
FULL_RUN=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: scripts/run_advisor_corpus.sh [OPTIONS]

Plan or execute the deterministic 64-family/280-instance advisor-v2 synthetic
benchmark corpus. Heavy execution requires all three of --full,
--approve-execution, and --approval-reference.

Options:
  --python PATH                 Python 3.12 interpreter.
  --registry-dir RELATIVE_DIR  Ignored project-relative registry directory.
  --shards N                   Deterministic shard count (default: 32).
  --replicates N               Replicates per instance (default: 5).
  --full                       Execute every shard after focused verification.
  --approve-execution          Record explicit human execution approval.
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
    --shards)
      [[ $# -ge 2 ]] || { echo "ERROR: --shards requires a number." >&2; exit 2; }
      SHARD_COUNT="$2"
      shift 2
      ;;
    --replicates)
      [[ $# -ge 2 ]] || { echo "ERROR: --replicates requires a number." >&2; exit 2; }
      REPLICATES="$2"
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
      [[ $# -ge 2 ]] || {
        echo "ERROR: --approval-reference requires a non-identifying value." >&2
        exit 2
      }
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

CURRENT_COMMAND="<python> -c <version-check>"
"${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 2)'

case "${SHARD_COUNT}" in
  ''|*[!0-9]*) echo "ERROR: --shards must be an integer from 1 to 256." >&2; exit 2 ;;
esac
case "${REPLICATES}" in
  ''|*[!0-9]*) echo "ERROR: --replicates must be an integer from 1 to 100." >&2; exit 2 ;;
esac
((SHARD_COUNT >= 1 && SHARD_COUNT <= 256)) || {
  echo "ERROR: --shards must be an integer from 1 to 256." >&2
  exit 2
}
((REPLICATES >= 1 && REPLICATES <= 100)) || {
  echo "ERROR: --replicates must be an integer from 1 to 100." >&2
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
    raise SystemExit("ERROR: --registry-dir must be a project-relative path without '..'.")
candidate = root
for part in relative.parts:
    candidate /= part
    if candidate.is_symlink():
        raise SystemExit("ERROR: --registry-dir cannot traverse symbolic links.")
resolved = Path(os.path.abspath(candidate))
if not resolved.is_relative_to(root) or (resolved.exists() and not resolved.is_dir()):
    raise SystemExit("ERROR: --registry-dir is outside the approved directory boundary.")
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

echo "[preflight] shell syntax"
CURRENT_COMMAND="bash -n scripts/run_advisor_corpus.sh"
bash -n "${BASH_SOURCE[0]}"

echo "[preflight] aggregate advisor-v2 plan"
CURRENT_COMMAND="mapel-linkage plan-advisor-corpus --shards <count> --replicates <count>"
run_cli plan-advisor-corpus --shards "${SHARD_COUNT}" --replicates "${REPLICATES}"

if [[ "${DRY_RUN}" == true ]]; then
  echo "[dry-run] focused verification and heavy shard execution were not run."
  echo "Changed: none (dry-run planning only)."
  echo "Next command: scripts/run_advisor_corpus.sh --full --approve-execution --approval-reference '<APPROVAL_REFERENCE>'"
  exit 0
fi

if [[ "${FULL_RUN}" == false ]]; then
  echo "Changed: none (planning only)."
  echo "Next command: scripts/run_advisor_corpus.sh --full --approve-execution --approval-reference '<APPROVAL_REFERENCE>'"
  exit 0
fi

echo "[verify] focused formatting and lint"
CURRENT_COMMAND="<python> -m ruff format --check <focused paths>"
"${PYTHON_BIN}" -m ruff format --check \
  src/mapel_linkage/benchmarking \
  src/mapel_linkage/recommendation/meta_ranker.py \
  src/mapel_linkage/cli/main.py \
  tests/benchmarking \
  tests/recommendation/test_meta_ranker.py
CURRENT_COMMAND="<python> -m ruff check <focused paths>"
"${PYTHON_BIN}" -m ruff check \
  --no-cache \
  src/mapel_linkage/benchmarking \
  src/mapel_linkage/recommendation/meta_ranker.py \
  src/mapel_linkage/cli/main.py \
  tests/benchmarking \
  tests/recommendation/test_meta_ranker.py

echo "[verify] focused typing"
CURRENT_COMMAND="<python> -m mypy <focused paths>"
"${PYTHON_BIN}" -m mypy --no-incremental \
  src/mapel_linkage/benchmarking \
  src/mapel_linkage/recommendation/meta_ranker.py \
  src/mapel_linkage/cli/main.py

echo "[verify] focused tests"
CURRENT_COMMAND="<python> -m pytest <focused tests> -q"
"${PYTHON_BIN}" -m pytest \
  -p no:cacheprovider \
  tests/benchmarking \
  tests/recommendation/test_meta_ranker.py \
  -q

echo "[execute] approved synthetic advisor-v2 corpus"
for ((shard_index = 0; shard_index < SHARD_COUNT; shard_index += 1)); do
  printf '[execute] shard %d of %d\n' "$((shard_index + 1))" "${SHARD_COUNT}"
  CURRENT_COMMAND="mapel-linkage run-advisor-corpus <approved shard arguments>"
  run_cli run-advisor-corpus \
    --project-root . \
    --registry-dir "${REGISTRY_DIR}" \
    --shards "${SHARD_COUNT}" \
    --shard-index "${shard_index}" \
    --replicates "${REPLICATES}" \
    --approve-execution \
    --approval-reference "${APPROVAL_REFERENCE}"
done

echo "[audit] aggregate corpus completion and three-adapter cell coverage"
CURRENT_COMMAND="mapel-linkage audit-advisor-corpus <aggregate arguments>"
run_cli audit-advisor-corpus \
  --project-root . \
  --registry-dir "${REGISTRY_DIR}" \
  --shards "${SHARD_COUNT}" \
  --replicates "${REPLICATES}"

echo "Changed: wrote or idempotently resumed aggregate synthetic advisor-v2 evidence in the configured ignored registry."
echo "Next command: return to Codex and report 'done' for aggregate completion and evidence review."
