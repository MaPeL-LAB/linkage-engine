#!/usr/bin/env bash
set -Eeuo pipefail

CURRENT_COMMAND="preflight"
trap 'echo "ERROR: all-model portfolio failed at line ${LINENO}; command: ${CURRENT_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CANONICAL_PYTHON="${HOME:-}/.venvs/mapel-linkage-engine-m2-py312/bin/python"
PYTHON_BIN=""
CONFIG_PATH="${REPO_ROOT}/configs/examples/synthetic_all_models.yaml"
ENTITY_COUNT=120
K_FOLDS=3
DRY_RUN=false
FULL_RUN=false

usage() {
  cat <<'EOF'
Usage: scripts/run_all_model_portfolio.sh [OPTIONS]

Run the configuration-driven synthetic all-model portfolio with real native
Splink, XGBoost, LightGBM, PyTorch, stacking, and ranking runtimes.

Options:
  --python PATH       Python 3.12 interpreter from the prepared environment.
  --entity-count N    Generated synthetic entity count (100-100000; default 120).
  --k-folds N         Supervised OOF fold count (2-10; default 3).
  --full              Add repository-wide lint, typing, tests, and verification.
  --dry-run           Validate inputs and print the bounded command plan only.
  -h, --help          Show this help.
EOF
}

while (($# > 0)); do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { echo "ERROR: --python requires a path." >&2; exit 2; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --entity-count)
      [[ $# -ge 2 ]] || { echo "ERROR: --entity-count requires a value." >&2; exit 2; }
      ENTITY_COUNT="$2"
      shift 2
      ;;
    --k-folds)
      [[ $# -ge 2 ]] || { echo "ERROR: --k-folds requires a value." >&2; exit 2; }
      K_FOLDS="$2"
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

[[ -f "${REPO_ROOT}/pyproject.toml" && -f "${REPO_ROOT}/AGENTS.md" ]] || {
  echo "ERROR: Could not resolve the Linkage Engine repository root." >&2
  exit 2
}
[[ -f "${CONFIG_PATH}" ]] || { echo "ERROR: The all-model configuration is missing." >&2; exit 2; }
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
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: The selected Python is not executable." >&2; exit 2; }
CLI_BIN="$(dirname "${PYTHON_BIN}")/mapel-linkage"
[[ -x "${CLI_BIN}" ]] || { echo "ERROR: The mapel-linkage CLI is not installed." >&2; exit 2; }
[[ "${ENTITY_COUNT}" =~ ^[0-9]+$ ]] || { echo "ERROR: --entity-count must be an integer." >&2; exit 2; }
[[ "${K_FOLDS}" =~ ^[0-9]+$ ]] || { echo "ERROR: --k-folds must be an integer." >&2; exit 2; }
((ENTITY_COUNT >= 100 && ENTITY_COUNT <= 100000)) || {
  echo "ERROR: --entity-count must be between 100 and 100000." >&2
  exit 2
}
((K_FOLDS >= 2 && K_FOLDS <= 10)) || {
  echo "ERROR: --k-folds must be between 2 and 10." >&2
  exit 2
}

"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit("ERROR: This workflow requires Python 3.12.")
for module in ("lightgbm", "torch", "splink", "xgboost"):
    importlib.import_module(module)
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

cd "${REPO_ROOT}"
export MAPEL_TEST_DATA_POLICY="synthetic_only"
export MAPEL_RANDOM_SEED="20260816"

echo "[preflight] Python 3.12 and all optional runtimes are importable."
echo "[preflight] Data policy is synthetic-only; deterministic seed is 20260816."
run "validate the all-model configuration" \
  "${CLI_BIN}" validate-config \
  --config "${CONFIG_PATH}" --project-root "${REPO_ROOT}"
run "execute the aggregate-only all-model CLI workflow" \
  "${CLI_BIN}" run-model-portfolio \
  --config "${CONFIG_PATH}" --project-root "${REPO_ROOT}" --synthetic-demo \
  --entity-count "${ENTITY_COUNT}" --k-folds "${K_FOLDS}"
run "verify all-model E2E and malicious-boundary cases" \
  "${PYTHON_BIN}" -m pytest \
  tests/end_to_end/test_synthetic_all_model_portfolio.py \
  tests/pipeline/test_native_replay_contract.py \
  tests/pipeline/test_score_evidence.py -q

if [[ "${FULL_RUN}" == true ]]; then
  run "check formatting" "${PYTHON_BIN}" -m ruff format --check .
  run "run lint" "${PYTHON_BIN}" -m ruff check .
  run "run strict typing" "${PYTHON_BIN}" -m mypy src tests
  run "run complete tests" "${PYTHON_BIN}" -m pytest
  run "synchronise configuration schema" \
    "${PYTHON_BIN}" scripts/generate_config_schema.py
  run "synchronise capability matrix" \
    "${PYTHON_BIN}" scripts/generate_capability_matrix.py
  run "synchronise repository manifest" \
    "${PYTHON_BIN}" scripts/generate_repository_manifest.py
  run "verify repository" "${PYTHON_BIN}" scripts/verify_repository.py
  run "verify distribution" "${PYTHON_BIN}" scripts/verify_distribution.py
fi

if [[ "${DRY_RUN}" == true ]]; then
  echo "Changed: nothing (dry run)."
  echo "Next command: scripts/run_all_model_portfolio.sh"
else
  echo "Changed: wrote only policy-approved synthetic fixtures and immutable aggregate artifacts."
  if [[ "${FULL_RUN}" == true ]]; then
    echo "Changed: regenerated the configuration schema, capability matrix, and repository manifest."
  fi
  echo "Next command: git diff --check && git status --short"
fi
