#!/usr/bin/env bash
set -Eeuo pipefail

CURRENT_COMMAND="preflight"
CACHE_ROOT=""
INITIAL_REPOSITORY_SNAPSHOT=""
REPOSITORY_SNAPSHOT_CAPTURED=false
REPOSITORY_STATE_CHECKED=false

on_error() {
  local exit_status="$1"
  local line_number="$2"
  printf 'ERROR: I1C verification failed at line %s; command: %s\n' \
    "${line_number}" "${CURRENT_COMMAND}" >&2
  exit "${exit_status}"
}

set_current_command() {
  local argument=""
  local quoted=""
  local rendered=()
  local position=0
  for argument in "$@"; do
    if ((position == 0)) && [[ -n "${PYTHON_BIN:-}" && "${argument}" == "${PYTHON_BIN}" ]]; then
      rendered+=("<python>")
    elif [[ -n "${REPO_ROOT:-}" && "${argument}" == "${REPO_ROOT}" ]]; then
      rendered+=(".")
    elif [[ -n "${REPO_ROOT:-}" && "${argument}" == "${REPO_ROOT}/"* ]]; then
      rendered+=("${argument#"${REPO_ROOT}/"}")
    elif [[ -n "${CACHE_ROOT}" && "${argument}" == "${CACHE_ROOT}"* ]]; then
      rendered+=("<verification-cache>${argument#"${CACHE_ROOT}"}")
    else
      rendered+=("${argument}")
    fi
    ((position += 1))
  done
  if [[ "${rendered[0]}" == "<python>" ]]; then
    CURRENT_COMMAND="<python>"
    if ((${#rendered[@]} > 1)); then
      printf -v quoted '%q ' "${rendered[@]:1}"
      CURRENT_COMMAND+=" ${quoted% }"
    fi
  else
    printf -v quoted '%q ' "${rendered[@]}"
    CURRENT_COMMAND="${quoted% }"
  fi
}

repository_snapshot() {
  local snapshot=""
  set_current_command "${PYTHON_BIN}" -c "<repository-snapshot-helper>" "${REPO_ROOT}"
  if ! snapshot="$("${PYTHON_BIN}" -c "${REPOSITORY_SNAPSHOT_CODE}" "${REPO_ROOT}" 2>/dev/null)"; then
    echo "ERROR: Could not capture repository content and Git state." >&2
    return 1
  fi
  if [[ ! "${snapshot}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: Repository state snapshot is invalid." >&2
    return 1
  fi
  printf '%s' "${snapshot}"
}

repository_state_matches() {
  local final_repository_snapshot=""
  if ! final_repository_snapshot="$(repository_snapshot)"; then
    return 1
  fi
  if [[ "${final_repository_snapshot}" != "${INITIAL_REPOSITORY_SNAPSHOT}" ]]; then
    echo "ERROR: Verification changed candidate repository content or Git/index state." >&2
    return 1
  fi
}

cleanup() {
  local exit_status=$?
  trap - ERR EXIT
  if [[ "${REPOSITORY_SNAPSHOT_CAPTURED}" == true && "${REPOSITORY_STATE_CHECKED}" == false ]]; then
    if ! repository_state_matches; then
      exit_status=1
    fi
  fi
  if [[ -n "${CACHE_ROOT}" ]]; then
    case "${CACHE_ROOT}" in
      "${REPO_ROOT}"|"${REPO_ROOT}"/*|/|"")
        echo "ERROR: Refusing unsafe verification-cache cleanup." >&2
        exit_status=1
        ;;
      *)
        if [[ -d "${CACHE_ROOT}" && ! -L "${CACHE_ROOT}" ]]; then
          rm -rf -- "${CACHE_ROOT}"
        fi
        ;;
    esac
  fi
  exit "${exit_status}"
}

trap 'on_error "$?" "${LINENO}"' ERR
trap cleanup EXIT

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_PARENT="${SCRIPT_PATH%/*}"
[[ "${SCRIPT_PARENT}" != "${SCRIPT_PATH}" ]] || SCRIPT_PARENT="."
SCRIPT_DIR="$(cd "${SCRIPT_PARENT}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
USER_HOME_DIR="${HOME:-}"
[[ -n "${USER_HOME_DIR}" && "${USER_HOME_DIR}" == /* ]] || {
  echo "ERROR: A valid user home is required to resolve the canonical Python environment." >&2
  exit 2
}
CANONICAL_PYTHON="${USER_HOME_DIR%/}/.venvs/mapel-linkage-engine-m2-py312/bin/python"
PYTHON_BIN=""
EXPLICIT_PYTHON=false
DRY_RUN=false
FULL_RUN=false

read -r -d '' REPOSITORY_SNAPSHOT_CODE <<'PY' || true
import hashlib
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
digest = hashlib.sha256()


def update(value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


candidate_files = subprocess.run(
    ("git", "ls-files", "-co", "--exclude-standard", "-z"),
    cwd=root,
    check=False,
    capture_output=True,
)
if candidate_files.returncode != 0:
    raise SystemExit(2)
for raw_relative in sorted(set(candidate_files.stdout.split(b"\0")) - {b""}):
    relative = Path(os.fsdecode(raw_relative))
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(2)
    path = root / relative
    update(raw_relative)
    if not os.path.lexists(path):
        update(b"missing")
        continue
    metadata = path.lstat()
    update(str(metadata.st_mode).encode("ascii"))
    if path.is_symlink():
        update(os.fsencode(os.readlink(path)))
    elif path.is_file():
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                update(chunk)
    else:
        update(b"non-file")

for command in (
    ("status", "--porcelain=v1", "--untracked-files=all"),
    ("ls-files", "-s", "-z"),
    ("diff", "--binary", "--no-ext-diff"),
    ("diff", "--cached", "--binary", "--no-ext-diff"),
):
    completed = subprocess.run(
        ("git", *command),
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SystemExit(2)
    update(" ".join(command).encode("ascii"))
    update(completed.stdout)

print(digest.hexdigest())
PY

read -r -d '' DISTRIBUTION_COPY_CODE <<'PY' || true
import os
import shutil
import subprocess
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
forbidden_parts = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "data",
    "dist",
    "private",
}

try:
    candidate_files = subprocess.run(
        ("git", "ls-files", "-co", "--exclude-standard", "-z"),
        cwd=source,
        check=False,
        capture_output=True,
    )
    if candidate_files.returncode != 0:
        raise ValueError
    destination.mkdir(parents=True, exist_ok=False)
    for raw_relative in sorted(set(candidate_files.stdout.split(b"\0")) - {b""}):
        relative = Path(os.fsdecode(raw_relative))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any(part in forbidden_parts or part.endswith(".egg-info") for part in relative.parts)
        ):
            continue
        candidate = source / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, target, follow_symlinks=False)
except (OSError, ValueError):
    raise SystemExit(2) from None
PY

usage() {
  echo "Usage: scripts/run_i1c_linkage_modes.sh [--python PATH] [--full] [--dry-run]"
}

while (($# > 0)); do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { echo "ERROR: --python requires a path." >&2; exit 2; }
      PYTHON_BIN="$2"
      EXPLICIT_PYTHON=true
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
      echo "ERROR: Unsupported option." >&2
      usage >&2
      exit 2
      ;;
  esac
done

for required_command in bash git mktemp mkdir rm; do
  command -v "${required_command}" >/dev/null 2>&1 || {
    echo "ERROR: A required verification command is unavailable." >&2
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

if [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}" 2>/dev/null || true)"
elif [[ "${PYTHON_BIN}" != /* ]]; then
  PYTHON_DIRECTORY="$(cd "${PYTHON_BIN%/*}" 2>/dev/null && pwd -P)" || {
    echo "ERROR: The selected Python interpreter path is invalid." >&2
    exit 2
  }
  PYTHON_BIN="${PYTHON_DIRECTORY}/${PYTHON_BIN##*/}"
fi
[[ -n "${PYTHON_BIN}" && -f "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || {
  echo "ERROR: The selected Python interpreter is not an executable file." >&2
  exit 2
}

PYTHON_MARKER_COMMAND=(
  "${PYTHON_BIN}"
  -c
  'import sys; print(f"mapel-i1c-python:{sys.version_info.major}.{sys.version_info.minor}")'
)
set_current_command "${PYTHON_MARKER_COMMAND[@]}"
PYTHON_MARKER="$("${PYTHON_MARKER_COMMAND[@]}" 2>/dev/null)" || {
  echo "ERROR: The selected executable could not run the Python 3.12 preflight." >&2
  exit 2
}
if [[ "${PYTHON_MARKER}" != "mapel-i1c-python:3.12" ]]; then
  echo "ERROR: The selected executable did not identify as Python 3.12." >&2
  exit 2
fi

TEMP_BASE="${TMPDIR:-/tmp}"
[[ "${TEMP_BASE}" == /* && -d "${TEMP_BASE}" ]] || {
  echo "ERROR: The temporary-directory base is invalid." >&2
  exit 2
}
TEMP_BASE="$(cd "${TEMP_BASE}" && pwd -P)"
case "${TEMP_BASE}" in
  "${REPO_ROOT}"|"${REPO_ROOT}"/*)
    echo "ERROR: Verification caches must remain outside the repository." >&2
    exit 2
    ;;
esac
set_current_command mktemp -d "${TEMP_BASE%/}/mapel-i1c-verify.XXXXXX"
CACHE_ROOT="$(mktemp -d "${TEMP_BASE%/}/mapel-i1c-verify.XXXXXX")"
[[ -d "${CACHE_ROOT}" && ! -L "${CACHE_ROOT}" ]] || {
  echo "ERROR: Could not create a safe verification-cache directory." >&2
  exit 2
}
set_current_command mkdir -p \
  "${CACHE_ROOT}/ruff" "${CACHE_ROOT}/mypy" "${CACHE_ROOT}/xdg" "${CACHE_ROOT}/tmp"
mkdir -p \
  "${CACHE_ROOT}/ruff" "${CACHE_ROOT}/mypy" "${CACHE_ROOT}/xdg" "${CACHE_ROOT}/tmp"

cd "${REPO_ROOT}"
INITIAL_REPOSITORY_SNAPSHOT="$(repository_snapshot)"
REPOSITORY_SNAPSHOT_CAPTURED=true

export MAPEL_TEST_DATA_POLICY="synthetic_only"
export MAPEL_RANDOM_SEED="20260816"
export PYTHONDONTWRITEBYTECODE="1"
export RUFF_CACHE_DIR="${CACHE_ROOT}/ruff"
export MYPY_CACHE_DIR="${CACHE_ROOT}/mypy"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="${CACHE_ROOT}/tmp"

run_check() {
  local label="$1"
  shift
  echo "[verify] ${label}"
  set_current_command "$@"
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[plan] ${CURRENT_COMMAND}"
  else
    "$@"
  fi
}

prepare_distribution_copy() {
  local destination="${CACHE_ROOT}/distribution-source"
  echo "[verify] external distribution source copy"
  set_current_command \
    "${PYTHON_BIN}" -c "<distribution-copy-helper>" "${REPO_ROOT}" "${destination}"
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[plan] ${CURRENT_COMMAND}"
  else
    "${PYTHON_BIN}" -c "${DISTRIBUTION_COPY_CODE}" "${REPO_ROOT}" "${destination}"
  fi
}

run_check "shell syntax" bash -n "${BASH_SOURCE[0]}"
run_check "I1C lint" "${PYTHON_BIN}" -m ruff check \
  src/mapel_linkage/configuration \
  src/mapel_linkage/capabilities.py \
  src/mapel_linkage/cli/main.py \
  src/mapel_linkage/candidate_generation \
  src/mapel_linkage/decisions \
  src/mapel_linkage/governance \
  src/mapel_linkage/pipeline \
  src/mapel_linkage/synthetic \
  tests/configuration/test_mode_orchestration_config.py \
  tests/configuration/test_schema.py \
  tests/candidate_generation/test_duckdb_candidate_generator.py \
  tests/decisions/test_relationship_policy.py \
  tests/governance/test_paths_logging_and_manifests.py \
  tests/pipeline/test_mode_artifacts.py \
  tests/pipeline/test_synthetic_mode_workflow.py \
  tests/synthetic/test_generator.py \
  tests/test_cli.py \
  tests/test_i1c_driver.py
run_check "I1C focused typing" "${PYTHON_BIN}" -m mypy \
  --cache-dir "${CACHE_ROOT}/mypy" \
  src/mapel_linkage/configuration/models.py \
  src/mapel_linkage/configuration/compiler.py \
  src/mapel_linkage/capabilities.py \
  src/mapel_linkage/cli/main.py \
  src/mapel_linkage/candidate_generation/duckdb_generator.py \
  src/mapel_linkage/decisions/policy.py \
  src/mapel_linkage/governance/paths.py \
  src/mapel_linkage/pipeline/mode_artifacts.py \
  src/mapel_linkage/pipeline/synthetic_mode_workflow.py \
  src/mapel_linkage/synthetic/__init__.py \
  src/mapel_linkage/synthetic/generator.py
run_check "I1C focused tests" "${PYTHON_BIN}" -m pytest -q -p no:cacheprovider \
  tests/configuration/test_mode_orchestration_config.py \
  tests/configuration/test_schema.py \
  tests/candidate_generation/test_duckdb_candidate_generator.py \
  tests/decisions/test_relationship_policy.py \
  tests/governance/test_paths_logging_and_manifests.py \
  tests/pipeline/test_inference_runner.py \
  tests/pipeline/test_native_replay_contract.py \
  tests/pipeline/test_mode_artifacts.py \
  tests/pipeline/test_synthetic_mode_workflow.py \
  tests/synthetic/test_generator.py \
  tests/test_capabilities.py \
  tests/test_cli.py \
  tests/test_i1c_driver.py

if [[ "${FULL_RUN}" == true ]]; then
  run_check "repository formatting" "${PYTHON_BIN}" -m ruff format --check .
  run_check "repository lint" "${PYTHON_BIN}" -m ruff check .
  run_check "repository typing" "${PYTHON_BIN}" -m mypy \
    --cache-dir "${CACHE_ROOT}/mypy-full" src tests
  run_check "complete test suite" "${PYTHON_BIN}" -m pytest -q -p no:cacheprovider
  run_check "repository verification" "${PYTHON_BIN}" scripts/verify_repository.py
  prepare_distribution_copy
  run_check "distribution verification from external candidate copy" \
    "${PYTHON_BIN}" "${CACHE_ROOT}/distribution-source/scripts/verify_distribution.py"
fi

if ! repository_state_matches; then
  exit 1
fi
REPOSITORY_STATE_CHECKED=true

if [[ "${EXPLICIT_PYTHON}" == true ]]; then
  FULL_FOLLOW_UP="scripts/run_i1c_linkage_modes.sh --python '<PYTHON_3_12_PATH>' --full"
  FULL_FOLLOW_UP_NOTE=" (replace the placeholder with the approved interpreter path)."
else
  FULL_FOLLOW_UP="scripts/run_i1c_linkage_modes.sh --full"
  FULL_FOLLOW_UP_NOTE=""
fi
if [[ "${DRY_RUN}" == true ]]; then
  echo "Changed: none (dry-run verification plan only)."
  echo "Next command: ${FULL_FOLLOW_UP}${FULL_FOLLOW_UP_NOTE}"
elif [[ "${FULL_RUN}" == true ]]; then
  echo "Changed: none (full verification preserved candidate content and Git/index state)."
  echo "Next command: none (return to Codex and say done)."
else
  echo "Changed: none (focused verification preserved candidate content and Git/index state)."
  echo "Next command: ${FULL_FOLLOW_UP}${FULL_FOLLOW_UP_NOTE}"
fi
