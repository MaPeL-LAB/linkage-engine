#!/usr/bin/env python3
"""Exercise an aggregate-only synthetic rollback against immutable Git snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import tomllib
import venv
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts"
TEMP_ROOT = Path(tempfile.gettempdir()).resolve(strict=True)
DEFAULT_OUTPUT = "artifacts/m8_rollback_drill_v1"
CANDIDATE_COMMIT = "81762675996eae77ccb16210936d630f092a3e7b"
BASELINE_COMMIT = "5050626583236fe1a7778eabc363a31764385285"
CONFIG_PATH = "configs/examples/synthetic_link_only.yaml"
CONSTRAINTS_PATH = "constraints/ci-py312.txt"
AUTHORITY_PATHS = (
    "src/mapel_linkage/assignment/contracts.py",
    "src/mapel_linkage/decisions/policy.py",
    "src/mapel_linkage/recommendation/contracts.py",
)
DRILL_ID = "m8_synthetic_rollback_v1"
SEED = 20260816
ENTITY_COUNT = 100
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_SOURCE_FILE_BYTES = 2 * 1024 * 1024
APPROVAL_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.:-]{2,79}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class RollbackDrillError(RuntimeError):
    """Fail-closed error with a privacy-safe public code and message."""

    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


def _fail(code: str, message: str) -> NoReturn:
    raise RollbackDrillError(code, message)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _compact_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _with_report_digest(payload: Mapping[str, object]) -> dict[str, object]:
    body = dict(payload)
    body["report_digest"] = _digest_bytes(_compact_json(body))
    return body


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    step: str,
    env: Mapping[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> bytes:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            input=input_bytes,
            check=False,
            capture_output=True,
        )
    except OSError:
        _fail("ML-ROLLBACK-003", f"Rollback drill failed closed during {step}.")
    if completed.returncode != 0:
        _fail("ML-ROLLBACK-003", f"Rollback drill failed closed during {step}.")
    return completed.stdout


def _git_bytes(arguments: Sequence[str], *, step: str) -> bytes:
    return _run(["git", *arguments], cwd=ROOT, step=step)


def _resolve_commit(commit: str) -> str:
    if COMMIT_PATTERN.fullmatch(commit) is None:
        _fail("ML-ROLLBACK-001", "Rollback drill requires a full immutable commit digest.")
    resolved = (
        _git_bytes(
            ["rev-parse", "--verify", f"{commit}^{{commit}}"],
            step="immutable commit validation",
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if resolved != commit:
        _fail("ML-ROLLBACK-001", "Rollback drill commit resolution was not immutable.")
    return resolved


def _commit_file(commit: str, relative: str) -> bytes:
    payload = _git_bytes(["show", f"{commit}:{relative}"], step="snapshot file loading")
    if not payload or len(payload) > MAX_SOURCE_FILE_BYTES:
        _fail("ML-ROLLBACK-002", "Rollback drill snapshot input is unavailable or oversized.")
    return payload


def _commit_timestamp(commit: str) -> str:
    value = (
        _git_bytes(
            ["show", "-s", "--format=%ct", commit],
            step="snapshot timestamp loading",
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if not value.isdigit():
        _fail("ML-ROLLBACK-002", "Rollback drill snapshot timestamp is invalid.")
    return value


def _bundle_digest(commit: str, paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        payload = _commit_file(commit, relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _package_version(commit: str) -> str:
    try:
        project = tomllib.loads(_commit_file(commit, "pyproject.toml").decode("utf-8"))["project"]
        version = project["version"]
    except (KeyError, TypeError, UnicodeError, tomllib.TOMLDecodeError):
        _fail("ML-ROLLBACK-002", "Rollback drill package metadata is invalid.")
    if not isinstance(version, str) or not version:
        _fail("ML-ROLLBACK-002", "Rollback drill package version is invalid.")
    return version


def _safe_output_path(relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        _fail("ML-ROLLBACK-001", "Rollback drill output must be repository-relative.")
    output = (ROOT / value).resolve(strict=False)
    artifact_root = ARTIFACT_ROOT.resolve(strict=False)
    if output == artifact_root or not output.is_relative_to(artifact_root):
        _fail("ML-ROLLBACK-001", "Rollback drill output must be below the artifact root.")
    cursor = output
    while cursor != ROOT:
        if cursor.is_symlink():
            _fail("ML-ROLLBACK-001", "Rollback drill output path is symlink-unsafe.")
        cursor = cursor.parent
    return output


def _implementation_digest() -> str:
    return _digest_file(Path(__file__).resolve())


def build_plan(candidate_commit: str, baseline_commit: str) -> dict[str, object]:
    candidate = _resolve_commit(candidate_commit)
    baseline = _resolve_commit(baseline_commit)
    if candidate == baseline:
        _fail("ML-ROLLBACK-001", "Rollback candidate and baseline must be distinct.")
    _git_bytes(
        ["merge-base", "--is-ancestor", baseline, candidate],
        step="rollback ancestry validation",
    )
    candidate_constraints = _commit_file(candidate, CONSTRAINTS_PATH)
    baseline_constraints = _commit_file(baseline, CONSTRAINTS_PATH)
    candidate_config = _commit_file(candidate, CONFIG_PATH)
    baseline_config = _commit_file(baseline, CONFIG_PATH)
    candidate_authority = _bundle_digest(candidate, AUTHORITY_PATHS)
    baseline_authority = _bundle_digest(baseline, AUTHORITY_PATHS)
    constraints_match = candidate_constraints == baseline_constraints
    authority_match = candidate_authority == baseline_authority
    if not constraints_match:
        _fail("ML-ROLLBACK-004", "Rollback dependency constraints require separate restoration.")
    if not authority_match:
        _fail("ML-ROLLBACK-004", "Rollback authority contracts changed across snapshots.")
    return {
        "schema_version": "1",
        "drill_id": DRILL_ID,
        "dry_run": True,
        "seed": SEED,
        "entity_count": ENTITY_COUNT,
        "candidate_commit": candidate,
        "baseline_commit": baseline,
        "candidate_version": _package_version(candidate),
        "baseline_version": _package_version(baseline),
        "candidate_constraints_digest": _digest_bytes(candidate_constraints),
        "baseline_constraints_digest": _digest_bytes(baseline_constraints),
        "constraints_match": constraints_match,
        "candidate_config_digest": _digest_bytes(candidate_config),
        "baseline_config_digest": _digest_bytes(baseline_config),
        "candidate_authority_contract_digest": candidate_authority,
        "baseline_authority_contract_digest": baseline_authority,
        "authority_contracts_unchanged": authority_match,
        "implementation_digest": _implementation_digest(),
        "installation_surface": "isolated_venv_with_verified_dependency_layer",
        "rollback_trigger": "approved_synthetic_drill_trigger",
        "data_policy": "synthetic_only",
        "report_classification": "aggregate_only",
        "contains_record_data": False,
        "contains_identifiers": False,
        "contains_candidate_pairs": False,
        "contains_local_paths": False,
        "decision_authority": "none",
        "assignment_authority": "none",
        "merge_authority": "none",
        "publication_authority": "none",
        "deployment_authority": "none",
        "release_authority": "none",
        "automatic_publication": "prohibited",
        "operational_validity": "not_established",
    }


def _extract_snapshot(commit: str, destination: Path) -> None:
    archive = _git_bytes(["archive", "--format=tar", commit], step="snapshot archival")
    if not archive or len(archive) > MAX_ARCHIVE_BYTES:
        _fail("ML-ROLLBACK-002", "Rollback snapshot archive is unavailable or oversized.")
    destination.mkdir(parents=True)
    with tempfile.SpooledTemporaryFile(max_size=MAX_ARCHIVE_BYTES) as handle:
        handle.write(archive)
        handle.seek(0)
        try:
            with tarfile.open(fileobj=handle, mode="r:") as bundle:
                for member in bundle.getmembers():
                    pure = PurePosixPath(member.name)
                    if (
                        pure.is_absolute()
                        or ".." in pure.parts
                        or member.issym()
                        or member.islnk()
                        or member.isdev()
                    ):
                        _fail("ML-ROLLBACK-002", "Rollback snapshot archive is path-unsafe.")
                    target = destination.joinpath(*pure.parts)
                    if not target.resolve(strict=False).is_relative_to(destination.resolve()):
                        _fail("ML-ROLLBACK-002", "Rollback snapshot archive escaped its boundary.")
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        source = bundle.extractfile(member)
                        if source is None:
                            _fail("ML-ROLLBACK-002", "Rollback snapshot archive is incomplete.")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open("wb") as output:
                            shutil.copyfileobj(source, output)
                    else:
                        _fail(
                            "ML-ROLLBACK-002",
                            "Rollback snapshot archive has unsupported entries.",
                        )
        except tarfile.TarError:
            _fail("ML-ROLLBACK-002", "Rollback snapshot archive is invalid.")


def _build_wheel(source: Path, destination: Path, *, timestamp: str) -> Path:
    destination.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        {
            "MAPEL_TEST_DATA_POLICY": "synthetic_only",
            "PYTHONHASHSEED": str(SEED),
            "SOURCE_DATE_EPOCH": timestamp,
        }
    )
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(destination),
        ],
        cwd=source,
        step="immutable wheel build",
        env=environment,
    )
    wheels = tuple(destination.glob("*.whl"))
    if len(wheels) != 1 or wheels[0].is_symlink():
        _fail("ML-ROLLBACK-005", "Rollback drill wheel build was not singular and immutable.")
    return wheels[0]


def _wheel_package_state(wheel: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    names: list[str] = []
    try:
        with zipfile.ZipFile(wheel) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                name = info.filename
                if info.is_dir() or not name.startswith("mapel_linkage/"):
                    continue
                if "__pycache__" in PurePosixPath(name).parts or name.endswith((".pyc", ".pyo")):
                    continue
                payload = archive.read(info)
                names.append(name)
                digest.update(name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(hashlib.sha256(payload).digest())
    except (OSError, zipfile.BadZipFile):
        _fail("ML-ROLLBACK-005", "Rollback drill wheel could not be inspected.")
    if not names:
        _fail("ML-ROLLBACK-005", "Rollback drill wheel contains no package tree.")
    return {"digest": digest.hexdigest(), "file_count": len(names), "names": tuple(names)}


def _create_drill_environment(directory: Path) -> Path:
    if sys.version_info[:2] != (3, 12):
        _fail("ML-ROLLBACK-001", "Rollback drill requires Python 3.12.")
    try:
        venv.EnvBuilder(with_pip=True, clear=False, symlinks=True).create(directory)
    except (OSError, subprocess.SubprocessError):
        _fail("ML-ROLLBACK-006", "Rollback drill isolated environment creation failed.")
    python = directory / "bin" / "python"
    if not python.is_file():
        _fail("ML-ROLLBACK-006", "Rollback drill isolated Python is unavailable.")
    purelib_raw = _run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        cwd=directory,
        step="isolated environment inspection",
    )
    try:
        purelib = Path(purelib_raw.decode("utf-8").strip()).resolve(strict=True)
        dependency_layer = Path(sysconfig.get_paths()["purelib"]).resolve(strict=True)
    except (OSError, UnicodeError):
        _fail("ML-ROLLBACK-006", "Rollback drill dependency layer is unavailable.")
    if not purelib.is_relative_to(directory.resolve()) or purelib == dependency_layer:
        _fail("ML-ROLLBACK-006", "Rollback drill installation boundary is invalid.")
    (purelib / "mapel_verified_dependency_layer.pth").write_text(
        str(dependency_layer) + "\n", encoding="utf-8"
    )
    return python


def _install_wheel(python: Path, wheel: Path, *, working: Path) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "MAPEL_TEST_DATA_POLICY": "synthetic_only",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONHASHSEED": str(SEED),
        }
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--force-reinstall",
            "--no-deps",
            str(wheel),
        ],
        cwd=working,
        step="isolated wheel installation",
        env=environment,
    )


_INSTALLED_STATE_SCRIPT = r"""
import hashlib
import json
import sys
from pathlib import Path

import mapel_linkage

root = Path(mapel_linkage.__file__).resolve().parent
prefix = Path(sys.prefix).resolve()
digest = hashlib.sha256()
names = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
        continue
    name = "mapel_linkage/" + path.relative_to(root).as_posix()
    names.append(name)
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print(json.dumps({
    "digest": digest.hexdigest(),
    "file_count": len(names),
    "inside_environment": root.is_relative_to(prefix),
    "version": mapel_linkage.__version__,
}, sort_keys=True))
"""


def _installed_package_state(python: Path, *, working: Path) -> dict[str, object]:
    raw = _run(
        [str(python), "-c", _INSTALLED_STATE_SCRIPT],
        cwd=working,
        step="installed package integrity inspection",
    )
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        _fail("ML-ROLLBACK-007", "Rollback drill installed package state is invalid.")
    if not isinstance(payload, dict):
        _fail("ML-ROLLBACK-007", "Rollback drill installed package state is invalid.")
    return payload


def _assert_installed_state(
    actual: Mapping[str, object], expected: Mapping[str, object], version: str
) -> None:
    if (
        actual.get("inside_environment") is not True
        or actual.get("digest") != expected.get("digest")
        or actual.get("file_count") != expected.get("file_count")
        or actual.get("version") != version
    ):
        _fail("ML-ROLLBACK-007", "Rollback drill installed package integrity check failed.")


def _prepare_workspace(root: Path, config: bytes) -> Path:
    for relative in ("private/config", "private/outputs", "data", "artifacts"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    config_path = root / "private" / "config" / "synthetic_link_only.yaml"
    config_path.write_bytes(config)
    if _digest_file(config_path) != _digest_bytes(config):
        _fail("ML-ROLLBACK-008", "Rollback drill configuration restoration failed.")
    return config_path


def _exercise_package(python: Path, workspace: Path, config: Path) -> tuple[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MAPEL_TEST_DATA_POLICY": "synthetic_only",
            "OMP_NUM_THREADS": "1",
            "PYTHONHASHSEED": str(SEED),
        }
    )
    doctor = _run(
        [str(python), "-m", "mapel_linkage", "doctor", "--project-root", str(workspace)],
        cwd=workspace,
        step="aggregate environment doctor",
        env=environment,
    )
    smoke = _run(
        [
            str(python),
            "-m",
            "mapel_linkage",
            "run",
            "--config",
            str(config),
            "--project-root",
            str(workspace),
            "--synthetic-demo",
            "--entity-count",
            str(ENTITY_COUNT),
        ],
        cwd=workspace,
        step="generated-synthetic smoke test",
        env=environment,
    )
    return _digest_bytes(doctor), _digest_bytes(smoke)


def _normalized_pins(payload: bytes, *, require_all_exact: bool) -> set[str]:
    pins: set[str] = set()
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError:
        _fail("ML-ROLLBACK-004", "Rollback dependency constraints are invalid.")
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            if require_all_exact:
                _fail("ML-ROLLBACK-004", "Rollback dependency constraints are not exact pins.")
            continue
        name, version = line.split("==", maxsplit=1)
        normalized_name = re.sub(r"[-_.]+", "-", name.strip()).lower()
        if not normalized_name or not version.strip():
            _fail("ML-ROLLBACK-004", "Rollback dependency constraints are invalid.")
        pins.add(f"{normalized_name}=={version.strip()}")
    if not pins:
        _fail("ML-ROLLBACK-004", "Rollback dependency constraints are empty.")
    return pins


def _dependency_environment_digest(expected_constraints: bytes) -> str:
    environment = os.environ.copy()
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    freeze = _run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=ROOT,
        step="dependency environment inventory",
        env=environment,
    )
    normalized_lines = sorted(line.strip() for line in freeze.splitlines() if line.strip())
    normalized = b"\n".join(normalized_lines)
    installed_pins = _normalized_pins(normalized, require_all_exact=False)
    if not _normalized_pins(expected_constraints, require_all_exact=True).issubset(installed_pins):
        _fail("ML-ROLLBACK-004", "Rollback dependency environment violates its constraints.")
    return _digest_bytes(normalized)


def _validate_retained_artifacts(output: Path, payload: Mapping[str, object]) -> None:
    for kind, digest_key in (
        ("candidate", "candidate_wheel_digest"),
        ("baseline", "baseline_wheel_digest"),
    ):
        directory = output / "retained" / kind
        wheels = tuple(directory.glob("*.whl")) if directory.is_dir() else ()
        if (
            directory.is_symlink()
            or len(wheels) != 1
            or wheels[0].is_symlink()
            or not wheels[0].is_file()
            or _digest_file(wheels[0]) != payload.get(digest_key)
        ):
            _fail("ML-ROLLBACK-009", "Retained rollback artifacts failed integrity checks.")


def _validate_existing_report(output: Path, plan: Mapping[str, object]) -> dict[str, object]:
    summary = output / "summary.json"
    if output.is_symlink() or summary.is_symlink() or not summary.is_file():
        _fail("ML-ROLLBACK-009", "Existing rollback evidence is incomplete or path-unsafe.")
    try:
        text = summary.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("ML-ROLLBACK-009", "Existing rollback evidence is invalid.")
    if not isinstance(payload, dict) or text != _canonical_json(payload):
        _fail("ML-ROLLBACK-009", "Existing rollback evidence is not canonical.")
    digest = payload.get("report_digest")
    body = {key: value for key, value in payload.items() if key != "report_digest"}
    if digest != _digest_bytes(_compact_json(body)):
        _fail("ML-ROLLBACK-009", "Existing rollback evidence digest is invalid.")
    for key in ("candidate_commit", "baseline_commit", "implementation_digest"):
        if payload.get(key) != plan.get(key):
            _fail("ML-ROLLBACK-009", "Existing rollback evidence conflicts with this plan.")
    if payload.get("status") != "verified_synthetic_rollback":
        _fail("ML-ROLLBACK-009", "Existing rollback evidence is not verified.")
    checks = payload.get("checks")
    if (
        payload.get("data_policy") != "synthetic_only"
        or payload.get("report_classification") != "aggregate_only"
        or payload.get("contains_record_data") is not False
        or payload.get("contains_identifiers") is not False
        or payload.get("contains_candidate_pairs") is not False
        or payload.get("contains_local_paths") is not False
        or payload.get("operational_validity") != "not_established"
        or payload.get("release_authority") != "none"
        or payload.get("failed_candidate_evidence_overwritten") is not False
        or payload.get("check_count") != payload.get("passed_check_count")
        or not isinstance(checks, dict)
        or len(checks) != payload.get("check_count")
        or any(value is not True for value in checks.values())
    ):
        _fail("ML-ROLLBACK-009", "Existing rollback evidence violates its safety boundary.")
    _validate_retained_artifacts(output, payload)
    return payload


def execute_drill(
    *,
    candidate_commit: str,
    baseline_commit: str,
    output_relative: str,
    approval_reference: str,
) -> dict[str, object]:
    if APPROVAL_PATTERN.fullmatch(approval_reference) is None:
        _fail("ML-ROLLBACK-001", "Rollback drill requires a bounded approval reference.")
    output = _safe_output_path(output_relative)
    plan = build_plan(candidate_commit, baseline_commit)
    if output.exists():
        return _validate_existing_report(output, plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate = str(plan["candidate_commit"])
    baseline = str(plan["baseline_commit"])
    candidate_constraints = _commit_file(candidate, CONSTRAINTS_PATH)
    dependency_digest = _dependency_environment_digest(candidate_constraints)
    candidate_config = _commit_file(candidate, CONFIG_PATH)
    baseline_config = _commit_file(baseline, CONFIG_PATH)

    with tempfile.TemporaryDirectory(prefix="mapel-rollback-work-", dir=TEMP_ROOT) as work_text:
        work = Path(work_text)
        candidate_source = work / "candidate-source"
        baseline_source = work / "baseline-source"
        _extract_snapshot(candidate, candidate_source)
        _extract_snapshot(baseline, baseline_source)
        candidate_build = work / "candidate-build"
        baseline_build = work / "baseline-build"
        candidate_wheel = _build_wheel(
            candidate_source,
            candidate_build,
            timestamp=_commit_timestamp(candidate),
        )
        baseline_wheel = _build_wheel(
            baseline_source,
            baseline_build,
            timestamp=_commit_timestamp(baseline),
        )
        candidate_wheel_state = _wheel_package_state(candidate_wheel)
        baseline_wheel_state = _wheel_package_state(baseline_wheel)
        candidate_names = set(cast(tuple[str, ...], candidate_wheel_state["names"]))
        baseline_names = set(cast(tuple[str, ...], baseline_wheel_state["names"]))
        candidate_only_count = len(candidate_names - baseline_names)
        if (
            candidate_wheel_state["digest"] == baseline_wheel_state["digest"]
            or candidate_only_count < 1
        ):
            _fail("ML-ROLLBACK-005", "Rollback snapshots do not provide a distinct package tree.")

        with tempfile.TemporaryDirectory(
            prefix=".m8-rollback-staging-", dir=output.parent
        ) as staging_text:
            staging = Path(staging_text)
            retained_candidate = staging / "retained" / "candidate" / candidate_wheel.name
            retained_baseline = staging / "retained" / "baseline" / baseline_wheel.name
            retained_candidate.parent.mkdir(parents=True)
            retained_baseline.parent.mkdir(parents=True)
            shutil.copy2(candidate_wheel, retained_candidate)
            shutil.copy2(baseline_wheel, retained_baseline)
            candidate_wheel_digest = _digest_file(retained_candidate)
            baseline_wheel_digest = _digest_file(retained_baseline)

            environment_python = _create_drill_environment(work / "candidate-environment")
            candidate_workspace = work / "candidate-workspace"
            candidate_config_path = _prepare_workspace(candidate_workspace, candidate_config)
            _install_wheel(environment_python, retained_candidate, working=candidate_workspace)
            installed_candidate = _installed_package_state(
                environment_python, working=candidate_workspace
            )
            _assert_installed_state(
                installed_candidate,
                candidate_wheel_state,
                str(plan["candidate_version"]),
            )
            candidate_doctor_digest, candidate_smoke_digest = _exercise_package(
                environment_python, candidate_workspace, candidate_config_path
            )

            baseline_workspace = work / "baseline-workspace"
            baseline_config_path = _prepare_workspace(baseline_workspace, baseline_config)
            _install_wheel(environment_python, retained_baseline, working=baseline_workspace)
            installed_baseline = _installed_package_state(
                environment_python, working=baseline_workspace
            )
            _assert_installed_state(
                installed_baseline,
                baseline_wheel_state,
                str(plan["baseline_version"]),
            )
            baseline_doctor_digest, baseline_smoke_digest = _exercise_package(
                environment_python, baseline_workspace, baseline_config_path
            )
            if (
                _digest_file(retained_candidate) != candidate_wheel_digest
                or _digest_file(retained_baseline) != baseline_wheel_digest
            ):
                _fail("ML-ROLLBACK-008", "Retained rollback artifacts failed integrity checks.")

            checks = {
                "authority_contracts_unchanged": True,
                "baseline_doctor_passed": True,
                "baseline_package_tree_restored": True,
                "baseline_synthetic_smoke_passed": True,
                "candidate_doctor_passed": True,
                "candidate_evidence_preserved": True,
                "candidate_package_tree_verified": True,
                "candidate_synthetic_smoke_passed": True,
                "config_restored": True,
                "constraints_restored": True,
                "isolated_installation_verified": True,
                "retained_artifact_digests_verified": True,
                "version_restored": True,
            }
            body: dict[str, object] = {
                **{key: value for key, value in plan.items() if key != "dry_run"},
                "dry_run": False,
                "status": "verified_synthetic_rollback",
                "approval_reference": approval_reference,
                "dependency_environment_digest": dependency_digest,
                "candidate_wheel_digest": candidate_wheel_digest,
                "baseline_wheel_digest": baseline_wheel_digest,
                "candidate_package_tree_digest": candidate_wheel_state["digest"],
                "baseline_package_tree_digest": baseline_wheel_state["digest"],
                "candidate_package_file_count": candidate_wheel_state["file_count"],
                "baseline_package_file_count": baseline_wheel_state["file_count"],
                "removed_candidate_file_count": candidate_only_count,
                "candidate_doctor_output_digest": candidate_doctor_digest,
                "candidate_synthetic_smoke_output_digest": candidate_smoke_digest,
                "baseline_doctor_output_digest": baseline_doctor_digest,
                "baseline_synthetic_smoke_output_digest": baseline_smoke_digest,
                "check_count": len(checks),
                "passed_check_count": sum(checks.values()),
                "checks": checks,
                "failed_candidate_evidence_overwritten": False,
                "dependency_layer_independently_isolated": False,
                "dependency_layer_reuse_note": (
                    "The isolated package surface reused the exact verified Python 3.12 "
                    "dependency layer bound by digest."
                ),
                "synthetic_testing_establishes_operational_validity": False,
            }
            report = _with_report_digest(body)
            (staging / "summary.json").write_text(_canonical_json(report), encoding="utf-8")
            staging.rename(output)
            return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exercise a fail-closed synthetic rollback against immutable commits."
    )
    parser.add_argument("--candidate-commit", default=CANDIDATE_COMMIT)
    parser.add_argument("--baseline-commit", default=BASELINE_COMMIT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--approval-reference")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        plan = build_plan(str(args.candidate_commit), str(args.baseline_commit))
        _safe_output_path(str(args.output_dir))
        if args.dry_run:
            print(_canonical_json(plan), end="")
            return 0
        if args.approval_reference is None:
            _fail("ML-ROLLBACK-001", "Rollback drill execution requires explicit approval.")
        report = execute_drill(
            candidate_commit=str(args.candidate_commit),
            baseline_commit=str(args.baseline_commit),
            output_relative=str(args.output_dir),
            approval_reference=str(args.approval_reference),
        )
    except RollbackDrillError as error:
        print(f"ERROR {error.code}: {error.public_message}", file=sys.stderr)
        return 2
    print(_canonical_json(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
