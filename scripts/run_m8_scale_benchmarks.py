#!/usr/bin/env python3
"""Plan, execute, and resume aggregate-only M8 synthetic scale evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
ALLOWED_CONFIG = ROOT / "configs" / "examples" / "synthetic_link_only.yaml"
DEFAULT_OUTPUT = "artifacts/m8_scale_benchmarks"
DEFAULT_COUNTS = "100,250,500,1000,2000"
DEFAULT_REPETITIONS = 2
DEFAULT_WORKERS = 10
MAXIMUM_WORKERS = 10
MAXIMUM_CASES = 50
MAXIMUM_ENTITY_COUNT = 100_000
BENCHMARK_ID = "m8_complete_synthetic_scale_v1"
PLAN_KEYS = {
    "benchmark_id",
    "cases",
    "configuration_digest",
    "contains_candidate_pairs",
    "contains_identifiers",
    "contains_local_paths",
    "contains_record_data",
    "implementation_digest",
    "maximum_workers",
    "operational_validity",
    "plan_digest",
    "plan_schema_version",
    "random_seed",
    "workers",
}
CASE_KEYS = {"case_id", "entity_count", "repetition"}
REPORT_KEYS = {
    "benchmark_id",
    "case_id",
    "child_system_cpu_seconds",
    "child_user_cpu_seconds",
    "contains_candidate_pairs",
    "contains_identifiers",
    "contains_local_paths",
    "contains_record_data",
    "elapsed_seconds",
    "entity_count",
    "maximum_resident_set_kib",
    "operational_validity",
    "plan_digest",
    "platform",
    "python_version",
    "random_seed",
    "repetition",
    "report_digest",
    "report_schema_version",
    "status",
    "workflow_summary_digest",
}


class ScaleBenchmarkError(ValueError):
    """Fail-closed error whose input values are never rendered."""


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_symlink_components(path: Path) -> None:
    lexical = path.absolute()
    if any(component.is_symlink() for component in (*reversed(lexical.parents), lexical)):
        raise ScaleBenchmarkError("A scale-benchmark path contains a symbolic link.")


def _resolve_output(raw: str) -> Path:
    relative = Path(raw)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] != "artifacts"
        or ".." in relative.parts
    ):
        raise ScaleBenchmarkError("Scale outputs must remain under the ignored artifact root.")
    candidate = ROOT / relative
    _reject_symlink_components(candidate)
    resolved = candidate.resolve(strict=False)
    artifact_root = (ROOT / "artifacts").resolve(strict=False)
    if not resolved.is_relative_to(artifact_root) or resolved == artifact_root:
        raise ScaleBenchmarkError("Scale outputs must use a bounded artifact subdirectory.")
    if resolved.exists() and not resolved.is_dir():
        raise ScaleBenchmarkError("The configured scale output is not a directory.")
    return resolved


def _resolve_config(raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ScaleBenchmarkError("Scale benchmarks require the package-owned synthetic config.")
    candidate = ROOT / relative
    _reject_symlink_components(candidate)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise ScaleBenchmarkError("The package-owned synthetic config is unavailable.") from None
    if resolved != ALLOWED_CONFIG.resolve(strict=True) or not resolved.is_file():
        raise ScaleBenchmarkError("Scale benchmarks require the package-owned synthetic config.")
    return resolved


def _resolve_python(raw: str) -> Path:
    candidate = Path(raw).expanduser()
    executable = Path(os.path.abspath(candidate))
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ScaleBenchmarkError("The selected Python interpreter is not executable.")
    completed = subprocess.run(
        [str(executable), "-c", "import json,sys; print(json.dumps(sys.version_info[:3]))"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        version = json.loads(completed.stdout)
    except json.JSONDecodeError:
        version = []
    if completed.returncode != 0 or version[:2] != [3, 12]:
        raise ScaleBenchmarkError("M8 scale benchmarks require Python 3.12.")
    return executable


def _parse_counts(raw: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item) for item in raw.split(","))
    except ValueError:
        raise ScaleBenchmarkError("Entity counts must be comma-separated integers.") from None
    if (
        not counts
        or tuple(sorted(set(counts))) != counts
        or any(count < 100 or count > MAXIMUM_ENTITY_COUNT for count in counts)
    ):
        raise ScaleBenchmarkError("Entity counts must be unique, increasing, and within bounds.")
    return counts


def _case_id(entity_count: int, repetition: int) -> str:
    return f"entities-{entity_count:06d}.repetition-{repetition:02d}"


def _build_plan(
    *,
    config: Path,
    counts: tuple[int, ...],
    repetitions: int,
    workers: int,
) -> dict[str, object]:
    if repetitions < 1 or repetitions > 5:
        raise ScaleBenchmarkError("Repetitions must be between one and five.")
    if workers < 1 or workers > MAXIMUM_WORKERS:
        raise ScaleBenchmarkError("Workers must be between one and ten.")
    cases = tuple(
        {
            "case_id": _case_id(entity_count, repetition),
            "entity_count": entity_count,
            "repetition": repetition,
        }
        for entity_count in counts
        for repetition in range(1, repetitions + 1)
    )
    if len(cases) > MAXIMUM_CASES:
        raise ScaleBenchmarkError("The bounded scale plan permits at most fifty cases.")
    payload: dict[str, object] = {
        "plan_schema_version": "1",
        "benchmark_id": BENCHMARK_ID,
        "configuration_digest": _sha256_file(config),
        "implementation_digest": _sha256_file(SCRIPT_PATH),
        "random_seed": 20260816,
        "workers": workers,
        "maximum_workers": MAXIMUM_WORKERS,
        "cases": cases,
        "contains_record_data": False,
        "contains_identifiers": False,
        "contains_candidate_pairs": False,
        "contains_local_paths": False,
        "operational_validity": "not_established",
    }
    return {**payload, "plan_digest": _digest(payload)}


def _write_once_or_same(path: Path, payload: object) -> None:
    _reject_symlink_components(path)
    text = _canonical_text(payload)
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            raise ScaleBenchmarkError("Conflicting aggregate scale evidence already exists.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ScaleBenchmarkError("A scale-evidence temporary path is unavailable.")
    try:
        temporary.write_text(text, encoding="utf-8")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                raise ScaleBenchmarkError(
                    "Conflicting aggregate scale evidence already exists."
                ) from None
        temporary.unlink()
    except ScaleBenchmarkError:
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise
    except OSError:
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise ScaleBenchmarkError("Aggregate scale evidence could not be written safely.") from None


def _load_case(path: Path, *, plan_digest: str, expected: dict[str, object]) -> dict[str, Any]:
    _reject_symlink_components(path)
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ScaleBenchmarkError("Retained scale evidence is unreadable.") from None
    if (
        not isinstance(payload, dict)
        or len(text.encode("utf-8")) > 64 * 1024
        or text != _canonical_text(payload)
        or set(payload) != REPORT_KEYS
    ):
        raise ScaleBenchmarkError("Retained scale evidence is not canonical.")
    report_digest = payload.get("report_digest")
    body = {key: value for key, value in payload.items() if key != "report_digest"}
    elapsed = payload.get("elapsed_seconds")
    user_cpu = payload.get("child_user_cpu_seconds")
    system_cpu = payload.get("child_system_cpu_seconds")
    maximum_rss = payload.get("maximum_resident_set_kib")
    if (
        report_digest != _digest(body)
        or payload.get("report_schema_version") != "1"
        or payload.get("benchmark_id") != BENCHMARK_ID
        or payload.get("plan_digest") != plan_digest
        or payload.get("case_id") != expected["case_id"]
        or payload.get("entity_count") != expected["entity_count"]
        or payload.get("repetition") != expected["repetition"]
        or payload.get("random_seed") != 20260816
        or payload.get("status") != "complete"
        or type(elapsed) not in {float, int}
        or elapsed < 0
        or type(user_cpu) not in {float, int}
        or user_cpu < 0
        or type(system_cpu) not in {float, int}
        or system_cpu < 0
        or type(maximum_rss) is not int
        or maximum_rss < 0
        or re.fullmatch(r"[0-9a-f]{64}", str(payload.get("workflow_summary_digest"))) is None
        or payload.get("platform") not in {"Darwin", "Linux"}
        or re.fullmatch(r"3\.12\.\d+", str(payload.get("python_version"))) is None
        or payload.get("contains_record_data") is not False
        or payload.get("contains_identifiers") is not False
        or payload.get("contains_candidate_pairs") is not False
        or payload.get("contains_local_paths") is not False
        or payload.get("operational_validity") != "not_established"
    ):
        raise ScaleBenchmarkError("Retained scale evidence failed its digest binding.")
    return payload


def _load_bound_plan(
    path: Path,
    *,
    expected_digest: str,
    config: Path,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    _reject_symlink_components(path)
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ScaleBenchmarkError("The retained scale plan is unreadable.") from None
    if (
        not isinstance(payload, dict)
        or len(text.encode("utf-8")) > 64 * 1024
        or text != _canonical_text(payload)
        or set(payload) != PLAN_KEYS
    ):
        raise ScaleBenchmarkError("The retained scale plan is not canonical.")
    body = {key: value for key, value in payload.items() if key != "plan_digest"}
    if (
        payload.get("plan_digest") != expected_digest
        or payload.get("plan_digest") != _digest(body)
        or payload.get("plan_schema_version") != "1"
        or payload.get("benchmark_id") != BENCHMARK_ID
        or payload.get("configuration_digest") != _sha256_file(config)
        or payload.get("implementation_digest") != _sha256_file(SCRIPT_PATH)
        or payload.get("random_seed") != 20260816
        or type(payload.get("workers")) is not int
        or not 1 <= payload["workers"] <= MAXIMUM_WORKERS
        or payload.get("maximum_workers") != MAXIMUM_WORKERS
        or payload.get("contains_record_data") is not False
        or payload.get("contains_identifiers") is not False
        or payload.get("contains_candidate_pairs") is not False
        or payload.get("contains_local_paths") is not False
        or payload.get("operational_validity") != "not_established"
    ):
        raise ScaleBenchmarkError("The retained scale plan failed its digest binding.")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not 1 <= len(raw_cases) <= MAXIMUM_CASES:
        raise ScaleBenchmarkError("The retained scale plan has an invalid case set.")
    cases: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict) or set(raw_case) != CASE_KEYS:
            raise ScaleBenchmarkError("The retained scale plan has an invalid case binding.")
        entity_count = raw_case.get("entity_count")
        repetition = raw_case.get("repetition")
        case_id = raw_case.get("case_id")
        if (
            type(entity_count) is not int
            or type(repetition) is not int
            or not 100 <= entity_count <= MAXIMUM_ENTITY_COUNT
            or not 1 <= repetition <= 5
            or case_id != _case_id(entity_count, repetition)
            or (entity_count, repetition) in seen
        ):
            raise ScaleBenchmarkError("The retained scale plan has an invalid case binding.")
        seen.add((entity_count, repetition))
        cases.append(
            {
                "case_id": case_id,
                "entity_count": entity_count,
                "repetition": repetition,
            }
        )
    if cases != sorted(cases, key=lambda case: (case["entity_count"], case["repetition"])):
        raise ScaleBenchmarkError("The retained scale plan case order is invalid.")
    return payload, cases


def _normalised_max_rss_kib(value: int) -> int:
    return max(0, value // 1024 if platform.system() == "Darwin" else value)


def _execute_case(
    *,
    python: Path,
    config: Path,
    output_root: Path,
    plan_digest: str,
    case: dict[str, object],
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    entity_count = int(case["entity_count"])
    repetition = int(case["repetition"])
    temporary_root = output_root / "temporary"
    _reject_symlink_components(temporary_root)
    temporary_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "MAPEL_TEST_DATA_POLICY": "synthetic_only",
            "MAPEL_RANDOM_SEED": "20260816",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with tempfile.TemporaryDirectory(prefix=f"{case_id}.", dir=temporary_root) as workspace_text:
        workspace = Path(workspace_text)
        started = time.perf_counter()
        completed = subprocess.run(
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
                str(entity_count),
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if completed.returncode != 0 or len(completed.stdout.encode("utf-8")) > 2 * 1024 * 1024:
        raise ScaleBenchmarkError("A synthetic scale case failed closed.")
    try:
        workflow_summary = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise ScaleBenchmarkError(
            "A synthetic scale case emitted an invalid aggregate summary."
        ) from None
    if not isinstance(workflow_summary, dict):
        raise ScaleBenchmarkError("A synthetic scale case emitted an invalid aggregate summary.")
    body: dict[str, object] = {
        "report_schema_version": "1",
        "benchmark_id": BENCHMARK_ID,
        "plan_digest": plan_digest,
        "case_id": case_id,
        "entity_count": entity_count,
        "repetition": repetition,
        "random_seed": 20260816,
        "status": "complete",
        "elapsed_seconds": round(elapsed, 6),
        "child_user_cpu_seconds": round(after.ru_utime - before.ru_utime, 6),
        "child_system_cpu_seconds": round(after.ru_stime - before.ru_stime, 6),
        "maximum_resident_set_kib": _normalised_max_rss_kib(after.ru_maxrss),
        "workflow_summary_digest": _digest(workflow_summary),
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "contains_record_data": False,
        "contains_identifiers": False,
        "contains_candidate_pairs": False,
        "contains_local_paths": False,
        "operational_validity": "not_established",
    }
    report = {**body, "report_digest": _digest(body)}
    _write_once_or_same(output_root / "cases" / f"{case_id}.json", report)
    return report


def _worker_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--python", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plan-digest", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--entity-count", type=int, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    args = parser.parse_args(arguments)
    try:
        python = _resolve_python(args.python)
        config = _resolve_config(args.config)
        output_root = _resolve_output(args.output_dir)
        _plan, cases = _load_bound_plan(
            output_root / "plan.json",
            expected_digest=args.plan_digest,
            config=config,
        )
        expected_id = _case_id(args.entity_count, args.repetition)
        expected_case = {
            "case_id": expected_id,
            "entity_count": args.entity_count,
            "repetition": args.repetition,
        }
        if args.case_id != expected_id or expected_case not in cases:
            raise ScaleBenchmarkError("The worker case binding is inconsistent.")
        _execute_case(
            python=python,
            config=config,
            output_root=output_root,
            plan_digest=args.plan_digest,
            case=expected_case,
        )
    except (OSError, ScaleBenchmarkError, json.JSONDecodeError):
        print("ERROR: An M8 scale benchmark case failed closed.", file=sys.stderr)
        return 2
    print(json.dumps({"case_id": args.case_id, "status": "complete"}, sort_keys=True))
    return 0


def _run_worker(
    *,
    python: Path,
    config: Path,
    output_root: Path,
    plan_digest: str,
    case: dict[str, object],
) -> None:
    relative_output = output_root.relative_to(ROOT)
    completed = subprocess.run(
        [
            str(python),
            str(SCRIPT_PATH),
            "_run-case",
            "--python",
            str(python),
            "--config",
            str(config.relative_to(ROOT)),
            "--output-dir",
            str(relative_output),
            "--plan-digest",
            plan_digest,
            "--case-id",
            str(case["case_id"]),
            "--entity-count",
            str(case["entity_count"]),
            "--repetition",
            str(case["repetition"]),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ScaleBenchmarkError("A parallel scale worker failed closed.")


def _aggregate_summary(plan: dict[str, object], reports: list[dict[str, Any]]) -> dict[str, object]:
    elapsed = [float(report["elapsed_seconds"]) for report in reports]
    rss = [int(report["maximum_resident_set_kib"]) for report in reports]
    body: dict[str, object] = {
        "summary_schema_version": "1",
        "benchmark_id": BENCHMARK_ID,
        "plan_digest": plan["plan_digest"],
        "case_count": len(reports),
        "median_elapsed_seconds": round(statistics.median(elapsed), 6),
        "maximum_elapsed_seconds": round(max(elapsed), 6),
        "maximum_resident_set_kib": max(rss),
        "case_report_digests": tuple(str(report["report_digest"]) for report in reports),
        "contains_record_data": False,
        "contains_identifiers": False,
        "contains_candidate_pairs": False,
        "contains_local_paths": False,
        "operational_validity": "not_established",
    }
    return {**body, "summary_digest": _digest(body)}


def _public_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--config", default="configs/examples/synthetic_link_only.yaml")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--entity-counts", default=DEFAULT_COUNTS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(arguments)
    try:
        python = _resolve_python(args.python)
        config = _resolve_config(args.config)
        output_root = _resolve_output(args.output_dir)
        counts = _parse_counts(args.entity_counts)
        plan = _build_plan(
            config=config,
            counts=counts,
            repetitions=args.repetitions,
            workers=args.workers,
        )
        if args.dry_run:
            print(_canonical_text({**plan, "dry_run": True}), end="")
            return 0

        output_root.mkdir(parents=True, exist_ok=True)
        _write_once_or_same(output_root / "plan.json", plan)
        plan_digest = str(plan["plan_digest"])
        plan, cases = _load_bound_plan(
            output_root / "plan.json",
            expected_digest=plan_digest,
            config=config,
        )
        reports_by_id: dict[str, dict[str, Any]] = {}
        pending: list[dict[str, object]] = []
        for case in cases:
            report_path = output_root / "cases" / f"{case['case_id']}.json"
            if report_path.exists():
                reports_by_id[str(case["case_id"])] = _load_case(
                    report_path,
                    plan_digest=plan_digest,
                    expected=case,
                )
            else:
                pending.append(case)

        if pending:
            with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
                futures = {
                    executor.submit(
                        _run_worker,
                        python=python,
                        config=config,
                        output_root=output_root,
                        plan_digest=plan_digest,
                        case=case,
                    ): case
                    for case in pending
                }
                for future in as_completed(futures):
                    future.result()
                    case = futures[future]
                    reports_by_id[str(case["case_id"])] = _load_case(
                        output_root / "cases" / f"{case['case_id']}.json",
                        plan_digest=plan_digest,
                        expected=case,
                    )

        reports = [reports_by_id[str(case["case_id"])] for case in cases]
        summary = _aggregate_summary(plan, reports)
        _write_once_or_same(output_root / "summary.json", summary)
        print(
            _canonical_text(
                {
                    **summary,
                    "newly_completed_case_count": len(pending),
                    "resumed_case_count": len(cases) - len(pending),
                }
            ),
            end="",
        )
        return 0
    except (OSError, ScaleBenchmarkError, TypeError, ValueError):
        print("ERROR: M8 scale benchmark planning or execution failed closed.", file=sys.stderr)
        return 2


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_run-case":
        return _worker_main(sys.argv[2:])
    return _public_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
