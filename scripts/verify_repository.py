#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "REPOSITORY_MANIFEST.txt"
FORBIDDEN_SUFFIXES = {
    ".arrow",
    ".cbm",
    ".csv",
    ".db",
    ".duckdb",
    ".feather",
    ".joblib",
    ".jsonl",
    ".ndjson",
    ".onnx",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".sqlite",
    ".tsv",
    ".ubj",
    ".xls",
    ".xlsx",
}
REQUIRED_FILES = {
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "docs/README.md",
    "docs/architecture/ARCHITECTURE.md",
    "docs/governance/PRIVACY_THREAT_MODEL.md",
    "docs/references/references.bib",
    "schemas/linkage-config.schema.json",
}
CHATGPT_MARKERS = ("\ue200cite", "\ue200filecite", "\ue200turn", "\ue201")


def candidate_files() -> list[Path]:
    excluded = {
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
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in excluded for part in path.relative_to(ROOT).parts)
    )


def validate_manifest() -> list[str]:
    errors: list[str] = []
    if not MANIFEST_PATH.is_file():
        return ["missing repository manifest"]
    entries: dict[str, tuple[str, int]] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-f]{64})\s+(\d+)\s+(.+)", line)
        if match is None:
            errors.append("malformed repository manifest entry")
            continue
        digest, size_text, relative = match.groups()
        if relative in entries:
            errors.append(f"duplicate repository manifest entry: {relative}")
            continue
        entries[relative] = (digest, int(size_text))

    files = [path for path in candidate_files() if path != MANIFEST_PATH]
    actual = {str(path.relative_to(ROOT)): path for path in files}
    for relative in sorted(set(actual) - set(entries)):
        errors.append(f"repository manifest missing entry: {relative}")
    for relative in sorted(set(entries) - set(actual)):
        errors.append(f"repository manifest has stale entry: {relative}")
    for relative in sorted(set(actual) & set(entries)):
        path = actual[relative]
        expected_digest, expected_size = entries[relative]
        payload = path.read_bytes()
        if len(payload) != expected_size:
            errors.append(f"repository manifest size mismatch: {relative}")
        if hashlib.sha256(payload).hexdigest() != expected_digest:
            errors.append(f"repository manifest digest mismatch: {relative}")
    return errors


def validate_repository() -> list[str]:
    errors: list[str] = []
    present = {str(path.relative_to(ROOT)) for path in candidate_files()}
    for required in sorted(REQUIRED_FILES - present):
        errors.append(f"missing required file: {required}")
    for path in candidate_files():
        relative_path = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden repository file type: {relative_path}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(marker in text for marker in CHATGPT_MARKERS) or re.search(
            r"turn\d+(?:search|view|file)\d+", text
        ):
            errors.append(f"transient citation marker: {relative_path}")
        placeholder = "[PRIVATE_GITHUB_" + "REPOSITORY_URL]"
        if placeholder in text and relative_path != Path("scripts/verify_repository.py"):
            errors.append(f"placeholder repository URL: {relative_path}")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if 'name = "mapel-linkage-engine"' not in pyproject:
        errors.append("distribution name mismatch")
    if '"Private :: Do Not Upload"' not in pyproject:
        errors.append("publication guard missing")
    bib = (ROOT / "docs/references/references.bib").read_text(encoding="utf-8")
    keys = re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE)
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        errors.append("duplicate BibTeX keys: " + ", ".join(duplicates))
    if bib.count("{") != bib.count("}"):
        errors.append("unbalanced BibTeX braces")
    citations: set[str] = set()
    for path in candidate_files():
        if path.suffix == ".md":
            markdown = path.read_text(encoding="utf-8")
            for block in re.findall(r"\[([^\]]*?@[^\]]*?)\]", markdown):
                citations.update(re.findall(r"@([A-Za-z][A-Za-z0-9_:-]+)", block))
    missing = sorted(citations - set(keys))
    if missing:
        errors.append("missing BibTeX keys: " + ", ".join(missing))
    errors.extend(validate_manifest())
    return errors


def members(path: Path) -> list[str]:
    if path.suffix in {".whl", ".zip"}:
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        with tarfile.open(path) as archive:
            return archive.getnames()
    return []


def validate_distributions(directory: Path) -> list[str]:
    errors: list[str] = []
    for archive in directory.glob("*"):
        if not archive.is_file():
            continue
        for member in members(archive):
            lower = member.lower()
            if any(part in lower.split("/") for part in ("private", "data", "artifacts")):
                errors.append(f"restricted directory in {archive.name}: {member}")
            if Path(lower).suffix in FORBIDDEN_SUFFIXES:
                errors.append(f"restricted file in {archive.name}: {member}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", type=Path)
    args = parser.parse_args()
    errors = validate_repository()
    if args.distribution:
        errors.extend(validate_distributions(args.distribution))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Repository verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
