#!/usr/bin/env python3
from __future__ import annotations
import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".arrow", ".cbm", ".csv", ".db", ".duckdb", ".feather", ".joblib", ".onnx", ".parquet", ".pickle", ".pkl", ".pt", ".pth", ".sqlite", ".tsv", ".ubj", ".xls", ".xlsx"}
REQUIRED_FILES = {"README.md", "AGENTS.md", "pyproject.toml", "docs/README.md", "docs/architecture/ARCHITECTURE.md", "docs/governance/PRIVACY_THREAT_MODEL.md", "docs/references/references.bib"}
CHATGPT_MARKERS = ("\ue200cite", "\ue200filecite", "\ue200turn", "\ue201")

def candidate_files() -> list[Path]:
    excluded = {".git", ".venv", "build", "dist", "private", "data", "artifacts"}
    return sorted(p for p in ROOT.rglob("*") if p.is_file() and (not p.relative_to(ROOT).parts or p.relative_to(ROOT).parts[0] not in excluded))

def validate_repository() -> list[str]:
    errors: list[str] = []
    present = {str(p.relative_to(ROOT)) for p in candidate_files()}
    for required in sorted(REQUIRED_FILES - present):
        errors.append(f"missing required file: {required}")
    for path in candidate_files():
        rel = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden repository file type: {rel}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(marker in text for marker in CHATGPT_MARKERS) or re.search(r"turn\d+(?:search|view|file)\d+", text):
            errors.append(f"transient citation marker: {rel}")
        placeholder = "[PRIVATE_GITHUB_" + "REPOSITORY_URL]"
        if placeholder in text and rel != Path("scripts/verify_repository.py"):
            errors.append(f"placeholder repository URL: {rel}")
    pyproject = (ROOT / "pyproject.toml").read_text()
    if 'name = "mapel-linkage-engine"' not in pyproject:
        errors.append("distribution name mismatch")
    if '"Private :: Do Not Upload"' not in pyproject:
        errors.append("publication guard missing")
    bib = (ROOT / "docs/references/references.bib").read_text()
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
