"""Filesystem policy for local-only configuration and artifacts."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from mapel_linkage.governance.errors import SafeError, SafeErrorCode

_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def _looks_remote(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if "\x00" in stripped:
        return True
    if _WINDOWS_DRIVE.match(stripped):
        return False
    if stripped.startswith(("\\\\", "//")):
        return True
    return _URI_SCHEME.match(stripped) is not None


def _resolve(base: Path, raw: str) -> Path:
    if raw != raw.strip():
        raise SafeError(
            SafeErrorCode.PATH_POLICY,
            "Paths must not contain leading or trailing whitespace.",
        )
    if _looks_remote(raw):
        raise SafeError(
            SafeErrorCode.PATH_POLICY,
            "Remote, URI, UNC, or malformed paths are not permitted.",
        )
    candidate = Path(raw)
    if "~" in candidate.parts:
        raise SafeError(
            SafeErrorCode.PATH_POLICY,
            "Home-directory expansion is not permitted in project configuration.",
        )
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=False)


def _is_within(candidate: Path, roots: tuple[Path, ...]) -> bool:
    return any(candidate == root or candidate.is_relative_to(root) for root in roots)


@dataclass(frozen=True, slots=True)
class PathPolicy:
    """Resolved host and project path allow-lists.

    Paths are intentionally hidden from ``repr`` because operational roots may
    reveal local infrastructure details.
    """

    project_root: Path = field(repr=False)
    input_roots: tuple[Path, ...] = field(repr=False)
    output_roots: tuple[Path, ...] = field(repr=False)
    host_input_roots: tuple[Path, ...] = field(repr=False)
    host_output_roots: tuple[Path, ...] = field(repr=False)

    @classmethod
    def build(
        cls,
        *,
        project_root: Path,
        configured_input_roots: Iterable[str],
        configured_output_roots: Iterable[str],
        host_input_roots: Iterable[Path] | None = None,
        host_output_roots: Iterable[Path] | None = None,
    ) -> PathPolicy:
        root = project_root.resolve(strict=False)
        default_host_inputs = (root / "data", root / "private")
        default_host_outputs = (root / "private", root / "artifacts")
        host_inputs = tuple(
            path.resolve(strict=False)
            for path in (host_input_roots or default_host_inputs)
        )
        host_outputs = tuple(
            path.resolve(strict=False)
            for path in (host_output_roots or default_host_outputs)
        )
        inputs = tuple(_resolve(root, raw) for raw in configured_input_roots)
        outputs = tuple(_resolve(root, raw) for raw in configured_output_roots)

        if not inputs or not outputs:
            raise SafeError(
                SafeErrorCode.PATH_POLICY,
                "At least one input root and one output root are required.",
            )
        if any(not _is_within(path, host_inputs) for path in inputs):
            raise SafeError(
                SafeErrorCode.PATH_POLICY,
                "A configured input root is outside the host-approved boundary.",
            )
        if any(not _is_within(path, host_outputs) for path in outputs):
            raise SafeError(
                SafeErrorCode.PATH_POLICY,
                "A configured output root is outside the host-approved boundary.",
            )
        return cls(root, inputs, outputs, host_inputs, host_outputs)

    def resolve_input(self, raw: str) -> Path:
        candidate = _resolve(self.project_root, raw)
        if not _is_within(candidate, self.input_roots):
            raise SafeError(
                SafeErrorCode.PATH_POLICY,
                "An input path is outside the configured input roots.",
            )
        return candidate

    def resolve_output(self, raw: str) -> Path:
        candidate = _resolve(self.project_root, raw)
        if not _is_within(candidate, self.output_roots):
            raise SafeError(
                SafeErrorCode.PATH_POLICY,
                "An output path is outside the configured output roots.",
            )
        return candidate

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "input_root_count": len(self.input_roots),
            "output_root_count": len(self.output_roots),
            "platform": os.name,
        }
