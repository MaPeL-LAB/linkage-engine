"""Strict helpers for package-owned SQL identifier construction."""

from __future__ import annotations

import re

from mapel_linkage.domain.errors import DataPlaneError

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(identifier: str) -> str:
    """Return a safe identifier or raise without echoing the submitted value."""

    if not _IDENTIFIER.fullmatch(identifier):
        raise DataPlaneError("ML-DATA-002", "An unsafe internal SQL identifier was rejected.")
    return identifier


def quote_identifier(identifier: str) -> str:
    """Quote a previously validated identifier for DuckDB SQL."""

    return f'"{validate_identifier(identifier)}"'


def quote_source_identifier(identifier: str) -> str:
    """Quote a configured source column without treating it as SQL syntax.

    Source columns may contain spaces or punctuation, so they cannot use the
    stricter internal identifier grammar. Double quotes are escaped according
    to SQL identifier rules. The submitted value is never included in errors.
    """

    if not identifier or "\x00" in identifier or len(identifier) > 256:
        raise DataPlaneError("ML-DATA-014", "An unsafe source column identifier was rejected.")
    return '"' + identifier.replace('"', '""') + '"'
