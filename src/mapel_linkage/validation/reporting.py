"""Aggregate-only synthetic evaluation report persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from mapel_linkage.domain.errors import ValidationReportError
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.paths import PathPolicy


class SafeSummary(Protocol):
    def safe_summary(self) -> Mapping[str, object]: ...


def write_aggregate_validation_report(
    *,
    reports: Mapping[str, SafeSummary],
    path: str,
    policy: PathPolicy,
    warning: str = (
        "Synthetic testing establishes software behaviour only; it does not validate "
        "linkage accuracy on real populations or systems."
    ),
) -> Path:
    destination = policy.resolve_output(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "0.1",
        "evaluation_scope": "synthetic_mechanical_evaluation",
        "real_data_validation_status": "not_established",
        "warning": warning,
        "reports": {name: dict(report.safe_summary()) for name, report in sorted(reports.items())},
    }
    try:
        atomic_write_text(destination, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except (OSError, TypeError, ValueError):
        raise ValidationReportError(
            "ML-VALID-009",
            "An aggregate validation report could not be written safely.",
        ) from None
    return destination
