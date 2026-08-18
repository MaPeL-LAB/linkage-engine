"""Restricted pipeline output persistence with aggregate-only manifests."""

from __future__ import annotations

import json
from pathlib import Path

from mapel_linkage.configuration.models import OutputConfig
from mapel_linkage.decisions import RelationshipDecision
from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.paths import PathPolicy


def write_relationship_decisions(
    *,
    decisions: tuple[RelationshipDecision, ...],
    output: OutputConfig,
    path: str,
    policy: PathPolicy,
) -> Path:
    """Write only explicitly permitted decision fields under an approved root."""

    destination = policy.resolve_output(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = "".join(
            json.dumps(decision.restricted_mapping(output.permitted_fields), sort_keys=True) + "\n"
            for decision in decisions
        )
        atomic_write_text(destination, text)
    except (OSError, TypeError, ValueError):
        raise PipelineError(
            "ML-PIPE-001", "Restricted relationship decisions could not be written safely."
        ) from None
    return destination


def write_run_manifest(
    *,
    payload: dict[str, object],
    path: str,
    policy: PathPolicy,
) -> Path:
    """Persist one aggregate-only run manifest."""

    destination = policy.resolve_output(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_text(destination, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except (OSError, TypeError, ValueError):
        raise PipelineError(
            "ML-PIPE-002", "The aggregate run manifest could not be written."
        ) from None
    return destination
