"""Value-safe YAML and JSON configuration loading."""

from __future__ import annotations

import json
import math
from collections.abc import Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode

from mapel_linkage.configuration.models import LinkageConfig
from mapel_linkage.governance.errors import (
    SafeError,
    SafeErrorCode,
    safe_error_from_validation,
)

_MAX_CONFIG_BYTES = 2 * 1024 * 1024
_MAX_YAML_ALIASES = 64
_MAX_STRUCTURE_DEPTH = 64
_MAX_STRUCTURE_NODES = 100_000


class _LimitedSafeLoader(yaml.SafeLoader):
    """SafeLoader with alias, duplicate-key, and merge-key controls."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._alias_count = 0

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(AliasEvent):
            self._alias_count += 1
            if self._alias_count > _MAX_YAML_ALIASES:
                raise yaml.YAMLError("YAML alias limit exceeded")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        if not isinstance(node, MappingNode):
            raise yaml.YAMLError("Expected a YAML mapping")
        mapping: dict[Hashable, Any] = {}
        for key_node, value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise yaml.YAMLError("YAML merge keys are not permitted")
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise yaml.YAMLError("YAML mapping keys must be strings")
            if key in mapping:
                raise yaml.YAMLError("Duplicate YAML mapping key")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclass(frozen=True, slots=True)
class LoadedConfiguration:
    config: LinkageConfig
    source_format: Literal["yaml", "json"]
    size_bytes: int
    source_path: Path | None = field(default=None, repr=False)


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError("Duplicate JSON mapping key")
        mapping[key] = value
    return mapping


def _reject_json_constant(_: str) -> object:
    raise ValueError("Non-finite JSON numbers are not permitted")


def _parse_text(text: str, source_format: Literal["yaml", "json"]) -> object:
    try:
        if source_format == "json":
            payload: object = json.loads(
                text,
                object_pairs_hook=_json_object,
                parse_constant=_reject_json_constant,
            )
            return payload
        payload = yaml.load(text, Loader=_LimitedSafeLoader)
        return payload
    except (json.JSONDecodeError, yaml.YAMLError, ValueError):
        raise SafeError(
            SafeErrorCode.CONFIG_PARSE,
            "Configuration could not be parsed as safe YAML or JSON.",
        ) from None


def _validate_structure(payload: object) -> None:
    """Require a bounded, acyclic, JSON-compatible configuration tree."""

    node_count = 0
    active_containers: set[int] = set()

    def visit(value: object, depth: int) -> None:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_STRUCTURE_NODES or depth > _MAX_STRUCTURE_DEPTH:
            raise SafeError(
                SafeErrorCode.CONFIG_PARSE,
                "Configuration structure exceeds permitted complexity.",
            )
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise SafeError(
                    SafeErrorCode.CONFIG_PARSE,
                    "Configuration contains an unsupported scalar value.",
                )
            return
        if isinstance(value, dict):
            identity = id(value)
            if identity in active_containers:
                raise SafeError(
                    SafeErrorCode.CONFIG_PARSE,
                    "Configuration must not contain recursive structures.",
                )
            active_containers.add(identity)
            try:
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise SafeError(
                            SafeErrorCode.CONFIG_PARSE,
                            "Configuration mapping keys must be strings.",
                        )
                    visit(item, depth + 1)
            finally:
                active_containers.remove(identity)
            return
        if isinstance(value, list):
            identity = id(value)
            if identity in active_containers:
                raise SafeError(
                    SafeErrorCode.CONFIG_PARSE,
                    "Configuration must not contain recursive structures.",
                )
            active_containers.add(identity)
            try:
                for item in value:
                    visit(item, depth + 1)
            finally:
                active_containers.remove(identity)
            return
        raise SafeError(
            SafeErrorCode.CONFIG_PARSE,
            "Configuration contains an unsupported scalar or container type.",
        )

    visit(payload, 0)


def load_config_text(
    text: str,
    *,
    source_format: Literal["yaml", "json"],
) -> LoadedConfiguration:
    encoded_size = len(text.encode("utf-8"))
    if encoded_size > _MAX_CONFIG_BYTES:
        raise SafeError(
            SafeErrorCode.CONFIG_READ,
            "Configuration exceeds the permitted size.",
        )
    payload = _parse_text(text, source_format)
    _validate_structure(payload)
    if not isinstance(payload, dict):
        raise SafeError(
            SafeErrorCode.CONFIG_PARSE,
            "Configuration must contain one top-level mapping.",
        )
    try:
        config = LinkageConfig.model_validate(payload)
    except ValidationError as error:
        raise safe_error_from_validation(error) from None
    return LoadedConfiguration(config=config, source_format=source_format, size_bytes=encoded_size)


def load_config(path: Path) -> LoadedConfiguration:
    suffix = path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise SafeError(
            SafeErrorCode.CONFIG_READ,
            "Configuration file type must be YAML or JSON.",
        )
    try:
        size = path.stat().st_size
        if size > _MAX_CONFIG_BYTES:
            raise SafeError(
                SafeErrorCode.CONFIG_READ,
                "Configuration exceeds the permitted size.",
            )
        text = path.read_text(encoding="utf-8")
    except SafeError:
        raise
    except (OSError, UnicodeError):
        raise SafeError(
            SafeErrorCode.CONFIG_READ,
            "Configuration could not be read as UTF-8 text.",
        ) from None
    loaded = load_config_text(
        text,
        source_format="json" if suffix == ".json" else "yaml",
    )
    return LoadedConfiguration(
        config=loaded.config,
        source_format=loaded.source_format,
        size_bytes=loaded.size_bytes,
        source_path=path.resolve(strict=False),
    )
