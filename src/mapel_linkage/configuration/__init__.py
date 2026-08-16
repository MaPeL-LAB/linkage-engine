"""Configuration schema, loading, registries, and compilation."""

from __future__ import annotations

from mapel_linkage.configuration.compiler import ExecutionPlan, compile_config
from mapel_linkage.configuration.loader import (
    LoadedConfiguration,
    load_config,
    load_config_text,
)
from mapel_linkage.configuration.models import LinkageConfig
from mapel_linkage.configuration.schema import (
    configuration_json_schema,
    write_configuration_json_schema,
)

__all__ = [
    "ExecutionPlan",
    "LinkageConfig",
    "LoadedConfiguration",
    "compile_config",
    "configuration_json_schema",
    "load_config",
    "load_config_text",
    "write_configuration_json_schema",
]
