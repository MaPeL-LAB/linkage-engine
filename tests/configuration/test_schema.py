from __future__ import annotations

import json

from mapel_linkage.configuration import configuration_json_schema
from tests.helpers import ROOT


def test_schema_forbids_unknown_top_level_properties() -> None:
    schema = configuration_json_schema()
    assert schema["additionalProperties"] is False
    assert "project" in schema["properties"]
    assert "datasets" in schema["properties"]


def test_schema_contains_discriminated_operation_definitions() -> None:
    schema = configuration_json_schema()
    definitions = schema["$defs"]
    assert "UnicodeNormalizeTransform" in definitions
    assert "PrefixEqualPredicate" in definitions
    assert "JaroWinklerComparison" in definitions


def test_committed_schema_matches_pydantic_model() -> None:
    committed = json.loads(
        (ROOT / "schemas/linkage-config.schema.json").read_text(encoding="utf-8")
    )
    assert committed == configuration_json_schema()
