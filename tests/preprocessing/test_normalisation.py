from __future__ import annotations

from datetime import date

import pytest

from mapel_linkage.configuration.models import VariableConfig
from mapel_linkage.domain import PreprocessingError
from mapel_linkage.preprocessing.normalisation import normalise_value


def _variable(**overrides: object) -> VariableConfig:
    payload: dict[str, object] = {
        "id": "generic_value",
        "data_type": "string",
        "source_columns": {"source_a": "source value"},
        "normalisation": (),
        "missingness": {
            "blank_is_missing": True,
            "comparison_policy": "explicit_missing_level",
        },
        "restricted_output": False,
    }
    payload.update(overrides)
    return VariableConfig.model_validate(payload)


def test_string_normalisation_is_allow_list_driven() -> None:
    variable = _variable(
        normalisation=(
            {"kind": "unicode_normalize", "form": "NFKC"},
            {"kind": "casefold"},
            {"kind": "strip"},
            {"kind": "collapse_whitespace"},
        )
    )

    assert normalise_value("  \uff21LPHA\t  Beta  ", variable) == "alpha beta"


def test_blank_value_becomes_explicit_missing() -> None:
    variable = _variable(normalisation=({"kind": "strip"},))

    assert normalise_value("   ", variable) is None
    assert normalise_value(None, variable) is None


def test_configured_date_formats_are_applied() -> None:
    variable = _variable(
        data_type="date",
        normalisation=({"kind": "parse_date", "formats": ("%d/%m/%Y",)},),
    )

    assert normalise_value("31/12/2025", variable) == date(2025, 12, 31)


def test_integer_conversion_rejects_fractional_values_without_echoing_them() -> None:
    sentinel = "123.75-SYNTHETIC-SENTINEL"
    variable = _variable(
        data_type="integer",
        normalisation=({"kind": "numeric_cast", "target": "integer"},),
    )

    with pytest.raises(PreprocessingError) as exc_info:
        normalise_value(sentinel, variable)

    assert exc_info.value.code == "ML-PREP-006"
    assert sentinel not in str(exc_info.value)


def test_boolean_conversion_is_bounded() -> None:
    variable = _variable(data_type="boolean")

    assert normalise_value("yes", variable) is True
    assert normalise_value("0", variable) is False
