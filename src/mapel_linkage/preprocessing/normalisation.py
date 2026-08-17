"""Allow-listed, value-safe canonical normalisation functions."""

from __future__ import annotations

import math
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from mapel_linkage.configuration.models import (
    CasefoldTransform,
    CollapseWhitespaceTransform,
    NumericCastTransform,
    ParseDateTransform,
    StripTransform,
    UnicodeNormalizeTransform,
    VariableConfig,
)
from mapel_linkage.domain.errors import PreprocessingError

type CanonicalValue = str | int | float | bool | date | None


def _is_blank(value: object) -> bool:
    return isinstance(value, str) and not value.strip()


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _parse_date(value: object, formats: tuple[str, ...]) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _as_text(value).strip()
    for date_format in formats:
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            continue
    raise PreprocessingError("ML-PREP-006", "A configured source value could not be normalised.")


def _parse_integer(value: object) -> int:
    if isinstance(value, bool):
        raise PreprocessingError(
            "ML-PREP-006", "A configured source value could not be normalised."
        )
    number = Decimal(_as_text(value).strip())
    if not number.is_finite() or number != number.to_integral_value():
        raise PreprocessingError(
            "ML-PREP-006", "A configured source value could not be normalised."
        )
    return int(number)


def _parse_float(value: object) -> float:
    if isinstance(value, bool):
        raise PreprocessingError(
            "ML-PREP-006", "A configured source value could not be normalised."
        )
    number = float(_as_text(value).strip())
    if not math.isfinite(number):
        raise PreprocessingError(
            "ML-PREP-006", "A configured source value could not be normalised."
        )
    return number


def _parse_boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    text = _as_text(value).strip().casefold()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise PreprocessingError("ML-PREP-006", "A configured source value could not be normalised.")


def normalise_value(value: object, variable: VariableConfig) -> CanonicalValue:
    """Apply only configuration-schema-approved transformations.

    The function deliberately reports a stable public error instead of the
    submitted value, source column, or transformation input.
    """

    if value is None or (variable.missingness.blank_is_missing and _is_blank(value)):
        return None

    current: object = value
    try:
        for transform in variable.normalisation:
            if isinstance(transform, StripTransform):
                current = _as_text(current).strip()
            elif isinstance(transform, CasefoldTransform):
                current = _as_text(current).casefold()
            elif isinstance(transform, UnicodeNormalizeTransform):
                current = unicodedata.normalize(transform.form, _as_text(current))
            elif isinstance(transform, CollapseWhitespaceTransform):
                current = " ".join(_as_text(current).split())
            elif isinstance(transform, ParseDateTransform):
                current = _parse_date(current, transform.formats)
            elif isinstance(transform, NumericCastTransform):
                current = (
                    _parse_integer(current)
                    if transform.target == "integer"
                    else _parse_float(current)
                )
            else:  # pragma: no cover - the discriminated schema prevents this branch.
                raise PreprocessingError(
                    "ML-PREP-002", "An unsupported normalisation operation was rejected."
                )

        if variable.missingness.blank_is_missing and _is_blank(current):
            return None

        if variable.data_type in {"string", "categorical"}:
            return _as_text(current)
        if variable.data_type == "date":
            if isinstance(current, datetime):
                return current.date()
            if isinstance(current, date):
                return current
            return date.fromisoformat(_as_text(current).strip())
        if variable.data_type == "integer":
            return _parse_integer(current)
        if variable.data_type == "float":
            return _parse_float(current)
        if variable.data_type == "boolean":
            return _parse_boolean(current)
    except PreprocessingError:
        raise
    except (InvalidOperation, UnicodeError, ValueError, TypeError, OverflowError):
        raise PreprocessingError(
            "ML-PREP-006", "A configured source value could not be normalised."
        ) from None

    raise PreprocessingError("ML-PREP-002", "An unsupported canonical data type was rejected.")
