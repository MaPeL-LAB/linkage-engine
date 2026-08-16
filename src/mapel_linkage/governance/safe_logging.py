"""Typed aggregate-only logging."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError

from mapel_linkage.governance.errors import SafeError, safe_error_from_validation

type SafeToken = Annotated[
    StrictStr,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$"),
]


class SafeLogEvent(BaseModel):
    """The complete allow-list of fields accepted by the application logger."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    event: SafeToken
    stage: SafeToken | None = None
    run_id: SafeToken | None = None
    model_id: SafeToken | None = None
    rule_id: SafeToken | None = None
    count: Annotated[StrictInt, Field(ge=0)] | None = None
    duration_ms: Annotated[StrictInt, Field(ge=0)] | None = None
    status: SafeToken | None = None
    version: SafeToken | None = None
    digest: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{12,64}$")] | None = None
    safe_error_code: (
        Annotated[
            StrictStr,
            Field(pattern=r"^ML-[A-Z]+-[0-9]{3}$"),
        ]
        | None
    ) = None


class SafeLogger:
    """A logger that accepts typed events rather than arbitrary messages."""

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def emit(self, event: SafeLogEvent) -> None:
        payload = event.model_dump(mode="json", exclude_none=True)
        self._logger.info(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def build_safe_log_event(payload: Mapping[str, object]) -> SafeLogEvent:
    """Validate an event while sanitizing rejected keys and values."""

    try:
        return SafeLogEvent.model_validate(dict(payload))
    except ValidationError as error:
        safe_error: SafeError = safe_error_from_validation(error)
        raise safe_error from None
