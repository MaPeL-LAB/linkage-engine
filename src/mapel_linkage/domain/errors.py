"""Value-safe runtime errors for the local linkage data plane."""

from __future__ import annotations


class LinkageRuntimeError(RuntimeError):
    """A stable, non-sensitive runtime error suitable for CLI translation."""

    __slots__ = ("code", "public_message")

    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code}: {public_message}")


class DataPlaneError(LinkageRuntimeError):
    """Raised when a local table operation fails without exposing row values."""


class CandidateGenerationError(LinkageRuntimeError):
    """Raised when safe candidate generation cannot complete."""


class CandidateBudgetExceeded(CandidateGenerationError):
    """Raised before materialisation when the configured pair budget is exceeded."""


class PreprocessingError(LinkageRuntimeError):
    """Raised when configured ingestion or normalisation cannot complete safely."""
