"""Value-safe errors for configuration and governance boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import ValidationError

_SAFE_LOCATION_FIELDS = frozenset(
    {
        "schema_version",
        "project",
        "project_id",
        "entity_type",
        "linkage_mode",
        "assignment_constraint",
        "random_seed",
        "runtime",
        "backend",
        "maximum_candidate_pairs",
        "deterministic_mode",
        "privacy",
        "allowed_input_roots",
        "allowed_output_roots",
        "allow_remote_uris",
        "allow_network_access",
        "log_policy",
        "include_tracebacks",
        "datasets",
        "id",
        "role",
        "path",
        "format",
        "record_id_column",
        "variables",
        "data_type",
        "source_columns",
        "normalisation",
        "missingness",
        "blank_is_missing",
        "comparison_policy",
        "restricted_output",
        "kind",
        "form",
        "formats",
        "target",
        "variable",
        "length",
        "maximum_days",
        "terms",
        "maximum_distance",
        "q",
        "unit",
        "scale",
        "minimum",
        "value",
        "deterministic_anchors",
        "predicate",
        "require_unique_left",
        "require_unique_right",
        "action",
        "allow_as_training_truth",
        "blocking",
        "rules",
        "comparisons",
        "function",
        "levels",
        "labels",
        "source",
        "entity_group_columns",
        "household_group_columns",
        "protocol_version",
        "permit_weak_labels_for_training",
        "permit_unverified_crosswalk",
        "models",
        "fellegi_sunter",
        "boosted_tree",
        "ranking",
        "neural",
        "enabled",
        "implementation",
        "model_id",
        "require_verified_labels",
        "query_side",
        "top_k",
        "calibration",
        "method",
        "source_model",
        "partition",
        "require_independent_partition",
        "model_selection",
        "mode",
        "selection_partition",
        "primary_metric",
        "test_partition_may_select_model",
        "assignment",
        "solver",
        "constraint",
        "no_match",
        "utility",
        "deterministic_tie_breaking",
        "decision_policy",
        "confirmed",
        "minimum_probability",
        "minimum_probability_margin",
        "require_assignment",
        "require_valid_calibration",
        "review_required",
        "maximum_top_probability",
        "require_complete_candidate_search",
        "unresolved",
        "fallback",
        "validation",
        "split",
        "training_fraction",
        "validation_fraction",
        "calibration_fraction",
        "decision_fraction",
        "test_fraction",
        "hard_negative_sampling",
        "verified_nonmatches_only",
        "candidate_recall_k",
        "outputs",
        "restricted_directory",
        "permitted_fields",
        "permitted_variable_values",
    }
)


def _safe_location(parts: Iterable[object]) -> str:
    rendered: list[str] = []
    for part in parts:
        if isinstance(part, int):
            rendered.append("*")
        elif isinstance(part, str) and part in _SAFE_LOCATION_FIELDS:
            rendered.append(part)
        else:
            rendered.append("*")
    return ".".join(rendered)


class SafeErrorCode(StrEnum):
    """Stable public error codes that never require sensitive values."""

    CONFIG_READ = "ML-CONFIG-001"
    CONFIG_PARSE = "ML-CONFIG-002"
    CONFIG_VALIDATION = "ML-CONFIG-003"
    CONFIG_UNSUPPORTED = "ML-CONFIG-004"
    CONFIG_SCHEMA_WRITE = "ML-CONFIG-005"
    PATH_POLICY = "ML-PATH-001"
    MANIFEST_WRITE = "ML-MANIFEST-001"
    SYNTHETIC_GENERATION = "ML-SYNTHETIC-001"


class SafeError(Exception):
    """An exception whose public representation excludes submitted values."""

    __slots__ = ("code", "locations", "message")

    def __init__(
        self,
        code: SafeErrorCode,
        message: str,
        locations: Iterable[str] = (),
    ) -> None:
        self.code = code
        self.message = message
        self.locations = tuple(dict.fromkeys(locations))
        super().__init__(message)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return f"SafeError(code={self.code!s}, locations={len(self.locations)})"

    def render(self) -> str:
        suffix = ""
        if self.locations:
            suffix = " Locations: " + ", ".join(self.locations[:8])
            if len(self.locations) > 8:
                suffix += ", …"
        return f"ERROR {self.code}: {self.message}{suffix}"


def safe_error_from_validation(error: ValidationError) -> SafeError:
    """Translate Pydantic validation details without copying input values."""

    locations: list[str] = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        location = _safe_location(item.get("loc", ()))
        if location:
            locations.append(location)
    return SafeError(
        SafeErrorCode.CONFIG_VALIDATION,
        "Configuration does not satisfy the validated schema.",
        locations,
    )
