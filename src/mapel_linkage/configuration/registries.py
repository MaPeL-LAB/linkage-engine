"""Immutable allow-list registries for configuration-selected operations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from mapel_linkage.governance.errors import SafeError, SafeErrorCode

type RegistryCategory = Literal[
    "transform",
    "predicate",
    "comparison",
    "pair_model",
    "ranker",
    "calibrator",
    "assignment_solver",
]


@dataclass(frozen=True, slots=True)
class OperationDescriptor:
    key: str
    category: RegistryCategory
    supported_types: tuple[str, ...] = ()


TRANSFORMS: Final = MappingProxyType(
    {
        key: OperationDescriptor(key, "transform", types)
        for key, types in {
            "strip": ("string", "categorical"),
            "casefold": ("string", "categorical"),
            "unicode_normalize": ("string", "categorical"),
            "collapse_whitespace": ("string", "categorical"),
            "parse_date": ("date",),
            "numeric_cast": ("integer", "float"),
        }.items()
    }
)
PREDICATES: Final = MappingProxyType(
    {
        key: OperationDescriptor(key, "predicate")
        for key in ("exact", "prefix_equal", "date_window", "all", "any")
    }
)
COMPARISONS: Final = MappingProxyType(
    {
        key: OperationDescriptor(key, "comparison", types)
        for key, types in {
            "exact": ("string", "date", "integer", "float", "boolean", "categorical"),
            "jaro_winkler": ("string", "categorical"),
            "levenshtein": ("string", "categorical"),
            "damerau_levenshtein": ("string", "categorical"),
            "qgram": ("string", "categorical"),
            "date_difference": ("date",),
            "numeric_difference": ("integer", "float"),
            "categorical": ("categorical", "string"),
        }.items()
    }
)
PAIR_MODELS: Final = MappingProxyType(
    {
        key: OperationDescriptor(key, "pair_model")
        for key in (
            "splink_duckdb",
            "xgboost_classifier",
            "lightgbm_classifier",
            "pytorch_pair_mlp",
            "stacking_logistic",
        )
    }
)
RANKERS: Final = MappingProxyType(
    {key: OperationDescriptor(key, "ranker") for key in ("xgboost_ranker", "lightgbm_ranker")}
)
CALIBRATORS: Final = MappingProxyType(
    {key: OperationDescriptor(key, "calibrator") for key in ("sigmoid", "isotonic", "beta")}
)
ASSIGNMENT_SOLVERS: Final = MappingProxyType(
    {
        key: OperationDescriptor(key, "assignment_solver")
        for key in ("ortools_min_cost_flow", "unconstrained")
    }
)

_REGISTRIES: Final = MappingProxyType(
    {
        "transform": TRANSFORMS,
        "predicate": PREDICATES,
        "comparison": COMPARISONS,
        "pair_model": PAIR_MODELS,
        "ranker": RANKERS,
        "calibrator": CALIBRATORS,
        "assignment_solver": ASSIGNMENT_SOLVERS,
    }
)


def resolve_operation(category: RegistryCategory, key: str) -> OperationDescriptor:
    """Resolve an operation without importing or reflecting user-supplied paths."""

    descriptor = _REGISTRIES[category].get(key)
    if descriptor is None:
        raise SafeError(
            SafeErrorCode.CONFIG_UNSUPPORTED,
            "Configuration references an unsupported allow-list operation.",
            (category,),
        )
    return descriptor


def registry_snapshot() -> dict[str, tuple[str, ...]]:
    return {category: tuple(sorted(registry)) for category, registry in sorted(_REGISTRIES.items())}


def registry_digest() -> str:
    payload = json.dumps(registry_snapshot(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
