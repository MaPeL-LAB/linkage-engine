"""Compile validated configuration into an immutable execution plan."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from mapel_linkage.configuration.models import (
    AllPredicate,
    AnyPredicate,
    BlockPredicate,
    LinkageConfig,
    SyntheticTruthSource,
)
from mapel_linkage.configuration.registries import (
    registry_digest,
    resolve_operation,
)
from mapel_linkage.governance.paths import PathPolicy


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Immutable M1 plan with local paths hidden from public representation."""

    configuration_digest: str
    registry_digest: str
    dataset_count: int
    variable_count: int
    random_seed: int
    dataset_paths: Mapping[str, Path] = field(repr=False)
    restricted_output_directory: Path = field(repr=False)
    label_source_path: Path | None = field(repr=False)
    path_policy: PathPolicy = field(repr=False)
    config: LinkageConfig = field(repr=False)

    def safe_summary(self) -> dict[str, int | str]:
        return {
            "configuration_digest": self.configuration_digest,
            "registry_digest": self.registry_digest,
            "dataset_count": self.dataset_count,
            "variable_count": self.variable_count,
            "random_seed": self.random_seed,
        }


def canonical_configuration_digest(config: LinkageConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_predicate(predicate: BlockPredicate) -> None:
    resolve_operation("predicate", predicate.kind)
    if isinstance(predicate, (AllPredicate, AnyPredicate)):
        for term in predicate.terms:
            _resolve_predicate(term)


def _resolve_registries(config: LinkageConfig) -> None:
    for variable in config.variables:
        for transform in variable.normalisation:
            resolve_operation("transform", transform.kind)
    predicates = [anchor.predicate for anchor in config.deterministic_anchors]
    predicates.extend(rule.predicate for rule in config.blocking.rules)
    for predicate in predicates:
        _resolve_predicate(predicate)
    for comparison in config.comparisons:
        resolve_operation("comparison", comparison.function.kind)
    resolve_operation("pair_model", config.models.fellegi_sunter.implementation)
    for model in (config.models.boosted_tree, config.models.neural):
        if model is not None:
            resolve_operation("pair_model", model.implementation)
    if config.models.ranking is not None:
        resolve_operation("ranker", config.models.ranking.implementation)
    resolve_operation("calibrator", config.calibration.method)
    resolve_operation("assignment_solver", config.assignment.solver)


def compile_config(
    config: LinkageConfig,
    *,
    project_root: Path,
    host_input_roots: Iterable[Path] | None = None,
    host_output_roots: Iterable[Path] | None = None,
) -> ExecutionPlan:
    """Compile configuration after Pydantic and semantic validation."""

    _resolve_registries(config)
    path_policy = PathPolicy.build(
        project_root=project_root,
        configured_input_roots=config.privacy.allowed_input_roots,
        configured_output_roots=config.privacy.allowed_output_roots,
        host_input_roots=host_input_roots,
        host_output_roots=host_output_roots,
    )
    dataset_paths = MappingProxyType(
        {dataset.id: path_policy.resolve_input(dataset.path) for dataset in config.datasets}
    )
    restricted_output = path_policy.resolve_output(config.outputs.restricted_directory)
    label_source_path: Path | None = None
    if config.labels is not None and not isinstance(config.labels.source, SyntheticTruthSource):
        label_source_path = path_policy.resolve_input(config.labels.source.path)
    return ExecutionPlan(
        configuration_digest=canonical_configuration_digest(config),
        registry_digest=registry_digest(),
        dataset_count=len(config.datasets),
        variable_count=len(config.variables),
        random_seed=config.project.random_seed,
        dataset_paths=dataset_paths,
        restricted_output_directory=restricted_output,
        label_source_path=label_source_path,
        path_policy=path_policy,
        config=config,
    )
