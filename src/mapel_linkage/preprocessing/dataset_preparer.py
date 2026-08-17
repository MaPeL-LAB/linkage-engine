"""Configuration-driven local ingestion and canonical dataset preparation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import cast

from mapel_linkage.configuration.compiler import ExecutionPlan
from mapel_linkage.configuration.models import DatasetConfig, VariableConfig
from mapel_linkage.domain.errors import DataPlaneError, PreprocessingError
from mapel_linkage.domain.table_refs import TableRef
from mapel_linkage.io.duckdb_store import ColumnSpec, DuckDBStore, SqlType
from mapel_linkage.preprocessing.normalisation import CanonicalValue, normalise_value


def _stable_column_suffix(variable_id: str) -> str:
    return hashlib.sha256(variable_id.encode("utf-8")).hexdigest()[:16]


def canonical_value_column(variable_id: str) -> str:
    """Return a deterministic safe internal value-column name."""

    return f"__ml_v_{_stable_column_suffix(variable_id)}"


def canonical_missingness_column(variable_id: str) -> str:
    """Return a deterministic safe internal missingness-column name."""

    return f"__ml_m_{_stable_column_suffix(variable_id)}"


def _sql_type(variable: VariableConfig) -> SqlType:
    return cast(
        SqlType,
        {
            "string": "VARCHAR",
            "categorical": "VARCHAR",
            "date": "DATE",
            "integer": "BIGINT",
            "float": "DOUBLE",
            "boolean": "BOOLEAN",
        }[variable.data_type],
    )


def _identity_text(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _surrogate_record_key(dataset_id: str, record_identifier: object) -> str:
    payload = f"{dataset_id}\x1f{_identity_text(record_identifier)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Structural reference to a canonical local dataset."""

    dataset_id: str
    table: TableRef
    variable_columns: Mapping[str, str] = field(repr=False)
    missingness_columns: Mapping[str, str] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variable_columns",
            MappingProxyType(dict(self.variable_columns)),
        )
        object.__setattr__(
            self,
            "missingness_columns",
            MappingProxyType(dict(self.missingness_columns)),
        )

    def safe_summary(self) -> dict[str, str | int]:
        return {
            "dataset_id": self.dataset_id,
            "row_count": self.table.row_count,
            "variable_count": len(self.variable_columns),
            "schema_digest": self.table.schema_digest,
        }


@dataclass(frozen=True, slots=True)
class PreparedDatasetCatalog:
    """Immutable catalog of canonical table references."""

    datasets: Mapping[str, PreparedDataset] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "datasets", MappingProxyType(dict(self.datasets)))

    def require(self, dataset_id: str) -> PreparedDataset:
        try:
            return self.datasets[dataset_id]
        except KeyError:
            raise PreprocessingError(
                "ML-PREP-009", "A prepared dataset reference is unavailable."
            ) from None

    def safe_summary(self) -> dict[str, int]:
        return {
            "dataset_count": len(self.datasets),
            "row_count": sum(dataset.table.row_count for dataset in self.datasets.values()),
        }


class ConfiguredDatasetPreparer:
    """Prepare configured local datasets without exposing source rows."""

    __slots__ = ("_store",)

    def __init__(self, store: DuckDBStore) -> None:
        self._store = store

    def prepare_all(self, plan: ExecutionPlan) -> PreparedDatasetCatalog:
        prepared = {
            dataset.id: self.prepare_dataset(plan, dataset) for dataset in plan.config.datasets
        }
        return PreparedDatasetCatalog(prepared)

    def prepare_dataset(
        self,
        plan: ExecutionPlan,
        dataset: DatasetConfig,
    ) -> PreparedDataset:
        variables = tuple(
            variable for variable in plan.config.variables if dataset.id in variable.source_columns
        )
        if not variables:
            raise PreprocessingError(
                "ML-PREP-001", "A configured dataset has no canonical variables."
            )
        if dataset.format == "duckdb":
            raise PreprocessingError(
                "ML-PREP-003", "Direct DuckDB source attachment is not enabled in this milestone."
            )

        path = plan.dataset_paths[dataset.id]
        source_columns = self._source_columns(dataset, variables)
        try:
            source_rows = self._store._read_local_rows(
                path,
                dataset.format,
                source_columns,
            )
        except DataPlaneError as error:
            raise PreprocessingError(error.code, error.public_message) from None

        source_index = {column: index for index, column in enumerate(source_columns)}
        value_columns = {variable.id: canonical_value_column(variable.id) for variable in variables}
        missingness_columns = {
            variable.id: canonical_missingness_column(variable.id) for variable in variables
        }
        column_specs: list[ColumnSpec] = [
            ColumnSpec("__ml_dataset_id", "VARCHAR"),
            ColumnSpec("__ml_record_key", "VARCHAR"),
        ]
        for variable in variables:
            column_specs.append(ColumnSpec(value_columns[variable.id], _sql_type(variable)))
            column_specs.append(ColumnSpec(missingness_columns[variable.id], "BOOLEAN"))

        prepared_rows: list[tuple[object, ...]] = []
        observed_keys: set[str] = set()
        for source_row in source_rows:
            record_identifier = source_row[source_index[dataset.record_id_column]]
            if record_identifier is None or (
                isinstance(record_identifier, str) and not record_identifier.strip()
            ):
                raise PreprocessingError(
                    "ML-PREP-004", "A configured record identifier is missing."
                )
            surrogate_key = _surrogate_record_key(dataset.id, record_identifier)
            if surrogate_key in observed_keys:
                raise PreprocessingError(
                    "ML-PREP-005", "Duplicate record identifiers were rejected."
                )
            observed_keys.add(surrogate_key)

            canonical_values: list[object] = [dataset.id, surrogate_key]
            for variable in variables:
                source_column = variable.source_columns[dataset.id]
                value: CanonicalValue = normalise_value(
                    source_row[source_index[source_column]],
                    variable,
                )
                canonical_values.extend((value, value is None))
            prepared_rows.append(tuple(canonical_values))

        table_suffix = hashlib.sha256(
            f"{plan.configuration_digest}|{dataset.id}".encode()
        ).hexdigest()[:16]
        table_name = f"__ml_prepared_{table_suffix}"
        try:
            table = self._store.create_table_from_rows(
                table_name,
                tuple(column_specs),
                prepared_rows,
            )
        except DataPlaneError as error:
            raise PreprocessingError(error.code, error.public_message) from None
        return PreparedDataset(
            dataset_id=dataset.id,
            table=table,
            variable_columns=value_columns,
            missingness_columns=missingness_columns,
        )

    @staticmethod
    def _source_columns(
        dataset: DatasetConfig,
        variables: tuple[VariableConfig, ...],
    ) -> tuple[str, ...]:
        ordered = [dataset.record_id_column]
        ordered.extend(variable.source_columns[dataset.id] for variable in variables)
        return tuple(dict.fromkeys(ordered))
