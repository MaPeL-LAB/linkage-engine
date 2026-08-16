"""A deliberately small local DuckDB data plane for row-bearing operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import duckdb

from mapel_linkage.domain.errors import DataPlaneError
from mapel_linkage.domain.sql_identifiers import quote_identifier, validate_identifier
from mapel_linkage.domain.table_refs import TableRef

SqlType = Literal["VARCHAR", "INTEGER", "BIGINT", "DOUBLE", "BOOLEAN", "DATE"]
_ALLOWED_TYPES: frozenset[str] = frozenset(
    {"VARCHAR", "INTEGER", "BIGINT", "DOUBLE", "BOOLEAN", "DATE"}
)


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """A column declaration drawn from a fixed SQL type allow-list."""

    name: str
    data_type: SqlType

    def __post_init__(self) -> None:
        validate_identifier(self.name)
        if self.data_type not in _ALLOWED_TYPES:
            raise DataPlaneError("ML-DATA-003", "An unsupported internal SQL type was rejected.")


class DuckDBStore:
    """Own a local DuckDB connection without exposing row-preview conveniences."""

    __slots__ = ("_connection", "_database_label")

    def __init__(self, database_path: str | Path = ":memory:") -> None:
        label = str(database_path)
        self._database_label = "<memory>" if label == ":memory:" else "<restricted>"
        try:
            self._connection: Any = duckdb.connect(label)
        except Exception:
            raise DataPlaneError(
                "ML-DATA-001", "The local DuckDB data plane could not be opened."
            ) from None

    def __enter__(self) -> DuckDBStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"DuckDBStore(database={self._database_label!r})"

    def close(self) -> None:
        """Close the local connection."""

        self._connection.close()

    def create_table_from_rows(
        self,
        table_name: str,
        columns: Sequence[ColumnSpec],
        rows: Iterable[Sequence[object]],
    ) -> TableRef:
        """Create a local table through parameterised inserts.

        This method accepts row values but never includes them in representations,
        errors, or logs.
        """

        validate_identifier(table_name)
        if not columns:
            raise DataPlaneError("ML-DATA-004", "At least one table column is required.")
        names = [column.name for column in columns]
        if len(names) != len(set(names)):
            raise DataPlaneError("ML-DATA-005", "Duplicate internal table columns were rejected.")

        quoted_table = quote_identifier(table_name)
        definitions = ", ".join(
            f"{quote_identifier(column.name)} {column.data_type}" for column in columns
        )
        placeholders = ", ".join("?" for _ in columns)
        insert_sql = f"INSERT INTO {quoted_table} VALUES ({placeholders})"
        materialised_rows = [tuple(row) for row in rows]
        if any(len(row) != len(columns) for row in materialised_rows):
            raise DataPlaneError(
                "ML-DATA-006", "A row did not match the declared internal schema."
            )

        try:
            self._connection.execute(f"DROP TABLE IF EXISTS {quoted_table}")
            self._connection.execute(f"CREATE TABLE {quoted_table} ({definitions})")
            if materialised_rows:
                self._connection.executemany(insert_sql, materialised_rows)
            return self.table_ref(table_name)
        except DataPlaneError:
            raise
        except Exception:
            self._connection.execute(f"DROP TABLE IF EXISTS {quoted_table}")
            raise DataPlaneError(
                "ML-DATA-007", "A local row-bearing table could not be created."
            ) from None

    def table_ref(self, table_name: str) -> TableRef:
        """Return an opaque reference with a deterministic schema digest."""

        quoted_table = quote_identifier(table_name)
        try:
            schema_rows = self._connection.execute(f"DESCRIBE {quoted_table}").fetchall()
            count_row = self._connection.execute(
                f"SELECT COUNT(*) FROM {quoted_table}"
            ).fetchone()
        except Exception:
            raise DataPlaneError(
                "ML-DATA-008", "The requested local table is unavailable."
            ) from None
        if count_row is None:
            raise DataPlaneError("ML-DATA-009", "The local table count could not be read.")
        schema_payload = [(str(row[0]), str(row[1])) for row in schema_rows]
        digest = hashlib.sha256(
            json.dumps(schema_payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return TableRef(
            table_name=table_name,
            schema_digest=digest,
            row_count=int(count_row[0]),
            _database_label=self._database_label,
        )

    def _scalar_int(self, package_owned_sql: str) -> int:
        try:
            row = self._connection.execute(package_owned_sql).fetchone()
        except Exception:
            raise DataPlaneError(
                "ML-DATA-010", "An internal aggregate query could not be completed."
            ) from None
        if row is None:
            raise DataPlaneError("ML-DATA-011", "An internal aggregate was unavailable.")
        return int(row[0])

    def _create_temp_table_as(self, table_name: str, package_owned_select: str) -> TableRef:
        quoted_table = quote_identifier(table_name)
        try:
            self._connection.execute(
                f"CREATE OR REPLACE TEMP TABLE {quoted_table} AS {package_owned_select}"
            )
            return self.table_ref(table_name)
        except Exception:
            raise DataPlaneError(
                "ML-DATA-012", "An internal candidate table could not be materialised."
            ) from None

    def _candidate_diagnostics(self, table_name: str) -> tuple[int, int, int]:
        quoted_table = quote_identifier(table_name)
        try:
            row = self._connection.execute(
                "SELECT COUNT(*), "
                "SUM(CASE WHEN retrieval_rule_count > 1 THEN 1 ELSE 0 END), "
                "COALESCE(MAX(retrieval_rule_count), 0) "
                f"FROM {quoted_table}"
            ).fetchone()
        except Exception:
            raise DataPlaneError(
                "ML-DATA-013", "Candidate diagnostics could not be calculated."
            ) from None
        if row is None:
            return (0, 0, 0)
        return (int(row[0]), int(row[1] or 0), int(row[2] or 0))
