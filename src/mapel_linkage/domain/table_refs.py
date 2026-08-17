"""Opaque references to local row-bearing tables."""

from __future__ import annotations

from dataclasses import dataclass, field

from mapel_linkage.domain.sql_identifiers import validate_identifier


@dataclass(frozen=True, slots=True)
class TableRef:
    """A privacy-safe handle that does not expose rows or database paths."""

    table_name: str
    schema_digest: str
    row_count: int
    contains_row_level_data: bool = True
    _database_label: str = field(default="<restricted>", repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_identifier(self.table_name)
        if self.row_count < 0:
            raise ValueError("row_count must be non-negative")

    def safe_summary(self) -> dict[str, str | int | bool]:
        """Return structural metadata only."""

        return {
            "table_name": self.table_name,
            "schema_digest": self.schema_digest,
            "row_count": self.row_count,
            "contains_row_level_data": self.contains_row_level_data,
        }
