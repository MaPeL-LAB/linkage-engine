from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest

from mapel_linkage.domain.errors import FellegiSunterError
from mapel_linkage.domain.table_refs import TableRef
from mapel_linkage.io import ColumnSpec, DuckDBStore
from mapel_linkage.models.fellegi_sunter import (
    SplinkCandidateParityChecker,
    SplinkSettingsPlan,
)
from mapel_linkage.preprocessing import PreparedDataset


def _prepared(dataset_id: str, table_name: str) -> PreparedDataset:
    return PreparedDataset(
        dataset_id,
        TableRef(
            table_name=table_name,
            schema_digest="a" * 64,
            row_count=1,
        ),
        {"value": "value"},
        {"value": "missing_value"},
    )


class _SettingsCreator:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _DuckDBAPI:
    pass


class _SplinkFrame:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self._records = records

    def as_record_dict(self) -> list[dict[str, object]]:
        return list(self._records)


class _Linker:
    output_records: ClassVar[list[dict[str, object]]] = [
        {"__ml_record_key_l": "left-surrogate", "__ml_record_key_r": "right-surrogate"}
    ]

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.inference = SimpleNamespace(
            deterministic_link=lambda: _SplinkFrame(list(self.output_records))
        )


def _plan() -> SplinkSettingsPlan:
    return SplinkSettingsPlan(
        settings_digest="b" * 64,
        comparison_count=1,
        blocking_rule_count=1,
        settings={"blocking_rules_to_generate_predictions": ["l.value = r.value"]},
    )


def test_real_splink_parity_uses_safe_aliases_for_reserved_dataset_ids() -> None:
    columns = (
        ColumnSpec("__ml_record_key", "VARCHAR"),
        ColumnSpec("__ml_dataset_id", "VARCHAR"),
        ColumnSpec("value", "VARCHAR"),
    )
    with DuckDBStore() as store:
        left = PreparedDataset(
            "left",
            store.create_table_from_rows(
                "reserved_alias_source",
                columns,
                (("left-surrogate", "left", "synthetic-equal"),),
            ),
            {"value": "value"},
            {"value": "missing_value"},
        )
        right = PreparedDataset(
            "right",
            store.create_table_from_rows(
                "reserved_alias_target",
                columns,
                (("right-surrogate", "right", "synthetic-equal"),),
            ),
            {"value": "value"},
            {"value": "missing_value"},
        )
        report = SplinkCandidateParityChecker.check(
            store=store,
            left=left,
            right=right,
            settings_plan=_plan(),
            expected_pairs=(("left-surrogate", "right-surrogate"),),
        )

    assert report.parity is True
    assert report.expected_pair_count == report.observed_pair_count == 1


def test_splink_candidate_parity_report_is_aggregate_only(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(
        Linker=_Linker,
        SettingsCreator=_SettingsCreator,
        DuckDBAPI=_DuckDBAPI,
    )
    monkeypatch.setattr(
        "mapel_linkage.models.fellegi_sunter.splink_adapter.importlib.import_module",
        lambda _: fake_module,
    )
    monkeypatch.setattr(
        "mapel_linkage.models.fellegi_sunter.splink_adapter.importlib.metadata.version",
        lambda _: "4.0.test",
    )
    monkeypatch.setattr(
        "mapel_linkage.models.fellegi_sunter.splink_adapter._prepared_records",
        lambda *_: [{"__ml_record_key": "private", "value": "sentinel"}],
    )

    report = SplinkCandidateParityChecker.check(
        store=cast(Any, object()),
        left=_prepared("left", "left_table"),
        right=_prepared("right", "right_table"),
        settings_plan=_plan(),
        expected_pairs=(("left-surrogate", "right-surrogate"),),
    )

    assert report.parity is True
    assert report.expected_pair_count == 1
    assert report.observed_pair_count == 1
    assert "left-surrogate" not in repr(report)
    assert "right-surrogate" not in repr(report)
    assert "sentinel" not in repr(report)


def test_splink_candidate_parity_rejects_pair_set_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(
        Linker=_Linker,
        SettingsCreator=_SettingsCreator,
        DuckDBAPI=_DuckDBAPI,
    )
    monkeypatch.setattr(
        "mapel_linkage.models.fellegi_sunter.splink_adapter.importlib.import_module",
        lambda _: fake_module,
    )
    monkeypatch.setattr(
        "mapel_linkage.models.fellegi_sunter.splink_adapter._prepared_records",
        lambda *_: [],
    )

    with pytest.raises(FellegiSunterError) as captured:
        SplinkCandidateParityChecker.check(
            store=cast(Any, object()),
            left=_prepared("left", "left_table"),
            right=_prepared("right", "right_table"),
            settings_plan=_plan(),
            expected_pairs=(("different-left", "different-right"),),
        )

    assert captured.value.code == "ML-FS-037"
    assert "different-left" not in str(captured.value)
    assert "different-right" not in str(captured.value)


def test_splink_candidate_parity_rejects_duplicate_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = SimpleNamespace(
        Linker=_Linker,
        SettingsCreator=_SettingsCreator,
        DuckDBAPI=_DuckDBAPI,
    )
    monkeypatch.setattr(
        "mapel_linkage.models.fellegi_sunter.splink_adapter.importlib.import_module",
        lambda _: fake_module,
    )
    pair = ("left-surrogate", "right-surrogate")
    with pytest.raises(FellegiSunterError, match="ML-FS-038"):
        SplinkCandidateParityChecker.check(
            store=cast(Any, object()),
            left=_prepared("left", "left_table"),
            right=_prepared("right", "right_table"),
            settings_plan=_plan(),
            expected_pairs=(pair, pair),
        )

    monkeypatch.setattr(
        _Linker,
        "output_records",
        [
            {"__ml_record_key_l": pair[0], "__ml_record_key_r": pair[1]},
            {"__ml_record_key_l": pair[0], "__ml_record_key_r": pair[1]},
        ],
    )
    monkeypatch.setattr(
        "mapel_linkage.models.fellegi_sunter.splink_adapter._prepared_records",
        lambda *_: [],
    )
    with pytest.raises(FellegiSunterError, match="ML-FS-036"):
        SplinkCandidateParityChecker.check(
            store=cast(Any, object()),
            left=_prepared("left", "left_table"),
            right=_prepared("right", "right_table"),
            settings_plan=_plan(),
            expected_pairs=(pair,),
        )
