from __future__ import annotations

from pathlib import Path

import pytest

from mapel_linkage.configuration.registries import (
    TRANSFORMS,
    registry_digest,
    registry_snapshot,
    resolve_operation,
)
from mapel_linkage.governance.errors import SafeError, SafeErrorCode


def test_registry_is_immutable_and_allow_list_only() -> None:
    assert resolve_operation("transform", "casefold").key == "casefold"
    with pytest.raises(TypeError):
        TRANSFORMS["untrusted"] = TRANSFORMS["casefold"]  # type: ignore[index]


def test_unknown_registry_key_is_value_safe() -> None:
    sentinel = "restricted.module:callable"
    with pytest.raises(SafeError) as caught:
        resolve_operation("transform", sentinel)
    assert caught.value.code == SafeErrorCode.CONFIG_UNSUPPORTED
    assert sentinel not in caught.value.render()


def test_registry_digest_is_stable() -> None:
    assert registry_digest() == registry_digest()
    assert registry_snapshot()["transform"] == tuple(sorted(TRANSFORMS))


def test_configuration_runtime_contains_no_dynamic_code_resolution() -> None:
    root = Path(__file__).resolve().parents[2] / "src/mapel_linkage/configuration"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    for token in ("eval(", "exec(", "__import__(", "importlib"):
        assert token not in source
