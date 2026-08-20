"""Unit tests for package metadata and distribution entrypoint registration."""

from __future__ import annotations

import mapel_linkage
from tests.helpers import ROOT


def test_package_version_consistency() -> None:
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{mapel_linkage.__version__}"' in pyproject_text


def test_package_entrypoint_defined_in_pyproject() -> None:
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'mapel-linkage = "mapel_linkage.cli.main:main"' in pyproject_text


def test_public_api_top_level_exports() -> None:
    assert hasattr(mapel_linkage, "__version__")
    import mapel_linkage.configuration as config_pkg

    assert hasattr(config_pkg, "compile_config")
    assert hasattr(config_pkg, "load_config")
