"""Unit tests for package metadata and distribution entrypoint registration."""

from __future__ import annotations

import tomllib

import mapel_linkage
from mapel_linkage.models.fellegi_sunter import SUPPORTED_SPLINK_VERSION
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


def test_splink_runtime_metadata_and_ci_constraint_are_exactly_aligned() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = f"splink=={SUPPORTED_SPLINK_VERSION}"
    optional_dependencies = pyproject["project"]["optional-dependencies"]

    for extra in ("core", "dev"):
        splink_requirements = [
            requirement
            for requirement in optional_dependencies[extra]
            if requirement.startswith("splink")
        ]
        assert splink_requirements == [expected]

    constraint_requirements = [
        line.strip()
        for line in (ROOT / "constraints" / "ci-py312.txt").read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("splink")
    ]
    assert constraint_requirements == [expected]
