from mapel_linkage import __version__


def test_version_is_integration_audit_pre_alpha() -> None:
    assert __version__ == "0.2.0.dev3"
