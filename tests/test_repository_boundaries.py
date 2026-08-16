from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ignored_roots_and_row_level_formats() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for value in ("/private/**", "/data/**", "/artifacts/**", "*.jsonl", "*.ndjson"):
        assert value in text


def test_canonical_names() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Repository | `linkage-engine`" in text
    assert "Python distribution | `mapel-linkage-engine`" in text


def test_publication_guard() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"Private :: Do Not Upload"' in pyproject


def test_machine_readable_schema_is_committed() -> None:
    assert (ROOT / "schemas/linkage-config.schema.json").is_file()


def test_example_source_columns_are_not_hard_coded_in_package_logic() -> None:
    package_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src/mapel_linkage").rglob("*.py")
    )
    for column in (
        "record_key_a",
        "record_key_b",
        "label_value_a",
        "label_value_b",
        "date_value_a",
        "date_value_b",
        "synthetic_entity_id_a",
        "synthetic_entity_id_b",
    ):
        assert column not in package_source
