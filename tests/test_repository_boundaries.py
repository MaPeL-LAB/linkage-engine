from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_ignored_roots() -> None:
    text = (ROOT / ".gitignore").read_text()
    for value in ("/private/**", "/data/**", "/artifacts/**"):
        assert value in text

def test_canonical_names() -> None:
    text = (ROOT / "README.md").read_text()
    assert "Repository | `linkage-engine`" in text
    assert "Python distribution | `mapel-linkage-engine`" in text

def test_publication_guard() -> None:
    assert '"Private :: Do Not Upload"' in (ROOT / "pyproject.toml").read_text()
