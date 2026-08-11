"""Unit tests for the AST chunker."""
from pathlib import Path

from app.ingest.chunker import chunk_python_file

SAMPLE = '''"""Module docstring."""
import os

CONSTANT = 1


def alpha(x):
    """Doc for alpha."""
    return x + 1


class Beta:
    """Doc for Beta."""

    def method(self):
        return 2
'''


def test_chunks_functions_and_classes(tmp_path: Path):
    f = tmp_path / "sample.py"
    f.write_text(SAMPLE)
    chunks = chunk_python_file(f, tmp_path, repo="test")
    symbols = {c.symbol for c in chunks}
    assert {"alpha", "Beta", "<module>"} <= symbols


def test_function_body_is_intact(tmp_path: Path):
    f = tmp_path / "sample.py"
    f.write_text(SAMPLE)
    chunk = next(c for c in chunk_python_file(f, tmp_path, "test") if c.symbol == "alpha")
    assert "return x + 1" in chunk.code
    assert chunk.docstring == "Doc for alpha."


def test_unparseable_file_is_skipped(tmp_path: Path):
    f = tmp_path / "broken.py"
    f.write_text("def oops(:\n")
    assert chunk_python_file(f, tmp_path, "test") == []