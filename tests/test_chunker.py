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


MULTI_METHOD_CLASS = '''class Widget:
    """A widget with several methods."""

    def start(self):
        """Start it."""
        return "started"

    def stop(self):
        """Stop it."""
        return "stopped"
'''


def test_class_methods_are_chunked_individually(tmp_path: Path):
    f = tmp_path / "sample.py"
    f.write_text(MULTI_METHOD_CLASS)
    chunks = chunk_python_file(f, tmp_path, "test")

    start_chunk = next(c for c in chunks if c.symbol == "Widget.start")
    assert start_chunk.kind == "method"
    assert 'return "started"' in start_chunk.code
    assert start_chunk.docstring == "Start it."

    stop_chunk = next(c for c in chunks if c.symbol == "Widget.stop")
    assert 'return "stopped"' in stop_chunk.code

    class_chunk = next(c for c in chunks if c.symbol == "Widget")
    assert class_chunk.kind == "class"
    assert class_chunk.docstring == "A widget with several methods."
    assert "def start" in class_chunk.code
    assert "def stop" in class_chunk.code
    # Class-level chunk carries signatures, not full method bodies.
    assert "started" not in class_chunk.code


NO_METHOD_CLASS = '''class Point:
    """Just data."""
    x: int
    y: int
'''


def test_class_without_methods_stays_one_chunk(tmp_path: Path):
    f = tmp_path / "sample.py"
    f.write_text(NO_METHOD_CLASS)
    chunks = chunk_python_file(f, tmp_path, "test")

    class_chunks = [c for c in chunks if c.symbol == "Point"]
    assert len(class_chunks) == 1
    assert "x: int" in class_chunks[0].code
    assert "y: int" in class_chunks[0].code