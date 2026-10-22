"""Unit tests for the AST chunker."""
from pathlib import Path

from app.ingest.chunker import (
    OVERVIEW_PATH,
    build_overview_chunks,
    chunk_markdown_file,
    chunk_python_file,
    iter_source_files,
)

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

OVERLOADS = '''import typing as t
from typing import overload


@overload
def coerce(x: int) -> int: ...
@overload
def coerce(x: str) -> str: ...
def coerce(x):
    return x


class Box:
    """Doc for Box."""

    @t.overload
    def __init__(self, v: int) -> None: ...
    @t.overload
    def __init__(self, v: str) -> None: ...
    def __init__(self, v):
        self.v = v
'''


def test_overload_stubs_are_skipped(tmp_path: Path):
    f = tmp_path / "overloads.py"
    f.write_text(OVERLOADS)
    chunks = chunk_python_file(f, tmp_path, "test")

    coerce = [c for c in chunks if c.symbol == "coerce"]
    assert len(coerce) == 1 and "return x" in coerce[0].code

    init = [c for c in chunks if c.symbol == "Box.__init__"]
    assert len(init) == 1 and "self.v = v" in init[0].code


def test_class_summary_excludes_overload_stubs(tmp_path: Path):
    f = tmp_path / "overloads.py"
    f.write_text(OVERLOADS)
    summary = next(c for c in chunk_python_file(f, tmp_path, "test") if c.symbol == "Box")
    assert "overload" not in summary.code
    assert summary.code.count("def __init__") == 1


README = '''# MyProj

A tool that does useful things.

## Setup

Run `make install`.

```bash
# not a heading -- inside a code fence
make install
```

## Usage
### CLI

Use `myproj run`.
'''


def test_markdown_chunks_one_per_section_with_heading_path(tmp_path: Path):
    f = tmp_path / "README.md"
    f.write_text(README)
    chunks = chunk_markdown_file(f, tmp_path, "test")
    by_symbol = {c.symbol: c for c in chunks}

    # "Usage" has no body of its own (straight into "### CLI"), so it's skipped.
    assert list(by_symbol) == ["MyProj", "MyProj > Setup", "MyProj > Usage > CLI"]
    assert all(c.kind == "doc" for c in chunks)
    assert "make install" in by_symbol["MyProj > Setup"].code
    assert "not a heading" in by_symbol["MyProj > Setup"].code  # fence kept in its section
    assert by_symbol["MyProj"].start_line == 1


def test_markdown_text_before_first_heading_is_intro(tmp_path: Path):
    f = tmp_path / "notes.md"
    f.write_text("Just some notes.\n\n# Later\n\nMore.\n")
    assert [c.symbol for c in chunk_markdown_file(f, tmp_path, "test")] == ["<intro>", "Later"]


def test_long_markdown_section_is_split_into_parts(tmp_path: Path):
    f = tmp_path / "big.md"
    f.write_text("# Big\n\n" + "".join(f"line {i} of filler text\n" for i in range(400)))
    chunks = chunk_markdown_file(f, tmp_path, "test", max_chars=1000)
    assert len(chunks) > 1
    assert chunks[0].symbol == "Big (part 1)"
    assert all(len(c.code) <= 1000 for c in chunks)
    # Parts cover the file contiguously.
    assert all(b.start_line == a.end_line + 1 for a, b in zip(chunks, chunks[1:]))


def test_overview_lists_files_with_descriptions(tmp_path: Path):
    (tmp_path / "README.md").write_text(README)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text('"""Core logic for MyProj.\n\nMore detail."""\n')
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text('"""Should not appear."""\n')

    files = list(iter_source_files(tmp_path))
    [overview] = build_overview_chunks(tmp_path, "test", files)
    assert overview.file_path == OVERVIEW_PATH and overview.kind == "overview"
    assert "README.md -- MyProj: A tool that does useful things." in overview.code
    assert "pkg/core.py -- Core logic for MyProj." in overview.code
    assert "pkg/__init__.py\n" in overview.code
    assert "ignored" not in overview.code


def test_hidden_and_tooling_dirs_are_skipped(tmp_path: Path):
    for rel in ["keep.py", "docs/guide.md", ".pytest_cache/README.md", ".github/ISSUE.md",
                ".venv/lib/x.py", "node_modules/pkg/y.py", ".hidden_root_file.py"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x = 1\n")
    found = {str(p.relative_to(tmp_path)) for p in iter_source_files(tmp_path)}
    # Hidden *dirs* are skipped; a hidden file at the root is still a source file.
    assert found == {"keep.py", "docs/guide.md", ".hidden_root_file.py"}
