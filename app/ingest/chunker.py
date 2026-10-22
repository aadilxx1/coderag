"""
AST-aware chunking.

This is the piece that makes code RAG different from document RAG.
Splitting a file every 500 characters cuts functions in half and destroys
retrieval quality. Instead we walk the Python AST and emit one chunk per
function / class / module-level block, carrying metadata that we can filter on
later (file path, symbol name, docstring, line span).

Start with Python only (stdlib `ast` module, zero dependencies). If you want
multi-language support later, swap this for tree-sitter -- the CodeChunk
interface stays the same.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

MARKDOWN_SUFFIXES = {".md", ".markdown"}

# Pseudo file_path for the generated per-repo overview chunk(s) -- see
# build_overview_chunks. Not a real file, so incremental ingest must never
# treat it as one.
OVERVIEW_PATH = "<overview>"


@dataclass
class CodeChunk:
    repo: str
    file_path: str          # relative to repo root, or OVERVIEW_PATH
    symbol: str              # function/class name, "<module>", a heading path, or "<overview>"
    kind: str                # "function" | "class" | "method" | "module" | "doc" | "overview"
    start_line: int
    end_line: int
    docstring: str | None
    code: str

    def to_row(self) -> dict:
        return asdict(self)

    def embed_text(self) -> str:
        """What we actually embed. Prepending the path + symbol meaningfully
        improves retrieval -- identifier names are half the signal in code."""
        header = f"# {self.file_path} :: {self.symbol} ({self.kind})"
        doc = f"\n# {self.docstring.strip()}" if self.docstring else ""
        return f"{header}{doc}\n{self.code}"


def _node_source(source_lines: list[str], node: ast.AST) -> tuple[str, int, int]:
    start = node.lineno
    end = getattr(node, "end_lineno", node.lineno)
    return "".join(source_lines[start - 1 : end]), start, end


def _is_overload(node: ast.AST) -> bool:
    """True for a @typing.overload stub (`@overload` or `@t.overload`).
    Stubs are signature-only (`...` body), and a function with N overloads
    would otherwise become N+1 chunks with the same symbol -- near-identical
    noise that crowds the real implementation out of the top-k."""
    return any(
        (isinstance(d, ast.Name) and d.id == "overload")
        or (isinstance(d, ast.Attribute) and d.attr == "overload")
        for d in getattr(node, "decorator_list", [])
    )


def _chunk_class(node: ast.ClassDef, lines: list[str], repo: str, rel: str,
                  max_chars: int) -> list[CodeChunk]:
    """One chunk per method, plus a class-level chunk (header + docstring +
    method signatures, not full bodies). A single chunk for the whole class
    truncates at max_chars for any class with more than a few methods,
    silently making later methods unretrievable -- splitting per method
    fixes that and matches the granularity we already give functions."""
    methods = [
        n for n in node.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _is_overload(n)
    ]

    chunks = [
        CodeChunk(
            repo=repo,
            file_path=rel,
            symbol=f"{node.name}.{method.name}",
            kind="method",
            start_line=method.lineno,
            end_line=getattr(method, "end_lineno", method.lineno),
            docstring=ast.get_docstring(method),
            code=_node_source(lines, method)[0][:max_chars],
        )
        for method in methods
    ]

    class_end = getattr(node, "end_lineno", node.lineno)
    # The header ends where the first def starts -- including skipped
    # overload stubs and decorators, which would otherwise leak into it.
    all_defs = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    header_end = (
        min([all_defs[0].lineno] + [d.lineno for d in all_defs[0].decorator_list]) - 1
        if all_defs else class_end
    )
    header_code = "".join(lines[node.lineno - 1 : header_end])
    signatures = "\n".join(
        f"    def {m.name}({', '.join(a.arg for a in m.args.args)}): ..." for m in methods
    )
    class_code = f"{header_code}{signatures}" if signatures else header_code

    chunks.append(
        CodeChunk(
            repo=repo,
            file_path=rel,
            symbol=node.name,
            kind="class",
            start_line=node.lineno,
            end_line=class_end,
            docstring=ast.get_docstring(node),
            code=class_code[:max_chars],
        )
    )
    return chunks


def chunk_python_file(path: Path, repo_root: Path, repo: str,
                      max_chars: int = 4000) -> list[CodeChunk]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []  # skip files we can't parse rather than crashing the ingest

    lines = source.splitlines(keepends=True)
    rel = str(path.relative_to(repo_root))
    chunks: list[CodeChunk] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            chunks.extend(_chunk_class(node, lines, repo, rel, max_chars))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _is_overload(node):
            code, start, end = _node_source(lines, node)
            chunks.append(
                CodeChunk(
                    repo=repo,
                    file_path=rel,
                    symbol=node.name,
                    kind="function",
                    start_line=start,
                    end_line=end,
                    docstring=ast.get_docstring(node),
                    code=code[:max_chars],
                )
            )

    # Module-level code (imports, constants) is often the answer to
    # "what does this module depend on?" -- keep it as one chunk.
    module_doc = ast.get_docstring(tree)
    top_level = [
        n for n in tree.body
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if top_level:
        first, last = top_level[0], top_level[-1]
        code = "".join(lines[first.lineno - 1 : getattr(last, "end_lineno", last.lineno)])
        if code.strip():
            chunks.append(
                CodeChunk(
                    repo=repo, file_path=rel, symbol="<module>", kind="module",
                    start_line=first.lineno,
                    end_line=getattr(last, "end_lineno", last.lineno),
                    docstring=module_doc, code=code[:max_chars],
                )
            )
    return chunks


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _split_lines(lines: list[str], first_line: int, max_chars: int) -> Iterator[tuple[str, int, int]]:
    """Greedily pack whole lines into pieces of at most max_chars, yielding
    (text, start_line, end_line). A single line longer than max_chars
    becomes its own (truncated) piece."""
    buf: list[str] = []
    start, size = first_line, 0
    for offset, line in enumerate(lines):
        if buf and size + len(line) > max_chars:
            yield "".join(buf), start, start + len(buf) - 1
            buf, start, size = [], first_line + offset, 0
        buf.append(line)
        size += len(line)
    if buf:
        yield "".join(buf)[:max_chars], start, start + len(buf) - 1


def chunk_markdown_file(path: Path, repo_root: Path, repo: str,
                        max_chars: int = 4000) -> list[CodeChunk]:
    """One chunk per heading section, so a README answers the "what is this
    project / how do I run it" questions no single function can.

    The symbol is the section's heading path ("Title > Setup > Docker"), which
    keeps a subsection's context when it's retrieved on its own. Text before
    the first heading is "<intro>". Headings inside fenced code blocks are
    ignored, as are sections with no body (e.g. a heading directly followed
    by a subheading). Sections over max_chars are split into parts.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    lines = source.splitlines(keepends=True)
    rel = str(path.relative_to(repo_root))

    # (heading path, first line index, end line index, has heading line)
    sections: list[tuple[list[str], int, int, bool]] = []
    stack: list[tuple[int, str]] = []
    cur_path: list[str] = []
    cur_start, cur_has_heading = 0, False
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        match = None if in_fence else _HEADING_RE.match(line)
        if match:
            sections.append((cur_path, cur_start, i, cur_has_heading))
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
            cur_path, cur_start, cur_has_heading = [t for _, t in stack], i, True
    sections.append((cur_path, cur_start, len(lines), cur_has_heading))

    chunks: list[CodeChunk] = []
    for heading_path, start, end, has_heading in sections:
        body = lines[start + 1 if has_heading else start : end]
        if not "".join(body).strip():
            continue
        symbol = " > ".join(heading_path) or "<intro>"
        parts = list(_split_lines(lines[start:end], start + 1, max_chars))
        for n, (text, first, last) in enumerate(parts, start=1):
            chunks.append(CodeChunk(
                repo=repo, file_path=rel,
                symbol=symbol if len(parts) == 1 else f"{symbol} (part {n})",
                kind="doc", start_line=first, end_line=last,
                docstring=None, code=text,
            ))
    return chunks


def chunk_file(path: Path, repo_root: Path, repo: str, max_chars: int = 4000) -> list[CodeChunk]:
    if path.suffix in MARKDOWN_SUFFIXES:
        return chunk_markdown_file(path, repo_root, repo, max_chars)
    return chunk_python_file(path, repo_root, repo, max_chars)


SKIP_DIRS = {"venv", "node_modules", "__pycache__", "build", "dist"}


def _skipped_dir(name: str) -> bool:
    # Hidden dirs are tooling, not project source: .git, .venv, .pytest_cache
    # (which ships its own README.md), .tox, .github issue templates, ...
    return name in SKIP_DIRS or name.startswith(".")


def iter_source_files(repo_root: Path) -> Iterator[Path]:
    """Every .py and Markdown file in the repo, outside skipped dirs."""
    for path in sorted(repo_root.rglob("*")):
        if path.suffix != ".py" and path.suffix not in MARKDOWN_SUFFIXES:
            continue
        # Check dirs relative to the root, not the absolute path -- otherwise
        # a repo that itself lives under e.g. a "build/" dir is skipped entirely.
        if path.is_file() and not any(_skipped_dir(p) for p in path.relative_to(repo_root).parts[:-1]):
            yield path


def chunk_repo(repo_root: Path, repo: str, max_chars: int = 4000) -> list[CodeChunk]:
    out: list[CodeChunk] = []
    for path in iter_source_files(repo_root):
        out.extend(chunk_file(path, repo_root, repo, max_chars))
    return out


def _first_paragraph(text: str, limit: int = 200) -> str:
    para = " ".join(text.strip().split("\n\n", 1)[0].split())
    return para if len(para) <= limit else para[: limit - 3].rstrip() + "..."


def describe_file(path: Path) -> str | None:
    """One-line description for the repo overview: a Python file's module
    docstring, or a Markdown file's title plus its first paragraph."""
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None

    if path.suffix in MARKDOWN_SUFFIXES:
        title, rest = None, source
        for i, line in enumerate(source.splitlines()):
            if match := _HEADING_RE.match(line):
                title, rest = match.group(2).strip(), "\n".join(source.splitlines()[i + 1:])
                break
        # First paragraph that isn't a heading, badge row, or code fence.
        para = next((p for p in rest.split("\n\n")
                     if p.strip() and not p.lstrip().startswith(("#", "[!", "![", "```", "~~~"))), "")
        para = _first_paragraph(para)
        return ": ".join(x for x in (title, para) if x) or None

    try:
        doc = ast.get_docstring(ast.parse(source))
    except SyntaxError:
        return None
    return _first_paragraph(doc) if doc else None


def build_overview_chunks(repo_root: Path, repo: str, files: list[Path],
                          max_chars: int = 4000) -> list[CodeChunk]:
    """A generated table of contents for the whole repo: every source file
    with its one-line description. Answers "what does this repo do / where
    is X" questions even for repos without a README, since no single
    function or section describes the project as a whole."""
    lines = [f"Repository overview: {repo}\n",
             f"{len(files)} source files, one per line with its description.\n", "\n"]
    for path in files:
        rel = str(path.relative_to(repo_root))
        desc = describe_file(path)
        lines.append(f"{rel} -- {desc}\n" if desc else f"{rel}\n")

    parts = list(_split_lines(lines, 1, max_chars))
    return [
        CodeChunk(
            repo=repo, file_path=OVERVIEW_PATH,
            symbol="<overview>" if len(parts) == 1 else f"<overview> (part {n})",
            kind="overview", start_line=first, end_line=last,
            docstring=f"Overview of the {repo} repository: its files and what each one does.",
            code=text,
        )
        for n, (text, first, last) in enumerate(parts, start=1)
    ]