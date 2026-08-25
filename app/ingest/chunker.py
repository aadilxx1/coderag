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
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class CodeChunk:
    repo: str
    file_path: str          # relative to repo root
    symbol: str              # function/class name, or "<module>"
    kind: str                # "function" | "class" | "module"
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


def _chunk_class(node: ast.ClassDef, lines: list[str], repo: str, rel: str,
                  max_chars: int) -> list[CodeChunk]:
    """One chunk per method, plus a class-level chunk (header + docstring +
    method signatures, not full bodies). A single chunk for the whole class
    truncates at max_chars for any class with more than a few methods,
    silently making later methods unretrievable -- splitting per method
    fixes that and matches the granularity we already give functions."""
    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

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
    header_end = methods[0].lineno - 1 if methods else class_end
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
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
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


def chunk_repo(repo_root: Path, repo: str, max_chars: int = 4000) -> list[CodeChunk]:
    skip = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist"}
    out: list[CodeChunk] = []
    for path in repo_root.rglob("*.py"):
        if any(part in skip for part in path.parts):
            continue
        out.extend(chunk_python_file(path, repo_root, repo, max_chars))
    return out