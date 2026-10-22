"""
End-to-end ingest: walk a repo, chunk every .py and Markdown file, embed
each chunk, and store everything in Postgres -- plus a generated overview
chunk listing every file (see chunker.build_overview_chunks).

Incremental: each file's content hash is stored in `ingested_files`, and a
re-ingest only chunks + embeds files that are new or whose hash changed,
and deletes rows for files that no longer exist. Hashes are of content (not
mtimes), so this also works for GitHub repos, which are freshly cloned each
time.

Shared by the CLI (scripts/ingest.py) and the API (POST /repos).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import text

from app.core.embeddings import embed_texts
from app.core.store import ensure_ingest_schema, get_engine
from app.ingest.chunker import OVERVIEW_PATH, build_overview_chunks, chunk_file, iter_source_files
from app.ingest.github import cloned_repo, parse_github_url

# Part of every file hash. Bump this whenever chunking or embedding changes
# (chunker rules, embedding model, MAX_SEQ_LENGTH, ...): otherwise unchanged
# files keep chunks/vectors built the old way, and the index silently mixes
# both. Bumping it makes the next ingest of every repo a full rebuild.
INDEX_VERSION = "1"


@dataclass
class FilePlan:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def to_reindex(self) -> list[str]:
        return self.added + self.modified


@dataclass
class IngestStats:
    files_added: int
    files_modified: int
    files_removed: int
    files_unchanged: int
    chunks_inserted: int
    chunks_total: int


def file_hash(path: Path) -> str:
    return hashlib.sha256(INDEX_VERSION.encode() + b"\0" + path.read_bytes()).hexdigest()


def plan_changes(current: dict[str, str], stored: dict[str, str],
                 chunk_paths: set[str]) -> FilePlan:
    """Diff the files on disk against what's in the DB.

    current/stored: file_path -> content hash. chunk_paths: files that have
    rows in code_chunks -- normally a subset of `stored`, but not for repos
    ingested before hashes were tracked, whose files then all count as
    modified (and their stale rows get replaced) on the first run.
    """
    known = stored.keys() | chunk_paths
    plan = FilePlan(removed=sorted(known - current.keys()))
    for path, digest in sorted(current.items()):
        if path not in known:
            plan.added.append(path)
        elif stored.get(path) != digest:
            plan.modified.append(path)
        else:
            plan.unchanged.append(path)
    return plan


def ingest_repo(repo_root: Path, repo_name: str, full: bool = False) -> IngestStats:
    """Ingest `repo_root` under `repo_name`. full=True re-embeds every file
    regardless of hashes."""
    ensure_ingest_schema()
    engine = get_engine()

    files = {str(p.relative_to(repo_root)): p for p in iter_source_files(repo_root)}
    current = {rel: file_hash(p) for rel, p in files.items()}

    with engine.connect() as conn:
        stored = dict(conn.execute(
            text("SELECT file_path, content_hash FROM ingested_files WHERE repo = :repo"),
            {"repo": repo_name},
        ).fetchall())
        chunk_paths = set(conn.execute(
            text("SELECT DISTINCT file_path FROM code_chunks WHERE repo = :repo"),
            {"repo": repo_name},
        ).scalars())
    if full:
        stored = {}

    # The overview isn't a file on disk, so keep it out of the file diff
    # (it would otherwise always look "removed").
    has_overview = OVERVIEW_PATH in chunk_paths
    plan = plan_changes(current, stored, chunk_paths - {OVERVIEW_PATH})
    print(f"{len(files)} source files: {len(plan.added)} added, {len(plan.modified)} modified, "
          f"{len(plan.removed)} removed, {len(plan.unchanged)} unchanged")

    chunks = [c for rel in plan.to_reindex
              for c in chunk_file(files[rel], repo_root, repo_name)]

    # The overview describes every file, so any change invalidates it.
    rebuild_overview = bool(full or plan.to_reindex or plan.removed or not has_overview)
    if rebuild_overview and files:
        chunks += build_overview_chunks(repo_root, repo_name, list(files.values()))
    if chunks:
        # Embed in one batch call -- much faster than one call per chunk.
        print(f"Embedding {len(chunks)} chunks (this may take a moment)...")
        vectors = embed_texts([c.embed_text() for c in chunks])
    else:
        vectors = []

    stale = plan.to_reindex + plan.removed + ([OVERVIEW_PATH] if rebuild_overview else [])
    with engine.begin() as conn:
        # One transaction: a failure mid-ingest leaves the previous state intact.
        if stale:
            for table in ("code_chunks", "ingested_files"):
                conn.execute(
                    text(f"DELETE FROM {table} WHERE repo = :repo AND file_path = ANY(:paths)"),
                    {"repo": repo_name, "paths": stale},
                )
        if chunks:
            conn.execute(
                text("""
                    INSERT INTO code_chunks
                        (repo, file_path, symbol, kind, start_line, end_line, docstring, code, embedding)
                    VALUES
                        (:repo, :file_path, :symbol, :kind, :start_line, :end_line, :docstring, :code, :embedding)
                """),
                [
                    {
                        "repo": chunk.repo,
                        "file_path": chunk.file_path,
                        "symbol": chunk.symbol,
                        "kind": chunk.kind,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "docstring": chunk.docstring,
                        "code": chunk.code,
                        "embedding": str(vector),  # pgvector accepts this text form via cast
                    }
                    for chunk, vector in zip(chunks, vectors)
                ],
            )
        if plan.to_reindex:
            conn.execute(
                text("INSERT INTO ingested_files (repo, file_path, content_hash) "
                     "VALUES (:repo, :file_path, :content_hash)"),
                [{"repo": repo_name, "file_path": rel, "content_hash": current[rel]}
                 for rel in plan.to_reindex],
            )
        total = conn.execute(
            text("SELECT count(*) FROM code_chunks WHERE repo = :repo"), {"repo": repo_name}
        ).scalar_one()

    print(f"Inserted {len(chunks)} chunks; '{repo_name}' now has {total}")
    return IngestStats(
        files_added=len(plan.added),
        files_modified=len(plan.modified),
        files_removed=len(plan.removed),
        files_unchanged=len(plan.unchanged),
        chunks_inserted=len(chunks),
        chunks_total=total,
    )


def delete_repo(repo_name: str) -> int:
    """Remove every chunk and file record for a repo. Returns chunks deleted."""
    ensure_ingest_schema()
    with get_engine().begin() as conn:
        deleted = conn.execute(
            text("DELETE FROM code_chunks WHERE repo = :repo"), {"repo": repo_name}
        ).rowcount
        conn.execute(text("DELETE FROM ingested_files WHERE repo = :repo"), {"repo": repo_name})
    return deleted


def ingest_github(url: str, repo_name: str | None = None, ref: str | None = None,
                  full: bool = False) -> tuple[str, IngestStats]:
    """Clone a public GitHub repo, ingest it, and clean up the clone.
    Returns (repo name it was stored under, ingest stats)."""
    gh = parse_github_url(url, ref)
    name = repo_name or gh.slug
    with cloned_repo(gh) as path:
        return name, ingest_repo(path, name, full=full)
