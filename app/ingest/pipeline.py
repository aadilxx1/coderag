"""
End-to-end ingest: walk a repo, chunk every .py file, embed each chunk,
and store everything in Postgres.

Shared by the CLI (scripts/ingest.py) and the API (POST /repos).
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from app.core.embeddings import embed_texts
from app.core.store import get_engine
from app.ingest.chunker import chunk_repo
from app.ingest.github import cloned_repo, parse_github_url


def ingest_repo(repo_root: Path, repo_name: str) -> int:
    chunks = chunk_repo(repo_root, repo_name)
    if not chunks:
        print("No chunks found -- is the path correct and does it contain .py files?")
        return 0

    print(f"Chunked {len(chunks)} symbols from {repo_root}")

    # Embed in one batch call -- much faster than one call per chunk.
    texts_to_embed = [c.embed_text() for c in chunks]
    print("Embedding chunks (this may take a moment)...")
    vectors = embed_texts(texts_to_embed)

    engine = get_engine()
    with engine.begin() as conn:
        # Re-ingesting a repo replaces its chunks entirely, so re-running
        # ingest always reflects the repo's current state -- no duplicates,
        # no leftover rows for files that were deleted or renamed.
        deleted = conn.execute(
            text("DELETE FROM code_chunks WHERE repo = :repo"), {"repo": repo_name}
        )
        if deleted.rowcount:
            print(f"Removed {deleted.rowcount} existing rows for repo '{repo_name}'")

        for chunk, vector in zip(chunks, vectors):
            conn.execute(
                text("""
                    INSERT INTO code_chunks
                        (repo, file_path, symbol, kind, start_line, end_line, docstring, code, embedding)
                    VALUES
                        (:repo, :file_path, :symbol, :kind, :start_line, :end_line, :docstring, :code, :embedding)
                """),
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
                },
            )

    print(f"Inserted {len(chunks)} rows into code_chunks")
    return len(chunks)


def ingest_github(url: str, repo_name: str | None = None, ref: str | None = None) -> tuple[str, int]:
    """Clone a public GitHub repo, ingest it, and clean up the clone.
    Returns (repo name it was stored under, number of chunks)."""
    gh = parse_github_url(url, ref)
    name = repo_name or gh.slug
    with cloned_repo(gh) as path:
        return name, ingest_repo(path, name)
