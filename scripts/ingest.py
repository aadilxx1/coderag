"""
End-to-end ingest: walk a repo, chunk every .py file, embed each chunk,
and store everything in Postgres.

Usage:
    python scripts/ingest.py /path/to/repo my-repo-name
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.core.embeddings import embed_texts
from app.core.store import get_engine
from app.ingest.chunker import chunk_repo


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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/ingest.py /path/to/repo repo-name")
        sys.exit(1)

    repo_path = Path(sys.argv[1]).resolve()
    repo_name = sys.argv[2]
    ingest_repo(repo_path, repo_name)