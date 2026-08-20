"""
Vector similarity search over stored code chunks.

Uses pgvector's cosine distance operator (<=>) -- smaller distance means
more semantically similar. We embed the query with the same model used to
embed the chunks (this is critical: query and chunk vectors must come from
the same embedding space to be comparable).
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text

from app.core.embeddings import embed_query
from app.core.store import get_engine


@dataclass
class RetrievedChunk:
    file_path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    docstring: str | None
    code: str
    distance: float  # lower = more similar


def retrieve(query: str, repo: str | None = None, top_k: int = 5) -> list[RetrievedChunk]:
    query_vector = embed_query(query)

    sql = """
        SELECT file_path, symbol, kind, start_line, end_line, docstring, code,
               embedding <=> :query_vector AS distance
        FROM code_chunks
        WHERE (CAST(:repo AS text) IS NULL OR repo = CAST(:repo AS text))
        ORDER BY distance ASC
        LIMIT :top_k
    """

    with get_engine().connect() as conn:
        rows = conn.execute(
            text(sql),
            {"query_vector": str(query_vector), "repo": repo, "top_k": top_k},
        ).fetchall()

    return [
        RetrievedChunk(
            file_path=r.file_path,
            symbol=r.symbol,
            kind=r.kind,
            start_line=r.start_line,
            end_line=r.end_line,
            docstring=r.docstring,
            code=r.code,
            distance=r.distance,
        )
        for r in rows
    ]