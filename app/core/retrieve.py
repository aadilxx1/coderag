"""
Hybrid retrieval: combines pgvector cosine similarity (semantic) with
Postgres full-text search (keyword/identifier matching), fused with
Reciprocal Rank Fusion.

Pure vector search misses exact keyword/identifier matches when the
embedding doesn't rank them highly -- e.g. a question containing "top_k"
or "FastAPI" should surface a chunk mentioning that literal term even if
its embedding isn't the closest. RRF combines both rankings without needing
the two scores (cosine distance vs. text-search rank) to be on the same
scale.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from sqlalchemy import text

from app.core.embeddings import embed_query
from app.core.store import get_engine

RRF_K = 60  # standard RRF smoothing constant
CANDIDATE_POOL = 20  # candidates pulled from each search method before fusion

# Test functions describe the behavior they test in the same words a
# question uses ("test_class_methods_are_chunked_individually"), so they
# often outrank the code that implements it. Their fused score is scaled
# down unless the question is itself about tests. Tuned on the retrieval
# eval (evals/run_retrieval_eval.py).
TEST_PENALTY = 0.4
_TEST_QUERY_RE = re.compile(r"\btest(s|ed|ing)?\b", re.IGNORECASE)

_VECTOR_SQL = """
    SELECT id, file_path, symbol, kind, start_line, end_line, docstring, code,
           embedding <=> :query_vector AS distance
    FROM code_chunks
    WHERE (CAST(:repo AS text) IS NULL OR repo = CAST(:repo AS text))
    ORDER BY distance ASC
    LIMIT :limit
"""

_KEYWORD_SQL = """
    SELECT id, file_path, symbol, kind, start_line, end_line, docstring, code,
           ts_rank(search_vector, to_tsquery('simple', :tsquery)) AS rank
    FROM code_chunks
    WHERE (CAST(:repo AS text) IS NULL OR repo = CAST(:repo AS text))
      AND search_vector @@ to_tsquery('simple', :tsquery)
    ORDER BY rank DESC
    LIMIT :limit
"""

# 'simple' has no stopword list (deliberately -- see store.py), so we filter
# common English connector words ourselves when building the query. Without
# this, a full-sentence question ANDs against "how"/"does"/"in" etc. and
# almost never matches any actual code chunk.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "in", "into", "is", "it", "of", "on", "or", "that",
    "the", "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "will", "with",
}
_QUERY_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _keyword_tsquery(query: str) -> str | None:
    """OR-combine the query's significant words so a chunk matching any one
    of them can surface, ranked by how many/how well it matches -- an AND
    of every word (the plainto_tsquery default) almost never matches a
    natural-language question against code."""
    terms = [w.lower() for w in _QUERY_WORD_RE.findall(query) if w.lower() not in _STOPWORDS]
    if not terms:
        return None
    return " | ".join(terms)


def _is_test_file(file_path: str) -> bool:
    path = PurePosixPath(file_path)
    return (
        any(part in {"test", "tests"} for part in path.parts[:-1])
        or path.name.startswith("test_")
        or path.name.endswith("_test.py")
        or path.name == "conftest.py"
    )


@dataclass
class RetrievedChunk:
    file_path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    docstring: str | None
    code: str
    distance: float | None  # lower = more similar; None if only matched by keyword search


def retrieve(query: str, repo: str | None = None, top_k: int = 5) -> list[RetrievedChunk]:
    query_vector = embed_query(query)

    tsquery = _keyword_tsquery(query)

    with get_engine().connect() as conn:
        vector_rows = conn.execute(
            text(_VECTOR_SQL),
            {"query_vector": str(query_vector), "repo": repo, "limit": CANDIDATE_POOL},
        ).fetchall()
        keyword_rows = (
            conn.execute(
                text(_KEYWORD_SQL),
                {"tsquery": tsquery, "repo": repo, "limit": CANDIDATE_POOL},
            ).fetchall()
            if tsquery
            else []
        )

    scores: dict[int, float] = {}
    rows_by_id = {}

    for rank, row in enumerate(vector_rows):
        scores[row.id] = scores.get(row.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        rows_by_id[row.id] = row

    for rank, row in enumerate(keyword_rows):
        scores[row.id] = scores.get(row.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        rows_by_id.setdefault(row.id, row)

    if not _TEST_QUERY_RE.search(query):
        for i in scores:
            if _is_test_file(rows_by_id[i].file_path):
                scores[i] *= TEST_PENALTY

    ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        RetrievedChunk(
            file_path=rows_by_id[i].file_path,
            symbol=rows_by_id[i].symbol,
            kind=rows_by_id[i].kind,
            start_line=rows_by_id[i].start_line,
            end_line=rows_by_id[i].end_line,
            docstring=rows_by_id[i].docstring,
            code=rows_by_id[i].code,
            distance=getattr(rows_by_id[i], "distance", None),
        )
        for i in ranked_ids
    ]
