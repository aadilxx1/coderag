"""
Cross-encoder reranking of retrieved chunks.

Hybrid retrieval scores the query and each chunk independently (embedding
distance, keyword rank), so it can't tell a method that *answers* the
question from its class summary or a sibling method that merely shares its
vocabulary. A cross-encoder reads the query and chunk together and scores
the pair, which is much better at that last step -- but too slow to run
over the whole repo, so it only reorders the top RERANK_CANDIDATES.

jina-reranker-v2-base-multilingual is trained partly on code search. Picked
over ms-marco-MiniLM-L6 and bge-reranker-base on the retrieval eval
(MRR 0.554 -> 0.688 vs 0.578 / 0.565 for the others, 10 candidates).

Note: the model is licensed CC-BY-NC-4.0 (non-commercial use only), and like
the embedding model it ships custom code, so loading it requires
trust_remote_code=True (and the `einops` package).
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.core.retrieve import RetrievedChunk

MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
# Candidates pulled from hybrid retrieval for reranking. 15 and 30 scored the
# same on the eval as 10, at 2x and 3x the latency (~1s/query at 10 on an M3).
RERANK_CANDIDATES = 10
MAX_LENGTH = 512
BATCH_SIZE = 8

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH, trust_remote_code=True)
    return _model


def _pair_text(chunk: RetrievedChunk) -> str:
    """Same shape as CodeChunk.embed_text -- path and symbol carry signal."""
    doc = f"\n{chunk.docstring.strip()}" if chunk.docstring else ""
    return f"{chunk.file_path} :: {chunk.symbol} ({chunk.kind}){doc}\n{chunk.code}"


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not chunks:
        return []
    scores = _get_model().predict([(query, _pair_text(c)) for c in chunks], batch_size=BATCH_SIZE)
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in order[:top_k]]
