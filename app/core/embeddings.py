"""
Local embeddings via sentence-transformers -- no API key, runs on your machine.

jina-embeddings-v2-base-code is trained specifically on code/docstring pairs
and code-to-code similarity (unlike general sentence models), so it should
give noticeably better retrieval quality for CodeRAG. Outputs 768-dim vectors.

Note: this model ships custom modeling code on the HF Hub, so loading it
requires trust_remote_code=True. That's normal for this model and widely
used, but worth knowing -- it means you're trusting the code in that repo,
not just the weights.

If you ever swap models, EMBEDDING_DIM must match the vector(N) column in
the database, and you'd need to re-embed + re-store everything (embeddings
from different models aren't comparable to each other).
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 768

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            "jinaai/jina-embeddings-v2-base-code",
            trust_remote_code=True,
        )
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = _get_model().encode(texts, convert_to_numpy=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]