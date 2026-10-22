"""
Local embeddings via sentence-transformers -- runs on your machine.

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

import logging
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768

# FastAPI runs sync endpoints in a threadpool, so an ingest and an /ask can
# run model inference at the same time -- and PyTorch's MPS (Apple GPU)
# backend isn't safe to drive from several threads at once. Every model call
# (embedding here, reranking in rerank.py) holds this lock. Throughput cost is
# small: a single local GPU runs one batch at a time anyway.
MODEL_LOCK = threading.Lock()

# The model accepts up to 8192 tokens, but attention memory grows with
# seq_len^2 x batch_size -- a batch of long chunks OOMs the GPU (seen on an
# M3: 3.3 GiB for one softmax). Chunks are capped at 4000 chars (~1000-1300
# tokens of code), so 1024 tokens covers nearly all of each chunk; only the
# embedding is truncated, the stored code is not.
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 8

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            "jinaai/jina-embeddings-v2-base-code",
            trust_remote_code=True,
        )
        _model.max_seq_length = MAX_SEQ_LENGTH
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    with MODEL_LOCK:
        model = _get_model()
        vectors = model.encode(texts, batch_size=BATCH_SIZE, convert_to_numpy=True)

        # The GPU backend has intermittently produced NaN vectors (seen once
        # via POST /repos while Ollama shared the GPU; not reproducible
        # since). Postgres rejects NaN in a vector column, failing the whole
        # ingest -- so recompute just the bad rows on CPU, which is slower
        # but numerically reliable.
        bad = np.flatnonzero(~np.isfinite(vectors).all(axis=1))
        if bad.size:
            device = model.device
            logger.warning("%d/%d embeddings were non-finite on %s; recomputing on CPU",
                           bad.size, len(texts), device)
            try:
                vectors[bad] = model.encode([texts[i] for i in bad], batch_size=BATCH_SIZE,
                                            convert_to_numpy=True, device="cpu")
            finally:
                # encode(device=...) moves the whole model and leaves it there;
                # without this every later call would silently run on CPU.
                model.to(device)
            if not np.isfinite(vectors[bad]).all():
                raise ValueError("Embedding model produced non-finite vectors, even on CPU")
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]