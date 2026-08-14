"""
pgvector storage.

The embedding column type is vector(N) -- N must match whatever embedding
model we use. We're using jinaai/jina-embeddings-v2-base-code, which
outputs 768-dim vectors.
"""
from __future__ import annotations

from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg://coderag:coderag@localhost:5432/coderag"
EMBEDDING_DIM = 768  # matches jina-embeddings-v2-base-code

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS code_chunks;

CREATE TABLE code_chunks (
    id           BIGSERIAL PRIMARY KEY,
    repo         TEXT NOT NULL,
    file_path    TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    kind         TEXT NOT NULL,
    start_line   INT  NOT NULL,
    end_line     INT  NOT NULL,
    docstring    TEXT,
    code         TEXT NOT NULL,
    embedding    vector({EMBEDDING_DIM})
);
"""


def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


def init_schema() -> None:
    with get_engine().begin() as conn:
        conn.execute(text(SCHEMA))