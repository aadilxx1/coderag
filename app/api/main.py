"""
FastAPI wrapper exposing the retrieve -> generate LangGraph pipeline over HTTP.
"""
from __future__ import annotations

import logging
from dataclasses import asdict

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from app.core.store import get_engine
from app.graph.pipeline import rag_graph
from app.ingest.chunker import OVERVIEW_PATH
from app.ingest.github import GitHubCloneError
from app.ingest.pipeline import delete_repo, ingest_github

logger = logging.getLogger(__name__)

app = FastAPI(title="CodeRAG")


class AskRequest(BaseModel):
    question: str
    repo: str | None = None
    top_k: int = 5


class Source(BaseModel):
    file_path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    code: str


class AskResponse(BaseModel):
    answer: str
    blocked: bool = False
    block_reason: str | None = None
    # The chunks the answer was generated from, so a client can show them.
    # Empty if the question was blocked before retrieval.
    sources: list[Source] = []


class RepoInfo(BaseModel):
    repo: str
    files: int
    chunks: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


class IngestRequest(BaseModel):
    url: str                  # https://github.com/owner/repo
    name: str | None = None   # defaults to "owner/repo"
    ref: str | None = None    # branch or tag; defaults to the repo's default branch
    full: bool = False        # re-embed every file, not just new/changed ones


class IngestResponse(BaseModel):
    repo: str
    files_added: int
    files_modified: int
    files_removed: int
    files_unchanged: int
    chunks_inserted: int
    chunks_total: int


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = rag_graph.invoke(request.model_dump())
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Can't reach the LLM -- is `ollama serve` running?")
    return AskResponse(
        answer=result["answer"],
        blocked=result.get("blocked", False),
        block_reason=result.get("block_reason"),
        sources=[
            Source(file_path=c.file_path, symbol=c.symbol, kind=c.kind,
                   start_line=c.start_line, end_line=c.end_line, code=c.code)
            for c in result.get("chunks", [])
        ],
    )


@app.get("/repos", response_model=list[RepoInfo])
def list_repos() -> list[RepoInfo]:
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT repo,
                   count(DISTINCT file_path) FILTER (WHERE file_path <> :overview) AS files,
                   count(*) AS chunks
            FROM code_chunks GROUP BY repo ORDER BY repo
        """), {"overview": OVERVIEW_PATH}).fetchall()
    return [RepoInfo(repo=r.repo, files=r.files, chunks=r.chunks) for r in rows]


# {repo:path} because repo names contain a slash ("pallets/itsdangerous").
@app.delete("/repos/{repo:path}")
def remove_repo(repo: str) -> dict:
    deleted = delete_repo(repo)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No repo named {repo!r} is ingested.")
    return {"repo": repo, "chunks_deleted": deleted}


# Synchronous: clone + embed can take a while on a big repo, but a plain
# `def` endpoint runs in FastAPI's threadpool, so it doesn't block /ask.
@app.post("/repos", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    try:
        repo, stats = ingest_github(request.url, request.name, request.ref, full=request.full)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except GitHubCloneError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Anything else is a server-side failure. The full exception can be
        # huge (a DB error echoes every bound row), so log it and return a
        # short message instead of a bare "Internal Server Error".
        logger.exception("Ingest of %s failed", request.url)
        raise HTTPException(status_code=500,
                            detail=f"Ingest failed ({type(e).__name__}) -- see the API server log.")
    return IngestResponse(repo=repo, **asdict(stats))
