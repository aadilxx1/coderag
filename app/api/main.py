"""
FastAPI wrapper exposing the retrieve -> generate LangGraph pipeline over HTTP.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.graph.pipeline import rag_graph
from app.ingest.github import GitHubCloneError
from app.ingest.pipeline import ingest_github

app = FastAPI(title="CodeRAG")


class AskRequest(BaseModel):
    question: str
    repo: str | None = None
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str


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
    result = rag_graph.invoke(request.model_dump())
    return AskResponse(answer=result["answer"])


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
    return IngestResponse(repo=repo, **asdict(stats))
