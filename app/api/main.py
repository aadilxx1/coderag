"""
FastAPI wrapper exposing the retrieve -> generate LangGraph pipeline over HTTP.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.graph.pipeline import rag_graph

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


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = rag_graph.invoke(request.model_dump())
    return AskResponse(answer=result["answer"])
