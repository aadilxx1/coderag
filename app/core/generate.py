"""
Generation step: takes a question + retrieved code chunks, builds a prompt,
and asks a local LLM (via Ollama) to answer using that code as grounding.
"""
from __future__ import annotations

from langchain_ollama import ChatOllama

from app.core.retrieve import RetrievedChunk, retrieve

MODEL_NAME = "llama3.2:3b"

SYSTEM_PROMPT = """You are a helpful assistant that explains code accurately.
Answer the user's question using ONLY the code snippets provided below as context.
If the provided code doesn't actually answer the question, say so honestly instead of guessing.
Be concise and specific -- reference actual function/file names from the context."""


def build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"File: {c.file_path}\nSymbol: {c.symbol} ({c.kind}), lines {c.start_line}-{c.end_line}\n"
            f"Docstring: {c.docstring or 'none'}\n"
            f"Code:\n{c.code}\n"
        )
    return "\n---\n".join(parts)


def answer(question: str, repo: str | None = None, top_k: int = 5) -> str:
    chunks = retrieve(question, repo=repo, top_k=top_k)
    if not chunks:
        return "No relevant code found for this question."

    context = build_context(chunks)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    response = llm.invoke(prompt)
    return response.content