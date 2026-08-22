"""
LangGraph wiring of the retrieve -> generate flow.

Each step is a named node operating on a shared state object, so later
steps (guardrails, tracing hooks, reranking) can be inserted without
rewriting the flow.
"""
from __future__ import annotations

from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from app.core.generate import MODEL_NAME, SYSTEM_PROMPT, build_context
from app.core.retrieve import RetrievedChunk, retrieve


class GraphState(TypedDict):
    question: str
    repo: str | None
    top_k: int
    chunks: list[RetrievedChunk]
    answer: str


def retrieve_node(state: GraphState) -> dict:
    chunks = retrieve(state["question"], repo=state.get("repo"), top_k=state.get("top_k", 5))
    return {"chunks": chunks}


def generate_node(state: GraphState) -> dict:
    chunks = state["chunks"]
    if not chunks:
        return {"answer": "No relevant code found for this question."}

    context = build_context(chunks)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {state['question']}\n\nAnswer:"

    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    response = llm.invoke(prompt)
    return {"answer": response.content}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


rag_graph = build_graph()


def answer(question: str, repo: str | None = None, top_k: int = 5) -> str:
    result = rag_graph.invoke({"question": question, "repo": repo, "top_k": top_k})
    return result["answer"]
