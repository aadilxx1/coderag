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
from app.guards.checks import check_groundedness, check_input, check_output_toxicity


class GraphState(TypedDict):
    question: str
    repo: str | None
    top_k: int
    chunks: list[RetrievedChunk]
    answer: str
    blocked: bool
    block_reason: str | None


def input_guard_node(state: GraphState) -> dict:
    reason = check_input(state["question"])
    if reason:
        return {"blocked": True, "block_reason": reason, "answer": f"Request blocked: {reason}"}
    return {"blocked": False, "block_reason": None}


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


def output_guard_node(state: GraphState) -> dict:
    reason = check_output_toxicity(state["answer"]) or check_groundedness(state["answer"], state["chunks"])
    if reason:
        return {
            "blocked": True,
            "block_reason": reason,
            "answer": f"Answer withheld by guardrail: {reason}",
        }
    return {"blocked": False, "block_reason": None}


def _route_after_input_guard(state: GraphState) -> str:
    return "blocked" if state["blocked"] else "retrieve"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("output_guard", output_guard_node)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard", _route_after_input_guard, {"blocked": END, "retrieve": "retrieve"}
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "output_guard")
    graph.add_edge("output_guard", END)
    return graph.compile()


rag_graph = build_graph()


def answer(question: str, repo: str | None = None, top_k: int = 5) -> str:
    result = rag_graph.invoke({"question": question, "repo": repo, "top_k": top_k})
    return result["answer"]
