"""
Custom RAG-quality eval: runs the testset through the LangGraph pipeline and
scores each answer two ways:
  - groundedness_overlap: the same word-overlap heuristic used by the
    output guardrail (app/guards/checks.py), as a continuous score.
  - relevancy: a 1-5 rating from the same local LLM used for generation,
    asked to respond with a single digit rather than structured JSON --
    small local models are unreliable at strict schema compliance.

Run: ./.venv/bin/python evals/run_eval.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_ollama import ChatOllama

from app.core.generate import MODEL_NAME
from app.graph.pipeline import rag_graph
from app.guards.checks import groundedness_overlap

TESTSET_PATH = Path(__file__).parent / "testsets" / "questions.json"
REPORT_PATH = Path(__file__).parent / "reports" / "latest.json"

JUDGE_PROMPT = """You are evaluating the quality of an AI assistant's answer to a question about a codebase.

Question: {question}

Answer: {answer}

Rate how well the answer addresses the question, on a scale of 1 (useless or wrong) to 5 (excellent, accurate, and directly answers the question).
Respond with ONLY a single digit from 1 to 5, nothing else."""


def judge_relevancy(question: str, answer: str) -> int | None:
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    response = llm.invoke(JUDGE_PROMPT.format(question=question, answer=answer))
    match = re.search(r"[1-5]", response.content)
    return int(match.group()) if match else None


def run() -> list[dict]:
    questions = json.loads(TESTSET_PATH.read_text())
    results = []

    for item in questions:
        question = item["question"]
        state = rag_graph.invoke(
            {"question": question, "repo": item.get("repo"), "top_k": item.get("top_k", 5)}
        )

        if state.get("blocked"):
            print(f"[BLOCKED] {question!r} -- {state['block_reason']}")
            results.append(
                {
                    "question": question,
                    "answer": state["answer"],
                    "blocked": True,
                    "block_reason": state["block_reason"],
                }
            )
            continue

        overlap = groundedness_overlap(state["answer"], state["chunks"])
        relevancy = judge_relevancy(question, state["answer"])

        print(f"[OK] {question!r} -- groundedness={overlap:.2f} relevancy={relevancy}/5")

        results.append(
            {
                "question": question,
                "answer": state["answer"],
                "blocked": False,
                "groundedness_overlap": overlap,
                "relevancy_1_to_5": relevancy,
            }
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote report to {REPORT_PATH}")
    return results


if __name__ == "__main__":
    run()
