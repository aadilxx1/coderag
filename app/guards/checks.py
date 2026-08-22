"""
Lightweight, dependency-free guardrail checks for the RAG pipeline.

Each check returns a rejection reason (str) if the input/output should be
blocked, or None if it passes. Plain heuristics, not ML classifiers --
good enough as a first line of defense, cheap to run on every request.
"""
from __future__ import annotations

import re

from app.core.retrieve import RetrievedChunk

MAX_QUESTION_LENGTH = 2000

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|any|previous|prior) instructions", re.IGNORECASE),
    re.compile(r"disregard (the|your|all) (system|previous) prompt", re.IGNORECASE),
    re.compile(r"reveal (your|the) (system prompt|instructions)", re.IGNORECASE),
    re.compile(r"you are now (a|an|no longer)", re.IGNORECASE),
    re.compile(r"pretend (to be|you are)", re.IGNORECASE),
]

_TOXIC_TERMS = {
    "kill yourself",
    "idiot",
    "stupid piece of",
}


def check_input(question: str) -> str | None:
    if not question or not question.strip():
        return "Question is empty."
    if len(question) > MAX_QUESTION_LENGTH:
        return "Question is too long."
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(question):
            return "Question looks like a prompt-injection attempt."
    return None


def check_output_toxicity(answer: str) -> str | None:
    lowered = answer.lower()
    for term in _TOXIC_TERMS:
        if term in lowered:
            return "Answer contains disallowed language."
    return None


_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")

MIN_GROUNDEDNESS_OVERLAP = 0.15


def check_groundedness(answer: str, chunks: list[RetrievedChunk]) -> str | None:
    """Heuristic: what fraction of the answer's distinct words also appear in
    the retrieved context. A paraphrased-but-grounded answer still shares
    plenty of vocabulary (symbol names, code terms) with its source chunks;
    an ungrounded/hallucinated one usually doesn't."""
    if not chunks:
        return None

    context_text = " ".join(
        f"{c.symbol} {c.file_path} {c.docstring or ''} {c.code}" for c in chunks
    )
    context_words = set(_WORD_RE.findall(context_text.lower()))
    answer_words = set(_WORD_RE.findall(answer.lower()))

    if not answer_words:
        return None

    overlap = len(answer_words & context_words) / len(answer_words)
    if overlap < MIN_GROUNDEDNESS_OVERLAP:
        return "Answer has little overlap with retrieved code -- possibly ungrounded."
    return None
