"""
Retrieval-only eval: for each question in the testset, checks where the
expected chunk(s) land in retrieve()'s ranking. No LLM involved, so it's
fast and deterministic -- unlike run_eval.py, a score change here can only
come from retrieval (chunking, embeddings, hybrid fusion, reranking).

Metrics, per question then averaged:
  - hit@k: 1 if any expected chunk is in the top k, else 0
  - MRR:   1 / rank of the first expected chunk found (0 if not in top MAX_K)

Expected chunks are written as "file_path::symbol". Before scoring, every
one is checked against the database: a typo or a stale index would
otherwise just look like a retrieval miss.

By default this scores retrieval the way the pipeline does it: the top
RERANK_CANDIDATES from hybrid search, reordered by the cross-encoder.
--no-rerank scores hybrid search alone. (With reranking, hit@10 can't
differ from --no-rerank: the reranker only reorders those 10 candidates.)

Run: ./.venv/bin/python evals/run_retrieval_eval.py [--no-rerank]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.core.rerank import RERANK_CANDIDATES, rerank
from app.core.retrieve import retrieve
from app.core.store import get_engine

TESTSET_PATH = Path(__file__).parent / "testsets" / "retrieval.json"
REPORT_PATH = Path(__file__).parent / "reports" / "retrieval_latest.json"

KS = (1, 3, 5, 10)
MAX_K = max(KS)


def _key(file_path: str, symbol: str) -> str:
    return f"{file_path}::{symbol}"


def check_expected_exist(items: list[dict]) -> list[str]:
    """Return the expected chunks that aren't in the database at all."""
    with get_engine().connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT repo, file_path, symbol FROM code_chunks")).fetchall()
    stored = {(r.repo, _key(r.file_path, r.symbol)) for r in rows}
    return [
        f"{item['repo']}: {exp}"
        for item in items
        for exp in item["expected"]
        if (item["repo"], exp) not in stored
    ]


def first_hit_rank(retrieved: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk that is expected."""
    for rank, key in enumerate(retrieved, start=1):
        if key in expected:
            return rank
    return None


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    summary = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in KS}
    summary["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    summary["n"] = n
    return summary


def _fmt(summary: dict) -> str:
    hits = "  ".join(f"hit@{k}={summary[f'hit@{k}']:.2f}" for k in KS)
    return f"{hits}  MRR={summary['mrr']:.3f}  (n={summary['n']})"


def run(use_rerank: bool = True) -> dict:
    items = json.loads(TESTSET_PATH.read_text())

    missing = check_expected_exist(items)
    if missing:
        print("Expected chunks not found in the database (typo, or repo not ingested / stale?):")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    results = []
    for item in items:
        chunks = retrieve(item["question"], repo=item["repo"], top_k=max(MAX_K, RERANK_CANDIDATES))
        if use_rerank:
            chunks = rerank(item["question"], chunks, top_k=MAX_K)
        retrieved = [_key(c.file_path, c.symbol) for c in chunks]
        rank = first_hit_rank(retrieved, item["expected"])

        status = f"rank {rank}" if rank else "MISS"
        line = f"[{status:>7}] {item['repo']}: {item['question']}"
        if not rank and retrieved:
            line += f"  (top: {retrieved[0]})"
        print(line)
        results.append({**item, "rank": rank, "retrieved": retrieved})

    by_repo = defaultdict(list)
    for r in results:
        by_repo[r["repo"]].append(r["rank"])

    overall = summarize([r["rank"] for r in results])
    per_repo = {repo: summarize(ranks) for repo, ranks in by_repo.items()}

    print()
    for repo, summary in per_repo.items():
        print(f"{repo:>22}: {_fmt(summary)}")
    print(f"{'overall':>22}: {_fmt(overall)}")

    report = {"rerank": use_rerank, "overall": overall, "per_repo": per_repo, "results": results}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run(use_rerank="--no-rerank" not in sys.argv[1:])
