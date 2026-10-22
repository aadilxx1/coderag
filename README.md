# CodeRAG — Code Repository Q&A Assistant

An LLM-powered system for asking natural-language questions about a codebase,
built with production practices around it: automated evaluation, tracing,
guardrails, and CI.

## Progress Log

### Step 1: AST-aware chunking ✅

- **The chunker** (`app/ingest/chunker.py`) — takes a Python file, parses it
  with `ast`, and splits it into clean chunks: one per function, one per
  class, plus a leftover `<module>` chunk for imports/constants. Each chunk
  carries metadata (file path, symbol name, docstring, exact line numbers).
- Classes are chunked per method (`ClassName.method`), not as one giant
  blob — a single whole-class chunk truncates at `max_chars` for any class
  with more than a few methods, silently losing the rest. A lightweight
  class-summary chunk (header + docstring + method signatures, no bodies)
  is still emitted alongside the per-method chunks.
- **Tests** (`tests/test_chunker.py`) — automated tests covering functions,
  classes with and without methods, and an edge case (a file with a syntax
  error doesn't crash the whole ingest).

### Step 2: Storage (Postgres + pgvector) ✅
- Postgres running via `docker-compose.yml` using the `pgvector/pgvector:pg16` image
- Schema (`app/core/store.py`): `code_chunks` table with a `vector(768)` embedding
  column and a generated `tsvector` column (`'simple'` config, no stemming —
  code identifiers aren't natural language) with a GIN index for keyword search
- `scripts/ingest.py` chunks a repo, embeds every chunk, and stores it —
  re-running it always reflects the repo's current state with no duplicate
  rows (incremental since Step 14: only changed files are re-embedded)

### Step 3: Embeddings ✅
- Local embeddings via `sentence-transformers`, no API key (`app/core/embeddings.py`)
- Model: `jinaai/jina-embeddings-v2-base-code` — trained on code/docstring
  pairs and code-to-code similarity, 768-dim vectors

### Step 4: Retrieval — hybrid (semantic + keyword) ✅
- `app/core/retrieve.py` runs pgvector cosine similarity search and Postgres
  full-text search in parallel, then merges the two rankings with
  Reciprocal Rank Fusion
- Catches queries a pure embedding search misses or under-ranks — e.g. a
  question naming a literal identifier or constant (`RRF_K`, `top_k`)
  surfaces the chunk that defines it via the keyword side, even when the
  embedding alone doesn't rank it highly

### Step 5: Generation ✅
- `app/core/generate.py` — builds a grounding prompt from retrieved chunks
  and answers via a local LLM through Ollama (`llama3.2:3b`)

### Step 6: LangGraph flow (retrieve → generate) ✅
- `app/graph/pipeline.py` — the retrieve/generate flow as a `StateGraph`
  with named nodes (`input_guard` → `retrieve` → `rerank` → `generate` → `output_guard`)
  sharing one state object, instead of one function doing everything —
  makes it straightforward to insert further steps (reranking, tracing)
  without rewriting the flow

### Step 7: Guardrails ✅
- `app/guards/checks.py` — dependency-free heuristic checks wired into the
  graph as `input_guard`/`output_guard` nodes:
  - input validation: rejects empty/oversized questions and common
    prompt-injection phrasings
  - output toxicity: blocklist screen on the generated answer
  - groundedness: word-overlap ratio between the answer and its retrieved
    chunks, tolerant of paraphrasing
- The official `guardrails-ai` Hub validators were tried first but are
  unmaintained against current `langchain`/`langgraph` versions — dropped
  in favor of the checks above

### Step 8: FastAPI wrapper ✅
- `app/api/main.py` — `GET /health` (liveness) and `POST /ask`
  (`question`, optional `repo`/`top_k`) calling the compiled LangGraph pipeline

### Step 9: Evaluation ✅
- `evals/run_eval.py` — runs a testset (`evals/testsets/questions.json`)
  through the pipeline and scores each answer with the same groundedness
  heuristic as the guardrail, plus a 1-5 relevancy rating from the local
  LLM (a plain digit, not strict JSON — small local models aren't reliable
  at strict structured output)
- `ragas` and `deepeval` were both tried first: `ragas` has a packaging bug
  incompatible with current `langgraph`, and `deepeval`'s LLM-judge metrics
  need stricter structured-output compliance than a local 3B model reliably
  gives — a custom script was used instead
- This eval suite is what caught two real bugs: a stale vector index (files
  added after the first ingest were never re-ingested) and the chunking gap
  described in Step 1

### Step 10: Ingest from GitHub ✅
- `app/ingest/github.py` shallow-clones (`--depth 1`) a public repo into a
  temp dir with the system `git`, which is deleted after ingest
- Only `https://github.com/owner/repo` URLs (optionally `.git` or
  `/tree/<branch>`) are accepted, and the clone URL is rebuilt from the
  parsed parts so `file://`, local paths and other hosts can't be cloned
- `POST /repos` (`url`, optional `name`/`ref`) and
  `python scripts/ingest.py https://github.com/owner/repo [name]`; the repo
  is stored as `owner/repo` by default
- Ingest logic moved from the script into `app/ingest/pipeline.py`, so the
  CLI and API share it

### Step 11: Retrieval metrics ✅
- `evals/run_retrieval_eval.py` scores retrieval alone -- no LLM, so it's
  fast and deterministic, and a score change can only come from retrieval
- Testset (`evals/testsets/retrieval.json`): 31 questions across `coderag`
  and `pallets/itsdangerous`, each naming the chunk(s) that should come back
  as `file_path::symbol`; the script first checks every expected chunk exists
  in the DB, so a typo or stale index fails loudly instead of looking like a miss
- Reports hit@1/3/5/10 and MRR, overall and per repo
- Baseline (hybrid retrieval): hit@1 0.29, hit@5 0.81, MRR 0.484 overall --
  coderag MRR 0.634, itsdangerous MRR 0.325. Most misses are near-misses
  (the class summary or a sibling method outranks the expected method);
  test functions and duplicate `@overload` stub chunks also crowd results

### Step 12: Retrieval noise fixes ✅
Each change measured on the retrieval eval (33 questions -- 2 test-intent
questions were added first, so down-ranking tests can't silently break them):

| | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| baseline | 0.30 | 0.55 | 0.82 | 0.94 | 0.495 |
| + skip `@overload` stubs | 0.33 | 0.58 | 0.76 | 0.94 | 0.515 |
| + down-rank test files | 0.33 | 0.76 | 0.85 | 0.97 | **0.554** |

- **Overload stubs** (`app/ingest/chunker.py`): `@overload` / `@t.overload`
  signature-only stubs are no longer chunked -- a function with N overloads
  was N+1 near-identical chunks (`Serializer.__init__` x6) crowding out real code
- **Test down-ranking** (`app/core/retrieve.py`): chunks from test files get
  their fused RRF score scaled by `TEST_PENALTY = 0.4`, unless the question
  mentions tests. Test names restate behavior in the question's own words, so
  they often outranked the implementation. Swept 1.0 -> 0.0; gains plateau at
  0.4, and the test-intent questions kept their ranks at every value
- hit@1 barely moved: the right chunk is now usually in the top 3 but behind
  a closely related one (class summary, sibling method) -- the reranker's job

### Step 13: Cross-encoder reranking ✅
- `app/core/rerank.py` + a `rerank` graph node: hybrid retrieval fetches 10
  candidates, a cross-encoder scores each (question, chunk) pair together,
  and the top `top_k` go to generation
- Three local models compared on the retrieval eval (10 candidates):

| | hit@1 | hit@3 | hit@5 | MRR | latency/query (M3) |
|---|---|---|---|---|---|
| no rerank | 0.33 | 0.76 | 0.85 | 0.554 | -- |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 0.33 | 0.76 | 0.91 | 0.578 | ~0.2s |
| `BAAI/bge-reranker-base` (20 candidates) | 0.36 | 0.79 | 0.85 | 0.565 | ~2.4s |
| **`jinaai/jina-reranker-v2-base-multilingual`** | **0.55** | **0.79** | **0.91** | **0.688** | ~1s |

- jina-reranker-v2 is trained partly on code search, which shows: itsdangerous
  MRR 0.377 -> 0.638. 15 and 30 candidates scored the same as 10 at 2-3x the
  latency
- **License: jina-reranker-v2 is CC-BY-NC-4.0 (non-commercial only).** Swap
  `MODEL_NAME` in `rerank.py` to the Apache-2.0 MiniLM model for commercial use
- `evals/run_retrieval_eval.py` now reranks by default; `--no-rerank` scores
  hybrid search alone

### Step 14: Incremental re-ingest ✅
- A new `ingested_files` table stores a SHA-256 of each file's content; a
  re-ingest only chunks + embeds files that are new or changed, deletes rows
  for files that are gone, and skips the rest. Content hashes (not mtimes)
  mean it also works for GitHub repos, which are freshly cloned each time
- `INDEX_VERSION` in `app/ingest/pipeline.py` is part of every hash -- bump it
  when chunking or embedding changes, so old chunks don't silently mix with new
- `--full` (CLI) / `"full": true` (`POST /repos`) re-embeds everything;
  `POST /repos` now returns per-file counts (added / modified / removed /
  unchanged) and chunks inserted / total
- All deletes and inserts for one ingest run in a single transaction, so a
  failure mid-ingest leaves the previous index intact
- Verified: an unchanged re-ingest embeds 0 chunks (coderag: ~6s, mostly
  model load); editing / deleting / adding one file each re-embeds only those;
  itsdangerous retrieval scores are identical to a full rebuild

### Environment
- Project dependencies are isolated in `.venv/` (pinned in `requirements.txt`)
  rather than the system/Anaconda Python — run scripts via `./.venv/bin/python`

### Upcoming
- Langfuse tracing
- Dockerize the app itself (Postgres is already containerized)
- GitHub Actions CI/CD with an eval gate
