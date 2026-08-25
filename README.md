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
  re-ingesting a repo replaces its rows entirely (delete-then-insert per
  `repo`), so re-running it always reflects the repo's current state with
  no duplicate rows

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
  with named nodes (`input_guard` → `retrieve` → `generate` → `output_guard`)
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

### Environment
- Project dependencies are isolated in `.venv/` (pinned in `requirements.txt`)
  rather than the system/Anaconda Python — run scripts via `./.venv/bin/python`

### Upcoming
- Langfuse tracing
- Dockerize the app itself (Postgres is already containerized)
- GitHub Actions CI/CD with an eval gate
