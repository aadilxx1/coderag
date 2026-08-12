cat > README.md << 'EOF'
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

- **Tests** (`tests/test_chunker.py`) — 3 automated tests proving the chunker
  works correctly, including an edge case (a file with a syntax error doesn't
  crash the whole thing).

### Step 2: Storage (Postgres + pgvector) — in progress

### Upcoming
- Embeddings
- Hybrid retrieval (semantic + keyword search)
- LangGraph flow (retrieve → generate)
- FastAPI wrapper
- Langfuse tracing
- Ragas evaluation suite
- Guardrails AI validation
- Docker + Docker Compose
- GitHub Actions CI/CD with an eval gate
EOF