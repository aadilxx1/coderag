"""
Flask web UI for CodeRAG.

A thin client over the FastAPI service (app/api/main.py): it renders the page
and forwards the browser's JSON requests to the API, so the browser only ever
talks to this server (no CORS) and all RAG logic stays in one place.

Run (with the API already running on :8000):
    ./.venv/bin/python ui/app.py
Then open http://127.0.0.1:5050
"""
from __future__ import annotations

import os

import httpx
from flask import Flask, jsonify, render_template, request

API_URL = os.environ.get("CODERAG_API_URL", "http://127.0.0.1:8000")
# Generation on a local 3B model, and ingesting a large repo, can both take
# minutes -- well past httpx's 5s default.
ASK_TIMEOUT_S = 300
INGEST_TIMEOUT_S = 900

app = Flask(__name__)


def _forward(method: str, path: str, timeout: float, json: dict | None = None):
    """Call the API and relay its response (status code included)."""
    try:
        resp = httpx.request(method, f"{API_URL}{path}", json=json, timeout=timeout)
    except httpx.ConnectError:
        return jsonify(detail=f"Can't reach the CodeRAG API at {API_URL} -- is it running?"), 502
    except httpx.TimeoutException:
        return jsonify(detail="The API took too long to respond."), 504

    try:
        body = resp.json()
    except ValueError:
        body = {"detail": resp.text or f"API returned HTTP {resp.status_code}"}
    return jsonify(body), resp.status_code


@app.get("/")
def index():
    try:
        repos = httpx.get(f"{API_URL}/repos", timeout=10).raise_for_status().json()
        api_error = None
    except httpx.HTTPError:
        repos, api_error = [], f"Can't reach the CodeRAG API at {API_URL} -- start it, then reload."
    return render_template("index.html", repos=repos, api_error=api_error)


@app.get("/api/repos")
def repos():
    return _forward("GET", "/repos", timeout=10)


@app.post("/api/ask")
def ask():
    return _forward("POST", "/ask", timeout=ASK_TIMEOUT_S, json=request.get_json(silent=True))


@app.post("/api/ingest")
def ingest():
    return _forward("POST", "/repos", timeout=INGEST_TIMEOUT_S, json=request.get_json(silent=True))


if __name__ == "__main__":
    # Port 5000 is taken by AirPlay Receiver on macOS.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5050)), debug=True)
