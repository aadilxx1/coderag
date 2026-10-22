// CodeRAG UI: asks questions and ingests repos through the Flask proxy (/api/*).
//
// Security note: answers come from an LLM and sources come from arbitrary
// ingested repos, so neither is trusted. Answer markdown is sanitized with
// DOMPurify before insertion; source code is only ever set via textContent.

const $ = (id) => document.getElementById(id);

const messages = $("messages");
const askForm = $("ask-form");
const question = $("question");
const askBtn = $("ask-btn");
const repoSelect = $("repo");
const topK = $("top-k");

topK.addEventListener("input", () => { $("top-k-value").textContent = topK.value; });

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    // FastAPI validation errors put a list in `detail`; everything else a string.
    const detail = Array.isArray(data.detail)
      ? data.detail.map((d) => d.msg).join("; ")
      : data.detail;
    throw new Error(detail || `Request failed (HTTP ${resp.status})`);
  }
  return data;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function addMessage(role) {
  $("empty-state")?.remove();
  const msg = el("div", `msg ${role}`);
  const bubble = el("div", "bubble");
  msg.appendChild(bubble);
  messages.appendChild(msg);
  messages.scrollTop = messages.scrollHeight;
  return { msg, bubble };
}

function renderMarkdown(bubble, markdown) {
  bubble.innerHTML = DOMPurify.sanitize(marked.parse(markdown));
  bubble.querySelectorAll("pre code").forEach((block) => hljs.highlightElement(block));
}

function renderSources(msg, sources) {
  if (!sources.length) return;
  const wrap = el("div", "sources");
  wrap.appendChild(el("p", "sources-label", `Sources (${sources.length})`));
  for (const s of sources) {
    const details = el("details", "source");
    const summary = el("summary");
    summary.appendChild(document.createTextNode(`${s.file_path}:${s.start_line}-${s.end_line} `));
    summary.appendChild(el("span", "symbol", s.symbol));
    const code = el("code", "language-python", s.code);
    const pre = el("pre");
    pre.appendChild(code);
    details.append(summary, pre);
    // Highlight lazily: most sources are never opened.
    details.addEventListener("toggle", () => {
      if (details.open && !code.dataset.highlighted) hljs.highlightElement(code);
    }, { once: true });
    wrap.appendChild(details);
  }
  msg.appendChild(wrap);
}

async function ask() {
  const text = question.value.trim();
  if (!text) return;

  addMessage("user").bubble.textContent = text;
  question.value = "";
  question.style.height = "";

  const { msg, bubble } = addMessage("assistant");
  bubble.appendChild(el("span", "thinking", "Thinking"));
  askBtn.disabled = true;

  try {
    const data = await postJSON("/api/ask", {
      question: text,
      repo: repoSelect.value || null,
      top_k: Number(topK.value),
    });
    if (data.blocked) {
      bubble.classList.add("blocked");
      bubble.textContent = data.answer;
    } else {
      renderMarkdown(bubble, data.answer);
    }
    renderSources(msg, data.sources || []);
  } catch (err) {
    bubble.classList.add("error");
    bubble.textContent = err.message;
  } finally {
    askBtn.disabled = false;
    messages.scrollTop = messages.scrollHeight;
    question.focus();
  }
}

askForm.addEventListener("submit", (e) => { e.preventDefault(); ask(); });

// Enter sends, Shift+Enter adds a newline; the box grows with its content.
question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    askForm.requestSubmit();
  }
});
question.addEventListener("input", () => {
  question.style.height = "auto";
  question.style.height = `${question.scrollHeight}px`;
});

// ---- ingest ----

const ingestForm = $("ingest-form");
const ingestBtn = $("ingest-btn");
const ingestStatus = $("ingest-status");

async function refreshRepos(selected) {
  const resp = await fetch("/api/repos");
  if (!resp.ok) return;
  const repos = await resp.json();
  repoSelect.replaceChildren(el("option", "", "All repos"));
  repoSelect.firstChild.value = "";
  for (const r of repos) {
    const opt = el("option", "", `${r.repo} (${r.files} files)`);
    opt.value = r.repo;
    repoSelect.appendChild(opt);
  }
  if (selected) repoSelect.value = selected;
}

ingestForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  ingestBtn.disabled = true;
  ingestStatus.className = "status";
  ingestStatus.textContent = "Cloning and embedding... this can take a minute.";

  try {
    const data = await postJSON("/api/ingest", {
      url: $("ingest-url").value.trim(),
      ref: $("ingest-ref").value.trim() || null,
      full: $("ingest-full").checked,
    });
    const changed = data.files_added + data.files_modified;
    ingestStatus.textContent =
      `${data.repo}: ${changed} files indexed, ${data.files_unchanged} unchanged, ` +
      `${data.files_removed} removed -- ${data.chunks_total} chunks total.`;
    await refreshRepos(data.repo);
  } catch (err) {
    ingestStatus.className = "status error";
    ingestStatus.textContent = err.message;
  } finally {
    ingestBtn.disabled = false;
  }
});
