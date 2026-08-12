# Setup Instructions

Step-by-step instructions for getting this project running from a fresh
`git clone` — for anyone pulling this repo for the first time. For the
"why" behind these choices, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Prerequisites

- **Python 3.10+**
- **Git**
- An **OpenAI API key** (required — the app calls `gpt-4o-mini` for
  simplification, key points, and Q&A; nothing works without it)
- A **LangSmith API key** (optional — observability tracing is skipped
  gracefully if you don't set one)
- Windows, if you want to use the `start.bat` convenience script below.
  Everything else (backend, frontend, all the setup steps) works on
  macOS/Linux too — just skip `start.bat` and run the two processes
  manually (step 6).

## 1. Clone the repo

```
git clone https://github.com/vishnupriya-cheemalamarri/AI_Document-Simplifier.git
cd AI_Document-Simplifier
```

## 2. Create a virtual environment

```
python -m venv .venv
```

Activate it:

```
.venv\Scripts\activate        REM Windows
source .venv/bin/activate     # macOS/Linux
```

## 3. Install dependencies

```
pip install -r requirements.txt
```

**This is a large install — 2GB+.** It pulls in `torch`, `transformers`,
and `sentence-transformers` for the local embedding model and the
prompt-injection classifier, on top of the usual FastAPI/Streamlit stack.
On a slow or flaky connection, add retry/timeout flags so a single stalled
download doesn't fail the whole install:

```
pip install --timeout 100 --retries 10 -r requirements.txt
```

## 4. Configure environment variables

```
copy .env.example .env        REM Windows
cp .env.example .env          # macOS/Linux
```

Open `.env` and set, at minimum:

- `OPENAI_API_KEY` — **required**

Optional, but worth knowing about:

- `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` — set
  these to enable LangSmith tracing; leave unset and it's silently skipped.
- `ENABLE_PROMPT_INJECTION_CHECK` — defaults to `true`. Set to `false` if
  you want to skip downloading the ~700MB prompt-injection classifier
  (step 5) while you're just getting the core pipeline running; flip it
  back to `true` later to test that guardrail.

## 5. First run downloads two ML models (one-time)

The **first** time the backend starts, it downloads and caches (to
`~/.cache/huggingface`) whatever's enabled:

| Model | Size | Always downloaded? |
|---|---|---|
| `BAAI/bge-small-en-v1.5` (embeddings) | ~127 MB | Yes, always |
| `protectai/deberta-v3-base-prompt-injection-v2` (prompt-injection guardrail) | ~700 MB | Only if `ENABLE_PROMPT_INJECTION_CHECK=true` (the default) |

This can take anywhere from under a minute to 10+ minutes depending on
your connection. **Every run after the first is instant** — the models are
cached to disk, not re-downloaded.

## 6. Run it

**Windows — quickest way:**

```
start.bat
```

This starts the backend, waits 30 seconds for it to finish loading models,
then starts the frontend — each in its own console window so you can watch
its live output. Close a window (or Ctrl+C inside it) to stop that server.

**Manual way (any OS):** two terminals, both from the repo root:

```
# Terminal 1 — backend
python -m uvicorn backend.main:app --reload --port 8001
```

```
# Terminal 2 — frontend
streamlit run frontend/app.py --server.headless true
```

`--server.headless true` matters here: without it, Streamlit's very first
run in a real interactive terminal blocks on an unanswered "send usage
statistics?" prompt and silently never comes up. `start.bat` already
includes this flag; add it yourself if running manually.

## 7. Verify it's working

```
curl http://localhost:8001/health
```

should return `{"status":"ok"}`. Then open **http://localhost:8501** in a
browser, upload a PDF or `.txt` file, wait for it to process, and try
asking it a question.

## Troubleshooting

- **`Client.__init__() got an unexpected keyword argument 'proxies'`** —
  an `openai`/`httpx` version mismatch. `requirements.txt` already pins
  `httpx==0.27.2` to prevent this on a fresh install; if you still hit it,
  run `pip install httpx==0.27.2` inside your venv.
- **Frontend window opens but `localhost:8501` never loads, no error
  shown** — Streamlit's first-run prompt (see step 6) blocking on
  unanswered input. Make sure `--server.headless true` is present.
- **pip install fails partway with a read-timeout** — flaky connection on
  a large package (usually `torch`). Re-run with the `--timeout --retries`
  flags from step 3; the download resumes/retries rather than starting
  over.
- **"No in-memory index for this document" when asking a question** — the
  backend was restarted since that document was uploaded. FAISS/BM25
  indices are in-memory only and don't survive a restart (see
  [ARCHITECTURE.md](ARCHITECTURE.md) > Known Limitations) —
  just re-upload the document.
- **Large document uploads take a long time** — simplification + key
  points run at upload time, batched `SIMPLIFY_BATCH_SIZE` (default 5)
  chunks per LLM call. A very large document (hundreds of chunks) can
  still take several minutes even batched; this is expected, not a hang.

## Where to go next

- [ARCHITECTURE.md](ARCHITECTURE.md) — full architecture, tech
  stack + rationale, data model, mermaid diagrams, known limitations.
- [README.md](README.md) — short-form quick reference.
