# AI Document Simplifier

Prodapt Hackathon — Group 17. Full design rationale, diagrams, and known
limitations live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — read that
first. This README is a quick reference; for full step-by-step setup
(fresh clone → running, with troubleshooting) see
[SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md).

## Prerequisites

- Python 3.10+
- An OpenAI API key (required — the app calls `gpt-4o-mini` for
  simplification, key points, and Q&A)
- A LangSmith API key (optional — observability tracing is skipped
  gracefully if unset)

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` installs everything for both the backend and frontend,
including `sentence-transformers`/`torch`/`transformers` for the local
embedding model and prompt-injection classifier — this is a large download
(~2GB+). See the note at the top of `requirements.txt` if you want a
CPU-only `torch` wheel instead of the default.

Then:

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

Fill in `OPENAI_API_KEY` (required) and, optionally, the `LANGCHAIN_*` vars
for LangSmith tracing.

## Running

**Quickest way (Windows):** once dependencies are installed and `.env` is set up (see Setup above), just run:

```
start.bat
```

from the repository root. It starts the backend, waits 30 seconds for it to finish loading models, then starts the frontend — each in its own console window so you can watch its live output. Close a window (or Ctrl+C inside it) to stop that server.

**Manual way** (any OS, or if you want more control): two processes, in two terminals, both from the repository root:

```bash
# Terminal 1 — backend (FastAPI)
python -m uvicorn backend.main:app --reload --port 8001
```

```bash
# Terminal 2 — frontend (Streamlit)
streamlit run frontend/app.py
```

The first backend startup downloads and loads the embedding model and the
prompt-injection classifier, which can take a while the first time. Open
the Streamlit URL it prints (typically http://localhost:8501), upload a
PDF or .txt file, and go.

## Project layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full annotated
folder tree, data model, and both pipeline diagrams (ingestion and
query-time).

## Note on `faiss-service/`

The repository also contains a standalone `faiss-service/` Docker
microservice (a generic FAISS HTTP wrapper with a single shared, persisted
index). It predates and doesn't match the architecture documented in
`docs/ARCHITECTURE.md` (one in-memory FAISS index per uploaded document,
built directly inside the FastAPI backend, not persisted). By design
decision, the backend does **not** use it — it's left in place but unused.
See `docker-compose.yml` / `faiss-service/` if you want to run it
separately; it isn't required for anything above.

## Known limitations

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#known-limitations) —
notably: in-memory FAISS indices don't survive a backend restart (you'll
need to re-upload documents), and there's no OCR support for scanned PDFs.
