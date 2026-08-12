"""
Central configuration, loaded from environment variables (see .env.example).

Per ARCHITECTURE.md:
- Embedding model is loaded once at app startup, not per-request (see main.py).
- PROMPT_INJECTION_MODEL_NAME's exact HuggingFace model ID should be verified
  before relying on it in a demo.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# --- Storage (prototype-appropriate local folder + TinyDB, per ARCHITECTURE.md) ---
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploaded_docs"))
DB_DIR = Path(os.getenv("DB_DIR", BASE_DIR / "db"))
DB_PATH = DB_DIR / "db.json"
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", BASE_DIR / "faiss_indices"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# --- Chunking: semantic, bounded to 10-20 sentences/chunk, 2-sentence overlap ---
CHUNK_MIN_SENTENCES = int(os.getenv("CHUNK_MIN_SENTENCES", "10"))
CHUNK_MAX_SENTENCES = int(os.getenv("CHUNK_MAX_SENTENCES", "20"))
CHUNK_OVERLAP_SENTENCES = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "2"))

# --- Embeddings: local, sentence-transformers, BAAI/bge-small-en-v1.5 ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# --- Retrieval: hybrid FAISS + BM25 via reciprocal rank fusion ---
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))

# Number of chunks combined into a single simplify + key-points LLM call, to
# cut OpenAI call volume on large documents (e.g. 542 chunks / 5 per call =
# ~109 calls instead of 542). See ARCHITECTURE.md > Tech Stack.
SIMPLIFY_BATCH_SIZE = int(os.getenv("SIMPLIFY_BATCH_SIZE", "5"))

# --- Guardrails ---
# NOTE: verify this exact model ID against HuggingFace before demo day (see ARCHITECTURE.md).
PROMPT_INJECTION_MODEL_NAME = os.getenv(
    "PROMPT_INJECTION_MODEL_NAME", "protectai/deberta-v3-base-prompt-injection-v2"
)
PROMPT_INJECTION_THRESHOLD = float(os.getenv("PROMPT_INJECTION_THRESHOLD", "0.5"))
# Dev-time escape hatch: set to "false" to skip downloading/loading the
# classifier entirely (e.g. on a slow connection while iterating on the core
# pipeline). Defaults to on, matching the documented architecture — this is a
# per-run toggle, not a change to that default.
ENABLE_PROMPT_INJECTION_CHECK = os.getenv("ENABLE_PROMPT_INJECTION_CHECK", "true").lower() == "true"
SCOPE_SIMILARITY_THRESHOLD = float(os.getenv("SCOPE_SIMILARITY_THRESHOLD", "0.3"))

# --- LLM: OpenAI gpt-4o-mini ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Observability: LangSmith reads LANGCHAIN_* env vars itself (see .env.example) ---
