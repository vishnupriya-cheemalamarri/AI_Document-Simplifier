"""FastAPI entrypoint. Run with: python -m uvicorn backend.main:app --reload --port 8001"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .guardrails import prompt_injection
from .pipeline import embeddings
from .routes import documents, qa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Document Simplifier", version="0.1.0")

# The Streamlit frontend calls this API server-side (via `requests`), not
# from browser JS, so CORS isn't strictly required — kept permissive here
# in case a browser-based client is added later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(qa.router)


@app.on_event("startup")
def on_startup() -> None:
    # Load the embedding model once at startup, not per-request (see
    # ARCHITECTURE.md > Tech Stack). Also preload the prompt-injection
    # classifier here so the first user question isn't slowed down by it.
    logger.info("Loading embedding model %s ...", config.EMBEDDING_MODEL_NAME)
    embeddings.load_model()
    if config.ENABLE_PROMPT_INJECTION_CHECK:
        logger.info("Loading prompt-injection classifier %s ...", config.PROMPT_INJECTION_MODEL_NAME)
        prompt_injection.load_classifier()
    else:
        logger.info(
            "ENABLE_PROMPT_INJECTION_CHECK=false — skipping prompt-injection classifier "
            "download/load. Input questions will not be screened for injection attempts."
        )
    logger.info("Startup complete.")


@app.get("/health")
def health():
    return {"status": "ok"}
