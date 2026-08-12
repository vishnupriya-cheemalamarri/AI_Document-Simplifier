"""
Local embeddings via sentence-transformers (see docs/ARCHITECTURE.md > Tech Stack).

Model: BAAI/bge-small-en-v1.5, chosen over an API-based embedding model to
avoid per-request network latency during upload and to remove a dependency
on venue wifi during the live demo. Loaded once (lazily, thread-safely) and
reused — call `load_model()` at app startup so the first request isn't slow.
"""

import threading
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .. import config

_model: SentenceTransformer | None = None
_lock = threading.Lock()


def load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Encode texts into L2-normalized float32 vectors (cosine-ready for IndexFlatIP)."""
    model = load_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")
