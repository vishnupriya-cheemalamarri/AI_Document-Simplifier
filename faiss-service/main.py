"""
FAISS vector DB microservice.

FAISS has no built-in server/API of its own, so this wraps it in a small
FastAPI app: your RAG app computes embeddings (with whatever model it likes)
and sends the raw vectors here for indexing/search. The index + metadata are
persisted to disk (mounted as a Docker volume) so data survives restarts.
"""

import os
import pickle
import threading
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/data/index.faiss")
METADATA_PATH = os.environ.get("FAISS_METADATA_PATH", "/data/metadata.pkl")
METRIC = os.environ.get("METRIC", "l2").lower()  # "l2" (Euclidean) or "ip" (cosine — vectors are auto-normalized)

app = FastAPI(title="FAISS Vector DB Service", version="1.0.0")

_lock = threading.Lock()


class State:
    index: Optional[faiss.Index] = None
    dim: Optional[int] = None
    metadata: Dict[int, Dict[str, Any]] = {}


state = State()


def _new_index(dim: int) -> faiss.Index:
    base = faiss.IndexFlatIP(dim) if METRIC == "ip" else faiss.IndexFlatL2(dim)
    return faiss.IndexIDMap2(base)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows to unit length so IP search becomes cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero for an all-zero vector
    return vectors / norms


def _save() -> None:
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(state.index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump({"dim": state.dim, "metadata": state.metadata}, f)


def _load() -> None:
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        state.index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            payload = pickle.load(f)
            state.dim = payload["dim"]
            state.metadata = payload["metadata"]


@app.on_event("startup")
def startup() -> None:
    _load()


class AddRequest(BaseModel):
    ids: List[int] = Field(..., description="Unique integer ID per vector")
    embeddings: List[List[float]]
    metadatas: Optional[List[Dict[str, Any]]] = None


class SearchRequest(BaseModel):
    embedding: List[float]
    k: int = 5


class SearchResult(BaseModel):
    id: int
    score: float
    metadata: Optional[Dict[str, Any]] = None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "index_initialized": state.index is not None,
        "metric": METRIC,
        "dimension": state.dim,
        "total_vectors": int(state.index.ntotal) if state.index is not None else 0,
    }


@app.post("/add")
def add(req: AddRequest) -> Dict[str, Any]:
    if not req.ids or not req.embeddings:
        raise HTTPException(400, "ids and embeddings must be non-empty")
    if len(req.ids) != len(req.embeddings):
        raise HTTPException(400, "ids and embeddings must be the same length")
    if req.metadatas is not None and len(req.metadatas) != len(req.ids):
        raise HTTPException(400, "metadatas must match ids length")

    vectors = np.array(req.embeddings, dtype="float32")
    dim = vectors.shape[1]
    if METRIC == "ip":
        vectors = _normalize(vectors)

    with _lock:
        if state.index is None:
            state.dim = dim
            state.index = _new_index(dim)
        elif dim != state.dim:
            raise HTTPException(
                400, f"embedding dimension {dim} does not match index dimension {state.dim}"
            )

        ids_arr = np.array(req.ids, dtype="int64")
        state.index.add_with_ids(vectors, ids_arr)

        for i, doc_id in enumerate(req.ids):
            state.metadata[doc_id] = (req.metadatas[i] if req.metadatas else {})

        _save()

    return {"added": len(req.ids), "total_vectors": int(state.index.ntotal)}


@app.post("/search", response_model=List[SearchResult])
def search(req: SearchRequest) -> List[SearchResult]:
    if state.index is None or state.index.ntotal == 0:
        return []
    if len(req.embedding) != state.dim:
        raise HTTPException(
            400, f"embedding dimension {len(req.embedding)} does not match index dimension {state.dim}"
        )

    query = np.array([req.embedding], dtype="float32")
    if METRIC == "ip":
        query = _normalize(query)
    with _lock:
        scores, ids = state.index.search(query, req.k)

    results: List[SearchResult] = []
    for score, doc_id in zip(scores[0], ids[0]):
        if doc_id == -1:
            continue
        results.append(
            SearchResult(id=int(doc_id), score=float(score), metadata=state.metadata.get(int(doc_id)))
        )
    return results


@app.delete("/reset")
def reset() -> Dict[str, str]:
    with _lock:
        state.index = None
        state.dim = None
        state.metadata = {}
        for path in (INDEX_PATH, METADATA_PATH):
            if os.path.exists(path):
                os.remove(path)
    return {"status": "reset"}


@app.get("/stats")
def stats() -> Dict[str, Any]:
    return {
        "total_vectors": int(state.index.ntotal) if state.index is not None else 0,
        "dimension": state.dim,
        "metric": METRIC,
        "documents_with_metadata": len(state.metadata),
    }
