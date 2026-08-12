"""
Vector + keyword indices, one per uploaded document (see
docs/ARCHITECTURE.md > Tech Stack).

FAISS: IndexFlatIP over L2-normalized vectors (cosine similarity).
BM25: rank_bm25's BM25Okapi over whitespace-lowercased tokens.

Both are cached in memory for the running process AND persisted to disk
(config.FAISS_INDEX_DIR/{doc_id}.*) at build time. A server restart no
longer requires re-uploading documents: get_index() transparently loads a
document's index from disk into memory the first time it's needed after a
restart. (This used to be a documented Known Limitation — it isn't
anymore; see docs/ARCHITECTURE.md.)
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .. import config


@dataclass
class DocumentIndex:
    chunk_ids: List[str] = field(default_factory=list)
    chunk_texts: List[str] = field(default_factory=list)
    faiss_index: faiss.Index = None
    bm25: BM25Okapi = None


_indices: Dict[str, DocumentIndex] = {}


def _faiss_path(doc_id: str) -> Path:
    return config.FAISS_INDEX_DIR / f"{doc_id}.faiss"


def _meta_path(doc_id: str) -> Path:
    """BM25 has no persistence of its own, so it's pickled here alongside
    chunk_ids/chunk_texts — everything needed to reconstruct a DocumentIndex
    other than the FAISS index itself, which FAISS serializes natively."""
    return config.FAISS_INDEX_DIR / f"{doc_id}.meta.pkl"


def build_index(doc_id: str, chunk_ids: List[str], chunk_texts: List[str], embeddings: np.ndarray) -> None:
    """Build a fresh FAISS + BM25 index pair, register it in memory, and
    persist it to disk so it survives a server restart."""
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    tokenized_corpus = [t.lower().split() for t in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    _indices[doc_id] = DocumentIndex(
        chunk_ids=list(chunk_ids),
        chunk_texts=list(chunk_texts),
        faiss_index=faiss_index,
        bm25=bm25,
    )
    _persist_index(doc_id)


def _persist_index(doc_id: str) -> None:
    doc_index = _indices[doc_id]
    faiss.write_index(doc_index.faiss_index, str(_faiss_path(doc_id)))
    with open(_meta_path(doc_id), "wb") as f:
        pickle.dump(
            {
                "chunk_ids": doc_index.chunk_ids,
                "chunk_texts": doc_index.chunk_texts,
                "bm25": doc_index.bm25,
            },
            f,
        )


def _load_from_disk(doc_id: str) -> Optional[DocumentIndex]:
    faiss_path, meta_path = _faiss_path(doc_id), _meta_path(doc_id)
    if not (faiss_path.exists() and meta_path.exists()):
        return None

    faiss_index = faiss.read_index(str(faiss_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    doc_index = DocumentIndex(
        chunk_ids=meta["chunk_ids"],
        chunk_texts=meta["chunk_texts"],
        faiss_index=faiss_index,
        bm25=meta["bm25"],
    )
    _indices[doc_id] = doc_index
    return doc_index


def has_index(doc_id: str) -> bool:
    """True if an index is available in memory or on disk. A disk hit is
    not loaded into memory by this check alone — see get_index()."""
    return doc_id in _indices or (_faiss_path(doc_id).exists() and _meta_path(doc_id).exists())


def get_index(doc_id: str) -> DocumentIndex:
    if doc_id in _indices:
        return _indices[doc_id]

    doc_index = _load_from_disk(doc_id)
    if doc_index is None:
        raise KeyError(
            f"No index found for doc_id={doc_id!r} (checked memory and "
            f"{config.FAISS_INDEX_DIR}). Re-upload the document to build one."
        )
    return doc_index
