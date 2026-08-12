"""
Vector + keyword indices, one per uploaded document, held in memory
(see docs/ARCHITECTURE.md > Tech Stack and > Known Limitations).

FAISS: IndexFlatIP over L2-normalized vectors (cosine similarity).
BM25: rank_bm25's BM25Okapi over whitespace-lowercased tokens.

These indices are NOT persisted across server restarts — a restart requires
re-uploading documents to rebuild them. TinyDB metadata (backend/db.py) does
persist, but is not enough on its own to answer questions again.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class DocumentIndex:
    chunk_ids: List[str] = field(default_factory=list)
    chunk_texts: List[str] = field(default_factory=list)
    faiss_index: faiss.Index = None
    bm25: BM25Okapi = None


_indices: Dict[str, DocumentIndex] = {}


def build_index(doc_id: str, chunk_ids: List[str], chunk_texts: List[str], embeddings: np.ndarray) -> None:
    """Build and register a fresh FAISS + BM25 index pair for one document."""
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


def has_index(doc_id: str) -> bool:
    return doc_id in _indices


def get_index(doc_id: str) -> DocumentIndex:
    if doc_id not in _indices:
        raise KeyError(
            f"No in-memory index for doc_id={doc_id!r}. The server may have "
            "restarted — re-upload the document to rebuild its index."
        )
    return _indices[doc_id]
