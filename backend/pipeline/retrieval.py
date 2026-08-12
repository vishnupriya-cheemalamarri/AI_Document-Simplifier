"""
Hybrid retrieval: BM25 (keyword) + FAISS (dense) combined via reciprocal
rank fusion (RRF) (see ARCHITECTURE.md > Tech Stack).

Also returns the raw top FAISS cosine-similarity score separately, since
that (not the fused RRF score) is what the scope guardrail thresholds
against (see backend/guardrails/scope.py).
"""

from typing import Dict, List, Tuple

import numpy as np

from . import embeddings, indexing
from .. import config


def hybrid_retrieve(doc_id: str, query: str, top_k: int | None = None) -> Tuple[List[Dict], float]:
    top_k = top_k or config.RETRIEVAL_TOP_K
    doc_index = indexing.get_index(doc_id)
    n = len(doc_index.chunk_ids)
    if n == 0:
        return [], 0.0

    k_dense = min(top_k * 3, n)

    # --- Dense (FAISS) ---
    query_vec = embeddings.embed_texts([query])
    faiss_scores, faiss_positions = doc_index.faiss_index.search(query_vec, k_dense)
    faiss_ranking = [int(pos) for pos in faiss_positions[0] if pos != -1]
    top_faiss_score = float(faiss_scores[0][0]) if len(faiss_scores[0]) else 0.0

    # --- Sparse (BM25) ---
    tokenized_query = query.lower().split()
    bm25_scores = doc_index.bm25.get_scores(tokenized_query)
    bm25_ranking = [int(pos) for pos in np.argsort(bm25_scores)[::-1][:k_dense]]

    # --- Reciprocal rank fusion ---
    rrf_scores: Dict[int, float] = {}
    for rank, pos in enumerate(faiss_ranking):
        rrf_scores[pos] = rrf_scores.get(pos, 0.0) + 1.0 / (config.RRF_K + rank + 1)
    for rank, pos in enumerate(bm25_ranking):
        rrf_scores[pos] = rrf_scores.get(pos, 0.0) + 1.0 / (config.RRF_K + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    results = [
        {
            "chunk_id": doc_index.chunk_ids[pos],
            "text": doc_index.chunk_texts[pos],
            "fused_score": score,
        }
        for pos, score in fused
    ]
    return results, top_faiss_score
