"""
Semantic sentence-based chunker (see ARCHITECTURE.md > Tech Stack).

Not LangChain's SemanticChunker or any other off-the-shelf implementation —
this is a small custom version, consistent with the rest of this pipeline
(also hand-rolled: chunking, RRF fusion, retrieval). It reuses the same
local sentence-transformers model already loaded for retrieval embeddings,
so there's no new dependency or model download.

How it works: split into sentences, embed every sentence locally, and walk
forward chunk by chunk. Within each chunk's allowed size band
(CHUNK_MIN_SENTENCES..CHUNK_MAX_SENTENCES), the cut point is placed at the
position of lowest similarity between consecutive sentences — i.e. the
biggest local topic shift — rather than always cutting at the max size.
That keeps chunk boundaries semantically meaningful while still bounding
chunk size to something retrieval-friendly (an unbounded semantic splitter
can produce pathologically tiny or huge chunks on uniform or very choppy
text).

Note: sentence embeddings computed here are for boundary-detection only —
transient, in-memory, never stored or sent anywhere. They're computed on
raw (pre-PII-redaction) sentence text, which is fine because this model
runs 100% locally (no network call); redaction still happens per-chunk
before anything is indexed, shown to a user, or sent to the LLM.
"""

import re
from typing import List, TypedDict

import numpy as np

from . import embeddings

# Split after a sentence-ending punctuation mark + whitespace, but only
# where the next sentence looks like it's actually starting (capital
# letter, digit, quote, or opening paren). This is a lightweight regex
# heuristic, not an NLP sentence tokenizer — it has no extra dependency or
# model download, but it can mis-split on abbreviations like "Dr." or
# "e.g." (see ARCHITECTURE.md > Known Limitations).
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(“])')


class Chunk(TypedDict):
    index: int
    text: str
    sentence_start: int
    sentence_end: int


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using a lightweight regex heuristic."""
    collapsed = re.sub(r"\s+", " ", text.strip())
    if not collapsed:
        return []
    sentences = _SENTENCE_BOUNDARY_RE.split(collapsed)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    min_sentences: int = 10,
    max_sentences: int = 20,
    overlap: int = 2,
) -> List[Chunk]:
    """Semantic chunking, bounded to [min_sentences, max_sentences] per chunk.

    Each chunk always has at least min_sentences sentences (except possibly
    the last one) and at most max_sentences. Within that band, the exact
    cut point is chosen at the local minimum of consecutive-sentence
    embedding similarity — the most likely real topic boundary — rather
    than mechanically always at max_sentences.
    """
    if overlap >= min_sentences:
        raise ValueError("overlap must be smaller than min_sentences")
    if min_sentences > max_sentences:
        raise ValueError("min_sentences must be <= max_sentences")

    sentences = split_sentences(text)
    n = len(sentences)
    if n == 0:
        return []
    if n <= max_sentences:
        return [{"index": 0, "text": " ".join(sentences), "sentence_start": 0, "sentence_end": n}]

    # Embed every sentence once and score consecutive-sentence similarity.
    # Vectors are pre-normalized (see embeddings.py), so a dot product is
    # cosine similarity. sims[i] = similarity(sentence i, sentence i+1); a
    # low value there means i/i+1 is a good place to cut.
    vectors = embeddings.embed_texts(sentences)
    sims = np.sum(vectors[:-1] * vectors[1:], axis=1)

    chunks: List[Chunk] = []
    index = 0
    start = 0
    while start < n:
        remaining = n - start
        if remaining <= max_sentences:
            end = n
        else:
            # sims index i corresponds to a cut giving chunk size (i+1-start).
            # Search the window where that size falls in [min_sentences, max_sentences].
            window_lo = start + min_sentences - 1
            window_hi = start + max_sentences  # slice end, exclusive
            window = sims[window_lo:window_hi]
            if len(window) == 0:
                end = min(start + max_sentences, n)
            else:
                end = window_lo + int(np.argmin(window)) + 1

        chunks.append(
            {
                "index": index,
                "text": " ".join(sentences[start:end]),
                "sentence_start": start,
                "sentence_end": end,
            }
        )
        index += 1
        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks
