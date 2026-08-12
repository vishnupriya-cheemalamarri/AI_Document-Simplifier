"""
Custom word-based chunker (see docs/ARCHITECTURE.md > Tech Stack).

150 words per chunk, 30 words overlap (20%) by default.
"""

from typing import List, TypedDict


class Chunk(TypedDict):
    index: int
    text: str
    word_start: int
    word_end: int


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> List[Chunk]:
    """Split text into overlapping word-count-based chunks."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: List[Chunk] = []
    index = 0
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(
            {
                "index": index,
                "text": " ".join(words[start:end]),
                "word_start": start,
                "word_end": end,
            }
        )
        index += 1
        if end == len(words):
            break
        start += step
    return chunks
