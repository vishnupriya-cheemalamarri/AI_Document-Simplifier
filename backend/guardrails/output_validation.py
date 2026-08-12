"""
Output validation: the LLM returns answers with a "citations" field listing
chunk IDs used; the backend verifies every cited chunk ID was actually part
of the retrieved context before the answer is shown to the user
(see docs/ARCHITECTURE.md > Guardrails).
"""

from typing import List, Tuple


def validate_citations(citations: List[str], retrieved_chunk_ids: List[str]) -> Tuple[List[str], List[str]]:
    """Split citations into (valid, invalid) based on membership in the
    chunk IDs that were actually retrieved and given to the LLM as context."""
    retrieved_set = set(retrieved_chunk_ids)
    valid = [c for c in citations if c in retrieved_set]
    invalid = [c for c in citations if c not in retrieved_set]
    return valid, invalid
