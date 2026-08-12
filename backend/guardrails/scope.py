"""
Scope validation: reject/flag queries whose top retrieval similarity score
falls below a threshold, indicating the question is likely off-topic for
the uploaded document (see ARCHITECTURE.md > Guardrails).
"""

from .. import config


def is_in_scope(top_similarity_score: float) -> bool:
    return top_similarity_score >= config.SCOPE_SIMILARITY_THRESHOLD
