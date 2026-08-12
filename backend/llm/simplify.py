"""
Simplification + key-point extraction, batched across multiple chunks per
LLM call to cut OpenAI call volume on large documents (see
docs/ARCHITECTURE.md > Tech Stack). Each call handles up to
config.SIMPLIFY_BATCH_SIZE chunks at once instead of one call per chunk.

Traced via LangSmith's @traceable decorator. Tracing is a no-op unless
LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are set (see .env.example),
so this works whether or not LangSmith is configured.
"""

import json
import logging
from typing import Dict, List

from langsmith import traceable

from . import client as llm_client
from .. import config

logger = logging.getLogger(__name__)

SIMPLIFY_BATCH_SYSTEM_PROMPT = (
    "You are a document simplification assistant. You will be given several "
    "passages, each labeled with a chunk_id. For EACH passage, rewrite it in "
    "plain, simple language a non-expert can understand, and extract its key "
    "points. Respond ONLY with JSON of the form: "
    '{"results": [{"chunk_id": "...", "simplified_text": "...", '
    '"key_points": ["...", "..."]}, ...]} '
    "with exactly one entry per passage given, using the exact chunk_id "
    "provided for each — do not skip or merge any."
)


@traceable(name="simplify_chunks_batch")
def simplify_chunks_batch(chunks: List[Dict[str, str]]) -> Dict[str, dict]:
    """chunks: [{"chunk_id": ..., "text": ...}, ...], up to SIMPLIFY_BATCH_SIZE
    long. Returns {chunk_id: {"simplified_text": ..., "key_points": [...]}}."""
    client = llm_client.get_client()
    passages_block = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in chunks)

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SIMPLIFY_BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": passages_block},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    data = json.loads(response.choices[0].message.content)
    results = {
        item.get("chunk_id"): {
            "simplified_text": item.get("simplified_text", ""),
            "key_points": item.get("key_points", []),
        }
        for item in data.get("results", [])
        if item.get("chunk_id")
    }

    # Fail soft on individual chunks the model dropped from a batch response,
    # rather than losing the whole batch's results.
    missing = [c["chunk_id"] for c in chunks if c["chunk_id"] not in results]
    if missing:
        logger.warning(
            "simplify_chunks_batch: model omitted %d/%d chunk_id(s) from its "
            "response (%s) — filling with empty simplified_text/key_points.",
            len(missing),
            len(chunks),
            missing,
        )
        for chunk_id in missing:
            results[chunk_id] = {"simplified_text": "", "key_points": []}

    return results
