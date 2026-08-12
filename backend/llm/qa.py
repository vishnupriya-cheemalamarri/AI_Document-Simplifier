"""
Grounded Q&A over retrieved chunks (see ARCHITECTURE.md > Guardrails >
Output validation). The LLM must cite the chunk_ids it used; the backend
(backend/guardrails/output_validation.py) verifies those citations before
the answer is shown to the user.

Traced via LangSmith's @traceable decorator (no-op unless LangSmith env
vars are set — see .env.example).
"""

import json
from typing import Dict, List

from langsmith import traceable

from . import client as llm_client
from .. import config

QA_SYSTEM_PROMPT = (
    "You are a grounded question-answering assistant. Answer the user's "
    "question ONLY using the provided context chunks. If the answer isn't "
    "contained in the context, say so explicitly instead of guessing. Every "
    "claim must be backed by a chunk you were given. Respond ONLY with JSON "
    'of the form: {"answer": "...", "citations": ["chunk_id", ...]} where '
    "citations lists the chunk_id values (exactly as given) of the chunks "
    "you actually used."
)


@traceable(name="answer_question")
def answer_question(question: str, context_chunks: List[Dict[str, str]]) -> dict:
    client = llm_client.get_client()
    context_block = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in context_chunks)
    user_content = f"Context:\n{context_block}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    data = json.loads(response.choices[0].message.content)
    return {
        "answer": data.get("answer", ""),
        "citations": data.get("citations", []),
    }
