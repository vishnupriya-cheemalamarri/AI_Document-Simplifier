"""
Query-time pipeline route (see ARCHITECTURE.md > System Architecture >
2. Query-Time Pipeline):

User question -> input guardrails (prompt injection, scope check) -> hybrid
retrieve (BM25 + FAISS, RRF fusion) -> LLM call (traced via LangSmith) ->
output validation (citation check) -> response to user.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db
from ..guardrails import output_validation, prompt_injection, scope
from ..llm import qa as llm_qa
from ..pipeline import indexing, retrieval

router = APIRouter(prefix="/documents", tags=["qa"])

BLOCK_MESSAGES = {
    "prompt_injection_detected": "This question looks like a prompt injection attempt and was blocked.",
    "out_of_scope": "This question doesn't appear to be answerable from the uploaded document.",
    "citation_validation_failed": "The model's answer couldn't be verified against the retrieved sources, so it was withheld.",
}


class AskRequest(BaseModel):
    question: str


@router.post("/{doc_id}/ask")
def ask_question(doc_id: str, req: AskRequest):
    doc = db.documents_table.get(db.DocQuery.doc_id == doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    if doc.get("status") != "ready":
        raise HTTPException(409, f"Document is not ready (status={doc.get('status')!r})")
    if not indexing.has_index(doc_id):
        raise HTTPException(
            409,
            "No index found for this document, in memory or on disk. It may not "
            "have finished processing, or its stored index files were removed — "
            "re-upload the document to rebuild it.",
        )

    question = req.question.strip()
    if not question:
        raise HTTPException(400, "question must not be empty")

    record = {
        "id": str(uuid.uuid4()),
        "doc_id": doc_id,
        "question": question,
        "answer": None,
        "citations": [],
        "blocked": False,
        "block_reason": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # --- Input guardrails ---
    injection_result = prompt_injection.check_prompt_injection(question)
    if injection_result["flagged"]:
        return _block(record, "prompt_injection_detected")

    retrieved, top_faiss_score = retrieval.hybrid_retrieve(doc_id, question)
    if not scope.is_in_scope(top_faiss_score):
        return _block(record, "out_of_scope")

    # --- LLM call (traced via LangSmith) ---
    try:
        result = llm_qa.answer_question(question, retrieved)
    except Exception as exc:
        raise HTTPException(500, f"LLM call failed: {exc}") from exc

    # --- Output validation ---
    retrieved_ids = [r["chunk_id"] for r in retrieved]
    valid_citations, invalid_citations = output_validation.validate_citations(
        result["citations"], retrieved_ids
    )
    if invalid_citations:
        return _block(record, "citation_validation_failed")

    record["answer"] = result["answer"]
    record["citations"] = valid_citations
    db.qa_history_table.insert(record)

    citation_chunks = [r for r in retrieved if r["chunk_id"] in valid_citations]
    return {
        "answer": result["answer"],
        "citations": citation_chunks,
        "blocked": False,
        "block_reason": None,
    }


@router.get("/{doc_id}/history")
def get_history(doc_id: str):
    return db.qa_history_table.search(db.DocQuery.doc_id == doc_id)


def _block(record: dict, reason: str) -> dict:
    record["blocked"] = True
    record["block_reason"] = reason
    db.qa_history_table.insert(record)
    return {
        "answer": BLOCK_MESSAGES[reason],
        "citations": [],
        "blocked": True,
        "block_reason": reason,
    }
