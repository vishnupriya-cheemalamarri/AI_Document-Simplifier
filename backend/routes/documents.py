"""
Ingestion/indexing pipeline route (see docs/ARCHITECTURE.md > System
Architecture > 1. Ingestion / Indexing Pipeline):

Upload PDF/txt -> extract + chunk -> PII redaction -> embed (local model)
-> index (FAISS + BM25) + store metadata (TinyDB) -> simplify each chunk
and extract key points.

Runs synchronously within the upload request (hackathon scope) — for a
document with many chunks this means several sequential LLM calls, which
could be parallelized in a non-prototype build.
"""

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from .. import config, db
from ..llm import simplify as llm_simplify
from ..pipeline import chunking, embeddings, extraction, indexing, pii_redaction

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, f"Unsupported file type {ext!r}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    doc_id = str(uuid.uuid4())
    dest_path = config.UPLOAD_DIR / f"{doc_id}{ext}"
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    doc_record = {
        "doc_id": doc_id,
        "filename": file.filename,
        "content_type": "pdf" if ext == ".pdf" else "txt",
        "upload_time": datetime.now(timezone.utc).isoformat(),
        "file_path": str(dest_path),
        "num_chunks": 0,
        "word_count": 0,
        "status": "processing",
        "error": None,
    }
    db.documents_table.insert(doc_record)

    try:
        raw_text = extraction.extract_text(str(dest_path), doc_record["content_type"])
        if not raw_text.strip():
            raise ValueError(
                "No extractable text found. Scanned/image-only PDFs are not "
                "supported (no OCR) — see docs/ARCHITECTURE.md > Known Limitations."
            )

        chunks = chunking.chunk_text(
            raw_text, config.CHUNK_MIN_SENTENCES, config.CHUNK_MAX_SENTENCES, config.CHUNK_OVERLAP_SENTENCES
        )
        if not chunks:
            raise ValueError("Document produced no chunks.")

        chunk_ids, chunk_texts, chunk_rows = [], [], []
        for c in chunks:
            redacted_text, was_redacted = pii_redaction.redact_pii(c["text"])
            chunk_id = f"{doc_id}::{c['index']}"
            chunk_ids.append(chunk_id)
            chunk_texts.append(redacted_text)
            chunk_rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_index": c["index"],
                    "text": redacted_text,
                    "sentence_start": c["sentence_start"],
                    "sentence_end": c["sentence_end"],
                    "pii_redacted": was_redacted,
                }
            )
        db.chunks_table.insert_multiple(chunk_rows)

        vectors = embeddings.embed_texts(chunk_texts)
        indexing.build_index(doc_id, chunk_ids, chunk_texts, vectors)

        key_point_rows = []
        for i in range(0, len(chunk_rows), config.SIMPLIFY_BATCH_SIZE):
            batch = chunk_rows[i : i + config.SIMPLIFY_BATCH_SIZE]
            batch_input = [{"chunk_id": row["chunk_id"], "text": row["text"]} for row in batch]
            results = llm_simplify.simplify_chunks_batch(batch_input)
            for row in batch:
                result = results[row["chunk_id"]]
                key_point_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "chunk_id": row["chunk_id"],
                        "chunk_index": row["chunk_index"],
                        "simplified_text": result["simplified_text"],
                        "key_points": result["key_points"],
                    }
                )
        db.key_points_table.insert_multiple(key_point_rows)

        db.documents_table.update(
            {"num_chunks": len(chunks), "word_count": len(raw_text.split()), "status": "ready"},
            db.DocQuery.doc_id == doc_id,
        )
    except Exception as exc:
        db.documents_table.update(
            {"status": "failed", "error": str(exc)}, db.DocQuery.doc_id == doc_id
        )
        raise HTTPException(500, f"Failed to process document: {exc}") from exc

    return get_document(doc_id)


@router.get("/{doc_id}")
def get_document(doc_id: str):
    doc = db.documents_table.get(db.DocQuery.doc_id == doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    sections = db.key_points_table.search(db.DocQuery.doc_id == doc_id)
    sections.sort(key=lambda s: s["chunk_index"])
    return {"document": doc, "sections": sections}


@router.get("")
def list_documents():
    return db.documents_table.all()
