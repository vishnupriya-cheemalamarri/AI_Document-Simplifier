"""
Text extraction from uploaded documents.

PDF extraction uses PyMuPDF (import name `fitz`). No OCR support: a
scanned/image-only PDF has no text layer, so extraction returns an empty
string for it (see docs/ARCHITECTURE.md > Known Limitations).
"""

import fitz  # PyMuPDF


def extract_text(file_path: str, content_type: str) -> str:
    """Extract raw text from a PDF or plain-text file."""
    if content_type == "pdf":
        parts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                parts.append(page.get_text())
        return "\n".join(parts).strip()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()
