"""
TinyDB-backed metadata/document store (see docs/ARCHITECTURE.md > Data Model).

Four tables:
- documents:  one record per uploaded document
- chunks:     one record per (PII-redacted) text chunk
- key_points: simplified text + key points, one record per chunk/"section"
- qa_history: one record per question asked in the chat box
"""

from tinydb import Query, TinyDB

from . import config

_db = TinyDB(config.DB_PATH)

documents_table = _db.table("documents")
chunks_table = _db.table("chunks")
key_points_table = _db.table("key_points")
qa_history_table = _db.table("qa_history")

DocQuery = Query()
