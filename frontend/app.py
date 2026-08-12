"""
Streamlit UI (see docs/ARCHITECTURE.md > Step-by-Step Workflow):

1. User uploads a PDF/txt document.
2. Backend processes it (extract, chunk, redact, embed, index, simplify).
3. Simplified sections + key points are shown.
4. User asks questions in a chat box; answers show cited source chunks.

Talks to the FastAPI backend over HTTP via `requests` (server-side calls
from the Streamlit process, not browser JS).
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")

st.set_page_config(page_title="AI Document Simplifier", page_icon="📄", layout="wide")
st.title("📄 AI Document Simplifier")
st.caption("Prodapt Hackathon — Group 17")

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "doc_data" not in st.session_state:
    st.session_state.doc_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    if uploaded_file is not None and st.button("Process document", type="primary"):
        with st.spinner("Extracting, chunking, redacting PII, embedding, indexing, simplifying..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                resp = requests.post(f"{BACKEND_URL}/documents/upload", files=files, timeout=600)
                resp.raise_for_status()
                data = resp.json()
                st.session_state.doc_id = data["document"]["doc_id"]
                st.session_state.doc_data = data
                st.session_state.chat_history = []
                st.success(f"Processed {uploaded_file.name}")
            except requests.RequestException as e:
                st.error(f"Upload failed: {e}")

if st.session_state.doc_id:
    doc_data = st.session_state.doc_data
    doc = doc_data["document"]

    st.subheader(f"📄 {doc['filename']}")
    st.caption(f"{doc['num_chunks']} chunks · {doc['word_count']} words · status: {doc['status']}")

    tab_simplified, tab_chat = st.tabs(["Simplified sections", "Ask a question"])

    with tab_simplified:
        for section in doc_data["sections"]:
            with st.expander(f"Section {section['chunk_index'] + 1}"):
                st.markdown("**Simplified:**")
                st.write(section["simplified_text"])
                if section["key_points"]:
                    st.markdown("**Key points:**")
                    for kp in section["key_points"]:
                        st.markdown(f"- {kp}")

    with tab_chat:
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.write(turn.get("answer"))
                if turn.get("blocked"):
                    st.warning(f"Blocked: {turn.get('block_reason')}")
                elif turn.get("citations"):
                    with st.expander("Cited source chunks"):
                        for c in turn["citations"]:
                            st.markdown(f"**{c['chunk_id']}**")
                            st.caption(c["text"])

        question = st.chat_input("Ask a question about this document...")
        if question:
            st.session_state.chat_history.append({"question": question})
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/documents/{st.session_state.doc_id}/ask",
                        json={"question": question},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                except requests.RequestException as e:
                    result = {
                        "answer": f"Request failed: {e}",
                        "citations": [],
                        "blocked": True,
                        "block_reason": "request_error",
                    }
                st.session_state.chat_history[-1].update(result)
            st.rerun()
else:
    st.info("Upload a PDF or TXT document from the sidebar to get started.")
