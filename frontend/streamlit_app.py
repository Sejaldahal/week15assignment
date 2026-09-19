"""
Streamlit UI: a chat window plus a document upload panel for RAG.
Talks to the FastAPI backend over HTTP only - no business logic lives here.
"""
from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload a document for RAG", type=["pdf", "txt", "docx"])
    if uploaded and st.button("Ingest document"):
        with st.spinner("Uploading and ingesting..."):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            try:
                resp = requests.post(f"{BACKEND_URL}/documents/upload", files=files, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                if data["status"] == "ingested":
                    st.success(f"Ingested '{data['document_name']}' into {data['chunk_count']} chunks.")
                else:
                    st.error(f"Ingestion failed: {data.get('message')}")
            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")

    st.divider()
    use_rag = st.checkbox("Use document context (RAG)", value=True)
    agent_mode = st.checkbox("Agent mode (W16 verification loop)", value=False)

    st.divider()
    st.header("System status")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        st.json(health)
    except requests.RequestException:
        st.warning("Backend unreachable.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "message": prompt,
                        "session_id": st.session_state.session_id,
                        "use_rag": use_rag,
                        "agent": agent_mode,
                    },
                    timeout=180 if agent_mode else 60,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                meta_parts = [f"provider: {data['provider']}", f"confidence: {data['confidence']:.2f}"]
                if data.get("tool_used"):
                    meta_parts.append(f"tool: {data['tool_used']}")
                if data.get("sources"):
                    meta_parts.append(f"{len(data['sources'])} source chunk(s)")
                if data.get("usage"):
                    meta_parts.append(f"{data['usage']['total_tokens']} tokens, {data['usage']['steps']} steps")
                if data.get("needs_clarification"):
                    meta_parts.append("asking you a question - reply to continue")
                meta = " · ".join(meta_parts)

                st.markdown(answer)
                st.caption(meta)
                if data.get("trace"):
                    with st.expander("Agent trace"):
                        for t in data["trace"]:
                            line = f"{t['step']}. **{t.get('tool') or 'model'}**"
                            if t.get("args"):
                                line += f" `{t['args']}`"
                            if t.get("error"):
                                line += f" - error: {str(t['error'])[:150]}"
                            elif t.get("note"):
                                line += f" - {t['note']}"
                            st.markdown(line)
                if data.get("sources"):
                    with st.expander("Sources"):
                        for s in data["sources"]:
                            st.write(f"- **{s['document_name']}** [{s['chunk_id']}]"
                                      f"{' p.' + str(s['page']) if s.get('page') else ''}"
                                      f" (similarity {s.get('similarity')})")

                st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
            except requests.RequestException as exc:
                error_msg = f"Request failed: {exc}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
