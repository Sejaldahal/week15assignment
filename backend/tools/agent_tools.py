"""
Agent-only tools. `search_documents` wraps the W15 retriever; the other
three are control tools the loop itself interprets:
  record_note  -> structured external notes (context engineering)
  ask_user     -> stop and ask for clarification
  finish       -> the only normal way to end the loop
"""
from __future__ import annotations

from backend.rag.retriever import retrieve_relevant_chunks

SEARCH_SCHEMA = {
    "name": "search_documents",
    "description": (
        "Search the uploaded documents. Returns up to a few short snippets with "
        "chunk_id, document name and similarity. Rephrase the query and call again "
        "if the snippets do not settle the question."
    ),
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to look for."}},
        "required": ["query"],
    },
}

NOTE_SCHEMA = {
    "name": "record_note",
    "description": (
        "Save one fact to your notes. Old tool results are cleared from context, "
        "so anything you will need later MUST be noted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "claim": {"type": "string", "description": "The fact, with numbers/dates."},
            "source": {"type": "string", "description": "chunk_id, tool name, or 'user'."},
            "verified": {
                "type": "boolean",
                "description": "True only if a tool result or document states it directly.",
            },
        },
        "required": ["claim", "source", "verified"],
    },
}

ASK_SCHEMA = {
    "name": "ask_user",
    "description": "Stop and ask the user one clarifying question when required information is missing.",
    "parameters": {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    },
}

FINISH_SCHEMA = {
    "name": "finish",
    "description": "Give the final answer. Confidence must reflect how well verified notes support it.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number", "description": "0.0 - 1.0"},
        },
        "required": ["answer", "confidence"],
    },
}

AGENT_TOOL_SCHEMAS = [SEARCH_SCHEMA, NOTE_SCHEMA, ASK_SCHEMA, FINISH_SCHEMA]
SNIPPET_CHARS = 500  # cap per retrieved chunk (context engineering: capped retrieval)
SEARCH_MAX_CHUNKS = 3


async def search_documents(provider, store, settings, args: dict) -> tuple[dict, list[dict]]:
    """Returns (compact result shown to the model, full chunk metadata for sources)."""
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return {"error": "search_documents requires a non-empty 'query' string"}, []
    chunks = await retrieve_relevant_chunks(
        provider, store, query,
        top_k=SEARCH_MAX_CHUNKS,
        min_similarity=settings.retrieval_min_similarity,
    )
    result = {
        "query": query,
        "n_results": len(chunks),
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "document": c["document_name"],
                "similarity": c["similarity"],
                "text": c["text"][:SNIPPET_CHARS],
            }
            for c in chunks
        ],
    }
    return result, chunks
