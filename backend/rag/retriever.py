"""
Similarity search + relevance filtering. Chunks below the configured
minimum similarity are dropped so weakly-related content isn't forced
into the prompt (this is what lets the assistant honestly say "the
documents don't contain that" instead of guessing from noise).
"""
from __future__ import annotations

from backend.llm.base import LLMProvider
from backend.rag.vector_store import VectorStore


async def retrieve_relevant_chunks(
    provider: LLMProvider,
    store: VectorStore,
    query: str,
    top_k: int,
    min_similarity: float,
) -> list[dict]:
    if store.count() == 0:
        return []
    [query_embedding] = await provider.embed([query])
    raw = store.query(query_embedding, top_k=top_k)

    chunks: list[dict] = []
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for _id, doc, meta, dist in zip(ids, docs, metas, distances):
        similarity = 1 - dist  # cosine distance -> similarity
        if similarity < min_similarity:
            continue
        chunks.append(
            {
                "chunk_id": meta.get("chunk_id", _id),
                "document_id": meta.get("document_id"),
                "document_name": meta.get("document_name"),
                "page": meta.get("page"),
                "source": meta.get("source"),
                "text": doc,
                "similarity": round(similarity, 4),
            }
        )
    return chunks
