"""
Thin wrapper around the LLM provider's embedding call, batching requests
so ingestion of a large document doesn't issue one API call per chunk.
"""
from __future__ import annotations

from backend.llm.base import LLMProvider

_BATCH_SIZE = 32


async def embed_texts(provider: LLMProvider, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        vectors.extend(await provider.embed(batch))
    return vectors
