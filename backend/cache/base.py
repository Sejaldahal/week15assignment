"""
Cache interface. The orchestrator depends only on this - swapping Redis
for the in-memory dev cache, or for another backend later, is transparent.
A cache outage must never break /chat; implementations should catch their
own backend errors and behave as a cache miss rather than raising.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod


class Cache(ABC):
    @abstractmethod
    async def get(self, key: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        raise NotImplementedError


def build_cache_key(*, query: str, context_version: str, provider: str, model: str) -> str:
    """
    Cache key considers the user query, the retrieved-context version
    (e.g. a hash of retrieved chunk IDs, or 'no-rag'), the provider, and
    the model - so a cache hit is only reused when all of these match.
    """
    raw = json.dumps(
        {"query": query.strip().lower(), "context": context_version, "provider": provider, "model": model},
        sort_keys=True,
    )
    return "chat:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
