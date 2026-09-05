"""Lightweight in-process cache used when Redis is not configured/available."""
from __future__ import annotations

import time

from backend.cache.base import Cache


class MemoryCache(Cache):
    def __init__(self):
        self._store: dict[str, tuple[str, float]] = {}

    async def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if not entry:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._store[key] = (value, time.time() + ttl_seconds)
