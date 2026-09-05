"""Redis-backed cache. Falls back gracefully (returns None / no-ops) on
any connection error so a Redis outage degrades to 'no cache', never a
5xx for the caller."""
from __future__ import annotations

import logging

import redis.asyncio as redis

from backend.cache.base import Cache

logger = logging.getLogger("ai_assistant.cache.redis")


class RedisCache(Cache):
    def __init__(self, url: str):
        self._client = redis.from_url(url, decode_responses=True, socket_connect_timeout=2)

    async def get(self, key: str) -> str | None:
        try:
            return await self._client.get(key)
        except Exception:  # noqa: BLE001
            logger.warning("Redis GET failed, treating as cache miss", exc_info=True)
            return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        try:
            await self._client.set(key, value, ex=ttl_seconds)
        except Exception:  # noqa: BLE001
            logger.warning("Redis SET failed, skipping cache write", exc_info=True)

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except Exception:  # noqa: BLE001
            return False
