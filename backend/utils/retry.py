"""
Generic async retry-with-exponential-backoff helper.

Only exceptions marked `retryable=True` (see llm/base.py:LLMError) trigger
a retry. Permanent errors (invalid API key, malformed request/response)
fail fast instead of wasting retries and latency.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")
logger = logging.getLogger("ai_assistant.retry")


async def retry_async(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 0.5,
    retryable_exc: type[Exception] = Exception,
) -> T:
    attempt = 0
    while True:
        try:
            return await func()
        except retryable_exc as exc:  # type: ignore[misc]
            is_retryable = getattr(exc, "retryable", True)
            attempt += 1
            if not is_retryable or attempt > max_retries:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, base_delay)
            logger.info(
                "Retryable error on attempt %s/%s, backing off %.2fs: %s",
                attempt, max_retries, delay, exc,
            )
            await asyncio.sleep(delay)
