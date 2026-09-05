"""
Simple fixed-window per-client rate limiter (default: 10 requests/minute).

This protects the backend from being overwhelmed and, just as importantly,
protects our external Gemini API quota from being burned by a single
misbehaving client. Uses in-memory state, which is sufficient for a
single-process deployment; for multi-instance deployments this should be
backed by Redis (the same cache connection already used elsewhere).
"""
from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int, window_seconds: int):
        super().__init__(app)
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in {"/health", "/docs", "/openapi.json"}:
            return await call_next(request)

        client_key = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self._window_seconds
        hits = [t for t in self._hits.get(client_key, []) if t > window_start]

        if len(hits) >= self._max_requests:
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limit_exceeded", "detail": "Too many requests, slow down."},
            )

        hits.append(now)
        self._hits[client_key] = hits
        return await call_next(request)
