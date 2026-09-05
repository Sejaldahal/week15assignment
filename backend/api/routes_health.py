from __future__ import annotations

from fastapi import APIRouter, Request

from backend.assistant.schemas import HealthStatus

router = APIRouter()


@router.get("/health", response_model=HealthStatus)
async def health(request: Request) -> HealthStatus:
    settings = request.app.state.settings
    llm_chain = request.app.state.llm_chain
    store = request.app.state.vector_store
    cache = request.app.state.cache

    provider_health = await llm_chain.health()
    vector_ok = store.health_check()

    redis_ok: bool | None = None
    if settings.redis_url:
        redis_ok = await cache.ping() if hasattr(cache, "ping") else None

    overall = provider_health.get("gemini", False) and vector_ok

    return HealthStatus(
        gemini=provider_health.get("gemini", False),
        vector_db=vector_ok,
        redis=redis_ok,
        vllm=provider_health.get("vllm"),
        overall=overall,
    )
