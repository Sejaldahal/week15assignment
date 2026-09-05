"""
FastAPI application entrypoint. Wires up configuration, logging,
middleware, the LLM fallback chain, the vector store, the cache, and the
orchestrator, then exposes the API routes.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import routes_chat, routes_config, routes_documents, routes_health
from backend.assistant.orchestrator import Orchestrator
from backend.cache import build_cache
from backend.config import get_settings
from backend.llm.factory import build_fallback_chain
from backend.middleware.rate_limit import RateLimitMiddleware
from backend.middleware.request_id import RequestIDMiddleware
from backend.rag.vector_store import VectorStore
from backend.utils.logging import configure_logging

logger = logging.getLogger("ai_assistant.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting ai-assistant (environment=%s)", settings.environment)

    app.state.settings = settings
    app.state.llm_chain = build_fallback_chain(settings)
    app.state.vector_store = VectorStore(settings.chroma_persist_dir, settings.chroma_collection_name)
    app.state.cache = build_cache(settings)
    app.state.orchestrator = Orchestrator(
        settings=settings,
        llm_chain=app.state.llm_chain,
        vector_store=app.state.vector_store,
        cache=app.state.cache,
    )
    yield
    logger.info("Shutting down ai-assistant")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="AI Assistant", version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    app.add_middleware(RequestIDMiddleware)

    app.include_router(routes_health.router, tags=["health"])
    app.include_router(routes_config.router, tags=["config"])
    app.include_router(routes_chat.router, tags=["chat"])
    app.include_router(routes_documents.router, tags=["documents"])

    return app


app = create_app()
