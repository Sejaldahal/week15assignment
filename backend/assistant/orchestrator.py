"""
Assistant Orchestrator: the single place that ties together caching, RAG
retrieval, tool calling, and the provider fallback chain. Provider-specific
logic never leaks in here - only the LLMProvider interface is used.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time

from backend.assistant.prompts import SYSTEM_PROMPT, build_context_block, build_user_turn
from backend.assistant.schemas import ChatResponse, SourceChunk
from backend.cache.base import Cache, build_cache_key
from backend.config import Settings
from backend.llm.factory import FallbackChain
from backend.rag.retriever import retrieve_relevant_chunks
from backend.rag.vector_store import VectorStore
from backend.tools.registry import TOOL_SCHEMAS, execute_tool

logger = logging.getLogger("ai_assistant.orchestrator")


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        llm_chain: FallbackChain,
        vector_store: VectorStore,
        cache: Cache,
    ):
        self._settings = settings
        self._llm = llm_chain
        self._store = vector_store
        self._cache = cache

    async def handle_chat(self, message: str, session_id: str, use_rag: bool) -> ChatResponse:
        retrieval_ms = 0.0
        chunks: list[dict] = []

        if use_rag:
            t0 = time.perf_counter()
            # NOTE: retrieval always uses the Gemini embedding model, even
            # if the vLLM fallback ends up generating the final answer -
            # embeddings are only implemented for Gemini (see llm/vllm.py).
            chunks = await retrieve_relevant_chunks(
                self._llm._primary,  # embeddings always via primary (Gemini)
                self._store,
                message,
                top_k=self._settings.retrieval_top_k,
                min_similarity=self._settings.retrieval_min_similarity,
            )
            retrieval_ms = (time.perf_counter() - t0) * 1000

        context_version = self._context_version(chunks)
        cache_key = build_cache_key(
            query=message,
            context_version=context_version,
            provider="auto",
            model=self._settings.gemini_model,
        )

        cached = await self._cache.get(cache_key)
        if cached:
            logger.info("Cache hit", extra={"event": "cache_hit"})
            return ChatResponse.model_validate(json.loads(cached))

        context_block = build_context_block(chunks)
        user_prompt = build_user_turn(message, context_block)

        t0 = time.perf_counter()
        response = await self._llm.generate_structured(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=TOOL_SCHEMAS,
            tool_executor=execute_tool,
        )
        llm_ms = (time.perf_counter() - t0) * 1000

        if chunks:
            response.sources = [
                SourceChunk(
                    document_id=c["document_id"],
                    document_name=c["document_name"],
                    chunk_id=c["chunk_id"],
                    page=c.get("page"),
                    source=c.get("source", c["document_name"]),
                    similarity=c.get("similarity"),
                )
                for c in chunks
            ]

        logger.info(
            "chat_completed",
            extra={
                "event": "chat_completed",
                "provider": response.provider,
                "latency_ms": round(llm_ms, 1),
            },
        )
        logger.info(
            "rag_retrieval",
            extra={"event": "rag_retrieval", "latency_ms": round(retrieval_ms, 1)},
        )

        await self._cache.set(cache_key, response.model_dump_json(), self._settings.cache_ttl_seconds)
        return response

    @staticmethod
    def _context_version(chunks: list[dict]) -> str:
        if not chunks:
            return "no-rag"
        ids = sorted(c["chunk_id"] for c in chunks)
        return hashlib.sha256(",".join(ids).encode()).hexdigest()[:16]
