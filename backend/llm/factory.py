"""
Builds the provider fallback chain: Gemini (primary) -> vLLM (fallback).
The orchestrator depends only on `FallbackChain`, never on a concrete
provider class.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from backend.assistant.schemas import ChatResponse
from backend.config import Settings
from backend.llm.base import AgentStep, LLMError, LLMProvider
from backend.llm.gemini import GeminiProvider
from backend.llm.groq import GroqProvider
from backend.llm.vllm import VLLMProvider
from backend.utils.retry import retry_async

logger = logging.getLogger("ai_assistant.llm.factory")


class FallbackChain:
    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider | None,
        settings: Settings,
        agent_llm: LLMProvider | None = None,
    ):
        self._primary = primary
        self._agent_llm = agent_llm or primary  # provider that drives agent steps
        self._fallback = fallback
        self._settings = settings

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict] | None = None,
        tool_executor: Callable[[str, dict], Any] | None = None,
    ) -> ChatResponse:
        try:
            return await retry_async(
                lambda: self._primary.generate_structured(system_prompt, user_prompt, tools, tool_executor),
                max_retries=self._settings.max_retries,
                base_delay=self._settings.retry_base_delay_seconds,
                retryable_exc=LLMError,
            )
        except LLMError as exc:
            logger.warning("Primary provider (%s) failed after retries: %s", self._primary.name, exc)
            if self._fallback is None:
                raise
            return await self._fallback.generate_structured(system_prompt, user_prompt, tools, tool_executor)

    async def step(self, system_prompt: str, messages: list[dict], tools: list[dict]) -> AgentStep:
        # Agent turns go to the agent provider (Gemini or Groq); retry transient errors.
        return await retry_async(
            lambda: self._agent_llm.step(system_prompt, messages, tools),
            max_retries=self._settings.max_retries,
            base_delay=self._settings.retry_base_delay_seconds,
            retryable_exc=LLMError,
        )

    async def health(self) -> dict:
        primary_ok = await self._primary.health_check()
        fallback_ok = await self._fallback.health_check() if self._fallback else None
        return {"gemini": primary_ok, "vllm": fallback_ok}


def build_fallback_chain(settings: Settings) -> FallbackChain:
    primary = GeminiProvider(settings)
    fallback = VLLMProvider(settings) if settings.vllm_enabled else None
    agent_llm = GroqProvider(settings) if settings.agent_provider == "groq" else None
    return FallbackChain(primary, fallback, settings, agent_llm=agent_llm)
