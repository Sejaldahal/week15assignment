"""
Fallback provider: a locally-served open-source instruct model (e.g.
Llama 3.x) served through vLLM's OpenAI-compatible HTTP API. Using the
OpenAI-compatible wire protocol means we can reuse `openai`'s async client
instead of writing a bespoke HTTP client.

vLLM already provides continuous batching, paged-attention KV caching,
and (optionally) quantization/tensor-parallelism at the serving layer, so
no additional optimization is implemented here - see README for details.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from openai import AsyncOpenAI, APIError, APIConnectionError, APITimeoutError

from backend.assistant.schemas import ChatResponse
from backend.config import Settings
from backend.llm.base import LLMError, LLMProvider

logger = logging.getLogger("ai_assistant.llm.vllm")

_STRUCTURED_INSTRUCTION = """
Respond with ONLY a single JSON object matching this schema, no extra text:
{"answer": string, "sources": [], "tool_used": string|null, "confidence": number, "provider": "vllm"}
"""


class VLLMProvider(LLMProvider):
    name = "vllm"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key="not-needed")

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict] | None = None,
        tool_executor: Callable[[str, dict], Any] | None = None,
    ) -> ChatResponse:
        messages = [
            {"role": "system", "content": system_prompt + _STRUCTURED_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ]
        try:
            resp = await self._client.chat.completions.create(
                model=self._settings.vllm_model,
                messages=messages,
                temperature=self._settings.temperature,
                top_p=self._settings.top_p,
                max_tokens=self._settings.max_output_tokens,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            data["provider"] = "vllm"
            return ChatResponse.model_validate(data)
        except (APIConnectionError, APITimeoutError) as exc:
            raise LLMError(f"vLLM unreachable: {exc}", retryable=True) from exc
        except APIError as exc:
            status = getattr(exc, "status_code", 500)
            raise LLMError(f"vLLM error: {exc}", retryable=status >= 500) from exc
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise LLMError(f"vLLM returned invalid structured output: {exc}", retryable=False) from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Embeddings are intentionally always served by Gemini (see README);
        # vLLM here is a chat-completion fallback only.
        raise LLMError("Embeddings are not supported by the vLLM fallback provider", retryable=False)

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:  # noqa: BLE001
            logger.info("vLLM health check failed (expected if not running)")
            return False
