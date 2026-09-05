"""
Gemini provider implementation using Google's current unified SDK,
`google-genai` (pip install google-genai / `from google import genai`).

This supersedes the older `google-generativeai` package. We use the
synchronous-style `client.models.generate_content(...)` surface (via the
async client) rather than the newer `interactions` API, because it gives
tighter control over combining structured output (response_schema) with a
manual tool-call loop, which is what the orchestrator's retry/fallback
logic needs.

Errors are classified so only transient failures (timeouts, 5xx, 429) are
marked retryable; permanent failures (invalid API key, invalid request)
are not, so the retry decorator and fallback chain behave correctly.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from backend.assistant.schemas import ChatResponse
from backend.config import Settings
from backend.llm.base import LLMError, LLMProvider

logger = logging.getLogger("ai_assistant.llm.gemini")

# Status codes that indicate a transient, retry-worthy failure.
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _to_function_declaration(tool_schema: dict) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=tool_schema["name"],
        description=tool_schema.get("description", ""),
        parameters=tool_schema.get("parameters", {"type": "object", "properties": {}}),
    )


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not configured", retryable=False)
        self._settings = settings
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def _gen_config(self, response_schema: type | None = None, tools=None) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = dict(
            temperature=self._settings.temperature,
            top_p=self._settings.top_p,
            max_output_tokens=self._settings.max_output_tokens,
        )
        if response_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = response_schema
        if tools:
            kwargs["tools"] = [types.Tool(function_declarations=[_to_function_declaration(t) for t in tools])]
        return types.GenerateContentConfig(**kwargs)

    def _classify(self, exc: Exception) -> LLMError:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        retryable = status in _RETRYABLE_STATUS_CODES if status else True
        # Invalid API key / bad request errors from the SDK are typically 400/401/403.
        if status in {400, 401, 403}:
            retryable = False
        return LLMError(f"Gemini error: {exc}", retryable=retryable)

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict] | None = None,
        tool_executor: Callable[[str, dict], Any] | None = None,
    ) -> ChatResponse:
        model = self._settings.gemini_model
        tool_used: str | None = None
        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part(text=user_prompt)])
        ]

        try:
            # Step 1: allow the model to request a tool call (plain-text turn,
            # since structured-output + function-calling together is unreliable
            # across SDK versions).
            if tools:
                first = await self._client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self._settings.temperature,
                        tools=[types.Tool(function_declarations=[_to_function_declaration(t) for t in tools])],
                    ),
                )
                call = self._extract_function_call(first)
                if call and tool_executor:
                    tool_used = call["name"]
                    result = tool_executor(call["name"], call["args"])
                    contents.append(types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(
                        name=call["name"], args=call["args"]))]))
                    contents.append(types.Content(role="user", parts=[types.Part(
                        function_response=types.FunctionResponse(name=call["name"], response={"result": result})
                    )]))

            # Step 2: final structured answer.
            final = await self._client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=self._gen_config(response_schema=ChatResponse),
            )
            data = json.loads(final.text)
            data["provider"] = "gemini"
            if tool_used:
                data["tool_used"] = tool_used
            return ChatResponse.model_validate(data)

        except genai_errors.APIError as exc:
            raise self._classify(exc) from exc
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            # Malformed structured output - not a transient network issue.
            raise LLMError(f"Gemini returned invalid structured output: {exc}", retryable=False) from exc

    @staticmethod
    def _extract_function_call(response) -> dict | None:
        try:
            for part in response.candidates[0].content.parts:
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    return {"name": fc.name, "args": dict(fc.args)}
        except (IndexError, AttributeError):
            return None
        return None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            result = await self._client.aio.models.embed_content(
                model=self._settings.gemini_embedding_model,
                contents=texts,
            )
            return [e.values for e in result.embeddings]
        except genai_errors.APIError as exc:
            raise self._classify(exc) from exc

    async def health_check(self) -> bool:
        try:
            await self._client.aio.models.generate_content(
                model=self._settings.gemini_model,
                contents="ping",
                config=types.GenerateContentConfig(max_output_tokens=5),
            )
            return True
        except Exception:  # noqa: BLE001 - health check must never raise
            logger.warning("Gemini health check failed", exc_info=True)
            return False
