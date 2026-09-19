"""
Groq provider (OpenAI-compatible API), used for the agent loop only via `step()`.
Embeddings and structured single-pass generation stay on Gemini.
"""
from __future__ import annotations

import json
import logging
import uuid

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI

from backend.assistant.schemas import ChatResponse
from backend.config import Settings
from backend.llm.base import AgentStep, LLMError, LLMProvider

logger = logging.getLogger("ai_assistant.llm.groq")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, settings: Settings):
        if not settings.groq_api_key:
            raise LLMError("GROQ_API_KEY is not configured", retryable=False)
        self._settings = settings
        self._client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=settings.groq_api_key)

    @staticmethod
    def _to_messages(system_prompt: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system_prompt}]
        last_id = ""
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["text"]})
            elif m["role"] == "model":
                last_id = (m.get("raw") or {}).get("id") or f"call_{uuid.uuid4().hex[:8]}"
                out.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": last_id, "type": "function",
                        "function": {"name": m["call"]["name"], "arguments": json.dumps(m["call"]["args"])},
                    }],
                })
            else:
                out.append({"role": "tool", "tool_call_id": last_id, "content": json.dumps(m["result"], default=str)})
        return out

    async def step(self, system_prompt: str, messages: list[dict], tools: list[dict]) -> AgentStep:
        try:
            resp = await self._client.chat.completions.create(
                model=self._settings.groq_model,
                messages=self._to_messages(system_prompt, messages),
                tools=[{"type": "function", "function": {
                    "name": t["name"], "description": t.get("description", ""), "parameters": t["parameters"],
                }} for t in tools],
                tool_choice="auto",  # "required" makes Groq 400 when the model wants to answer in prose
                temperature=self._settings.temperature,
                parallel_tool_calls=False,
                extra_body={"reasoning_effort": "low"} if "gpt-oss" in self._settings.groq_model else None,
            )
        except APIStatusError as exc:
            # Some models sometimes emit a malformed tool call (400 tool_use_failed); a resample usually fixes it.
            transient = exc.status_code in {408, 500, 502, 503, 504} or "tool_use_failed" in str(exc)
            raise LLMError(f"Groq error: {exc}", retryable=transient) from exc
        except (APIConnectionError, APITimeoutError) as exc:
            raise LLMError(f"Groq connection error: {exc}", retryable=True) from exc
        except APIError as exc:
            raise LLMError(f"Groq error: {exc}", retryable=False) from exc

        msg = resp.choices[0].message
        call = raw = None
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            call, raw = {"name": tc.function.name, "args": args if isinstance(args, dict) else {}}, {"id": tc.id}
        usage = resp.usage
        return AgentStep(
            call=call, text=msg.content or "", raw=raw,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    async def generate_structured(self, system_prompt, user_prompt, tools=None, tool_executor=None) -> ChatResponse:
        raise LLMError("GroqProvider is only used for agent steps", retryable=False)

    async def embed(self, texts):
        raise LLMError("Groq has no embedding model; embeddings use Gemini", retryable=False)

    async def health_check(self) -> bool:
        return bool(self._settings.groq_api_key)
