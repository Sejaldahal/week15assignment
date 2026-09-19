

"""
Gemini provider implementation using Google's current unified SDK,
`google-genai`.

This provider supports:
- Structured JSON output
- Function/tool calling
- Gemini embeddings
- Retry/error classification
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Callable

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from backend.assistant.schemas import ChatResponse
from backend.config import Settings
from backend.llm.base import AgentStep, LLMError, LLMProvider

logger = logging.getLogger("ai_assistant.llm.gemini")

# Status codes that indicate a transient, retry-worthy failure.
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _normalize_schema(schema: dict) -> dict:
    """
    Convert a normal JSON Schema into the format expected by
    Google's Gemini SDK for function declarations.
    """
    if not isinstance(schema, dict):
        return schema

    normalized = dict(schema)

    # Gemini expects uppercase schema types.
    if isinstance(normalized.get("type"), str):
        normalized["type"] = normalized["type"].upper()

    # Recursively normalize object properties.
    if isinstance(normalized.get("properties"), dict):
        normalized["properties"] = {
            key: _normalize_schema(value)
            for key, value in normalized["properties"].items()
        }

    # Recursively normalize array item schemas.
    if isinstance(normalized.get("items"), dict):
        normalized["items"] = _normalize_schema(normalized["items"])

    return normalized


def _to_function_declaration(tool_schema: dict) -> types.FunctionDeclaration:
    """
    Convert the application's JSON-schema tool definition into
    Gemini's FunctionDeclaration format.
    """
    parameters = tool_schema.get(
        "parameters",
        {"type": "OBJECT", "properties": {}},
    )

    parameters = _normalize_schema(parameters)

    return types.FunctionDeclaration(
        name=tool_schema["name"],
        description=tool_schema.get("description", ""),
        parameters=parameters,
    )


def _build_gemini_response_schema(response_schema: type) -> dict:
    """
    Convert a Pydantic JSON schema into a Gemini-compatible schema.

    Pydantic may generate:
      - $defs
      - $ref
      - anyOf
      - NULL types

    Gemini's Schema type does not accept those forms directly, so
    references are resolved and nullable fields are simplified.
    """

    raw_schema = response_schema.model_json_schema()

    # Copy so we never modify Pydantic's original schema.
    schema = copy.deepcopy(raw_schema)

    definitions = schema.pop("$defs", {})

    def resolve_reference(ref: str) -> dict:
        """
        Resolve references such as:
        #/$defs/SourceChunk
        """
        prefix = "#/$defs/"

        if ref.startswith(prefix):
            name = ref[len(prefix):]

            if name in definitions:
                return copy.deepcopy(definitions[name])

        return {}

    def clean(obj: Any) -> Any:
        if isinstance(obj, list):
            return [clean(item) for item in obj]

        if not isinstance(obj, dict):
            return obj

        # Resolve $ref before doing anything else.
        if "$ref" in obj:
            resolved = resolve_reference(obj["$ref"])

            # Preserve any additional fields alongside the reference.
            extra = {
                key: value
                for key, value in obj.items()
                if key != "$ref"
            }

            resolved.update(extra)
            return clean(resolved)

        result = {}

        # Handle nullable Pydantic fields such as Optional[str].
        if "anyOf" in obj:
            variants = obj["anyOf"]

            non_null_variants = [
                variant
                for variant in variants
                if not (
                    isinstance(variant, dict)
                    and variant.get("type") == "null"
                )
            ]

            # Example:
            #
            # anyOf:
            #   - type: string
            #   - type: null
            #
            # becomes:
            #
            # type: STRING
            # nullable: true
            if len(non_null_variants) == 1:
                base = clean(non_null_variants[0])

                if isinstance(base, dict):
                    base["nullable"] = True

                # Preserve description if present.
                if "description" in obj:
                    base["description"] = obj["description"]

                return base

            # If there are multiple actual variants, keep the
            # supported variants rather than passing NULL to Gemini.
            obj = dict(obj)
            obj["anyOf"] = non_null_variants

        for key, value in obj.items():
            # Gemini does not accept these JSON Schema keywords.
            if key in {"$defs", "$ref", "title", "default"}:
                continue

            # Gemini schema types are uppercase.
            if key == "type" and isinstance(value, str):
                if value.lower() == "null":
                    continue

                result[key] = value.upper()
                continue

            result[key] = clean(value)

        return result

    return clean(schema)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise LLMError(
                "GEMINI_API_KEY is not configured",
                retryable=False,
            )

        self._settings = settings
        self._client = genai.Client(
            api_key=settings.gemini_api_key
        )

    def _gen_config(
        self,
        response_schema: type | None = None,
        tools=None,
    ) -> types.GenerateContentConfig:

        kwargs: dict[str, Any] = dict(
            temperature=self._settings.temperature,
            top_p=self._settings.top_p,
            max_output_tokens=self._settings.max_output_tokens,
        )

        if response_schema is not None:
            kwargs["response_mime_type"] = "application/json"

            # Convert Pydantic schema into Gemini-compatible schema.
            kwargs["response_schema"] = _build_gemini_response_schema(
                response_schema
            )

        if tools:
            kwargs["tools"] = [
                types.Tool(
                    function_declarations=[
                        _to_function_declaration(tool)
                        for tool in tools
                    ]
                )
            ]

        return types.GenerateContentConfig(**kwargs)

    def _classify(self, exc: Exception) -> LLMError:
        status = getattr(exc, "status_code", None) or getattr(
            exc,
            "code",
            None,
        )

        retryable = (
            status in _RETRYABLE_STATUS_CODES
            if status
            else True
        )
        # Do not immediately retry rate-limit errors.
        if status == 429:
          retryable = False


        # Invalid API key / bad request errors are not retryable.
        if status in {400, 401, 403}:
            retryable = False

        return LLMError(
            f"Gemini error: {exc}",
            retryable=retryable,
        )

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
            types.Content(
                role="user",
                parts=[
                    types.Part(text=user_prompt)
                ],
            )
        ]

        try:
            # Step 1: allow the model to request a tool call.
            if tools:
                first = await self._client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self._settings.temperature,
                        tools=[
                            types.Tool(
                                function_declarations=[
                                    _to_function_declaration(tool)
                                    for tool in tools
                                ]
                            )
                        ],
                    ),
                )

                call = self._extract_function_call(first)

                if call and tool_executor:
                    tool_used = call["name"]

                    result = tool_executor(
                        call["name"],
                        call["args"],
                    )

                    contents.append(
                        types.Content(
                            role="model",
                            parts=[
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=call["name"],
                                        args=call["args"],
                                    )
                                )
                            ],
                        )
                    )

                    contents.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(
                                    function_response=types.FunctionResponse(
                                        name=call["name"],
                                        response={
                                            "result": result
                                        },
                                    )
                                )
                            ],
                        )
                    )

            # Step 2: final structured answer.
            final = await self._client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=self._gen_config(
                    response_schema=ChatResponse
                ),
            )

            data = json.loads(final.text)

            data["provider"] = "gemini"

            if tool_used:
                data["tool_used"] = tool_used

            return ChatResponse.model_validate(data)

        except genai_errors.APIError as exc:
            raise self._classify(exc) from exc

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:

            raise LLMError(
                f"Gemini returned invalid structured output: {exc}",
                retryable=False,
            ) from exc

    @staticmethod
    def _to_contents(messages: list[dict]) -> list[types.Content]:
        contents: list[types.Content] = []
        for m in messages:
            if m["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m["text"])]))
            elif m["role"] == "model":
                # Echo the original turn back untouched when we have it.
                contents.append(
                    m.get("raw")
                    or types.Content(
                        role="model",
                        parts=[types.Part(function_call=types.FunctionCall(**m["call"]))],
                    )
                )
            else:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=m["name"], response={"result": m["result"]}
                                )
                            )
                        ],
                    )
                )
        return contents

    async def step(self, system_prompt: str, messages: list[dict], tools: list[dict]) -> AgentStep:
        try:
            resp = await self._client.aio.models.generate_content(
                model=self._settings.gemini_model,
                contents=self._to_contents(messages),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self._settings.temperature,
                    tools=[types.Tool(function_declarations=[_to_function_declaration(t) for t in tools])],
                    # Every turn must be a tool call, so the loop never has to parse prose.
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="ANY")
                    ),
                ),
            )
        except genai_errors.APIError as exc:
            raise self._classify(exc) from exc

        usage = getattr(resp, "usage_metadata", None)
        try:
            raw = resp.candidates[0].content
            text = "".join(p.text for p in raw.parts if getattr(p, "text", None))
        except (IndexError, AttributeError, TypeError):
            raw, text = None, ""
        return AgentStep(
            call=self._extract_function_call(resp),
            text=text,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            raw=raw,
        )

    @staticmethod
    def _extract_function_call(response) -> dict | None:
        try:
            for part in response.candidates[0].content.parts:

                if getattr(part, "function_call", None):
                    fc = part.function_call

                    return {
                        "name": fc.name,
                        "args": dict(fc.args),
                    }

        except (IndexError, AttributeError):
            return None

        return None

    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        try:
            result = await self._client.aio.models.embed_content(
                model=self._settings.gemini_embedding_model,
                contents=texts,
            )

            return [
                embedding.values
                for embedding in result.embeddings
            ]

        except genai_errors.APIError as exc:
            raise self._classify(exc) from exc

    async def health_check(self) -> bool:
        try:
            await self._client.aio.models.generate_content(
                model=self._settings.gemini_model,
                contents="ping",
                config=types.GenerateContentConfig(
                    max_output_tokens=5
                ),
            )

            return True

        except Exception:
            logger.warning(
                "Gemini health check failed",
                exc_info=True,
            )

            return False