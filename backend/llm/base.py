"""
Common interface every LLM provider (Gemini, vLLM, ...) must implement.
The orchestrator only ever talks to this interface, never to a concrete
provider directly - this is what makes fallback and testing simple.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from backend.assistant.schemas import ChatResponse


class LLMError(Exception):
    """Base class for provider errors."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class AgentStep:
    """One model turn in the agent loop: either a tool call or plain text."""

    call: dict | None = None  # {"name": str, "args": dict}
    text: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None  # provider-specific turn object, echoed back in history


class LLMProvider(ABC):
    name: str = "base"

    async def step(self, system_prompt: str, messages: list[dict], tools: list[dict]) -> AgentStep:
        """
        One turn of the agent loop. `messages` is provider-neutral:
        {"role": "user", "text"} | {"role": "model", "call", "raw"} |
        {"role": "tool", "name", "result"}. Optional: only providers used
        for the agent need it.
        """
        raise NotImplementedError(f"{self.name} does not support agent steps")

    @abstractmethod
    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict] | None = None,
        tool_executor: Any | None = None,
    ) -> ChatResponse:
        """
        Generate a response constrained to the ChatResponse schema.

        If `tools` are provided and the model requests a tool call,
        implementations should invoke `tool_executor(name, args) -> Any`
        and feed the result back before producing the final structured
        answer, then set `tool_used` accordingly.
        """
        raise NotImplementedError

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        raise NotImplementedError
