"""
Pydantic schemas shared across the API layer, the orchestrator, and the
LLM providers. ChatResponse is the predictable structured contract the
assistant always returns, regardless of which provider answered.
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(..., min_length=1, max_length=128)
    use_rag: bool = Field(
        default=True,
        description="If true, retrieve relevant document chunks before answering.",
    )
    agent: bool = Field(
        default=False,
        description="If true, answer with the agentic verification loop instead of the single-pass pipeline.",
    )


class SourceChunk(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    page: int | None = None
    source: str
    similarity: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    tool_used: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    provider: Literal["gemini", "vllm", "none"] = "gemini"


class AgentChatResponse(ChatResponse):
    """ChatResponse plus agent-loop metadata. Kept as a subclass so the W15
    single-pass structured-output schema sent to Gemini is unchanged."""

    needs_clarification: bool = False
    trace: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None


class UploadResponse(BaseModel):
    document_id: str
    document_name: str
    status: Literal["queued", "ingested", "failed"]
    chunk_count: int | None = None
    message: str | None = None


class HealthStatus(BaseModel):
    api: bool = True
    gemini: bool
    vector_db: bool
    redis: bool | None
    vllm: bool | None
    overall: bool


class ToolCallResult(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result: Any
