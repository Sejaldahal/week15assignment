from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from backend.assistant.schemas import AgentChatResponse, ChatRequest
from backend.llm.base import LLMError

logger = logging.getLogger("ai_assistant.api.chat")
router = APIRouter()


@router.post("/chat", response_model=AgentChatResponse)
async def chat(payload: ChatRequest, request: Request) -> AgentChatResponse:
    orchestrator = request.app.state.orchestrator
    try:
        return await orchestrator.handle_chat(
            message=payload.message, session_id=payload.session_id, use_rag=payload.use_rag,
            agent=payload.agent,
        )
    except LLMError as exc:
        logger.error("LLM error while handling chat", exc_info=True)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
