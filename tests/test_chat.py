from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.assistant.schemas import ChatResponse


def test_chat_returns_structured_response():
    with patch("backend.llm.gemini.genai.Client"):
        from backend.main import app

        with TestClient(app) as client:
            fake_response = ChatResponse(answer="42", provider="gemini", confidence=0.9)
            app.state.orchestrator.handle_chat = AsyncMock(return_value=fake_response)

            resp = client.post("/chat", json={"message": "what is 6*7", "session_id": "s1"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["answer"] == "42"
            assert data["provider"] == "gemini"


def test_chat_rejects_empty_message():
    with patch("backend.llm.gemini.genai.Client"):
        from backend.main import app

        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": "", "session_id": "s1"})
            assert resp.status_code == 422


def test_chat_handles_llm_failure_gracefully():
    from backend.llm.base import LLMError

    with patch("backend.llm.gemini.genai.Client"):
        from backend.main import app

        with TestClient(app) as client:
            app.state.orchestrator.handle_chat = AsyncMock(side_effect=LLMError("boom", retryable=False))
            resp = client.post("/chat", json={"message": "hi", "session_id": "s1"})
            assert resp.status_code == 502
