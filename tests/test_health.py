from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def test_health_endpoint():
    with patch("backend.llm.gemini.genai.Client"):
        from backend.main import app

        with TestClient(app) as client:
            app.state.llm_chain.health = AsyncMock(return_value={"gemini": True, "vllm": None})
            resp = client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert "gemini" in body
            assert "vector_db" in body
