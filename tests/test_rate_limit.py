from unittest.mock import patch

from fastapi.testclient import TestClient


def test_rate_limit_blocks_after_threshold(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")

    with patch("backend.llm.gemini.genai.Client"):
        from backend.config import get_settings
        get_settings.cache_clear()
        from backend.main import create_app

        app = create_app()
        with TestClient(app) as client:
            for _ in range(2):
                r = client.get("/config")
                assert r.status_code == 200
            r = client.get("/config")
            assert r.status_code == 429
