import io
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def test_upload_rejects_unsupported_extension():
    with patch("backend.llm.gemini.genai.Client"):
        from backend.main import app

        with TestClient(app) as client:
            files = {"file": ("malware.exe", io.BytesIO(b"not a real doc"))}
            resp = client.post("/documents/upload", files=files)
            assert resp.status_code == 400


def test_upload_accepts_txt_and_ingests():
    with patch("backend.llm.gemini.genai.Client"), \
         patch("backend.api.routes_documents.embed_texts", new=AsyncMock(return_value=[[0.1] * 8])):
        from backend.main import app

        with TestClient(app) as client:
            files = {"file": ("notes.txt", io.BytesIO(b"This is a short test document about llamas."))}
            resp = client.post("/documents/upload", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ingested"
            assert data["chunk_count"] >= 1
