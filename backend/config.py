"""
Centralized application configuration.

All configuration is sourced from environment variables (see .env.example).
Nothing secret is hardcoded. This module is also the single source of truth
for GET /config (which must only expose non-secret values).

NOTE: every env-derived field uses `default_factory` so the environment is
re-read at Settings() instantiation time (important for tests that set env
vars via monkeypatch and then construct a fresh Settings/app).
"""
from __future__ import annotations

import os
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field

# Load variables from a .env file in the current working directory (or any
# parent directory) into the process environment. This is what makes
# `cp .env.example .env` actually take effect - without it, `os.getenv(...)`
# below would never see values that only live in the .env file.
load_dotenv()


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


class Settings(BaseModel):
    # --- App ---
    app_name: str = "ai-assistant"
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # --- Gemini (primary provider) ---
    gemini_api_key: str | None = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.6-flash"))
    gemini_embedding_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    )

    # Generation defaults. See README "Prompt Engineering" for rationale.
    temperature: float = Field(default_factory=lambda: _float("LLM_TEMPERATURE", 0.3))
    top_p: float = Field(default_factory=lambda: _float("LLM_TOP_P", 0.9))
    max_output_tokens: int = Field(default_factory=lambda: _int("LLM_MAX_OUTPUT_TOKENS", 1024))

    # --- vLLM (fallback provider) ---
    vllm_enabled: bool = Field(default_factory=lambda: _bool("VLLM_ENABLED", "false"))
    vllm_base_url: str = Field(default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1"))
    vllm_model: str = Field(
        default_factory=lambda: os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    )

    # --- Agent loop ---
    # Which provider drives the agent loop: "gemini" or "groq" (embeddings always use Gemini).
    agent_provider: str = Field(default_factory=lambda: os.getenv("AGENT_PROVIDER", "gemini").lower())
    groq_api_key: str | None = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    agent_max_steps: int = Field(default_factory=lambda: _int("AGENT_MAX_STEPS", 8))
    # "notes" = clear old tool results + keep structured notes; "full" = keep everything (baseline)
    agent_context_mode: str = Field(default_factory=lambda: os.getenv("AGENT_CONTEXT_MODE", "notes"))

    # --- Retry ---
    max_retries: int = Field(default_factory=lambda: _int("MAX_RETRIES", 3))
    retry_base_delay_seconds: float = Field(default_factory=lambda: _float("RETRY_BASE_DELAY_SECONDS", 0.5))

    # --- Rate limiting ---
    rate_limit_requests: int = Field(default_factory=lambda: _int("RATE_LIMIT_REQUESTS", 10))
    rate_limit_window_seconds: int = Field(default_factory=lambda: _int("RATE_LIMIT_WINDOW_SECONDS", 60))

    # --- Request timeout ---
    request_timeout_seconds: float = Field(default_factory=lambda: _float("REQUEST_TIMEOUT_SECONDS", 30.0))

    # --- Cache ---
    redis_url: str | None = Field(default_factory=lambda: os.getenv("REDIS_URL") or None)
    cache_ttl_seconds: int = Field(default_factory=lambda: _int("CACHE_TTL_SECONDS", 600))

    # --- Vector store / RAG ---
    chroma_persist_dir: str = Field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_data"))
    chroma_collection_name: str = Field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "documents")
    )
    chunk_size: int = Field(default_factory=lambda: _int("CHUNK_SIZE", 800))
    chunk_overlap: int = Field(default_factory=lambda: _int("CHUNK_OVERLAP", 120))
    retrieval_top_k: int = Field(default_factory=lambda: _int("RETRIEVAL_TOP_K", 4))
    retrieval_min_similarity: float = Field(default_factory=lambda: _float("RETRIEVAL_MIN_SIMILARITY", 0.25))

    # --- Uploads ---
    upload_dir: str = Field(default_factory=lambda: os.getenv("UPLOAD_DIR", "./data"))
    max_upload_size_mb: int = Field(default_factory=lambda: _int("MAX_UPLOAD_SIZE_MB", 20))
    allowed_upload_extensions: tuple[str, ...] = (".pdf", ".txt", ".docx")

    # --- CORS ---
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    )

    def safe_public_config(self) -> dict:
        """Return only non-secret configuration, for GET /config."""
        return {
            "app_name": self.app_name,
            "environment": self.environment,
            "gemini_model": self.gemini_model,
            "gemini_embedding_model": self.gemini_embedding_model,
            "gemini_configured": bool(self.gemini_api_key),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "vllm_enabled": self.vllm_enabled,
            "vllm_model": self.vllm_model if self.vllm_enabled else None,
            "max_retries": self.max_retries,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window_seconds": self.rate_limit_window_seconds,
            "cache_backend": "redis" if self.redis_url else "memory",
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_top_k": self.retrieval_top_k,
            "max_upload_size_mb": self.max_upload_size_mb,
            "allowed_upload_extensions": list(self.allowed_upload_extensions),
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
