(output.png)
# AI Assistant — Production-Oriented Reference Implementation

A demonstration AI assistant built to showcase how a real assistant is designed and
productionized: LLM integration, structured output, tool calling, RAG, provider
fallback, caching, rate limiting, retries, and containerization — not a toy chatbot.

## 1. Project Overview

The assistant answers general questions, answers questions grounded in
user-uploaded documents (RAG), and can call tools (a calculator and a
date/time utility) when the question requires it. Every response is a
predictable, validated Pydantic object. **Google Gemini** is the primary
LLM; a **locally-served open-source model via vLLM** is an optional
fallback used only if Gemini is unavailable.

## 2. Features

- Gemini API integration via the current `google-genai` SDK
- Structured JSON output validated against a Pydantic schema
- Tool/function calling (calculator, current date/time), model-decided
- RAG: PDF/TXT/DOCX ingestion → cleaning → chunking → embeddings → ChromaDB → retrieval
- Provider fallback: Gemini → local vLLM model, behind one common interface
- Retry with exponential backoff for transient failures only
- Per-client rate limiting
- Response caching (Redis, with an automatic in-memory fallback)
- Structured JSON logging with request-ID tracing
- FastAPI backend + Streamlit UI
- Dockerized, with an optional GPU profile for vLLM
- Pytest suite mocking all external calls

## 3. Architecture

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for full Mermaid diagrams
(system overview, ingestion flow, and the chat/RAG/tool-calling sequence).

```
Streamlit UI → FastAPI → Assistant Orchestrator
                              ├── Cache (Redis / in-memory)
                              ├── RAG Retriever → ChromaDB
                              ├── Tool Registry (calculator, datetime)
                              └── LLM Fallback Chain
                                      ├── Gemini API (primary)
                                      └── vLLM local model (fallback, optional)
```

The orchestrator never contains provider-specific code — it only calls the
`LLMProvider` interface (`llm/base.py`), which `GeminiProvider` and
`VLLMProvider` both implement. This is what makes fallback, retries, and
mocking in tests all straightforward.

## 4. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI + Pydantic + Uvicorn | async-native, schema validation doubles as API contract |
| Primary LLM | Gemini (`google-genai`) | current, unified official SDK; structured output + function calling + embeddings in one client |
| Fallback LLM | vLLM (OpenAI-compatible server) | continuous batching + paged KV cache for efficient local serving |
| Embeddings | Gemini `text-embedding-004` | no separate provider/API key needed |
| Vector DB | ChromaDB | zero-infra local persistence, swappable interface |
| Cache | Redis (in-memory fallback) | shared across instances in production, works with zero setup in dev |
| Frontend | Streamlit | fastest usable UI for a chat + upload demo |
| Containerization | Docker / Docker Compose | CPU profile by default, optional GPU profile for vLLM |

## 5. Folder Structure

```
ai-assistant/
├── backend/
│   ├── main.py                  # FastAPI app, wiring
│   ├── config.py                # env-driven Settings
│   ├── api/                     # routes_chat, routes_documents, routes_health, routes_config
│   ├── llm/                     # base (interface), gemini, vllm, factory (fallback chain)
│   ├── rag/                     # ingestion, chunking, embeddings, vector_store, retriever
│   ├── tools/                   # calculator, time_tool, registry
│   ├── assistant/               # orchestrator, prompts, schemas
│   ├── middleware/               # rate_limit, request_id
│   ├── cache/                    # base, memory_cache, redis_cache
│   └── utils/                    # logging, retry, errors
├── frontend/streamlit_app.py
├── tests/                        # pytest, all external calls mocked
├── data/                         # uploaded documents (gitignored)
├── chroma_data/                  # persisted vector index (gitignored)
├── docker/                       # Dockerfile.backend / .frontend / .vllm
├── docker-compose.yml            # CPU-only stack
├── docker-compose.vllm.yml       # optional GPU override
├── requirements.txt
├── .env.example
└── ARCHITECTURE.md
```

## 6. Environment Variables

See [`.env.example`](./.env.example) for the full, commented list. Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | **required**, never hardcoded |
| `GEMINI_MODEL` | `gemini-2.5-flash` | primary chat model |
| `LLM_TEMPERATURE` | `0.3` | low variance, favors grounded answers |
| `LLM_TOP_P` | `0.9` | slight diversity without drift |
| `LLM_MAX_OUTPUT_TOKENS` | `1024` | enough for a grounded answer + citations |
| `VLLM_ENABLED` | `false` | opt-in fallback, off by default (no GPU assumed) |
| `MAX_RETRIES` | `3` | retry attempts for transient Gemini errors |
| `RATE_LIMIT_REQUESTS` / `_WINDOW_SECONDS` | `10` / `60` | 10 req/min/client by default |
| `REDIS_URL` | empty | leave unset to use the in-memory dev cache |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | see RAG section below for rationale |

## 7. Local Setup

**Requires Python 3.11 or 3.12** (see `.python-version`). `chromadb` has a
hard dependency on `onnxruntime`, which does not yet publish wheels for
very new Python releases (e.g. 3.13/3.14 at the time of writing) —
installing on an unsupported Python version will fail trying to compile
`tokenizers` from source via a Rust/PyO3 toolchain that also doesn't
support the newest CPython yet. This is an upstream ecosystem-lag issue,
not a project bug — check `pip index versions onnxruntime` if you want to
confirm current wheel support before bumping the pin.

```bash
# macOS, if you don't already have 3.12: brew install python@3.12
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY
```

## 8. Gemini API Setup

1. Create a key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Put it in `.env` as `GEMINI_API_KEY=...` — never commit this file or
   hardcode the key anywhere in source.
3. The app uses `google-genai` (`from google import genai`), Google's
   current unified SDK, superseding the older `google-generativeai`
   package.

## 9. Running Without vLLM (default)

```bash
uvicorn backend.main:app --reload --port 8000
streamlit run frontend/streamlit_app.py
```

With `VLLM_ENABLED=false` (the default), the app runs entirely on Gemini.
`GET /health` reports `"vllm": null` and everything else works normally.

## 10. Running With vLLM (optional, GPU required)

Requirements:
- An NVIDIA GPU with enough VRAM for your chosen model (e.g. ~16GB for an
  8B model at fp16).
- The [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) installed on the host so Docker can access the GPU.

```bash
docker compose -f docker-compose.yml -f docker-compose.vllm.yml up --build
```

This starts a `vllm` service serving an OpenAI-compatible API on port
`8001` and sets `VLLM_ENABLED=true` for the backend. If the GPU/service is
unavailable, the backend's fallback chain simply skips it and Gemini
continues to serve all requests — nothing else needs to change.

## 11. Docker Setup

```bash
docker build -f docker/Dockerfile.backend -t ai-assistant-backend .
docker build -f docker/Dockerfile.frontend -t ai-assistant-frontend .
```

## 12. Docker Compose

```bash
docker compose up --build          # CPU-only: backend + frontend + redis
```

Add `-f docker-compose.vllm.yml` to also start the optional GPU-backed
vLLM service (see §10).

## 13. API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | `{message, session_id, use_rag}` → `ChatResponse` |
| `POST` | `/documents/upload` | multipart file upload → ingests into ChromaDB |
| `GET` | `/health` | status of API, Gemini, vector DB, Redis, vLLM |
| `GET` | `/config` | safe, non-secret configuration |
| `GET` | `/docs` | interactive OpenAPI docs (FastAPI default) |

## 14. RAG Pipeline Explanation

`Document → Loader → Text extraction → Cleaning → Chunking → Embedding →
ChromaDB → Retriever → Relevant chunks → Prompt construction → Gemini →
Structured response`

- **Chunking**: 800 characters, 120 character (~15%) overlap. Large
  enough to usually hold a complete paragraph/thought (reduces
  fragmentation); small enough that 4–6 retrieved chunks fit comfortably
  in the prompt budget. Overlap prevents losing a boundary-spanning
  sentence from both neighboring chunks.
- **Metadata per chunk**: `document_id`, `document_name`, `chunk_id`,
  `page` (when available, e.g. PDFs), `source`.
- **Retrieval**: top-k (default 4) cosine similarity search, with a
  minimum similarity cutoff (`0.25`) so weakly related chunks are
  excluded — this is what lets the assistant honestly say "the documents
  don't cover that" instead of forcing an answer from noise.
- **Grounding**: the system prompt instructs the model to cite
  `[chunk_id]` references and separate document-grounded claims from
  general-knowledge claims.

## 15. Tool-Calling Explanation

Two tools: a **calculator** (AST-restricted arithmetic evaluator — no
`eval()`/`exec()`, so arbitrary code execution is impossible) and a
**current date/time** utility. Both validate their arguments with a
Pydantic model before executing. The model decides when to call a tool;
the orchestrator executes it and feeds the result back for a final
structured answer, recording which tool was used in `tool_used`.

## 16. Structured Output Explanation

Every chat response conforms to:

```json
{
  "answer": "...",
  "sources": [],
  "tool_used": null,
  "confidence": 0.0,
  "provider": "gemini"
}
```

Gemini is asked to emit JSON via `response_schema=ChatResponse` (using
the SDK's native structured-output support, not manual string parsing).
If the model still returns something invalid, the API treats it as a
provider error (not silently passed through) and either retries or falls
back.

## 17. Retry / Fallback Architecture

- **Retry**: exponential backoff (`utils/retry.py`), only for errors
  marked `retryable=True` — timeouts, 5xx, and 429 rate-limit responses.
  Invalid API key / malformed request errors (4xx auth/validation) are
  never retried. `MAX_RETRIES` is configurable (default 3).
- **Fallback**: `llm/factory.py` builds a `FallbackChain` that retries
  Gemini first, and only falls through to the local vLLM model after
  retries are exhausted. Both providers implement the same
  `LLMProvider` interface, so the orchestrator has zero provider-specific
  branching.

## 18. Rate Limiting

Default: **10 requests/minute per client IP** (`RATE_LIMIT_REQUESTS` /
`RATE_LIMIT_WINDOW_SECONDS`), implemented as fixed-window middleware.
This protects both the backend and the (metered) Gemini API quota from a
single client overwhelming the system. `/health` and docs endpoints are
exempt.

## 19. Caching

Cache key is a hash of `{query, retrieved-context version, provider,
model}` — so a cache hit only ever fires when all of these match,
preventing stale answers when the underlying document set changes.
Backend is Redis when `REDIS_URL` is set; otherwise an in-memory TTL
cache is used automatically, so local development needs no extra
services. Cache errors degrade to a miss, never a request failure.

## 20. Testing

```bash
pytest tests/ -v
```

All external calls (Gemini, vLLM) are mocked — no API quota is consumed
running the suite. Covers: health, chat (success/validation/failure),
calculator (including a rejected code-injection attempt), chunking edge
cases, retry/backoff behavior (transient vs. permanent errors), rate
limiting, and document upload/ingestion (valid and rejected file types).

## 21. Performance Considerations

- **I/O-bound** (async, non-blocking): Gemini/vLLM API calls, Redis
  reads/writes, ChromaDB queries.
- **CPU-bound** (offloaded via `asyncio.to_thread`): PDF/DOCX text
  extraction during ingestion — this keeps the event loop free to serve
  concurrent chat requests while a large document is being parsed.
- Request timeouts are enforced at the HTTP client level
  (`REQUEST_TIMEOUT_SECONDS`) so a hung upstream call can't block a
  worker indefinitely.

## 22. ONNX Justification

Gemini is a hosted API model — there is no local model file to convert,
so ONNX conversion does not apply to the primary provider. For the local
fallback model served via vLLM, ONNX is intentionally **not** used
either: vLLM already provides continuous batching, paged-attention KV
caching, and (for supported models) quantization and tensor parallelism
at the serving layer, which is a better fit for a chat-serving workload
than a static ONNX graph — ONNX conversion mainly pays off for
latency-critical, fixed-shape inference (e.g. edge deployment), which is
not this project's use case.

## 23. Limitations

- ChromaDB's local persistence is fine for a demo corpus but not built
  for multi-instance horizontal scaling — would move to a managed vector
  DB in a larger deployment.
- The in-memory cache and in-memory rate limiter are per-process; a
  multi-instance deployment needs the Redis-backed versions everywhere.
- vLLM cold-start latency on first fallback use is a known, undocumented
  cost — acceptable for a demo, worth pre-warming in production.
- Tool set is intentionally minimal (2 tools) per the assessment scope.

## 24. Future Improvements

- Streaming responses (SSE/websockets) instead of single-shot JSON.
- Swap ChromaDB for a managed vector store (pgvector/Qdrant) via the
  existing `VectorStore` interface.
- Add per-session conversation history/context window management.
- Add authentication and per-API-key rate limiting instead of per-IP.
- Add OpenTelemetry tracing spans on top of the existing structured logs.
