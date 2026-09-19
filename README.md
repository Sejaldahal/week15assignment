
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
│   ├── llm/                     # base (interface), gemini, groq, vllm, factory (fallback chain)
│   ├── rag/                     # ingestion, chunking, embeddings, vector_store, retriever
│   ├── tools/                   # calculator, time_tool, registry, agent_tools (W16)
│   ├── assistant/               # orchestrator, prompts, schemas, agent (W16 loop)
│   ├── middleware/               # rate_limit, request_id
│   ├── cache/                    # base, memory_cache, redis_cache
│   └── utils/                    # logging, retry, errors
├── frontend/streamlit_app.py
├── tests/                        # pytest, all external calls mocked
├── eval/                         # W16 evaluation harness, cases, docs, results.md
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
| `POST` | `/chat` | `{message, session_id, use_rag, agent}` → `ChatResponse` (`agent: true` runs the W16 loop, see §25) |
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

## 25. W16: Agentic Verification Loop

`POST /chat` with `"agent": true` runs `AgentRunner` (`backend/assistant/agent.py`) instead of the single-pass
pipeline. Each turn the model must call one tool: `search_documents`, `calculator`, `current_datetime`,
`record_note`, `ask_user` or `finish`. The loop ends on `finish`, on `ask_user` (the reply arrives as the next
message with the same `session_id`), or after `AGENT_MAX_STEPS` (default 8), which returns a partial answer from the
verified notes at confidence 0.2. Diagram: [`ARCHITECTURE.md`](./ARCHITECTURE.md). Config: `AGENT_PROVIDER`
(`gemini` or `groq`; embeddings always use Gemini), `GROQ_API_KEY`, `GROQ_MODEL`, `AGENT_MAX_STEPS`, `AGENT_CONTEXT_MODE`.

**Why a fixed pipeline is not enough.** Our test documents contain an old and a newer Pro price and separate seat
limits, so what to look up next (the newer source, another document, a calculation, or the user) depends on what
the previous result showed, which a fixed retrieve-then-answer sequence cannot know in advance.

### a. Context engineering technique
**Structured external notes + clearing old tool results + capped retrieval**, applied in `AgentRunner._render()`,
which rebuilds the model's context before every step. Problem: each `search_documents` result is a few hundred
tokens, and a multi-search task (comparison, annual price) runs 5-9 steps; a normal chat history would re-send every
earlier result on every later step, so the prompt grows with each iteration. Now only the latest tool result stays
in full; older ones become one-line stubs (`search_documents(...) -> 3 chunks: id1, id2, id3`), and anything needed
later must be saved with `record_note` (claim, source chunk, verified flag), which is re-injected each turn.
Retrieval is capped at 3 chunks of 500 characters. `AGENT_CONTEXT_MODE=full` turns the clearing off and is the
baseline in the eval. Measured effect: **none in our eval.** Average tokens per query were 4,706 with clearing (`notes`) vs
4,579 without (`full`), and completion was 9/12 vs 11/12 (one run each, so partly noise). Our tasks are short (3-6
steps, results of a few hundred tokens), so there is little history to clear, and the extra `record_note` turns cost
about what the clearing saves. In `notes` mode the agent also skipped the calculator twice (right number, wrong tool
use); in `full` mode it never did. Clearing keeps prompt size from growing with each search, but we have not shown
a benefit on this workload; `AGENT_CONTEXT_MODE=full` is the better setting until longer tasks are tested.

### b. Agentic pattern: single-agent loop
Every step depends on the previous result (search, then note, then calculate), so there is nothing to run in
parallel. Context saturation is handled by the notes and clearing above, which is cheaper than a sub-agent that
would add coordination tokens. Six small tools keep skill dilution low. The self-verification paradox is reduced
because claims are checked against retrieved chunks and tool output (the `verified` flag), and a code guard caps
confidence when a tool failed and nothing was verified. The single point of failure is accepted and mitigated by the
step cap, errors returned as observations, and a limitation report instead of a guess.

### c. Evaluation harness (`python -m eval.run_eval`, no framework)
12 queries in `eval/cases.json` over four synthetic documents in `eval/docs/` (single fact, conflicting sources,
retrieve-then-calculate, comparison, calculator-only, datetime-only, not-in-docs, ambiguous, two-turn clarification),
run against the real API and written to `eval/results.md` / `results.json`. Per query it records:
- **Task completion:** expected keywords in the answer, or the expected action (ask the user / abstain).
- **Tool-call correctness:** expected tools were used, plus the share of calls with schema-valid arguments.
- **Trajectory length:** model turns, checked against an expected range per case.
- **Tokens:** prompt + output tokens summed over the query (embedding calls not counted).
- **Failure class:** for any case that is not both completed and tool-correct. **Hard** = no usable answer (model
  unreachable or step cap hit). **Cascading soft** = an earlier bad step (tool error, invalid arguments, empty
  retrieval) carried into a wrong answer. **Soft** = wrong or unsupported answer, or wrong tool use, with no earlier
  bad step. These are our working definitions of the class taxonomy.

**Results** (`openai/gpt-oss-20b` on Groq, one run per mode; full tables in `eval/results.md`):

| mode | completion | tool-correct | valid args | avg steps | in range | avg tokens |
|---|---|---|---|---|---|---|
| `notes` (clearing on) | 9/12 | 10/12 | 100% | 4.5 | 10/12 | 4,706 |
| `full` (baseline) | 11/12 | 12/12 | 100% | 4.0 | 12/12 | 4,579 |

`notes` failures: 2 hard (`compare_recommend`, `not_in_docs` hit the step cap; on `not_in_docs` the agent kept
re-searching instead of concluding the documents lack the answer), 3 soft (`retrieve_then_calc` and `annual_discount`
gave the right number without the calculator; `ambiguous_asks` guessed the Starter plan instead of asking), 0 cascading
soft. `full` failures: 1 hard (`compare_recommend`: Groq rejected a malformed tool call even after retries, so the
model call failed), 0 soft, 0 cascading soft. None of the failures came from a bad tool result carried forward; they
are step-cap exhaustion, a model-call error, or the model choosing the wrong action (skipping the calculator, guessing
instead of asking).

### Failure injection
`AgentRunner` accepts an optional `tool_hook`, used only by the eval and tests (nothing in `backend/` sets it). We
break `search_documents` two ways, on three queries each: it times out (`search_down`), or returns corrupted output
(`malformed`, rejected by the output check). The agent detected the failure in 12/12 runs (6 per mode), retried or rephrased, and
ended with an explicit "could not verify" answer at confidence 0.0-0.3, never a confident figure; all 12 passed. The first live
injection run exposed a real gap: after retrying, the agent ended with `ask_user`, echoing the user's own question
back. We added a guard so that after a tool failure with nothing verified, `ask_user` or a prose reply becomes a
limitation report (`tests/test_agent.py::test_failed_tool_then_ask_user_reports_limitation`).

### Skill vs Agent
The verification behaviour could not be a Skill, because a Skill only loads instructions into context while
retrieval, arithmetic and the stop-or-ask decision need tool execution and control flow; the guidance text in our
system prompt could be a Skill, but it is short and needed on every turn, so progressive disclosure would save nothing.

### Tool vs agent boundary
The external services (the LLM API, the embedding API, ChromaDB) are modelled as bounded tool calls, not as agents.
`search_documents` is one call with a timeout, a capped result and an output check; it holds no task state and returns
to our loop after a single round trip. The multi-step behaviour (rephrase, search another document, retry) lives in our
own loop where we can cap it and trace it, and the only state is the per-session notes and pending question that
`AgentRunner` keeps in memory. Agent-to-agent delegation would add coordination cost for a service that never needs to
decide anything.

### Known limitations
12 cases, one run per mode on a small model, so numbers are indicative, not statistically meaningful. Two step ranges
(`calc_only`, `datetime_only`) and `AGENT_MAX_STEPS` (6 to 8) were adjusted after a first run showed they were too
tight. Session state is in memory (lost on restart). Run the harness with `python -m eval.run_eval --context both`.

