"""
Agentic verification loop (W16). A single agent repeatedly asks the model for
its next action; the model chooses among search_documents / calculator /
current_datetime / record_note / ask_user / finish based on what the previous
result showed. The loop is bounded by `agent_max_steps`.

Context engineering (see README): structured external notes + clearing of old
tool results + capped retrieval snippets. `context_mode="full"` disables the
clearing and is used only as the baseline in the eval.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from backend.assistant.prompts import AGENT_SYSTEM_PROMPT
from backend.assistant.schemas import AgentChatResponse, SourceChunk
from backend.config import Settings
from backend.llm.base import LLMError
from backend.tools.agent_tools import AGENT_TOOL_SCHEMAS, search_documents
from backend.tools.registry import TOOL_SCHEMAS, execute_tool
from backend.rag.vector_store import VectorStore

logger = logging.getLogger("ai_assistant.agent")

ALL_SCHEMAS = TOOL_SCHEMAS + AGENT_TOOL_SCHEMAS
_SCHEMA_BY_NAME = {s["name"]: s for s in ALL_SCHEMAS}
_JSON_TYPES = {"string": str, "boolean": bool, "number": (int, float), "integer": int}
MAX_NOTES = 12
LIMITATION = (
    "I could not verify an answer: a required tool failed or returned invalid data, "
    "and I have no reliable evidence. Please try again later."
)

# A hook may replace a tool's real behaviour (used by the failure-injection tests).
# Return a result to substitute it, None to run the real tool, or raise to simulate a crash.
ToolHook = Callable[[str, dict], Awaitable[Any] | Any]


def validate_args(name: str, args: Any) -> str | None:
    """Check tool arguments against the tool's JSON schema. Returns an error string or None."""
    schema = _SCHEMA_BY_NAME.get(name)
    if schema is None:
        return f"unknown tool '{name}'"
    if not isinstance(args, dict):
        return "arguments must be an object"
    params = schema["parameters"]
    for req in params.get("required", []):
        if req not in args:
            return f"missing required argument '{req}'"
    for key, spec in params.get("properties", {}).items():
        if key in args:
            expected = _JSON_TYPES.get(spec.get("type"))
            value = args[key]
            if expected and (not isinstance(value, expected) or (spec["type"] != "boolean" and isinstance(value, bool))):
                return f"argument '{key}' must be {spec['type']}"
    return None


def _valid_search_result(result: Any) -> bool:
    return (
        isinstance(result, dict)
        and isinstance(result.get("chunks"), list)
        and all(isinstance(c, dict) and c.get("chunk_id") and isinstance(c.get("text"), str) for c in result["chunks"])
    )


def _short(value: Any, n: int = 90) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= n else text[: n - 1] + "…"


class AgentRunner:
    def __init__(
        self,
        settings: Settings,
        llm_chain,
        vector_store: VectorStore,
        max_steps: int | None = None,
        context_mode: str | None = None,
        tool_hook: ToolHook | None = None,
    ):
        self._settings = settings
        self._llm = llm_chain
        self._store = vector_store
        self.max_steps = max_steps or settings.agent_max_steps
        self.context_mode = context_mode or settings.agent_context_mode
        self.tool_hook = tool_hook
        self._sessions: dict[str, dict] = {}  # session_id -> state saved by ask_user

    # ------------------------------------------------------------------ loop
    async def run(self, message: str, session_id: str) -> AgentChatResponse:
        pending = self._sessions.pop(session_id, None)
        notes: list[dict] = list(pending["notes"]) if pending else []
        question = (
            f"{pending['question']}\n(You asked: \"{pending['asked']}\" — the user replied: \"{message}\")"
            if pending
            else message
        )

        history: list[dict] = []
        trace: list[dict] = []
        sources: dict[str, dict] = {}
        tokens = {"prompt_tokens": 0, "output_tokens": 0}
        had_error = False
        nudged = False

        for step_no in range(1, self.max_steps + 1):
            prompt = self._render(question, notes, history, step_no)
            try:
                step = await asyncio.wait_for(
                    self._llm.step(AGENT_SYSTEM_PROMPT, prompt, ALL_SCHEMAS),
                    timeout=self._settings.request_timeout_seconds,
                )
            except (LLMError, asyncio.TimeoutError) as exc:
                logger.warning("agent model call failed: %s", exc)
                trace.append({"step": step_no, "tool": None, "error": f"model call failed: {exc}"})
                return self._response(
                    "The assistant could not reach the model, so no answer was produced.",
                    0.0, trace, tokens, provider="none",
                )
            tokens["prompt_tokens"] += step.prompt_tokens
            tokens["output_tokens"] += step.output_tokens

            if step.call is None:
                if not nudged and step_no < self.max_steps:
                    # Prose instead of a tool call: ask once for a proper tool call (e.g. finish).
                    nudged = True
                    trace.append({"step": step_no, "tool": None, "note": "no tool call; nudged to call a tool"})
                    history.append({"role": "user", "text": "Respond by calling a tool. To give your answer, call finish."})
                    continue
                trace.append({"step": step_no, "tool": None, "note": "no tool call; text treated as final"})
                if had_error and not any(n["verified"] for n in notes):
                    return self._response(LIMITATION, 0.1, trace, tokens)
                return self._response(step.text or "No answer produced.", 0.3, trace, tokens)

            name, args = step.call["name"], step.call["args"]
            entry: dict[str, Any] = {"step": step_no, "tool": name, "args": args}
            trace.append(entry)

            if name == "finish":
                return self._finish(args, notes, sources, trace, tokens, had_error)
            if name == "ask_user":
                if had_error and not any(n["verified"] for n in notes):
                    # Asking the user can't fix a broken tool: report the limitation instead.
                    entry["note"] = "ask_user after tool failure converted to limitation report"
                    return self._response(LIMITATION, 0.1, trace, tokens)
                if validate_args(name, args) is None:
                    self._sessions[session_id] = {"question": question, "asked": args["question"], "notes": notes}
                    return self._response(args["question"], 0.0, trace, tokens, clarify=True)

            result = await self._execute(name, args, notes, sources)
            failed = isinstance(result, dict) and "error" in result
            had_error = had_error or failed
            entry["error"] = result["error"] if failed else None
            entry["result"] = _short(result)
            history.append({"role": "model", "call": step.call, "raw": step.raw})
            history.append({"role": "tool", "name": name, "args": args, "result": result})

        # Stopping condition: step budget exhausted without finish/ask_user.
        trace.append({"step": self.max_steps, "tool": None, "note": "step budget exhausted"})
        verified = [n["claim"] for n in notes if n["verified"]]
        answer = f"I could not finish within {self.max_steps} steps."
        if verified:
            answer += " What I verified so far: " + "; ".join(verified)
        return self._response(answer, 0.2, trace, tokens)

    # ------------------------------------------------------------- internals
    async def _execute(self, name: str, args: dict, notes: list[dict], sources: dict) -> Any:
        error = validate_args(name, args)
        if error:
            return {"error": f"invalid arguments: {error}"}
        try:
            if self.tool_hook is not None:
                hooked = self.tool_hook(name, args)
                hooked = await hooked if asyncio.iscoroutine(hooked) else hooked
                if hooked is not None:
                    return self._check_result(name, hooked)
            if name == "record_note":
                if len(notes) >= MAX_NOTES:
                    return {"error": f"note limit ({MAX_NOTES}) reached"}
                notes.append({k: args[k] for k in ("claim", "source", "verified")})
                return {"ok": True, "notes": len(notes)}
            if name == "search_documents":
                result, chunks = await asyncio.wait_for(
                    search_documents(self._llm._primary, self._store, self._settings, args),
                    timeout=self._settings.request_timeout_seconds,
                )
                for c in chunks:
                    sources[c["chunk_id"]] = c
                return self._check_result(name, result)
            return execute_tool(name, args)
        except Exception as exc:  # noqa: BLE001 - a broken tool must not crash the loop
            return {"error": f"{name} failed: {type(exc).__name__}: {exc}"}

    @staticmethod
    def _check_result(name: str, result: Any) -> Any:
        if name == "search_documents" and not (
            (isinstance(result, dict) and "error" in result) or _valid_search_result(result)
        ):
            return {"error": "search_documents returned malformed output"}
        return result

    def _render(self, question: str, notes: list[dict], history: list[dict], step_no: int) -> list[dict]:
        """Build the model's context for this turn (the context-engineering point)."""
        note_lines = "\n".join(
            f"- [{'verified' if n['verified'] else 'unverified'}] {n['claim']} (source: {n['source']})" for n in notes
        ) or "(none yet)"
        head = (
            f"QUESTION:\n{question}\n\nNOTES SO FAR:\n{note_lines}\n\n"
            f"Step {step_no} of {self.max_steps}."
        )
        messages: list[dict] = [{"role": "user", "text": head}]
        last_tool = max((i for i, m in enumerate(history) if m["role"] == "tool"), default=-1)
        for i, m in enumerate(history):
            if m["role"] == "tool" and self.context_mode == "notes" and i != last_tool:
                m = {**m, "result": self._stub(m)}
            messages.append(m)
        return messages

    @staticmethod
    def _stub(m: dict) -> dict:
        r = m["result"]
        if m["name"] == "search_documents" and isinstance(r, dict) and "chunks" in r:
            summary = f"{r.get('n_results', len(r['chunks']))} chunks: " + ", ".join(c["chunk_id"] for c in r["chunks"])
        else:
            summary = _short(r, 60)
        return {"cleared": f"{m['name']}({_short(m.get('args', {}), 60)}) -> {summary}. Use notes."}

    def _finish(self, args, notes, sources, trace, tokens, had_error) -> AgentChatResponse:
        if validate_args("finish", args) is not None:
            return self._response(str(args.get("answer", "No answer produced.")), 0.3, trace, tokens)
        answer, confidence = args["answer"], max(0.0, min(1.0, float(args["confidence"])))
        # Guard: a tool failed and nothing was verified -> never let a confident answer through.
        if had_error and not any(n["verified"] for n in notes):
            confidence = min(confidence, 0.3)
            answer = "[Some sources were unavailable or returned invalid data; this is unverified.] " + answer
            trace[-1]["note"] = "confidence capped: tool failure with no verified notes"
        cited = {n["source"] for n in notes}
        return self._response(answer, confidence, trace, tokens, sources=[sources[c] for c in sources if c in cited])

    def _response(self, answer, confidence, trace, tokens, provider="gemini", clarify=False, sources=()) -> AgentChatResponse:
        used = [t["tool"] for t in trace if t.get("tool") and t["tool"] not in ("finish", "ask_user", "record_note")]
        return AgentChatResponse(
            answer=answer,
            confidence=confidence,
            provider=provider,
            needs_clarification=clarify,
            tool_used=used[-1] if used else None,
            sources=[
                SourceChunk(
                    document_id=c["document_id"], document_name=c["document_name"], chunk_id=c["chunk_id"],
                    page=c.get("page"), source=c.get("source", c["document_name"]), similarity=c.get("similarity"),
                )
                for c in sources
            ],
            trace=trace,
            usage={**tokens, "total_tokens": tokens["prompt_tokens"] + tokens["output_tokens"], "steps": len(trace)},
        )
