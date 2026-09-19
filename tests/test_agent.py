"""Agent loop tests with a scripted fake model: no network, no quota."""
import asyncio

import pytest

from backend.assistant.agent import AgentRunner, validate_args
from backend.config import Settings
from backend.llm.base import AgentStep


class FakeStore:
    def count(self):
        return 1

    def query(self, embedding, top_k):
        return {
            "ids": [["c1"]],
            "documents": [["Pro costs $35 per seat per month. " * 30]],
            "metadatas": [[{"chunk_id": "c1", "document_id": "d1", "document_name": "price.txt", "source": "price.txt"}]],
            "distances": [[0.1]],
        }


class FakePrimary:
    async def embed(self, texts):
        return [[0.0] for _ in texts]


class ScriptedChain:
    """Plays back a fixed list of tool calls and records the context it was given."""

    def __init__(self, calls):
        self._primary = FakePrimary()
        self.calls = list(calls)
        self.seen: list[list[dict]] = []

    async def step(self, system_prompt, messages, tools):
        self.seen.append(messages)
        name, args = self.calls.pop(0) if self.calls else ("search_documents", {"query": "again"})
        return AgentStep(call={"name": name, "args": args}, prompt_tokens=10, output_tokens=5)


def runner(calls, **kw):
    chain = ScriptedChain(calls)
    return AgentRunner(Settings(), chain, FakeStore(), **kw), chain


def run(r, msg="q", sid="s"):
    return asyncio.run(r.run(msg, sid))


def test_multi_step_loop_uses_results_and_finishes():
    r, _ = runner([
        ("search_documents", {"query": "pro price"}),
        ("record_note", {"claim": "Pro $35/seat", "source": "c1", "verified": True}),
        ("calculator", {"expression": "35*8"}),
        ("finish", {"answer": "280 [c1]", "confidence": 0.9}),
    ])
    resp = run(r)
    assert resp.answer == "280 [c1]" and resp.confidence == 0.9
    assert [t["tool"] for t in resp.trace] == ["search_documents", "record_note", "calculator", "finish"]
    assert resp.usage["total_tokens"] == 60 and resp.tool_used == "calculator"
    assert [s.chunk_id for s in resp.sources] == ["c1"]


def test_step_cap_stops_a_looping_agent():
    r, chain = runner([], max_steps=3)  # scripted chain searches forever
    resp = run(r)
    assert len(chain.seen) == 3
    assert resp.confidence == 0.2 and "could not finish within 3 steps" in resp.answer


def test_ask_user_then_resume_in_same_session():
    r, chain = runner([("ask_user", {"question": "Which plan?"}), ("finish", {"answer": "$35", "confidence": 0.8})])
    first = run(r, "How much is it?")
    assert first.needs_clarification and first.answer == "Which plan?"
    second = run(r, "Pro")
    assert not second.needs_clarification
    assert "How much is it?" in chain.seen[1][0]["text"] and "Pro" in chain.seen[1][0]["text"]


def test_old_tool_results_are_cleared_but_latest_kept():
    r, chain = runner([("search_documents", {"query": "a"}), ("search_documents", {"query": "b"}),
                       ("search_documents", {"query": "c"}), ("finish", {"answer": "x", "confidence": 0.5})])
    run(r)
    tools = [m for m in chain.seen[3] if m["role"] == "tool"]
    assert "cleared" in tools[0]["result"] and "cleared" in tools[1]["result"]
    assert "chunks" in tools[2]["result"]


def test_full_context_mode_keeps_everything():
    r, chain = runner([("search_documents", {"query": "a"}), ("search_documents", {"query": "b"}),
                       ("finish", {"answer": "x", "confidence": 0.5})], context_mode="full")
    run(r)
    assert all("chunks" in m["result"] for m in chain.seen[2] if m["role"] == "tool")


def test_injected_tool_failure_caps_confidence():
    def down(name, args):
        if name == "search_documents":
            raise TimeoutError("search backend timed out")

    r, _ = runner([("search_documents", {"query": "x"}), ("finish", {"answer": "It costs $12", "confidence": 0.95})],
                  tool_hook=down)
    resp = run(r)
    assert "timed out" in resp.trace[0]["error"]
    assert resp.confidence <= 0.3 and "unverified" in resp.answer


def test_malformed_retrieval_output_is_rejected():
    r, _ = runner([("search_documents", {"query": "x"}), ("finish", {"answer": "y", "confidence": 0.9})],
                  tool_hook=lambda n, a: {"chunks": "\x00garbage"} if n == "search_documents" else None)
    resp = run(r)
    assert "malformed" in resp.trace[0]["error"] and resp.confidence <= 0.3


def test_invalid_arguments_are_reported_not_executed():
    assert validate_args("calculator", {}) == "missing required argument 'expression'"
    assert validate_args("calculator", {"expression": 5}) == "argument 'expression' must be string"
    assert validate_args("nope", {}) == "unknown tool 'nope'"
    assert validate_args("finish", {"answer": "a", "confidence": 0.5}) is None


def test_prose_reply_is_nudged_once_then_accepted():
    class ProseThenFinish(ScriptedChain):
        async def step(self, system_prompt, messages, tools):
            self.seen.append(messages)
            if len(self.seen) == 1:
                return AgentStep(text="The price is $35.")
            return AgentStep(call={"name": "finish", "args": {"answer": "$35", "confidence": 0.8}})

    chain = ProseThenFinish([])
    resp = asyncio.run(AgentRunner(Settings(), chain, FakeStore()).run("q", "s"))
    assert resp.answer == "$35" and "call finish" in chain.seen[1][-1]["text"]


def test_failed_tool_then_ask_user_reports_limitation():
    def down(name, args):
        if name == "search_documents":
            raise TimeoutError("search backend timed out")

    r, _ = runner([("search_documents", {"query": "x"}), ("search_documents", {"query": "y"}),
                   ("ask_user", {"question": "What is the price?"})], tool_hook=down)
    resp = run(r)
    assert not resp.needs_clarification and resp.confidence <= 0.1
    assert "could not verify" in resp.answer and "$" not in resp.answer
    assert sum(1 for t in resp.trace if t.get("error")) == 2  # detected and retried
