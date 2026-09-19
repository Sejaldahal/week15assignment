"""
Evaluation harness for the W16 agent (written from scratch, no eval framework).

  python -m eval.run_eval                      # notes-mode context, normal cases + failure injection
  python -m eval.run_eval --context both       # also run the full-context baseline (token comparison)
  python -m eval.run_eval --only single_fact --inject none

Runs against the real Gemini API using a throwaway Chroma store built from eval/docs.
Measures: task completion, tool-call correctness, trajectory length, tokens; logs failures
as hard / soft / cascading soft; runs failure-injection cases.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.assistant.agent import AgentRunner
from backend.config import Settings
from backend.llm.base import LLMError
from backend.llm.factory import build_fallback_chain
from backend.rag.chunking import chunk_text, clean_text, new_chunk_id, new_document_id
from backend.rag.embeddings import embed_texts
from backend.rag.vector_store import VectorStore

HERE = Path(__file__).parent
ABSTAIN = re.compile(r"not (found|contain|mention|available|specif|covered|include)|no information|could ?n[o']t|cannot|can't|unable|unverified|do(es)? ?n[o']t", re.I)
INJECT_CASES = ["single_fact", "retrieve_then_calc", "compare_recommend"]


# ---- failure injections: replace search_documents' behaviour -------------------
def _search_down(name, args):
    if name == "search_documents":
        raise TimeoutError("search backend timed out")


def _malformed(name, args):
    if name == "search_documents":
        return {"chunks": "\x00\x00 ### corrupted ###", "n_results": "??"}


FAULTS = {"search_down": _search_down, "malformed": _malformed}


# ---- rate-limit friendly wrappers (free Gemini tier) ---------------------------
class Paced:
    def __init__(self, inner, delay: float):
        self._inner, self._delay, self._last = inner, delay, 0.0

    async def _wait(self):
        gap = self._delay - (time.monotonic() - self._last)
        if gap > 0:
            await asyncio.sleep(gap)
        self._last = time.monotonic()

    async def embed(self, texts):
        await self._wait()
        return await self._inner.embed(texts)


class PacedChain:
    def __init__(self, chain, delay: float):
        self._chain, self._delay, self._last = chain, delay, 0.0
        self._primary = Paced(chain._primary, delay)

    async def step(self, system_prompt, messages, tools):
        for attempt in range(4):
            gap = self._delay - (time.monotonic() - self._last)
            if gap > 0:
                await asyncio.sleep(gap)
            self._last = time.monotonic()
            try:
                return await self._chain.step(system_prompt, messages, tools)
            except LLMError as exc:
                if "429" in str(exc) and attempt < 3:  # free-tier quota: wait it out, then retry
                    await asyncio.sleep(25)
                    continue
                raise


# ---- setup ---------------------------------------------------------------------
async def build_store(settings: Settings, chain) -> VectorStore:
    persist = str(HERE / ".chroma")
    shutil.rmtree(persist, ignore_errors=True)
    store = VectorStore(persist, "eval_docs")
    for path in sorted((HERE / "docs").glob("*.txt")):
        doc_id, idx = new_document_id(), 0
        pieces = chunk_text(clean_text(path.read_text()), settings.chunk_size, settings.chunk_overlap)
        ids = [new_chunk_id(doc_id, i) for i in range(len(pieces))]
        metas = [{"document_id": doc_id, "document_name": path.name, "chunk_id": cid, "source": path.name} for cid in ids]
        store.add(ids=ids, embeddings=await embed_texts(chain._primary, pieces), documents=pieces, metadatas=metas)
    return store


# ---- scoring -------------------------------------------------------------------
def _calls(traces):
    return [t for t in traces if t.get("tool") and t["tool"] not in ("finish", "ask_user")]


def _flagged(entry) -> bool:
    """An intermediate step that went wrong: tool error/invalid args or an empty search."""
    return bool(entry.get("error")) or '"n_results": 0' in str(entry.get("result", ""))


def score(case: dict, responses: list, fault: str | None = None) -> dict:
    final = responses[-1]
    traces = [t for r in responses for t in (r.trace or [])]
    calls = _calls(traces)
    used = {c["tool"] for c in calls}
    tokens = sum((r.usage or {}).get("total_tokens", 0) for r in responses)
    steps = sum((r.usage or {}).get("steps", 0) for r in responses)
    year = str(datetime.now(timezone.utc).year)
    text = re.sub(r"[\u00a0\u202f\u2009]", " ", final.answer).replace("$", "")
    capped = any("step budget exhausted" in str(t.get("note", "")) for t in traces)
    hard = final.provider == "none" or capped
    abstained = final.confidence <= 0.4 or bool(ABSTAIN.search(final.answer))

    action = case.get("action")
    if fault:
        ok = not hard and not final.needs_clarification and abstained
    elif action == "ask_user":
        ok = final.needs_clarification
    elif action == "abstain":
        ok = not hard and not final.needs_clarification and abstained
    else:
        kws = [k.replace("{year}", year) for k in case.get("keywords_all", [])]
        ok = not hard and not final.needs_clarification and all(re.search(k, text, re.I) for k in kws)

    invalid = [c for c in calls if str(c.get("error", "")).startswith(("invalid arguments", "unknown tool"))]
    tools_ok = all(t in used for t in case.get("tools_all", [])) and (
        not case.get("tools_any") or bool(used & set(case["tools_any"]))
    )
    if fault:
        tools_ok = True  # tool selection is not what failure injection tests
    lo, hi = case["steps"]
    return {
        "id": case["id"], "fault": fault, "completed": ok, "tools_ok": tools_ok,
        "call_valid_rate": round(1 - len(invalid) / len(calls), 2) if calls else 1.0,
        "n_calls": len(calls), "steps": steps, "steps_ok": lo <= steps <= hi, "tokens": tokens,
        "confidence": final.confidence, "answer": final.answer[:160],
        "failure_recognized": any(t.get("error") for t in traces) if fault else None,
        "tools": [c["tool"] for c in calls], "hard": hard,
        "first_error": next((str(t["error"])[:200] for t in traces if t.get("error")), None),
        "early_flag": any(_flagged(t) for t in traces if t.get("tool") != "finish"),
    }


def classify(row: dict) -> str | None:
    """Failure taxonomy: hard = no usable answer (crash/timeout/step cap/model unreachable);
    cascading soft = an earlier bad step (tool error, invalid args, empty retrieval) propagated to a
    wrong or unsupported final answer; soft = wrong/unsupported answer or wrong tool with no earlier bad step."""
    if row["completed"] and row["tools_ok"]:
        return None
    if row["hard"]:
        return "hard"
    return "cascading soft" if row["early_flag"] else "soft"


async def run_case(runner: AgentRunner, case: dict, fault: str | None = None) -> dict:
    responses, sid = [], f"{case['id']}-{time.time_ns()}"
    try:
        for turn in case["turns"]:
            responses.append(await runner.run(turn, sid))
    except Exception as exc:  # noqa: BLE001 - harness must survive a crashing case
        row = {"id": case["id"], "fault": fault, "completed": False, "tools_ok": False, "hard": True,
               "early_flag": False, "steps": 0, "steps_ok": False, "tokens": 0, "n_calls": 0,
               "call_valid_rate": 0, "confidence": 0, "tools": [], "answer": f"CRASH {type(exc).__name__}: {exc}"}
        row["class"] = "hard"
        return row
    row = score(case, responses, fault)
    row["class"] = classify(row)
    return row


# ---- report --------------------------------------------------------------------
def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    return "\n".join(out)


def summarize(rows):
    n = len(rows) or 1
    return {
        "n": len(rows),
        "completion_rate": round(sum(r["completed"] for r in rows) / n, 2),
        "tools_ok_rate": round(sum(r["tools_ok"] for r in rows) / n, 2),
        "arg_valid_rate": round(sum(r["call_valid_rate"] for r in rows) / n, 2),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 1),
        "steps_in_range": round(sum(r["steps_ok"] for r in rows) / n, 2),
        "avg_tokens": round(sum(r["tokens"] for r in rows) / n),
        "total_tokens": sum(r["tokens"] for r in rows),
    }


def _agent_model() -> str:
    st = Settings()
    return f"groq/{st.groq_model}" if st.agent_provider == "groq" else f"gemini/{st.gemini_model}"


def report(results: dict) -> str:
    md = [f"# W16 agent evaluation\n\nAgent model: `{_agent_model()}` - generated {datetime.now():%Y-%m-%d %H:%M}\n"]
    summaries = {m: summarize(rows) for m, (rows, _) in results.items()}
    md.append("## Summary\n\n" + table([{"context_mode": m, **s} for m, s in summaries.items()], ["context_mode", *next(iter(summaries.values())).keys()]))
    for mode, (rows, inj) in results.items():
        md.append(f"\n## Per-case results - context mode `{mode}`\n\n" + table(rows, ["id", "completed", "tools_ok", "call_valid_rate", "steps", "steps_ok", "tokens", "confidence", "tools"]))
        fails = [r for r in rows if r["class"]]
        md.append(f"\n### Failure log (`{mode}`)\n\n" + (table([{**r, "answer": r["answer"][:100], "first_error": str(r.get("first_error"))[:80]} for r in fails], ["id", "class", "first_error", "tools", "confidence", "answer"]) if fails else "No failures."))
        if inj:
            md.append(f"\n### Failure injection (`{mode}`)\n\nPass = the agent recognized the failure and did not give a confident answer.\n\n"
                      + table([{**r, "pass": r["completed"]} for r in inj], ["id", "fault", "failure_recognized", "confidence", "pass", "steps", "tokens", "answer"]))
    if len(results) == 2:
        a, b = (summaries[m] for m in results)
        (ma, mb) = list(results)
        md.append(f"\n## Context-engineering effect\n\nAverage tokens per query: `{ma}` = {a['avg_tokens']}, `{mb}` = {b['avg_tokens']} "
                  f"(completion {a['completion_rate']} vs {b['completion_rate']}). Tokens are Gemini prompt+output tokens for the agent loop; embedding calls are not counted.")
    return "\n".join(md) + "\n"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", choices=["notes", "full", "both"], default="notes")
    ap.add_argument("--inject", choices=["none", "all", *FAULTS], default="all")
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--delay", type=float, default=4.5, help="seconds between model calls (free-tier pacing)")
    args = ap.parse_args()

    cases = json.loads((HERE / "cases.json").read_text())
    if args.only:
        cases = [c for c in cases if c["id"] in args.only.split(",")]
    settings = Settings()
    settings.request_timeout_seconds = 120  # PacedChain may wait out a 429 inside a step
    chain = PacedChain(build_fallback_chain(settings), args.delay)
    store = await build_store(settings, chain)
    modes = ["notes", "full"] if args.context == "both" else [args.context]
    faults = [] if args.inject == "none" else list(FAULTS) if args.inject == "all" else [args.inject]

    results = {}
    for mode in modes:
        rows, inj = [], []
        for case in cases:
            rows.append(await run_case(AgentRunner(settings, chain, store, context_mode=mode), case))
            print(f"[{mode}] {case['id']}: completed={rows[-1]['completed']} steps={rows[-1]['steps']} tokens={rows[-1]['tokens']}", flush=True)
        for fault in faults:
            for case in (c for c in cases if c["id"] in INJECT_CASES):
                runner = AgentRunner(settings, chain, store, context_mode=mode, tool_hook=FAULTS[fault])
                inj.append(await run_case(runner, case, fault))
                print(f"[{mode}] inject {fault} on {case['id']}: pass={inj[-1]['completed']} conf={inj[-1]['confidence']}", flush=True)
        results[mode] = (rows, inj)
        # Save after every mode so a quota failure in a later mode cannot lose finished work.
        (HERE / "results.md").write_text(report(results))
        (HERE / "results.json").write_text(json.dumps({m: {"cases": r, "injection": i} for m, (r, i) in results.items()}, indent=1, default=str))
        print(f"wrote eval/results.md (through mode {mode})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
