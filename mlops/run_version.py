"""Run one prompt/config version end to end: agent eval -> traces -> Evidently regression -> MLflow.

  uv run python -m mlops.run_version v1                 # full run
  uv run python -m mlops.run_version v2 --skip-judge    # agent + traces only
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.assistant.agent import AgentRunner  # noqa: E402
from backend.config import Settings  # noqa: E402
from backend.llm.factory import build_fallback_chain  # noqa: E402
from eval.run_eval import (PacedChain, _agent_model, build_store, classify, score, summarize)  # noqa: E402
from mlops.regression import JUDGE_MODEL, run_regression  # noqa: E402

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT = "agent-prompt-versions"
RUNS = ROOT / "mlops" / "runs"


def termination_reason(responses) -> str:
    """Why the loop ended, derived from the final trace step."""
    final = responses[-1]
    last = (final.trace or [{}])[-1]
    if final.provider == "none":
        return "model_call_failed"
    if "step budget exhausted" in str(last.get("note", "")):
        return "max_steps_hit"
    if final.needs_clarification:
        return "ask_user"
    if last.get("tool") == "finish":
        return "finish"
    return "prose_answer_no_finish"


async def run_agent(cfg: dict, delay: float) -> list[dict]:
    settings = Settings()
    settings.request_timeout_seconds = 120
    settings.temperature = cfg.get("temperature", settings.temperature)
    settings.groq_model = cfg.get("agent_model", settings.groq_model)
    chain = PacedChain(build_fallback_chain(settings), delay)
    store = await build_store(settings, chain)
    prompt = (ROOT / cfg["prompt_file"]).read_text()
    cases = json.loads((ROOT / "eval" / "cases.json").read_text())
    records = []
    for case in cases:
        runner = AgentRunner(settings, chain, store, max_steps=cfg["max_steps"],
                             context_mode=cfg["context_mode"], system_prompt=prompt)
        responses, sid = [], f"{case['id']}-{len(records)}"
        try:
            for turn in case["turns"]:
                responses.append(await runner.run(turn, sid))
        except Exception as exc:  # noqa: BLE001
            print(f"  {case['id']}: CRASH {exc}")
            continue
        row = score(case, responses)
        row["class"] = classify(row)
        trace = [dict(t, turn=i + 1) for i, r in enumerate(responses) for t in (r.trace or [])]
        records.append({"row": row, "case": case, "trace": trace, "final_answer": responses[-1].answer,
                        "iterations": sum((r.usage or {}).get("steps", 0) for r in responses),
                        "termination": termination_reason(responses)})
        print(f"  {case['id']}: completed={row['completed']} steps={row['steps']} term={records[-1]['termination']}", flush=True)
    return records


def pick_traces(records: list[dict]) -> list[dict]:
    """1 clean success + up to 2 failures (worst first)."""
    ok = [r for r in records if not r["row"]["class"]]
    bad = sorted((r for r in records if r["row"]["class"]), key=lambda r: r["row"]["class"] != "hard")
    return ok[:1] + bad[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--delay", type=float, default=4.5)
    ap.add_argument("--from-saved", action="store_true",
                    help="reuse mlops/runs/<version>/results.json instead of re-running the agent")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "mlops" / "versions.json").read_text())[args.version]
    out = RUNS / args.version
    out.mkdir(parents=True, exist_ok=True)

    if args.from_saved:
        records = json.loads((out / "results.json").read_text())
    else:
        records = asyncio.run(run_agent(cfg, args.delay))
    rows = [r["row"] for r in records]
    (out / "results.json").write_text(json.dumps(records, indent=1, default=str))
    summary = summarize(rows)
    terms = {}
    for r in records:
        terms[r["termination"]] = terms.get(r["termination"], 0) + 1
    print("summary:", summary, "\ntermination:", terms)

    regression = None
    if not args.skip_judge:
        answers = {r["case"]["id"]: r["final_answer"] for r in records}
        regression = run_regression(answers, out / "evidently_report.html")
        (out / "regression.json").write_text(json.dumps(regression, indent=1, default=str))
        print("regression:", regression["metrics"])

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    st = Settings()
    prompt_path = ROOT / cfg["prompt_file"]
    with mlflow.start_run(run_name=f"agent_{args.version}"):
        mlflow.log_params({
            "prompt_version": args.version, "prompt_file": cfg["prompt_file"],
            "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest()[:12],
            "prompt_chars": len(prompt_path.read_text()), "max_steps": cfg["max_steps"],
            "context_mode": cfg["context_mode"], "agent_model": f"groq/{cfg.get('agent_model', st.groq_model)}",
            "temperature": cfg.get("temperature", st.temperature), "retrieval_top_k": 3,
            "judge_model": JUDGE_MODEL, "provider_fix": cfg.get("provider_fix", "none"),
        })
        mlflow.set_tag("description", cfg["description"])
        mlflow.log_metrics({f"harness_{k}": v for k, v in summary.items()})
        mlflow.log_metrics({f"term_{k}": v for k, v in terms.items()})
        mlflow.log_metric("harness_hard_failures", sum(r["class"] == "hard" for r in rows))
        if regression:
            mlflow.log_metrics(regression["metrics"])
        mlflow.log_artifact(str(prompt_path), artifact_path="prompt")
        mlflow.log_artifact(str(out / "results.json"), artifact_path="harness")
        for r in pick_traces(records):
            tag = "success" if not r["row"]["class"] else f"failure_{r['row']['class'].replace(' ', '_')}"
            mlflow.log_dict({
                "case": r["case"]["id"], "query_turns": r["case"]["turns"], "outcome": tag,
                "iterations": r["iterations"], "termination_reason": r["termination"],
                "final_answer": r["final_answer"], "steps": r["trace"],
            }, f"traces/{tag}_{r['case']['id']}.json")
        if regression:
            mlflow.log_artifact(str(out / "regression.json"), artifact_path="evidently")
            mlflow.log_artifact(str(out / "evidently_report.html"), artifact_path="evidently")
    print("logged to MLflow")


if __name__ == "__main__":
    main()
