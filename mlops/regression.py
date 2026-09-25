"""Evidently regression Test Suite: LLM-as-judge comparison of new answers against golden answers.

Two judge checks per test case:
  1. correctness  - reference-based: does the answer contradict or lose information from the golden answer?
  2. no_fabrication - does the answer avoid stating specific facts that are absent from the golden answer?
A test case passes only if BOTH checks pass. `pct_tests_passed` = share of cases passing both.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
# Judge backend: "groq" (Qwen via Groq's OpenAI-compatible API) or "gemini". Both are a different model family
# from the gpt-oss agent, which limits self-preference bias. Free-tier quotas decide which one is usable.
BACKEND = os.getenv("JUDGE_BACKEND", "groq")
RPM = int(os.getenv("JUDGE_RPM", "4"))
if BACKEND == "gemini":
    JUDGE_PROVIDER, JUDGE_MODEL = "gemini", os.getenv("JUDGE_MODEL", "gemini-3.6-flash")
else:
    JUDGE_PROVIDER, JUDGE_MODEL = "openai", os.getenv("JUDGE_MODEL", "qwen/qwen3.8-27b")


def _judge_options():
    from evidently.llm.utils.wrapper import GeminiOptions, OpenAIKey, RateLimits
    limits = RateLimits(rpm=RPM)
    if BACKEND == "gemini":
        return [GeminiOptions(api_key=os.environ["GEMINI_API_KEY"], limits=limits)]
    os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"  # openai client reads this
    return [OpenAIKey(api_key=os.environ["GROQ_API_KEY"], limits=limits)]


def load_regression_set() -> list[dict]:
    return json.loads((HERE / "regression_set.json").read_text())


def run_regression(answers: dict[str, str], html_path: Path) -> dict:
    """answers: case id -> the agent's final answer. Returns metrics + per-case verdicts."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.descriptors import LLMEval
    from evidently.llm.templates import BinaryClassificationPromptTemplate
    from evidently.presets import TextEvals
    from evidently.tests import eq

    cases = load_regression_set()
    df = pd.DataFrame({
        "id": [c["id"] for c in cases],
        "question": [c["query"] for c in cases],
        "reference": [c["golden"] for c in cases],
        "answer": [answers.get(c["id"], "") for c in cases],
    })

    correctness = BinaryClassificationPromptTemplate(
        criteria=(
            "You compare a NEW ANSWER to a REFERENCE answer approved by a human. The answer is CORRECT if it "
            "conveys the same facts as the reference (same numbers, same recommendation, same abstention or "
            "clarifying-question behaviour) and does not contradict it or drop a key fact. Wording may differ. "
            "It is INCORRECT if a key number/decision differs, it gives up or says it could not finish when the "
            "reference has an answer, or it answers confidently where the reference says the answer is unavailable "
            "or a clarification is needed.\n"
            "REFERENCE:\n{reference}\nQUESTION:\n{question}"
        ),
        target_category="correct", non_target_category="incorrect",
        include_reasoning=True, uncertainty="unknown",
    )
    fabrication = BinaryClassificationPromptTemplate(
        criteria=(
            "Check the NEW ANSWER for FABRICATION. It is CLEAN if every specific fact it states (prices, limits, "
            "dates, policies) is consistent with the REFERENCE or is a plain arithmetic consequence of it. It is "
            "FABRICATED if it states a specific fact that the REFERENCE does not support or that contradicts it. "
            "Admitting inability to answer is CLEAN.\n"
            "REFERENCE:\n{reference}\nQUESTION:\n{question}"
        ),
        target_category="clean", non_target_category="fabricated",
        include_reasoning=True, uncertainty="unknown",
    )
    descriptors = [
        LLMEval("answer", template=correctness, provider=JUDGE_PROVIDER, model=JUDGE_MODEL,
                additional_columns={"reference": "reference", "question": "question"},
                alias="correctness", tests=[eq("correct", column="correctness", alias="correctness_test")]),
        LLMEval("answer", template=fabrication, provider=JUDGE_PROVIDER, model=JUDGE_MODEL,
                additional_columns={"reference": "reference", "question": "question"},
                alias="no_fabrication", tests=[eq("clean", column="no_fabrication", alias="no_fabrication_test")]),
    ]
    definition = DataDefinition(text_columns=["answer", "question", "reference"], id_column=None)
    # Free-tier friendly pacing of judge calls.
    options = _judge_options()
    dataset = Dataset.from_pandas(df, data_definition=definition, descriptors=descriptors, options=options)
    snapshot = Report([TextEvals()], include_tests=True).run(dataset, None)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))

    out = dataset.as_dataframe()
    c_cat, f_cat = "correctness", "no_fabrication"
    c_rsn, f_rsn = "correctness reasoning", "no_fabrication reasoning"
    verdicts = []
    for i, r in out.iterrows():
        ok_c, ok_f = r[c_cat] == "correct", r[f_cat] == "clean"
        verdicts.append({"id": df.loc[i, "id"], "correctness": r[c_cat], "no_fabrication": r[f_cat],
                         "passed": bool(ok_c and ok_f), "correctness_reason": r[c_rsn], "fabrication_reason": r[f_rsn],
                         "answer": df.loc[i, "answer"]})
    n = len(verdicts)
    metrics = {
        "pct_tests_passed": round(100 * sum(v["passed"] for v in verdicts) / n, 1),
        "pct_correct": round(100 * sum(v["correctness"] == "correct" for v in verdicts) / n, 1),
        "pct_no_fabrication": round(100 * sum(v["no_fabrication"] == "clean" for v in verdicts) / n, 1),
        "n_tests": n, "n_failed": sum(not v["passed"] for v in verdicts),
    }
    return {"metrics": metrics, "verdicts": verdicts}
