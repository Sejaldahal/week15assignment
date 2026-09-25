"""Export the MLflow run comparison (all prompt versions side by side) to mlops/run_comparison.md."""
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
exp = mlflow.get_experiment_by_name("agent-prompt-versions")
runs = mlflow.search_runs([exp.experiment_id]).sort_values("params.prompt_version")
cols = {
    "params.prompt_version": "version", "params.agent_model": "agent model", "params.max_steps": "max_steps",
    "metrics.harness_completion_rate": "completion", "metrics.harness_tools_ok_rate": "tools_ok",
    "metrics.harness_avg_steps": "avg_steps", "metrics.harness_avg_tokens": "avg_tokens",
    "metrics.harness_hard_failures": "hard_fail", "metrics.term_model_call_failed": "model_call_failed",
    "metrics.term_max_steps_hit": "max_steps_hit", "metrics.pct_correct": "pct_correct",
    "metrics.pct_no_fabrication": "pct_no_fabrication", "metrics.pct_tests_passed": "pct_tests_passed",
}
table = runs[list(cols)].rename(columns=cols).fillna(0)
out = ROOT / "mlops" / "run_comparison.md"
out.write_text(table.round(2).to_markdown(index=False) + "\n")
print(out.read_text())
