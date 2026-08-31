"""
benchmark_runner.py
Task 3 (second half): formulates a single comprehensive prompt asking the
agent to autonomously evaluate 3 algorithms across 2 datasets, perform
cross-validation for each, and produce a Markdown experimental summary
table as its Final Answer.

Usage:
    python benchmark_runner.py
"""

import os
import datetime

from react_agent import run_agent_loop, LOG_DIR

DEFAULT_ALGORITHMS = ["decision_tree", "random_forest", "logistic_regression"]
DEFAULT_DATASETS = ["wine", "breast_cancer"]


def build_benchmark_prompt(algorithms=None, datasets=None) -> str:
    algorithms = algorithms or DEFAULT_ALGORITHMS
    datasets = datasets or DEFAULT_DATASETS
    return (
        f"Autonomously evaluate these {len(algorithms)} algorithms: {', '.join(algorithms)} "
        f"across these {len(datasets)} datasets: {', '.join(datasets)}. "
        "For each (algorithm, dataset) pair, call the appropriate tool and record its test "
        "accuracy, 5-fold cross-validation mean accuracy, and CV standard deviation. "
        "After gathering all results, write your Final Answer as a Markdown table with columns: "
        "Algorithm | Dataset | Test Accuracy | CV Mean Accuracy | CV Std, followed by 2-3 "
        "sentences recommending the best model per dataset and discussing the bias/variance "
        "trade-off you observed across the results."
    )


def main():
    prompt = build_benchmark_prompt()
    # 3 algorithms x 2 datasets = 6 tool calls minimum, plus the Final Answer
    # step and headroom for self-correction retries.
    result = run_agent_loop(prompt, max_iterations=12)

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(LOG_DIR, f"benchmark_{ts}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark Result ({ts})\n\n")
        f.write(result)

    print(f"\n[benchmark summary saved to {out_path}]")


if __name__ == "__main__":
    main()
