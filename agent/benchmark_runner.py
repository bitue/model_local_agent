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

    # Spell out the exact required calls as a checklist rather than leaving
    # the (algorithm x dataset) cross-product for the model to derive itself
    # turn by turn. The small 3B model was observed to re-derive ("I need to
    # gather the dataset summaries first...") and re-run its own plan from
    # scratch on every turn of an open-ended multi-dataset prompt instead of
    # advancing through it -- an explicit checklist gives it a fixed list of
    # remaining work items to check off instead of a plan to keep re-deriving.
    call_list = "\n".join(
        f'{i}. train_sklearn_model with Action Input: {{"dataset_name": "{ds}", "model_type": "{algo}"}}'
        for i, (algo, ds) in enumerate(
            ((algo, ds) for ds in datasets for algo in algorithms), start=1
        )
    )

    return (
        f"Autonomously evaluate these {len(algorithms)} algorithms: {', '.join(algorithms)} "
        f"across these {len(datasets)} datasets: {', '.join(datasets)}.\n\n"
        f"You must make exactly these {len(algorithms) * len(datasets)} train_sklearn_model calls, "
        f"each one time, in any order (do not call load_dataset_summary -- it is not needed here, "
        f"train_sklearn_model already reports everything you need):\n"
        f"{call_list}\n\n"
        "For each call, record its test accuracy, 5-fold cross-validation mean accuracy, and CV "
        f"standard deviation from the Observation. Once you have all {len(algorithms) * len(datasets)} "
        "results (and not before), write your Final Answer as a Markdown table with columns: "
        "Algorithm | Dataset | Test Accuracy | CV Mean Accuracy | CV Std, followed by 2-3 "
        "sentences recommending the best model per dataset and discussing the bias/variance "
        "trade-off you observed across the results."
    )


def main():
    prompt = build_benchmark_prompt()
    # 3 algorithms x 2 datasets = 6 tool calls minimum, plus the Final Answer
    # step and headroom for self-correction retries -- the 3B model has been
    # observed to need a couple of nudged retries per dataset before it
    # stops re-fetching a summary it already has, so this budget covers 6
    # real calls + up to ~3 wasted nudge turns per dataset + the final step.
    result = run_agent_loop(prompt, max_iterations=16)

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(LOG_DIR, f"benchmark_{ts}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark Result ({ts})\n\n")
        f.write(result)

    print(f"\n[benchmark summary saved to {out_path}]")


if __name__ == "__main__":
    main()
