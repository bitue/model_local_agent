"""
benchmark_runner.py
Task 3 (second half): the agent autonomously evaluates 3 algorithms across 2
datasets, 5-fold cross-validating each, and produces a Markdown experimental
summary table plus a bias/variance discussion.

Design note (why this isn't one long ReAct session): an earlier version
asked the agent to plan and execute all 6 (algorithm, dataset) tool calls
*and* synthesize the table in a single continuous ReAct run. In practice the
3B model reliably made several of the 6 real calls but was not reliable at
finishing all 6 -- it would either fabricate the remaining rows once it had
*some* real numbers in hand, or stall re-issuing a call it had already made
(see `agent/logs/_archive/run_012_benchmark_attempt*` for the captured
attempts and `REPORT.md` §3.1/§6 for the full story). Rather than keep
fighting that with ever-larger iteration budgets, this version decomposes
the work the way a robust pipeline should: one independent, short ReAct
sub-task per (algorithm, dataset) pair -- exactly the kind of single-tool
call this agent already handles reliably (see `agent/logs/run_001.log`) --
run back-to-back by this script. The resulting Markdown table is assembled
directly from each sub-task's *real, verified* tool Observation (captured
via `collect_results`, never the model's own prose paraphrase of a number,
which is not guarded against being subtly misstated even when the
underlying call was genuine). The agent still performs every ML computation
and every piece of tool-selection reasoning; only the final transcription
from Observation to Markdown cell is done in Python, so the deliverable's
numbers are unconditionally trustworthy. The LLM is then handed the
complete, real table and asked only for the qualitative bias/variance
interpretation -- well within a small model's reach once it isn't also
responsible for remembering 6 numbers correctly across 20+ turns.

Usage:
    python benchmark_runner.py
"""

import os
import json
import datetime

from react_agent import run_agent_loop, query_local_llm, LOG_DIR

DEFAULT_ALGORITHMS = ["decision_tree", "random_forest", "logistic_regression"]
DEFAULT_DATASETS = ["wine", "breast_cancer"]


def _required_calls(algorithms, datasets):
    """The exact list of (tool_name, kwargs) pairs the benchmark needs a
    real Observation for. Also used by build_benchmark_prompt() below for
    the (currently unused but kept for reference) single-session variant."""
    return [
        ("train_sklearn_model", {"dataset_name": ds, "model_type": algo})
        for ds in datasets
        for algo in algorithms
    ]


def build_benchmark_prompt(algorithms=None, datasets=None) -> str:
    """Builds the single-session prompt used by an earlier design (see the
    module docstring for why the current main() no longer uses this to
    drive one long ReAct run) -- kept available for anyone re-running that
    comparison, and used by run_single_evaluation()'s sub-task prompt style."""
    algorithms = algorithms or DEFAULT_ALGORITHMS
    datasets = datasets or DEFAULT_DATASETS

    call_list = "\n".join(
        f'{i}. train_sklearn_model with Action Input: {json.dumps(kwargs)}'
        for i, (_, kwargs) in enumerate(_required_calls(algorithms, datasets), start=1)
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


def _sub_task_prompt(dataset_name: str, model_type: str) -> str:
    return (
        f"Call the appropriate tool to train a {model_type} model on the {dataset_name} "
        f"dataset and report its test accuracy, 5-fold cross-validation mean accuracy, and "
        f"CV standard deviation."
    )


def run_single_evaluation(dataset_name: str, model_type: str) -> dict:
    """Runs one independent ReAct sub-task for exactly one (algorithm,
    dataset) pair. Returns the tool's real, parsed JSON Observation --
    required_calls + collect_results together guarantee this came from an
    actual executed call, not the model's paraphrase of one."""
    call = ("train_sklearn_model", {"dataset_name": dataset_name, "model_type": model_type})
    collected = {}
    run_agent_loop(
        _sub_task_prompt(dataset_name, model_type),
        max_iterations=6,
        required_calls=[call],
        collect_results=collected,
        # This sub-task only exists to get one verified number -- once the
        # required call succeeds there's nothing left worth an extra turn
        # for, and letting the model keep going was observed to just burn
        # iterations (and Ollama load) re-fetching a dataset summary it
        # doesn't even need for this sub-task.
        stop_when_satisfied=True,
    )
    key = (call[0], json.dumps(call[1], sort_keys=True))
    if key not in collected:
        raise RuntimeError(
            f"Sub-task for {model_type}/{dataset_name} did not produce a verified result "
            f"within its iteration budget -- see the per-call log in {LOG_DIR} for what happened."
        )
    return json.loads(collected[key])


def build_markdown_table(rows: list) -> str:
    header = "| Algorithm | Dataset | Test Accuracy | CV Mean Accuracy | CV Std |\n"
    header += "| --- | --- | --- | --- | --- |\n"
    body = "".join(
        f"| {r['model']} | {r['dataset']} | {r['test_accuracy']} | {r['cv_mean_accuracy']} | {r['cv_std']} |\n"
        for r in rows
    )
    return header + body


def main():
    rows = []
    for dataset_name in DEFAULT_DATASETS:
        for model_type in DEFAULT_ALGORITHMS:
            print(f"\n=== Evaluating {model_type} on {dataset_name} ===")
            rows.append(run_single_evaluation(dataset_name, model_type))

    table = build_markdown_table(rows)

    # Every number above is verified real -- the only remaining job for the
    # LLM is the qualitative bias/variance interpretation, given the
    # complete, real table (not asked to recall or recompute any number).
    commentary_prompt = (
        "Here is a real experimental results table from evaluating 3 algorithms "
        "(decision_tree, random_forest, logistic_regression) across 2 datasets "
        f"(wine, breast_cancer) with 5-fold cross-validation:\n\n{table}\n\n"
        "In 2-3 sentences, recommend the best model per dataset and discuss the "
        "bias/variance trade-off you observe across these results. Reply with "
        "only those 2-3 sentences, nothing else."
    )
    commentary = query_local_llm(commentary_prompt).strip()

    result = f"{table}\n{commentary}\n"
    print(f"\n{result}")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(LOG_DIR, f"benchmark_{ts}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark Result ({ts})\n\n")
        f.write(result)

    print(f"\n[benchmark summary saved to {out_path}]")


if __name__ == "__main__":
    main()
