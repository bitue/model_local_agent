"""
react_agent.py
A from-scratch ReAct (Reason + Act) loop: a local LLM served by Ollama
autonomously reasons about which tool (from ml_tools.py) to call, calls it,
reads the result back as an Observation, and repeats until it emits a
Final Answer.

Usage:
    python react_agent.py "your query here"      # one-shot
    python react_agent.py                         # interactive REPL

Environment variables (set automatically by docker-compose.yml):
    OLLAMA_URL   REST endpoint, default http://127.0.0.1:11434/api/generate
    MODEL_NAME   Ollama model tag, default llama3.2:3b
    LOG_DIR      directory execution traces are written to, default 'logs'
"""

import os
import re
import sys
import json
import datetime

import requests

from ml_tools import AVAILABLE_TOOLS

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2:3b")
LOG_DIR = os.environ.get("LOG_DIR", "logs")
# CPU-only inference slows down noticeably as the ReAct prompt accumulates
# Observations across steps, so give later turns a generous ceiling.
LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", "900"))
# The 3B quantized model sometimes re-verifies information it already has
# (re-fetching a summary, an unrequested PCA pass) before it actually
# concludes, so one-shot CLI queries get a more generous default than the
# library default of 8 used when run_agent_loop() is called directly.
DEFAULT_MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "8"))

SYSTEM_PROMPT = """You are an expert Autonomous Machine Learning Assistant.
You solve machine learning problems by thinking step-by-step and invoking external tools.

You have access to the following tools:
1. load_dataset_summary(dataset_name: str) -> JSON summary of a dataset (iris, wine, breast_cancer).
2. train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2) -> JSON test/CV scores.
   model_type: decision_tree, logistic_regression, random_forest.
3. train_pytorch_mlp(dataset_name: str, hidden_dim: int = 32, epochs: int = 50, lr: float = 0.01) -> JSON
   PyTorch training/evaluation results.
4. tune_hyperparameters(dataset_name: str, model_type: str, search_type: str = "grid", cv: int = 5) -> JSON
   best params + best CV score. model_type: svc, decision_tree. search_type: grid, random.
5. reduce_dimensionality(dataset_name: str, method: str = "pca", n_components: int = 2) -> JSON explained
   variance ratio (pca) or selected feature names (sequential). method: pca, sequential.
6. train_deep_classifier(dataset_name: str, hidden_dims: list = [64, 32], dropout: float = 0.3,
   use_batchnorm: bool = true, epochs: int = 50, lr: float = 0.01, scheduler: str = "steplr") -> JSON
   training/evaluation results for a regularized deep MLP. scheduler: steplr, plateau.

To use a tool, you MUST strictly use this format, and then STOP -- do not write an Observation
yourself, do not write more than one Action per turn, and do not guess what the result will be.
The real tool result will be given back to you as an Observation before your next turn:
Thought: Describe your reasoning about what to do next.
Action: <tool_name>
Action Input: {"param_name": "value"}

If an Observation reports an error (bad parameter, shape mismatch, or a NaN loss), do NOT repeat
the same Action Input. Read the error message, reason about what went wrong, and retry the same
tool with corrected parameters (e.g. a smaller learning rate, a valid model_type, or a valid
n_components).

Do NOT write "Final Answer" until you have a real Observation for every tool call the task
requires. Writing "Final Answer" and then continuing to write more Action/Action Input text is
invalid and will be rejected -- if the task has multiple parts, finish gathering every
Observation first, THEN write exactly one "Final Answer" and stop.

When you have received the observation(s) and are ready to give the complete answer to the user,
format your output as:
Thought: I have gathered all necessary experimental data.
Final Answer: <your complete answer, including any requested tables or comparisons>

Example of ONE correct turn (notice it stops right after Action Input -- it never writes
"Observation:" itself and never fabricates a Final Answer before real data comes back):
Thought: I need the shape and class balance of the wine dataset before choosing a model.
Action: load_dataset_summary
Action Input: {"dataset_name": "wine"}

Begin!
"""


def query_local_llm(prompt: str) -> str:
    """Queries the local Ollama instance running the configured model."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "stop": ["Observation:"],
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama error: {response.text}")
    return response.json().get("response", "")


def _next_log_path() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    existing = [f for f in os.listdir(LOG_DIR) if f.startswith("run_") and f.endswith(".log")]
    return os.path.join(LOG_DIR, f"run_{len(existing) + 1:03d}.log")


def run_agent_loop(user_query: str, max_iterations: int = 8) -> str:
    """Runs the Thought/Action/Action Input/Observation loop until a Final
    Answer is produced or max_iterations is exhausted. Returns the final
    answer text (or the full trace if no Final Answer was reached), and
    writes the full trace to a log file for the execution-log deliverable."""
    lines = []

    def emit(text: str):
        print(text)
        lines.append(text)

    emit("=" * 70)
    emit(f"USER QUERY: {user_query}")
    emit(f"TIMESTAMP: {datetime.datetime.now().isoformat()}")
    emit("=" * 70)

    prompt = f"{SYSTEM_PROMPT}\nUser Query: {user_query}\n"
    final_answer = None
    # Repeat-call guard: as the prompt grows across steps, the small
    # quantized model can lose track of what it already did and re-issue
    # the *identical* Action/Action Input turn after turn instead of
    # progressing -- left unchecked this both wastes the iteration budget
    # and keeps re-appending duplicate Observations, ballooning the prompt
    # until Ollama's memory use gets the container OOM-killed. We track the
    # last executed call and short-circuit an exact repeat with a nudge
    # instead of re-running the tool, aborting outright after a few in a row.
    last_call = None
    repeat_count = 0

    for step in range(1, max_iterations + 1):
        emit(f"\n--- Step {step} ---")
        try:
            raw_output = query_local_llm(prompt)
        except requests.exceptions.RequestException as e:
            # Transient network/timeout failure talking to Ollama itself
            # (not a tool error) -- retry once before giving up on this run,
            # rather than letting the whole script crash mid-trace.
            emit(f"\n>>> LLM request failed ({e}); retrying once...")
            try:
                raw_output = query_local_llm(prompt)
            except requests.exceptions.RequestException as e2:
                emit(f"\n>>> LLM request failed again ({e2}); aborting this run.")
                break

        action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", raw_output)
        input_match = re.search(r"Action Input:\s*(\{.*?\})", raw_output, re.DOTALL)
        final_idx = raw_output.find("Final Answer:")
        has_action = bool(action_match and input_match)

        # The small quantized model sometimes ignores the "one Action per
        # turn, then stop" instruction and hallucinates the rest of the
        # dialogue in a single completion. This shows up two ways: (a) a
        # fabricated Final Answer *after* announcing an Action ("I'll call
        # tool X... Final Answer: <fake result>"), or (b) a premature Final
        # Answer *before* describing Actions it still intends to run
        # ("Final Answer: ... First I will call tool X ... Then tool Y...").
        # In both cases a real, un-executed Action is present somewhere in
        # the text -- the strict format never legitimately mixes the two, so
        # whenever ANY Action is attempted it always wins over any Final
        # Answer text. We truncate everything after that first Action Input
        # (including the premature/fabricated Final Answer) so a REAL
        # Observation replaces it on the next turn instead of letting the
        # agent "coast" on fabricated or merely-announced results.
        if has_action:
            llm_output = raw_output[:input_match.end()]
        else:
            llm_output = raw_output

        emit(llm_output)
        prompt += llm_output

        if final_idx != -1 and not has_action:
            emit("\n>>> Task Completed Successfully!")
            final_answer = llm_output.split("Final Answer:", 1)[1].strip()
            break

        if has_action:
            tool_name = action_match.group(1).strip()
            raw_input = input_match.group(1).strip()

            call_key = (tool_name, raw_input)
            repeat_count = repeat_count + 1 if call_key == last_call else 0
            last_call = call_key

            if repeat_count >= 3:
                emit("\n>>> Aborting: agent repeated the same Action Input 3 turns in a row "
                     "without progressing.")
                break

            if repeat_count >= 1:
                observation = (
                    f"\nObservation: You already called {tool_name} with this exact input in "
                    f"the previous step and already have that result. Do not repeat it -- use "
                    f"what you already learned and move on to the next tool this task still "
                    f"needs, or write your Final Answer if you have everything.\n"
                )
                emit(observation)
                prompt += observation
                continue

            try:
                kwargs = json.loads(raw_input)
            except json.JSONDecodeError:
                observation = "\nObservation: Error parsing Action Input as JSON. Ensure it is valid JSON.\n"
                emit(observation)
                prompt += observation
                continue

            if tool_name in AVAILABLE_TOOLS:
                try:
                    tool_res = AVAILABLE_TOOLS[tool_name](**kwargs)
                    observation = f"\nObservation: {tool_res}\n"
                except Exception as e:
                    # Self-healing (Task 3): the LLM sees *what* failed and is
                    # explicitly told to retry with corrected parameters.
                    observation = (
                        f"\nObservation: Tool execution error: {str(e)}. "
                        f"Reconsider your parameters and retry the same tool.\n"
                    )
            else:
                observation = (
                    f"\nObservation: Tool '{tool_name}' not recognized. "
                    f"Valid tools: {list(AVAILABLE_TOOLS.keys())}.\n"
                )

            emit(observation)
            prompt += observation
        else:
            reminder = "\nObservation: Please respond with a Thought + Action + Action Input, or a Final Answer.\n"
            emit(reminder)
            prompt += reminder
    else:
        emit("\n>>> Max iterations reached without a Final Answer.")

    log_file = _next_log_path()
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[log saved to {log_file}]")

    return final_answer or "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_agent_loop(query, max_iterations=DEFAULT_MAX_ITERATIONS)
        return

    print("Autonomous ML Agent -- interactive mode. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            query = input(">>> ").strip()
        except EOFError:
            break
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        run_agent_loop(query, max_iterations=DEFAULT_MAX_ITERATIONS)


if __name__ == "__main__":
    main()
