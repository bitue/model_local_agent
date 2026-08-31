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
import time
import json
import inspect
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
# (re-fetching a summary, an unrequested PCA pass) or blends two tools'
# parameter names together before it actually concludes, so one-shot CLI
# queries get a more generous default than the library default of 8 used
# when run_agent_loop() is called directly.
DEFAULT_MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "10"))

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

Only call one of the 6 tools listed above -- never invent a tool name that is not in that list.

Before calling a tool, check whether you already received an Observation that answers the exact
same question (e.g. you already have the dataset summary, or you already trained that exact
model). If you already have the information, do NOT call that tool again -- reuse the numbers
you already have and move on to the next distinct step the task still needs.

Do NOT write "Final Answer" until you have a real Observation for every tool call the task
requires. Writing "Final Answer" and then continuing to write more Action/Action Input text is
invalid and will be rejected -- if the task has multiple parts, finish gathering every
Observation first, THEN write exactly one "Final Answer" and stop. As soon as you have every
Observation the task needs, write the Final Answer immediately -- do not perform extra
confirmatory or repeated tool calls "just to be sure".

When you have received the observation(s) and are ready to give the complete answer to the user,
format your output as:
Thought: I have gathered all necessary experimental data.
Final Answer: <your complete answer, including any requested tables or comparisons>

Example showing how you PROGRESS across turns (illustrative only -- this whole block, including
its Observations, is just a demonstration of the pattern; it is not part of the real
conversation, and the real Observation text will differ). Notice turn 1 stops right after its
Action Input -- it never writes "Observation:" itself and never fabricates a Final Answer before
real data comes back -- and notice turn 2 does NOT repeat turn 1's action, it moves on to a new
tool now that it has the summary it needed:

Thought: I need the shape and class balance of the wine dataset before choosing a model.
Action: load_dataset_summary
Action Input: {"dataset_name": "wine"}

Observation: {"dataset": "wine", "n_samples": 178, "n_features": 13, "classes": ["0", "1", "2"], "missing_values": 0}

Thought: I already have the dataset summary above -- calling it again would teach me nothing new.
Now that I know the shape and class balance, I will train a model on it.
Action: train_sklearn_model
Action Input: {"dataset_name": "wine", "model_type": "random_forest"}

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
            # Bound a single turn's generation. Without this the model can
            # ramble on unbounded once it drifts off the strict Thought/
            # Action format (observed producing 200+ token free-form
            # responses on this host, which twice coincided with the local
            # Ollama server itself crashing under CPU/memory pressure mid-
            # generation) -- one real turn never legitimately needs more
            # than a few hundred tokens.
            "num_predict": 400,
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama error: {response.text}")
    return response.json().get("response", "")


def _next_log_path() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    # Use the highest existing run number + 1, not a plain count -- once any
    # earlier run is archived out of LOG_DIR (e.g. into logs/_archive/), a
    # count-based name would collide with/reuse an already-archived number.
    existing_nums = [
        int(m.group(1))
        for f in os.listdir(LOG_DIR)
        if (m := re.match(r"run_(\d+)\.log$", f))
    ]
    next_num = max(existing_nums, default=0) + 1
    return os.path.join(LOG_DIR, f"run_{next_num:03d}.log")


def run_agent_loop(
    user_query: str,
    max_iterations: int = 8,
    required_calls=None,
    collect_results=None,
    stop_when_satisfied: bool = False,
) -> str:
    """Runs the Thought/Action/Action Input/Observation loop until a Final
    Answer is produced or max_iterations is exhausted. Returns the final
    answer text (or the full trace if no Final Answer was reached), and
    writes the full trace to a log file for the execution-log deliverable.

    required_calls: optional list of (tool_name, kwargs_dict) pairs that
    MUST have a real, successfully-executed Observation before a Final
    Answer is accepted. Without this, the model was observed to sometimes
    skip straight to a Final Answer containing plausible-looking but
    partially fabricated numbers once it had *some* real data in hand (e.g.
    1 real tool call plus 5 invented ones in a 6-call benchmark) -- since no
    un-executed Action is present in that completion, the existing "an
    Action always wins over a Final Answer" guard has nothing to catch. This
    is the only defense against that: for open-ended queries (no fixed
    required call list) leave this None.

    collect_results: optional dict, mutated in place as {call_key:
    raw_tool_json_string} for every required_calls entry that actually
    executes. Lets a caller (e.g. benchmark_runner.py) read back the exact,
    verified tool Observation for a known call instead of trusting the
    model's own prose paraphrase of a number, which -- unlike the number
    itself -- is not guarded against being subtly misstated.

    stop_when_satisfied: if True, return as soon as every required_calls
    entry has succeeded, instead of continuing to wait for the model to
    also produce a well-formed Final Answer. Useful when the caller (e.g. a
    benchmark_runner.py sub-task with exactly one required call) only cares
    about collect_results and not about a synthesized answer -- otherwise
    the model tends to keep rambling for several more turns after it
    already has everything, wasting iteration budget and Ollama load.
    """
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
    # an *identical* Action/Action Input instead of progressing -- not only
    # back-to-back, but also several steps later after other tools ran in
    # between (e.g. re-fetching a dataset summary it already has). Left
    # unchecked this both wastes the iteration budget and keeps
    # re-appending duplicate Observations, ballooning the prompt until
    # Ollama's memory use gets the container OOM-killed. We remember every
    # (tool, input) pair actually executed so far and short-circuit ANY
    # repeat of one with a nudge instead of re-running the tool, aborting
    # outright once nudging itself stalls out.
    executed_calls = set()
    used_tool_names = set()
    consecutive_nudges = 0
    required_list = list(required_calls) if required_calls else None
    # Every successfully-executed (tool_name, kwargs_dict) pair, in order.
    # Kept separate from executed_calls/repeat-guard's exact-string keys
    # because "does this satisfy a required call" needs *subset* matching,
    # not exact matching -- the model was observed adding an extra
    # explicit parameter at its default value (e.g. test_size=0.2) to an
    # otherwise-correct call, which an exact-JSON-string comparison would
    # wrongly treat as not satisfying the requirement at all.
    successful_call_records = []

    def missing_required():
        """None if required_list wasn't given; otherwise the required
        (tool, kwargs) pairs that no successful call has satisfied yet, in
        required_list's original order."""
        if required_list is None:
            return None
        return [
            (tool, req_kwargs)
            for tool, req_kwargs in required_list
            if not any(
                t == tool and all(actual.get(k) == v for k, v in req_kwargs.items())
                for t, actual in successful_call_records
            )
        ]

    for step in range(1, max_iterations + 1):
        emit(f"\n--- Step {step} ---")
        # Real per-turn LLM latency, for the technical report's latency
        # benchmark section -- CPU-only inference slows down as the prompt
        # accumulates Observations across steps, so this is worth capturing
        # turn-by-turn rather than just once for the whole run.
        turn_started = time.monotonic()
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
        llm_latency = time.monotonic() - turn_started
        emit(f"[LLM latency: {llm_latency:.2f}s, prompt length: {len(prompt)} chars]")

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

        # Always show what the model actually produced in the log/console,
        # even a detected duplicate -- but decide *below* whether it also
        # gets appended to the working `prompt` that feeds the next turn.
        emit(llm_output)

        all_required_satisfied_this_step = False

        if has_action:
            tool_name = action_match.group(1).strip()
            raw_input = input_match.group(1).strip()

            try:
                kwargs = json.loads(raw_input)
            except json.JSONDecodeError:
                # Not a duplicate-detection case -- this is genuine new
                # content (even if malformed), so it belongs in the prompt.
                prompt += llm_output
                observation = "\nObservation: Error parsing Action Input as JSON. Ensure it is valid JSON.\n"
                emit(observation)
                prompt += observation
                continue

            # Canonicalize on the *parsed* kwargs (sorted keys), not the raw
            # regex-captured string -- two calls that are semantically
            # identical but differ in key order or whitespace would
            # otherwise dodge the repeat-call guard below.
            call_key = (tool_name, json.dumps(kwargs, sort_keys=True))

            if call_key in executed_calls:
                consecutive_nudges += 1
                if consecutive_nudges >= 7:
                    emit("\n>>> Aborting: agent kept repeating Action Inputs it already has "
                         "results for, even after repeated nudges to move on.")
                    break
                # Critical: do NOT append this duplicate llm_output to
                # `prompt`. Small quantized models fall into self-reinforcing
                # repetition once an identical block appears twice in their
                # own context -- feeding the duplicate back in (even
                # alongside a nudge) was observed to make a 3rd, 4th, ... copy
                # more likely rather than less. Only the nudge Observation is
                # appended, so the model's own repeated text never accumulates
                # in what it reads back next turn.
                still_missing = missing_required()
                if still_missing:
                    # required_calls is set and there's a specific call still
                    # outstanding -- point at it by name. A generic "tools you
                    # haven't used" hint (below) is actively counterproductive
                    # here: it's phrased in terms of tool *names*, so once
                    # e.g. train_sklearn_model has been called at all it drops
                    # off that list -- even though a benchmark task like this
                    # needs many more calls to that exact same tool with
                    # different arguments. This was observed steering the
                    # model away from finishing a benchmark it was most of
                    # the way through.
                    next_tool, next_kwargs = still_missing[0]
                    next_hint = (
                        f" Your next Action must be exactly this one: "
                        f"Action: {next_tool}\nAction Input: {json.dumps(next_kwargs)}"
                    )
                else:
                    untried = [t for t in AVAILABLE_TOOLS if t not in used_tool_names]
                    next_hint = f" Tools you have not used yet in this task: {untried}." if untried else ""
                observation = (
                    f"\nObservation: You already called {tool_name} with this exact input "
                    f"earlier in this task and already have that result. Do not repeat it -- "
                    f"use what you already learned instead.{next_hint} Or write your Final "
                    f"Answer now if you already have everything the task needs.\n"
                )
                emit(observation)
                prompt += observation
                continue

            prompt += llm_output
            consecutive_nudges = 0
            executed_calls.add(call_key)
            used_tool_names.add(tool_name)

            if tool_name in AVAILABLE_TOOLS:
                try:
                    tool_res = AVAILABLE_TOOLS[tool_name](**kwargs)
                    observation = f"\nObservation: {tool_res}\n"
                    # Only a call that actually ran without raising counts as
                    # "real data in hand" for the required-calls gate below --
                    # ml_tools' own {"error": ...} JSON payloads for expected
                    # bad input still land here (no Python exception), but
                    # required_calls is only ever populated with pre-validated
                    # argument combinations, so that's not a concern in
                    # practice.
                    successful_call_records.append((tool_name, kwargs))
                    if collect_results is not None and required_list is not None:
                        for req_tool, req_kwargs in required_list:
                            req_key = (req_tool, json.dumps(req_kwargs, sort_keys=True))
                            if (
                                req_key not in collect_results
                                and tool_name == req_tool
                                and all(kwargs.get(k) == v for k, v in req_kwargs.items())
                            ):
                                collect_results[req_key] = tool_res
                    if stop_when_satisfied and required_list is not None and not missing_required():
                        all_required_satisfied_this_step = True
                except Exception as e:
                    # Self-healing (Task 3): the LLM sees *what* failed and is
                    # explicitly told to retry with corrected parameters. A
                    # bare exception message (e.g. "unexpected keyword
                    # argument 'hidden_dim'") tells it a param name is wrong
                    # but not what's actually valid -- the 3B model has been
                    # observed blending two tools' parameter names together
                    # (e.g. passing train_pytorch_mlp's hidden_dim/epochs
                    # into train_sklearn_model). Echoing the tool's real
                    # signature turns that into a self-correctable Observation
                    # instead of a second guess.
                    try:
                        sig = str(inspect.signature(AVAILABLE_TOOLS[tool_name]))
                    except (TypeError, ValueError):
                        sig = ""
                    observation = (
                        f"\nObservation: Tool execution error: {str(e)}. "
                        f"The actual signature of {tool_name} is {tool_name}{sig}. "
                        f"Reconsider your parameters and retry -- either call {tool_name} again "
                        f"with only its real parameters, or call a different tool if this one "
                        f"isn't the right one for what you're trying to do.\n"
                    )
            else:
                observation = (
                    f"\nObservation: Tool '{tool_name}' not recognized. "
                    f"Valid tools: {list(AVAILABLE_TOOLS.keys())}.\n"
                )

            emit(observation)
            prompt += observation

            if all_required_satisfied_this_step:
                emit("\n>>> All required tool calls satisfied -- stopping early "
                     "(stop_when_satisfied=True).")
                break
        elif final_idx != -1:
            prompt += llm_output

            missing = missing_required()
            if missing:
                # Reject a Final Answer that skips required tool calls -- this
                # is what actually stops fabricated numbers, not just a
                # format check. Observed directly: given 1 real tool result,
                # the model wrote a complete Final Answer table with 5
                # invented rows rather than making the other 5 calls, since
                # no un-executed Action was present for the existing
                # Action-vs-Final-Answer guard (§ above) to catch.
                #
                # Naming only the single next call (not the full missing
                # list) mirrors the repeat-guard nudge above: listing all N
                # missing calls at once was observed to overwhelm the model
                # into abandoning the checklist entirely and falling back to
                # re-calling a tool it had already used successfully.
                next_tool, next_kwargs = missing[0]
                observation = (
                    f"\nObservation: Your Final Answer is rejected -- {len(missing)} of the "
                    f"required tool calls are still missing, so some of your numbers would not "
                    f"be real. Your very next Action must be exactly this one: "
                    f"Action: {next_tool}\nAction Input: {json.dumps(next_kwargs)}\n"
                    f"Do not write a Final Answer again until every required call has a real "
                    f"result.\n"
                )
                emit(observation)
                prompt += observation
                continue

            emit("\n>>> Task Completed Successfully!")
            final_answer = llm_output.split("Final Answer:", 1)[1].strip()
            break
        else:
            # Not a duplicate-action case -- this is genuine (if unhelpful)
            # new content, so it belongs in the prompt like any other turn.
            prompt += llm_output
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
