
# Technical Report — Autonomous Local LLM ML Agent

**Course:** CSE445 – Machine Learning | Section 6 | Instructor: Dr. Mohammad Abdul Qayum
**Assignment:** #3 — Building an Autonomous Local LLM ML Agent
**Author:** [YOUR NAME] · [YOUR ID]

---

## 1. Overview

This project implements a fully local, privacy-preserving **ReAct** (Reason + Act) agent.
A quantized instruction-tuned LLM, served locally by **Ollama**, reasons in natural language
about which of six Python/Scikit-Learn/PyTorch tools to call in order to solve a machine
learning task; a hand-written controller (`react_agent.py`) parses that reasoning, executes
the requested tool, feeds the real result back as an `Observation`, and repeats until the
model produces a `Final Answer`. No external API and no cloud service is used anywhere in
the pipeline — every model weight, every dataset, and every training run stays on the local
machine.

### 1.1 Deviation from the assignment brief: Docker Compose instead of native WSL2

The assignment brief (`step.md`) specifies a native **WSL2 + Ubuntu + Python venv** setup.
This implementation instead uses **Docker Compose** with three services — `ollama` (the
inference engine), `ollama-pull` (a one-shot init container that pulls the model), and
`agent` (the Python 3.10 ReAct controller + ML tools). Docker Desktop on Windows runs its
containers on top of the WSL2 backend, so the underlying execution environment is still a
Linux kernel inside WSL2 — this substitution keeps every property the rubric cares about
(local inference, no cloud API, Linux runtime, reproducibility) while making the environment
one `docker compose up` away from reproducible on any machine, rather than depending on a
hand-configured Ubuntu distro. All commands, ports (`11434`), and model tags match the brief
exactly; only the process-isolation mechanism (containers vs. a venv on a WSL distro)
differs. **Flag this substitution to the instructor explicitly**, since the rubric literally
checks "WSL & Local LLM Setup" (15 marks, CO3).

---

## 2. System Architecture

```
                              User Query (CLI arg or REPL)
                                          │
                                          ▼
                    ┌───────────────────────────────────────────┐
                    │   agent container — react_agent.py         │
                    │   (Python 3.10, ReAct controller)           │
                    │                                              │
                    │   1. build prompt = SYSTEM_PROMPT + history  │
                    │   2. POST prompt to Ollama /api/generate     │
                    │   3. regex-parse "Action:" / "Action Input:" │
                    │   4. reject a "Final Answer" that still      │
                    │      contains an un-executed Action          │
                    │      (anti-hallucination guard)               │
                    │   5. detect + short-circuit a repeated        │
                    │      (tool, input) call (repeat-loop guard)   │
                    │   6. json.loads the Action Input              │
                    │   7. dispatch to AVAILABLE_TOOLS[name]        │
                    │   8. catch any exception → Observation with   │
                    │      the tool's real signature echoed back    │
                    │      ("self-healing" retry path)              │
                    │   9. append Observation, loop                 │
                    │  10. stop on a clean "Final Answer:"          │
                    │  11. persist full trace → logs/run_NNN.log    │
                    └───────────────┬───────────────┬───────────────┘
                                    │               │
                    HTTP POST       │               │  Python call
                    :11434/api/     │               │  (in-process)
                    generate        ▼               ▼
                    ┌──────────────────────┐  ┌───────────────────────────┐
                    │  ollama container     │  │  ml_tools.py               │
                    │  Ollama server         │  │  (Scikit-Learn + PyTorch   │
                    │  model: llama3.2:3b    │  │   + Pandas + NumPy)        │
                    │  (quantized, 2.0 GB)   │  │                             │
                    └──────────────────────┘  │  1. load_dataset_summary    │
                                                │  2. train_sklearn_model     │
                                                │  3. train_pytorch_mlp       │
                                                │  4. tune_hyperparameters    │
                                                │  5. reduce_dimensionality   │
                                                │  6. train_deep_classifier   │
                                                └───────────────────────────┘
```

| Component | Technology | Role |
|---|---|---|
| OS / Subsystem | Docker Desktop → WSL2 backend (Ubuntu-based container images) | Linux kernel runtime, container isolation |
| Inference Engine | Ollama (`ollama/ollama` image), REST API on `:11434` | Serves the quantized `llama3.2:3b` model |
| Agent Logic | Python 3.10, `react_agent.py` (from-scratch ReAct loop, no LangChain) | Prompt construction, regex parsing, tool dispatch, repeat-loop guard, self-healing, logging |
| ML Frameworks | PyTorch 2.x (CPU), Scikit-Learn 1.7, Pandas, NumPy | Preprocessing, model fitting, cross-validation, metrics |

---

## 3. Prompt Engineering

The entire tool-calling contract is enforced through the `SYSTEM_PROMPT` string in
`react_agent.py` — there is no function-calling API on the Ollama `/api/generate` endpoint,
so the model has to be taught the `Thought / Action / Action Input` grammar purely through
instructions and a worked example.

Techniques used, in order of impact:

1. **Low temperature (`0.1`)** — a 3-billion-parameter quantized model is noticeably less
   reliable at following a rigid output grammar than a frontier model; low temperature keeps
   the format close to the example rather than drifting into free-form prose.
2. **`stop: ["Observation:"]`** — prevents the model from writing its own fabricated
   Observation and continuing the dialogue with itself in a single completion.
3. **A two-turn worked example embedded in the system prompt** — showing not just the
   correct `Thought → Action → Action Input` grammar but *progression*: turn 1 fetches a
   dataset summary, turn 2 (after an illustrative Observation) moves on to training a model
   rather than re-fetching the summary. An earlier one-turn version of this example (showing
   only the `load_dataset_summary` call) measurably worsened a repetition failure mode (§3.1)
   — the model appeared to anchor on the one tool it had seen demonstrated.
4. **An explicit anti-premature-conclusion rule** and an explicit **anti-repetition rule**
   ("check whether you already received an Observation that answers the exact same
   question... do NOT call that tool again") — both added after the failure modes below were
   observed directly in this project's own logs.
5. **`num_predict: 400`** — caps a single turn's generation length. Once the model drifts off
   the strict format it can ramble indefinitely; bounding generation length limits the blast
   radius of a bad turn and reduces load on the local CPU inference server.
6. **Explicit numbered tool catalogue with full signatures** (name, parameter names, types,
   defaults, valid enum values) directly in the prompt — updated every time a tool was added
   in Task 2, since the model has no other way to discover what tools exist.

### 3.1 Observed failure modes and the engineering response

This is the most instructive part of the project: with a 3B quantized model, **prompt text
alone was not sufficient** to guarantee correct agent behaviour. Four distinct failure modes
were captured directly in this project's own logs (`agent/logs/_archive/`) and each drove a
round of hardening in `run_agent_loop` or the prompt:

- **Trailing hallucination** (`_archive/run_002_stale_prefix_bug.log`) — after a real
  `Action Input`, the model kept generating rather than stopping, narrating fake tool
  results in prose (carefully avoiding the literal string `"Observation:"`, so the stop
  sequence never triggered) and ending in a fabricated `Final Answer` with numbers that were
  never actually computed. **Fix:** any well-formed, un-executed `Action:`/`Action Input:`
  pair anywhere in a completion always takes priority over any `Final Answer:` text in that
  same completion — the controller truncates the completion right after the Action Input and
  only accepts a `Final Answer` when no residual Action is present.
- **Premature conclusion** (`_archive/run_003_stale_premature_final.log`) — the model
  announced `Final Answer:` after only the *first* of several requested sub-tasks, then kept
  generating a to-do list of Actions it never executed underneath its own "answer". Fixed by
  the same truncation rule above, plus an explicit prompt rule to gather every Observation
  before concluding.
- **Repeated-action loop (the hardest failure mode)** — even after real tool calls were
  working correctly, the model would re-issue an **identical** `Action`/`Action Input` several
  turns running instead of progressing (`_archive/run_004_repeat_guard_v1_partial.log` through
  `_archive/run_010_temperature_bump_still_stuck.log` document six iterations of diagnosis).
  Root-causing this took several attempts:
  - A per-tool call-history guard (nudge the model instead of re-executing an exact repeat)
    reduced but did not eliminate it — non-consecutive repeats (e.g. re-fetching a summary
    two steps later) still slipped through a "last call only" check.
  - Raising the Ollama `repeat_penalty` was tried and made things categorically *worse*
    (`_archive/run_008_repeat_penalty_experiment_broke_coherence.log`) — at the strength
    needed to suppress the repeat, it also destroyed the model's ability to produce valid
    JSON/format tokens at all, producing incoherent word-salad output.
  - **Root cause, found by inspecting the actual accumulated prompt:** the controller was
    re-appending the model's *duplicate* Thought/Action text into the prompt even when the
    duplicate call was not re-executed. This meant the exact same block of text appeared
    **twice** in the model's own context — a well-documented degenerate-repetition trigger
    for small LLMs, which then reproduced it a third and fourth time. **Fix:** a detected
    duplicate call is still logged for transparency but is *never* appended to the working
    prompt — only a corrective Observation naming the tools not yet used is appended. This
    fix (not a sampling-parameter change) is what actually resolved the loop.
  - The benchmark's more open-ended "evaluate 3 algorithms across 2 datasets" prompt
    triggered the same failure more persistently than a single concrete query, because it
    left the (algorithm × dataset) cross-product for the model to re-derive on every turn.
    **Fix:** `benchmark_runner.py` now generates an explicit numbered checklist of the exact
    tool calls required, so the model has a fixed list of remaining work items to check off
    instead of a plan to keep re-deriving from scratch.
- **Parameter/tool confusion** — the model was observed calling `train_sklearn_model` with
  `hidden_dim`/`epochs` (parameters that belong to `train_pytorch_mlp`), receiving only a bare
  "unexpected keyword argument" error, and not reliably recovering. **Fix:** the tool-error
  Observation now echoes the tool's real signature (via `inspect.signature`) alongside the
  exception message, and explicitly suggests trying a different tool — this generalizes to
  any tool without hardcoding domain knowledge about which tool is "correct."
- **Fabricated Final Answer with *no* dangling Action to catch (the most serious failure
  mode, caught just before this was written up)** — once the repeated-action loop above was
  fixed, the benchmark run reached a *new* failure: given 1 genuine `train_sklearn_model`
  result, the model wrote a complete `Final Answer` Markdown table with plausible-looking
  numbers for all 6 required rows, having only ever made 1 real call. Because the completion
  contained no un-executed `Action:` text, the existing "an Action always wins over a Final
  Answer" guard (which only inspects the *current* completion) had nothing to catch — the
  fabrication happened entirely inside the Final Answer text itself. **Fix:** `run_agent_loop`
  now accepts an optional `required_calls` list of exact (tool, kwargs) pairs; a Final Answer
  is rejected outright — with a corrective Observation naming the single next required call —
  until every one of them has a real, successfully-executed Observation behind it (tracked via
  **subset matching** on the kwargs, not exact-string matching, after a related bug where the
  model added an extra explicit default parameter like `test_size=0.2` and an exact match
  wrongly failed to recognize an otherwise-correct call). This is the only defense in the
  system against a fluent-sounding but partially invented result, and it is the fix that
  finally made the benchmark deliverable trustworthy.
- **Long-horizon unreliability, solved by decomposition, not more prompting** — even with
  fabrication blocked, driving all 6 required calls to completion in a *single* continuous
  ReAct session proved unreliable: the model would make 1-5 of the 6 real calls correctly but
  then stall re-issuing a call it already had (`agent/logs/_archive/run_012_benchmark_attempt*`
  documents ten such attempts, with iteration budgets raised as high as 22 and repeat-abort
  thresholds raised as high as 7, still without a reliable 6/6 completion). The lesson: **no
  amount of prompt engineering reliably fixes a small model's long-horizon state-tracking**.
  `benchmark_runner.py` was restructured to run 6 independent, single-purpose ReAct sub-tasks
  (one per algorithm/dataset pair) instead of one long session — exactly the kind of one-shot
  tool call this agent already handles reliably (`agent/logs/run_001.log`). Each sub-task's
  real Observation is captured directly (`collect_results`, plus `stop_when_satisfied=True` so
  a sub-task stops the instant its one required call succeeds instead of rambling on), the
  Markdown table is assembled in Python from these six verified results, and the LLM is asked
  only for the closing bias/variance commentary given the complete real table — a much smaller
  ask than remembering 6 numbers across 20+ turns. This is a legitimate agent-engineering
  pattern (decompose an unreliable long-horizon task into reliable independent sub-tasks), not
  a workaround that reduces how "autonomous" the result is: the LLM still performs every piece
  of tool-selection reasoning and every interpretation in the deliverable.

### 3.2 Infrastructure note: local Ollama stability under load

Independent of the agent logic, the local `ollama` container was observed to crash (process
exit, connection refused on the next request) on this development machine several times
during long benchmark runs — most likely CPU/memory pressure from sustained local inference
inside a resource-constrained Docker Desktop WSL2 VM (~7.6 GB allocated) rather than a bug in
the agent code, since it always recovered cleanly after a container restart and the Python
side already retries a failed request once before giving up. This is a real-world
consequence of "fully local, no cloud" inference: unlike a hosted API, the inference
service's reliability is now bounded by the host machine's own resources. `num_predict: 400`
(§3) was added partly to reduce the chance of an unbounded runaway generation contributing to
this.

### 3.3 Latency

CPU-only inference on `llama3.2:3b` inside the container, measured from this project's own
`logs/run_*.log` files:

| Trace | Steps | Approx./exact total wall-clock | Latency / LLM turn |
|---|---|---|---|
| `run_001.log` (single tool: `load_dataset_summary` → Final Answer) | 2 | ~30–60s | ~15–30s (approx.) |
| `run_011.log` (multi-tool: 5 distinct tools + 1 self-healing retry) | 10 | ~2–3 min | ~12–18s (approx.) |
| `run_012.log`–`run_017.log` (benchmark sub-tasks, 1 real tool call each) | 1 each | 9.24s–55.43s | see below (exact) |

(`run_001`/`run_011` predate the per-turn latency instrumentation added to `react_agent.py`
in this pass — `[LLM latency: X.XXs, prompt length: N chars]` is now printed after every
turn — so their figures above are wall-clock-derived approximations. `run_012`–`run_017`
carry exact per-turn numbers: 55.43s on the first call after a fresh container start
(one-time model warm-up inside Ollama), then 9.24s–10.57s per call once warm — a useful
illustration of local-inference cold-start cost, which a cloud API user would rarely notice
but a "fully local" deployment has to account for.)

Per-turn latency grows across a single run because the full conversation (system prompt +
every prior Thought/Action/Observation) is resent as the prompt on every turn — there is no
KV-cache reuse across separate HTTP calls in this simple `/api/generate`-based design (Ollama
does opportunistically reuse a matching prompt prefix internally, which is why later turns in
the raw server log show "cached n_tokens" close to the new prompt's length — but each call is
still a fresh HTTP request/response round trip).

---

## 4. Tool Registry (`ml_tools.py`)

| Tool | Task | ML technique | Notes |
|---|---|---|---|
| `load_dataset_summary` | 1 | — | Shape / feature names / class balance / missing values for `iris`, `wine`, `breast_cancer` |
| `train_sklearn_model` | 1 | Decision Tree / Logistic Regression / Random Forest | 80/20 stratified split + 5-fold CV |
| `train_pytorch_mlp` | 1 | `Linear → ReLU → Linear`, Adam, CrossEntropyLoss | Standardized features, NaN-loss guard |
| `tune_hyperparameters` | 2 | `GridSearchCV` / `RandomizedSearchCV` over `SVC` or `DecisionTreeClassifier` | Returns best params + best CV score |
| `reduce_dimensionality` | 2 | `PCA` or `SequentialFeatureSelector` | Explained variance ratio, or selected feature names |
| `train_deep_classifier` | 2 | Configurable MLP with `Dropout`, `BatchNorm1d`, `StepLR`/`ReduceLROnPlateau` | Parameterized depth/width/dropout, training curve summary |

Every tool returns a flat, JSON-serializable string and never raises on *expected* bad input
(unknown dataset name, invalid enum, out-of-range hyperparameter) — it returns
`{"error": "..."}` with an actionable message instead, so the agent can self-correct without
ever seeing a Python traceback. Genuinely unexpected exceptions (e.g., a bad keyword argument
the LLM invents) are still allowed to propagate and are caught one layer up, in
`react_agent.py`, which now also echoes the tool's real signature back (§3.1).

---

## 5. Self-Correction (Task 3)

`react_agent.py` wraps every tool call in `try/except`. On failure it feeds back an
`Observation` that names the failure, echoes the tool's real parameter signature, and
explicitly instructs the model to reconsider its parameters, rather than a bare stack trace.

**Captured proof trace:** `agent/logs/run_011.log`, steps 5–6 — the agent called
`tune_hyperparameters` with `model_type="random_forest"`, which is not one of that tool's
supported models:

1. Step 5: `Action: tune_hyperparameters`, `Action Input: {"dataset_name": "breast_cancer", "model_type": "random_forest", "search_type": "grid", "cv": 5}`
2. `Observation: {"error": "Unsupported model 'random_forest' for tuning. Valid options: ['svc', 'decision_tree']."}`
3. Step 6 **Thought**: *"The error message indicates that the 'random_forest' model type is
   not supported for tuning. I will try using the 'decision_tree' model type instead."* — the
   model explicitly reasons about *why* the call failed before acting.
4. Step 6 **Action**: retries `tune_hyperparameters` with `model_type="decision_tree"` — a
   genuinely corrected parameter, not a repeat of the failing input — and the call succeeds
   (`best_params`: `criterion=entropy, max_depth=4, min_samples_split=2`, `best_cv_score=0.942`).

This demonstrates genuine reasoning about the failure (not just exception suppression): the
LLM received the failure `Observation`, stated *what* was wrong with its own prior Action in
its next Thought, and changed its next Action Input in response.

**A limitation worth reporting honestly:** later in the same trace (step 8), the model's
`Action Input` for `train_deep_classifier` used the Python literal `True` instead of the JSON
literal `true` (`"use_batchnorm": True`), which failed the (deliberately strict) JSON parser.
Rather than retrying `train_deep_classifier` with corrected JSON, the model abandoned that
tool entirely and moved on to `reduce_dimensionality` instead. Self-correction on a *JSON
syntax* error was not as reliable as self-correction on a *semantic* error (bad `model_type`)
in this run — a genuine limitation of a 3B model under `Observation: Error parsing Action
Input as JSON` feedback, worth noting rather than hiding.

---

## 6. Model Comparison & Statistical Analysis (`benchmark_runner.py`)

The agent autonomously evaluated **3 algorithms** (`decision_tree`, `random_forest`,
`logistic_regression`) across **2 datasets** (`wine`, `breast_cancer`), 5-fold
cross-validating each — as 6 independent ReAct sub-tasks per the decomposed design in §3.1,
each sub-task's number captured directly from its verified tool Observation, never from the
model's prose. Every cell below is real, executed data (see `agent/logs/run_012.log` through
`run_017.log` for each sub-task's individual trace).

**Result** (`agent/logs/benchmark_20260831_091312.md`):

| Algorithm | Dataset | Test Accuracy | CV Mean Accuracy | CV Std |
| --- | --- | --- | --- | --- |
| decision_tree | wine | 0.9444 | 0.8937 | 0.0472 |
| random_forest | wine | 1.0 | 0.9610 | 0.0221 |
| logistic_regression | wine | 0.9722 | 0.9611 | 0.0333 |
| decision_tree | breast_cancer | 0.9386 | 0.9209 | 0.0202 |
| random_forest | breast_cancer | 0.9561 | 0.9543 | 0.0244 |
| logistic_regression | breast_cancer | 0.9649 | 0.9526 | 0.0142 |

The agent's own closing commentary, given this real table (asked only to interpret it, not
to recall or recompute any number):

> Based on the results, the best model for the wine dataset is random_forest with a test
> accuracy of 1.0, and for the breast_cancer dataset, it is logistic_regression with a test
> accuracy of 0.9649. The random_forest model performs exceptionally well on the wine
> dataset, while the logistic_regression model outperforms the other two models on the
> breast_cancer dataset, suggesting that the dataset's complexity and nature may favor a
> model that can handle non-linear relationships, such as logistic_regression.

(The agent's own phrase "handle non-linear relationships" mischaracterizes logistic
regression, which is a *linear* classifier — a genuine reasoning slip worth flagging rather
than silently editing out. The numeric picks it made, best-per-dataset by test accuracy, are
correct; see the more careful discussion below for *why* each model wins where it does.)

### 6.1 Bias/variance discussion (CO2)

- **Decision Tree has the lowest CV mean on *both* datasets** (0.8937 on wine, 0.9209 on
  breast_cancer) and the single highest CV std of all six rows (0.0472, on wine) — the
  textbook signature of a high-bias, high-variance single shallow tree (`max_depth=4`): its
  decision boundary is the most sensitive of the three to exactly which rows land in each
  training fold, especially on wine's smaller sample size (~178 rows across 3 classes leaves
  less data per fold).
- **Random Forest wins outright on wine** (test accuracy 1.0, the only perfect score in the
  table) and comfortably reduces variance relative to the single tree on wine (CV std 0.0221
  vs. decision tree's 0.0472) — consistent with ensembling trading a little bias for a large
  variance reduction. Notably, this pattern does **not** repeat on breast_cancer: random
  forest's CV std there (0.0244) is actually the *highest* of the three algorithms on that
  dataset, higher than both decision tree (0.0202) and logistic regression (0.0142). This is
  a genuine, honestly-reported deviation from the generic "ensembles always reduce variance"
  expectation — with only 5 folds and no repeated CV, a single unlucky fold split can dominate
  the variance estimate, so this should be read as one experimental run rather than a
  guaranteed property of random forests on this dataset.
- **Logistic Regression is the most stable model on breast_cancer** — its CV std (0.0142) is
  the lowest in the entire table, and it achieves the highest test accuracy on breast_cancer
  (0.9649) of all three algorithms, consistent with `breast_cancer`'s classes being close to
  linearly separable in this 30-feature space. On wine, however, it is the *least* stable of
  the three (CV std 0.0333, higher than random forest's 0.0221), suggesting the boundary
  between wine's 3 cultivar classes is less linear across its 13 chemical-composition
  features than breast_cancer's binary malignant/benign boundary.
- **Recommendation per dataset:** random_forest for wine (best accuracy *and* far lower
  variance than the decision tree); logistic_regression for breast_cancer (best accuracy *and*
  the lowest variance in the table) — both agree with the agent's own picks above, despite its
  flawed one-line justification for *why* logistic regression won.

---

## 7. Reproducibility

```bash
docker compose up -d ollama
docker compose up ollama-pull
docker compose build agent
docker compose run --rm agent python react_agent.py "<query>"
docker compose run --rm agent python benchmark_runner.py
```

Every ReAct run (including each of `benchmark_runner.py`'s 6 sub-task calls) appends a full
trace to its own `agent/logs/run_NNN.log`, and `benchmark_runner.py` additionally writes the
assembled table + commentary to `logs/benchmark_*.md` — all via the bind-mounted
`./agent/logs` volume, exactly the execution-log deliverable required by the assignment.
Superseded/invalid runs captured during development and debugging (e.g. the repeat-loop and
fabrication failure modes discussed in §3.1) are kept for transparency under
`agent/logs/_archive/`, clearly named for what they demonstrate, rather than deleted.

---

## 8. Conclusion

The agent satisfies all three graded tasks: a working local ReAct loop over a from-scratch
regex parser, hardened against six distinct hallucination/repetition/fabrication failure
modes observed in this project's own logs (Task 1); six JSON-serializable Scikit-Learn/PyTorch
tools spanning baseline models, hyperparameter tuning, dimensionality reduction, and a
regularized deep classifier (Task 2); and a proven self-healing retry loop (genuine reasoning
about a bad `model_type`, captured in `run_011.log`) plus a fully autonomous, fully-verified
3-algorithm × 2-dataset benchmark with agent-authored statistical commentary (Task 3). The
most instructive engineering lessons were, in the order they were learned the hard way: (1) a
small quantized local model cannot be trusted to reliably self-manage its own conversational
state — the controller, not the prompt alone, has to be the final authority on whether an
Action is genuinely new versus a stale repeat, and on whether a `Final Answer` is genuine or
partially fabricated; (2) decoding-parameter fixes (e.g. `repeat_penalty`) are a poor
substitute for fixing the actual bug (duplicate content re-entering the model's own context)
— the more surgical, root-caused fix was both simpler and the only one that actually worked;
and (3) reliability over a long multi-turn horizon is not something a 3B model can be
prompted into indefinitely — decomposing the benchmark into 6 independent single-purpose
ReAct sub-tasks, each verified before being trusted, succeeded where raising iteration
budgets and abort thresholds on one long session repeatedly did not.
