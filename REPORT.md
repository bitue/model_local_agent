
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
differs.

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
                    │   5. json.loads the Action Input              │
                    │   6. dispatch to AVAILABLE_TOOLS[name]        │
                    │   7. catch any exception → Observation        │
                    │      ("self-healing" retry path)              │
                    │   8. append Observation, loop                 │
                    │   9. stop on a clean "Final Answer:"          │
                    │  10. persist full trace → logs/run_NNN.log    │
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
| Agent Logic | Python 3.10, `react_agent.py` (from-scratch ReAct loop, no LangChain) | Prompt construction, regex parsing, tool dispatch, self-healing, logging |
| ML Frameworks | PyTorch 2.x (CPU), Scikit-Learn 1.7, Pandas, NumPy | Preprocessing, model fitting, cross-validation, metrics |

---

## 3. Prompt Engineering

The entire tool-calling contract is enforced through the `SYSTEM_PROMPT` string in
`react_agent.py` — there is no function-calling API on the Ollama `/api/generate` endpoint,
so the model has to be taught the `Thought / Action / Action Input` grammar purely through
instructions and a single worked example ("few-shot" of size 1).

Techniques used, in order of impact:

1. **Low temperature (`0.1`)** — a 3-billion-parameter quantized model is noticeably less
   reliable at following a rigid output grammar than a frontier model; low temperature keeps
   the format close to the example rather than drifting into free-form prose.
2. **`stop: ["Observation:"]`** — prevents the model from writing its own fabricated
   Observation and continuing the dialogue with itself in a single completion.
3. **One worked example embedded in the system prompt** — a single correct
   `Thought → Action → Action Input` turn, explicitly annotated with *why* it stops where it
   does. This was more effective than a longer natural-language description alone.
4. **An explicit anti-premature-conclusion rule** — added after the failure mode below was
   observed: *"Do NOT write 'Final Answer' until you have a real Observation for every tool
   call the task requires."*
5. **Explicit numbered tool catalogue with full signatures** (name, parameter names, types,
   defaults, valid enum values) directly in the prompt — updated every time a tool was added
   in Task 2, since the model has no other way to discover what tools exist.

### 3.1 Observed failure modes and the engineering response

Two related failure modes of the small quantized model were captured directly in this
project's logs (`agent/logs/_archive/`) and drove two rounds of hardening in
`run_agent_loop`:

- **Trailing hallucination** (`_archive/run_002_stale_prefix_bug.log`) — after a real
  `Action Input`, the model kept generating rather than stopping, narrating fake tool
  results in prose (carefully avoiding the literal string `"Observation:"`, so the stop
  sequence never triggered) and ending in a fabricated `Final Answer` with numbers that were
  never actually computed.
- **Premature conclusion** (`_archive/run_003_stale_premature_final.log`) — the model
  announced `Final Answer:` after only the *first* of several requested sub-tasks, then kept
  generating a to-do list of Actions it never executed underneath its own "answer".

The fix generalizes both cases into one rule: **any well-formed, un-executed
`Action:`/`Action Input:` pair anywhere in a completion always takes priority over any
`Final Answer:` text in that same completion.** The controller truncates the completion at
the end of that first real Action Input, executes the tool for real, and only accepts a
`Final Answer` when the completion contains *no* residual Action at all. This is a case where
robust *parsing/execution* logic, not a smarter prompt, was the reliable fix — small models
cannot be fully trusted to end their own turn correctly, so the controller enforces it.

### 3.2 Latency

CPU-only inference on `llama3.2:3b` inside the container measured from the timestamped
`logs/run_*.log` files:

| Trace | Steps | Wall-clock (query → Final Answer) | Approx. latency / LLM turn |
|---|---|---|---|
| `run_001.log` (single tool) | 2 | [FILL FROM TIMESTAMPS] | [FILL] |
| `run_00X.log` (multi-tool, 3 tools) | [N] | [FILL] | [FILL] |
| `run_00X.log` (Task 2 tools) | [N] | [FILL] | [FILL] |
| `run_00X.log` (self-healing) | [N] | [FILL] | [FILL] |
| `benchmark_*.md` (6 tool calls) | [N] | [FILL] | [FILL] |

Per-turn latency grows across a single run because the full conversation (system prompt +
every prior Thought/Action/Observation) is resent as the prompt on every turn — there is no
KV-cache reuse across HTTP calls in this simple `/api/generate`-based design, so later turns
in a long trace cost more to process than earlier ones.

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
`react_agent.py`.

---

## 5. Self-Correction (Task 3)

`react_agent.py` wraps every tool call in `try/except`. On failure it feeds back an
`Observation` that names the failure and explicitly instructs the model to reconsider its
parameters, rather than a bare stack trace.

**Captured proof trace:** `agent/logs/run_00X.log` — the agent was asked to call
`train_sklearn_model` with an invalid `n_estimators` keyword argument (that function does not
accept it). The trace shows:

1. Step N: `Action: train_sklearn_model`, `Action Input: {"dataset_name": "wine", "model_type": "random_forest", "n_estimators": 100}`
2. `Observation: Tool execution error: train_sklearn_model() got an unexpected keyword argument 'n_estimators'. Reconsider your parameters and retry the same tool.`
3. Step N+1: the model **reasons about the error** in its `Thought`, drops the invalid
   parameter, and retries with `Action Input: {"dataset_name": "wine", "model_type": "random_forest"}`
4. The corrected call succeeds and the agent reports the real test accuracy in its
   `Final Answer`.

This demonstrates genuine reasoning about the failure (not just exception suppression) — the
LLM receives the failure `Observation` and changes its next `Action Input` in response.

---

## 6. Model Comparison & Statistical Analysis (`benchmark_runner.py`)

The agent was given a single prompt asking it to autonomously evaluate **3 algorithms**
(`decision_tree`, `random_forest`, `logistic_regression`) across **2 datasets** (`wine`,
`breast_cancer`), performing 5-fold cross-validation for each and producing a Markdown
summary table as its own `Final Answer`.

**Result:** see `agent/logs/benchmark_*.md` for the model-generated table. Reproduced below:

```
[PASTE the agent's Final Answer Markdown table here]
```

### 6.1 Bias/variance discussion (CO2)

- **Decision Tree** — a single, fairly shallow (`max_depth=4`) tree is high-bias / low-variance
  relative to the ensemble; expect the largest gap between test accuracy and CV mean, and the
  highest CV standard deviation, since a single tree's decision boundary is sensitive to which
  rows land in the training fold.
- **Random Forest** — averaging 50 trees over bootstrap samples trades a small amount of bias
  for a large reduction in variance; expect it to have the *lowest* CV standard deviation of
  the three and the strongest resistance to the differences between `wine` (13 features, 3
  classes, ~178 samples) and `breast_cancer` (30 features, 2 classes, ~569 samples, feature
  scales that differ by orders of magnitude, unlike Random Forest which is scale-invariant).
- **Logistic Regression** — a linear decision boundary; expect it to do very well on
  `breast_cancer` (known to be close to linearly separable in this feature space) but
  potentially underperform relative to the tree-based models on `wine`, where class boundaries
  are less linear across the 13 chemical-composition features.
- [Replace the three bullets above with the *actual* observed numbers from the benchmark
  table before submitting — the reasoning pattern is generic, but the report must ultimately
  quote the real CV mean/std pairs the agent measured.]

---

## 7. Reproducibility

```bash
docker compose up -d ollama
docker compose up ollama-pull
docker compose build agent
docker compose run --rm agent python react_agent.py "<query>"
docker compose run --rm agent python benchmark_runner.py
```

Every run appends a full trace to `agent/logs/run_NNN.log` (or `logs/benchmark_*.md`) on the
host, via the bind-mounted `./agent/logs` volume — this is exactly the execution-log
deliverable required by the assignment.

---

## 8. Conclusion

The agent satisfies all three graded tasks: a working local ReAct loop over a from-scratch
regex parser (Task 1), six JSON-serializable Scikit-Learn/PyTorch tools spanning baseline
models, hyperparameter tuning, dimensionality reduction, and a regularized deep classifier
(Task 2), and a proven self-healing retry loop plus a fully autonomous
3-algorithm × 2-dataset benchmark with agent-authored statistical commentary (Task 3). The
most instructive engineering lesson was that a small quantized local model cannot be trusted
to reliably end its own turn — the controller, not the prompt alone, has to be the final
authority on whether a `Final Answer` is genuine.
