# Progress Tracker — CSE445 Assignment #3

> Auto-derived from `step.md` + current repo state. I'll update this every time code changes.
> Last checked: 2026-08-31

Legend: ✅ done · ⚠️ partial / code exists but unverified · ❌ not started

---

## Architecture note (deviation from step.md)
`step.md` asks for native **WSL2 + Ubuntu + venv**. This repo instead uses **Docker Compose**
(`ollama` + `ollama-pull` + `agent` services) to achieve the same "local, no-cloud" goal.
Functionally equivalent, but flag this explicitly in your report/README since the rubric
literally checks "WSL & Local LLM Setup" (15 marks, CO3) — make sure Dr. Qayum is OK with the
Docker substitution, or run it inside WSL2 too if it must match literally.

---

## Task 1 — Environment & Baseline ReAct Engine (25 marks)

- ⚠️ WSL2 + Ubuntu installed and working → **replaced with Docker Compose** (`docker-compose.yml`, not verified running)
- ⚠️ Ollama installed / serving / model pulled → configured (`ollama` + `ollama-pull` services, `.env` sets model), not confirmed running
- ✅ Python deps (PyTorch + Scikit-Learn, etc.) → `agent/requirements.txt` complete
- ✅ Baseline ReAct loop implemented → `react_agent.py` (`SYSTEM_PROMPT`, `query_local_llm`, `run_agent_loop`) matches §3.5 spec
- ❌ Verify single-tool execution trace → no `agent/logs/` directory exists yet — never run
- ❌ Verify multi-tool execution trace → same, not run
- ❌ Save traces for deliverables → nothing captured yet

## Task 2 — Advanced ML Tool Expansion (40 marks)

- ✅ Hyperparameter tuning tool → `tune_hyperparameters()` (GridSearchCV/RandomizedSearchCV, SVC + DecisionTree)
- ✅ Feature selection / dimensionality reduction tool → `reduce_dimensionality()` (PCA + SequentialFeatureSelector)
- ✅ Deep PyTorch classifier with regularization → `train_deep_classifier()` (Dropout, BatchNorm1d, StepLR/ReduceLROnPlateau)
- ✅ All 3 registered in `AVAILABLE_TOOLS`
- ✅ All 3 described in `SYSTEM_PROMPT` (tool list items 4–6)

**Task 2 code is functionally complete.**

## Task 3 — Self-Correction & Benchmark (35 marks)

- ⚠️ Self-healing logic in `react_agent.py` → try/except around tool calls feeds a corrective `Observation` back (code present), **but not yet proven with a captured trace** — you still need to deliberately trigger an error and show the LLM retrying
- ✅ `benchmark_runner.py` exists → 3 algorithms (`decision_tree`, `random_forest`, `logistic_regression`) × 2 datasets (`wine`, `breast_cancer`), 5-fold CV, Markdown table prompt
- ❌ Benchmark actually run / Markdown summary captured → no output file found yet

## Submission Deliverables

- ❌ **GitHub repo pushed** → this folder is **not a git repo yet** (no `.git`)
- ✅ `ml_tools.py`
- ✅ `react_agent.py`
- ✅ `benchmark_runner.py`
- ✅ `requirements.txt`
- ✅ Docstrings / comments present in all three files
- ❌ **Execution logs** (3+ multi-step traces) → none captured
- ❌ **Technical Report** (3–5 pages) → not started (no report file found)
  - ❌ Local LLM architecture / prompt engineering / latency benchmarks
  - ❌ Statistical comparison of models (bias/variance, CV mean/std)
  - ❌ Architecture diagram

---

## What's actually left to do (in order)

1. Run `docker compose up -d ollama && docker compose up ollama-pull && docker compose build agent` — confirm the model pulls and Ollama responds.
2. Run the sample query from `react_agent.py` (or README §2) → this both verifies Task 1 and produces your first execution-log trace.
3. Run a query that exercises a Task 2 tool (tuning / PCA / deep classifier) → second trace.
4. Deliberately trigger a tool error (bad `model_type`, bad `n_components`, etc.) and confirm in the saved log that the LLM receives the error `Observation` and retries with corrected input → third trace, satisfies Task 3 self-healing proof.
5. Run `benchmark_runner.py` → capture the Markdown summary table output.
6. Write the technical report using the 3+ captured logs and the architecture diagram from `step.md` §2.
7. `git init`, commit, push to GitHub; double check `requirements.txt` via `pip freeze` (or confirm the Docker-based one is accurate).

---

*This file is maintained alongside the code — whenever I make changes to `ml_tools.py`, `react_agent.py`, `benchmark_runner.py`, or run/capture logs, I'll update the checkboxes above.*
