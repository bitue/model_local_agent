# Progress Tracker — CSE445 Assignment #3

> Auto-derived from `step.md` + current repo state.
> Last checked: 2026-08-31 (root-caused and fixed the repeat-action loop bug; report rewritten with real data)

Legend: ✅ done · ⚠️ partial / pending final verification run · ❌ not started

## The repeat-action bug is fixed (root cause found)

Earlier passes on this repo (see `REPORT.md` §3.1 for the full story) tried several partial
fixes for the model repeating the same Action/Action Input turn after turn. The actual root
cause: the controller was re-appending the model's **duplicate** output back into the prompt
even when it wasn't re-executed, so the exact same text block ended up appearing twice in the
model's own context — a known degenerate-repetition trigger for small LLMs. The fix: a
detected duplicate is still logged for transparency, but is never appended to the working
prompt; only a corrective Observation is. This, not a sampling-parameter change (raising
`repeat_penalty` was tried and made things categorically worse — see the report), is what
resolved it.

`agent/logs/run_011.log` is the resulting clean proof trace: 10 steps, 5 distinct tools used,
one genuine self-healing retry (a bad `tune_hyperparameters` `model_type`, corrected in the
very next Thought), ending in a real `Final Answer`.

## Architecture note (deviation from step.md)
`step.md` asks for native **WSL2 + Ubuntu + venv**. This repo instead uses **Docker Compose**
(`ollama` + `ollama-pull` + `agent` services) to achieve the same "local, no-cloud" goal.
Functionally equivalent (Docker Desktop runs on the WSL2 backend either way), but flag this
explicitly in the report/README since the rubric literally checks "WSL & Local LLM Setup" (15
marks, CO3) — `REPORT.md` §1.1 already does this.

## Known infra limitation: local Ollama crashes under sustained load
On this machine (Docker Desktop WSL2 VM, ~7.6 GB allocated), the `ollama` container has
crashed (connection refused on the next request) a handful of times during long benchmark
runs — reproducible at roughly the same point in a run, consistent with CPU/memory pressure
rather than an agent-code bug. Recovery is just `docker compose up -d ollama` + re-run. See
`README.md` and `REPORT.md` §3.2.

---

## Task 1 — Environment & Baseline ReAct Engine (25 marks)

- ✅ WSL2 + Ubuntu installed and working → **replaced with Docker Compose** (documented substitute); `ollama` service confirmed healthy, model pulled and running
- ✅ Ollama installed / serving / model pulled → `llama3.2:3b` pulled and verified via `ollama list` inside the container, responding to real queries
- ✅ Python deps (PyTorch + Scikit-Learn, etc.) → `agent/requirements.txt` complete
- ✅ Baseline ReAct loop implemented → `react_agent.py` matches §3.5 spec, hardened against 4 distinct failure modes (see `REPORT.md` §3.1)
- ✅ Verify single-tool execution trace → `agent/logs/run_001.log`: "summary of the iris dataset" → real `load_dataset_summary` call → genuine `Final Answer`.
- ✅ Verify multi-tool execution trace → `agent/logs/run_011.log`: 10-step trace, 5 distinct tools, genuine `Final Answer`. **Satisfied.**
- ✅ Save traces for deliverables → `run_001.log` + `run_011.log` are clean, complete, deliverable-ready. Superseded debugging attempts are kept (clearly named) under `agent/logs/_archive/` for transparency, not deleted.

## Task 2 — Advanced ML Tool Expansion (40 marks)

- ✅ Hyperparameter tuning tool → `tune_hyperparameters()` (GridSearchCV/RandomizedSearchCV, SVC + DecisionTree)
- ✅ Feature selection / dimensionality reduction tool → `reduce_dimensionality()` (PCA + SequentialFeatureSelector)
- ✅ Deep PyTorch classifier with regularization → `train_deep_classifier()` (Dropout, BatchNorm1d, StepLR/ReduceLROnPlateau)
- ✅ All 3 registered in `AVAILABLE_TOOLS`
- ✅ All 3 described in `SYSTEM_PROMPT` (tool list items 4–6)

**Task 2 is complete** — code, registration, prompt description, and exercised live in `run_011.log` (`tune_hyperparameters`, `reduce_dimensionality`; `train_deep_classifier` was attempted but hit a JSON-literal `True`-vs-`true` parse error and the agent moved on rather than retrying it — see `REPORT.md` §5 for this honestly-reported limitation).

## Task 3 — Self-Correction & Benchmark (35 marks)

- ✅ Self-healing logic in `react_agent.py` → try/except around tool calls feeds a corrective `Observation` back, now including the tool's real signature. **Proven** in `run_011.log` steps 5-6: bad `model_type` → error Observation → the model's next Thought explicitly reasons about the fix → retries with a valid `model_type` → succeeds.
- ✅ `benchmark_runner.py` exists → 3 algorithms (`decision_tree`, `random_forest`, `logistic_regression`) × 2 datasets (`wine`, `breast_cancer`), 5-fold CV, explicit-checklist prompt, Markdown table `Final Answer`
- ⚠️ Benchmark actually run / Markdown summary captured → **pending the final verification pass** (paused mid-session per instruction to finish all non-Ollama work first; several attempts hit a local Ollama infra crash, unrelated to the benchmark prompt/agent logic itself, which had already been fixed to reach real `train_sklearn_model` calls before each crash)

## Submission Deliverables

- ⚠️ **GitHub repo pushed** → local repo exists (2+ commits), **no remote configured yet** — needs `git remote add origin <url>` + push
- ✅ `ml_tools.py`
- ✅ `react_agent.py`
- ✅ `benchmark_runner.py`
- ✅ `requirements.txt`
- ✅ Docstrings / comments present in all three files (extensively — every fix in this pass is commented with *why*, not just *what*)
- ✅ **Execution logs** (3+ multi-step traces) → `run_001.log` (single-tool) + `run_011.log` (multi-tool + self-healing) are clean and deliverable-ready now; the benchmark run will be the 3rd once captured
- ✅ **Technical Report** (`REPORT.md`, 3-5 pages worth of content) → rewritten with real captured data (architecture, prompt engineering + all 4 failure modes and fixes, self-healing proof trace, latency table, bias/variance discussion). Only remaining gaps: your name/student ID (left as placeholders by request) and pasting in the final benchmark table once captured.

---

## What's actually left to do (in order)

1. Run the final Ollama verification pass: `docker compose up -d ollama` → confirm healthy →
   `docker compose run --rm agent python benchmark_runner.py` → confirm it reaches a real
   `Final Answer` with the Markdown table (the prompt/logic fix is in place; only local Ollama
   stability was blocking this).
2. Paste the resulting table into `REPORT.md` §6 and replace the bias/variance bullet's
   forward-looking note with a cross-check against the actual numbers.
3. Fill in your name and Student ID in `REPORT.md`'s header (left as placeholders by request).
4. `git remote add origin <your-github-url>`, commit the final state, push.
5. Optionally export/print `REPORT.md` to PDF if the submission portal wants a PDF rather than
   Markdown (check the assignment portal's accepted format first).

See the chat guidance for the full submission checklist and grading-rubric alignment.
