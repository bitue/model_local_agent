# Progress Tracker — CSE445 Assignment #3

> Auto-derived from `step.md` + current repo state. I'll update this every time code changes.
> Last checked: 2026-08-31 (evening pass — two sessions worked on this in parallel today)

Legend: ✅ done · ⚠️ partial / code exists but unverified · ❌ not started

## ⚠️ Heads up: two real bugs found today, one still open

Two sessions (this one + another Claude Code window) worked on this repo in parallel today and
both landed fixes in `react_agent.py`. Net result — the good news first:

- **Bug #1 (fixed, archived as `agent/logs/_archive/run_002_stale_prefix_bug.log`):** the LLM was
  hallucinating the entire Thought→Action→Observation→Final Answer dialogue in one shot with
  fabricated results, never calling a real tool. Fixed by making any attempted `Action` always take
  priority over `Final Answer` text wherever it appears in the raw completion.
- **Bug #2 (fixed, archived as `run_003_stale_premature_final.log`):** the model was declaring
  `Final Answer` before actually finishing the requested work. Fixed via both a code-level guard and
  a `SYSTEM_PROMPT` instruction telling it not to.

**Bug #3 (found just now, NOT fixed yet):** with #1 and #2 fixed, real tool calls are happening —
but the model now gets stuck **repeating the identical Action verbatim**, even after a successful
Observation, instead of progressing to the next step:
- `agent/logs/run_002.log` (wine dataset, multi-tool query): real tool results throughout (PCA,
  hyperparameter tuning, etc. all genuine), but at steps 5–7 it retried the exact same malformed
  `Action Input` three times unchanged, then never reached a `Final Answer` (`Max iterations
  reached`).
- `agent/logs/run_003.log` + `agent/logs/benchmark_20260831_054016.md` (the Task 3 benchmark run):
  called `load_dataset_summary("wine")` **12 times in a row**, identical Thought/Action/Observation
  every time, never advanced to training any of the 3 required algorithms, never reached a
  `Final Answer`.

So as of this check, **no multi-step run has actually completed successfully end-to-end** —
single-tool works, multi-tool tool-calling works, but the small 3B model doesn't reliably know when
to stop repeating itself. Likely fix: detect when the same `(tool_name, kwargs)` pair repeats with a
non-error Observation and inject a corrective nudge ("you already have this result, move to the next
step") — not yet implemented by either session as of this check.

---

## Architecture note (deviation from step.md)
`step.md` asks for native **WSL2 + Ubuntu + venv**. This repo instead uses **Docker Compose**
(`ollama` + `ollama-pull` + `agent` services) to achieve the same "local, no-cloud" goal.
Functionally equivalent, but flag this explicitly in your report/README since the rubric
literally checks "WSL & Local LLM Setup" (15 marks, CO3) — make sure Dr. Qayum is OK with the
Docker substitution, or run it inside WSL2 too if it must match literally.

---

## Task 1 — Environment & Baseline ReAct Engine (25 marks)

- ✅ WSL2 + Ubuntu installed and working → **replaced with Docker Compose** (agreed substitute); `ollama` service confirmed healthy, model pulled and running
- ✅ Ollama installed / serving / model pulled → `llama3.2:3b` pulled and verified via `ollama list` inside the container, responding to real queries
- ✅ Python deps (PyTorch + Scikit-Learn, etc.) → `agent/requirements.txt` complete (added missing `torchaudio` to match `step.md`'s exact pip install)
- ✅ Baseline ReAct loop implemented → `react_agent.py` (`SYSTEM_PROMPT`, `query_local_llm`, `run_agent_loop`) matches §3.5 spec, hardened against two hallucination bugs (see callout above)
- ✅ Verify single-tool execution trace → `agent/logs/run_001.log`: "summary of the iris dataset" → real `load_dataset_summary` call → genuine `Final Answer`. **Clean, satisfies checkpoint.**
- ❌ Verify multi-tool execution trace → **not yet satisfied**. Real tool calls confirmed happening (no more hallucination), but every multi-step attempt so far (`run_002.log`, `run_003.log`) ends in `Max iterations reached without a Final Answer` due to Bug #3 above (model repeats the same Action instead of progressing)
- ⚠️ Save traces for deliverables → 3 logs captured, but only 1 of them (`run_001.log`) is a genuinely clean, complete trace — need at least one clean multi-tool trace before this is deliverable-ready

## Task 2 — Advanced ML Tool Expansion (40 marks)

- ✅ Hyperparameter tuning tool → `tune_hyperparameters()` (GridSearchCV/RandomizedSearchCV, SVC + DecisionTree)
- ✅ Feature selection / dimensionality reduction tool → `reduce_dimensionality()` (PCA + SequentialFeatureSelector)
- ✅ Deep PyTorch classifier with regularization → `train_deep_classifier()` (Dropout, BatchNorm1d, StepLR/ReduceLROnPlateau)
- ✅ All 3 registered in `AVAILABLE_TOOLS`
- ✅ All 3 described in `SYSTEM_PROMPT` (tool list items 4–6)

**Task 2 code is functionally complete.**

## Task 3 — Self-Correction & Benchmark (35 marks)

- ⚠️ Self-healing logic in `react_agent.py` → try/except around tool calls feeds a corrective `Observation` back (code present). Partially observed in `run_002.log`: a malformed `Action Input` was correctly caught and reported as an error Observation — but the model then repeated the *identical* malformed input three times instead of correcting it, so genuine self-correction ("agent reasons about the fix and changes its Action Input") is **not yet proven**
- ✅ `benchmark_runner.py` exists → 3 algorithms (`decision_tree`, `random_forest`, `logistic_regression`) × 2 datasets (`wine`, `breast_cancer`), 5-fold CV, Markdown table prompt
- ❌ Benchmark actually run / Markdown summary captured → `benchmark_20260831_054016.md` exists but is **not a valid deliverable** — the run got stuck repeating `load_dataset_summary("wine")` 12 times and never trained a single model or produced the required Markdown table

## Submission Deliverables

- ❌ **GitHub repo pushed** → this folder is **not a git repo yet** (no `.git`)
- ✅ `ml_tools.py`
- ✅ `react_agent.py`
- ✅ `benchmark_runner.py`
- ✅ `requirements.txt`
- ✅ Docstrings / comments present in all three files
- ⚠️ **Execution logs** (3+ multi-step traces) → 3 captured (`run_001/002/003.log`), only 1 is genuinely clean end-to-end — need the repeat-action bug fixed and at least 2 more clean traces
- ❌ **Technical Report** (3–5 pages) → not started (no report file found)
  - ❌ Local LLM architecture / prompt engineering / latency benchmarks
  - ❌ Statistical comparison of models (bias/variance, CV mean/std)
  - ❌ Architecture diagram

---

## What's actually left to do (in order)

1. **Fix Bug #3 (repeat-action loop)** in `react_agent.py` — the agent needs to detect when the same
   `(tool_name, kwargs)` pair repeats with a non-error Observation and inject a corrective nudge
   ("you already have this result — move to the next required step") instead of letting the model
   spin. Without this, no multi-step query reliably reaches a `Final Answer`.
2. Re-run the multi-tool sample query (breast_cancer RF + MLP comparison, or the wine query) →
   confirm it now reaches a genuine `Final Answer` → this is your real Task 1 multi-tool trace.
3. Re-run `benchmark_runner.py` → confirm it actually trains all 6 (algorithm × dataset) combos and
   produces the real Markdown table as `Final Answer`, not another stuck loop.
4. Deliberately trigger a tool error (bad `model_type`, bad `n_components`, etc.) and confirm the log
   shows the LLM receiving the error Observation and **changing** its next Action Input (not
   repeating it) → captures genuine Task 3 self-healing proof.
5. Once you have 3+ genuinely clean traces, write the technical report using them + the architecture
   diagram from `step.md` §2.
6. `git init`, commit, push to GitHub; double check `requirements.txt` via `pip freeze` (or confirm
   the Docker-based one is accurate).

> Coordination note: two sessions touched `react_agent.py` today. Before either side makes further
> changes, do a quick diff/read of the current file to make sure you're both working from the same
> version — the file has already accumulated fixes from both sides once (compatible so far, but
> worth confirming next time too).

---

*This file is maintained alongside the code — whenever I make changes to `ml_tools.py`, `react_agent.py`, `benchmark_runner.py`, or run/capture logs, I'll update the checkboxes above.*
