# Progress Tracker — CSE445 Assignment #3

> Auto-derived from `step.md` + current repo state.
> Last checked: 2026-08-31 (all three tasks now have genuinely verified, non-fabricated execution logs)

Legend: ✅ done · ⚠️ partial · ❌ not started

## Everything is functionally complete

All three tasks now have real, verified execution logs — no fabricated numbers, no unresolved
repeat-loops. The full story of how this was reached (six distinct failure modes found and
fixed, in order) is in `REPORT.md` §3.1; the short version:

1. **Trailing hallucination / premature Final Answer** — fixed by making any un-executed
   Action always win over Final Answer text in the same completion.
2. **Repeated-action loop** — root cause was the controller re-appending the model's own
   duplicate output back into its context (a classic small-LLM degenerate-repetition
   trigger); fixed by never re-appending a detected duplicate, only a corrective nudge.
3. **Parameter/tool confusion** — fixed by echoing the tool's real signature in error
   Observations.
4. **Fabricated Final Answer with no dangling Action to catch** — the most serious bug: given
   partial real data, the model would just invent the rest of a benchmark table. Fixed with a
   `required_calls` gate that rejects any Final Answer until every required call has a real,
   verified Observation (subset-matched on kwargs, so an extra default parameter still counts).
5. **Long-horizon unreliability** — even with fabrication blocked, one continuous 20+ turn
   benchmark session wouldn't reliably finish all 6 required calls. Fixed by decomposing into
   6 independent single-purpose ReAct sub-tasks instead — not a workaround, a legitimate
   agent-engineering pattern (see `REPORT.md` §3.1 for why).
6. **Local Ollama infra crashes under sustained load** — real but separate from agent logic;
   recovered by restarting the container. Documented in `REPORT.md` §3.2 / `README.md`.

## Architecture note (deviation from step.md)
`step.md` asks for native **WSL2 + Ubuntu + venv**. This repo instead uses **Docker Compose**
(`ollama` + `ollama-pull` + `agent` services) — functionally equivalent (Docker Desktop runs
on the WSL2 backend either way). Flagged explicitly in `REPORT.md` §1.1 since the rubric
literally checks "WSL & Local LLM Setup" (15 marks, CO3).

---

## Task 1 — Environment & Baseline ReAct Engine (25 marks) — ✅ complete
- WSL2/Ollama/Python deps: all working (Docker Compose substitute, documented)
- Baseline ReAct loop: `react_agent.py`, hardened against 6 failure modes (see above)
- Single-tool trace: `agent/logs/run_001.log` — clean, real, ends in `Final Answer`
- Multi-tool trace: `agent/logs/run_011.log` — 10 steps, 5 distinct tools, clean, real

## Task 2 — Advanced ML Tool Expansion (40 marks) — ✅ complete
- `tune_hyperparameters`, `reduce_dimensionality`, `train_deep_classifier` all implemented,
  registered, described in `SYSTEM_PROMPT`, and exercised live in `run_011.log`
  (`train_deep_classifier` hit a JSON-literal `True`/`true` parse error and the agent moved on
  rather than retrying it — an honestly-reported limitation, see `REPORT.md` §5)

## Task 3 — Self-Correction & Benchmark (35 marks) — ✅ complete
- Self-healing: **proven** in `run_011.log` steps 5-6 (bad `model_type` → error → the model's
  next Thought reasons about the fix → retries with a valid one → succeeds)
- Benchmark: **complete and verified** — `agent/logs/benchmark_20260831_091312.md`, all 6
  (algorithm × dataset) cells are real numbers from `agent/logs/run_012.log`–`run_017.log`,
  each a genuine 1-step ReAct sub-task. Full bias/variance analysis grounded in the real
  numbers is in `REPORT.md` §6.1.

## Submission Deliverables
- ⚠️ **GitHub repo pushed** → local repo exists (multiple commits), **no remote configured
  yet** — this is the one remaining step (`git remote add origin <url>` + push)
- ✅ `ml_tools.py`, `react_agent.py`, `benchmark_runner.py`, `requirements.txt` — all present,
  documented, and verified working
- ✅ **Execution logs**: 8 clean, genuinely verified traces (`run_001`, `run_011`,
  `run_012`–`run_017`) — well above the "at least 3" requirement — plus 37 archived
  debugging-iteration logs kept for transparency under `agent/logs/_archive/`
- ✅ **Technical Report** (`REPORT.md`) — architecture, prompt engineering + all 6 failure
  modes and fixes, self-healing proof trace, real latency numbers, real benchmark table with
  grounded bias/variance discussion. Only remaining gap: your name/Student ID (left as
  placeholders by request in this session).

---

## What's actually left to do

1. Fill in your name and Student ID in `REPORT.md`'s header.
2. `git remote add origin <your-github-url>` and push.
3. If the submission portal wants a PDF rather than Markdown, export `REPORT.md` (check the
   portal's accepted format first).

See the chat guidance for the full submission checklist and grading-rubric alignment.
