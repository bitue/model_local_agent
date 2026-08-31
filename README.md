# CSE445 Assignment #3 — Autonomous Local LLM ML Agent (Dockerized)

Fully containerized implementation. See [`step.md`](step.md) for the full
assignment breakdown/checklist this repo satisfies.

## Stack

- `ollama` — Ollama server (`ollama/ollama` image), REST API on `:11434`
- `ollama-pull` — one-shot init container that pulls the model once
- `agent` — Python 3.10 container with the ReAct agent + ML tools (CPU-only PyTorch)

## Prerequisites

- Docker Desktop (with Compose v2, i.e. `docker compose`, not the old `docker-compose`)
- ~4 GB free disk for the model + CPU torch wheels

## 1. First-time setup

```bash
# Start Ollama and let it pull the model (llama3.2:3b by default, see .env)
docker compose up -d ollama
docker compose up ollama-pull        # waits for ollama healthy, then pulls, then exits 0

# Build the agent image
docker compose build agent
```

Check the model is present:
```bash
docker compose exec ollama ollama list
```

## 2. Run the agent

One-shot query (recommended — easiest way to capture execution-log traces):
```bash
docker compose run --rm agent python react_agent.py "Analyze the breast_cancer dataset, train a Random Forest and a PyTorch MLP on it, compare their accuracies, and recommend the best model for clinical screening."
```

Interactive REPL:
```bash
docker compose run --rm -it agent python react_agent.py
```

Task 3 benchmark (3 algorithms × 2 datasets, Markdown summary table):
```bash
docker compose run --rm agent python benchmark_runner.py
```

Every run writes a full trace to `agent/logs/run_NNN.log` (or `benchmark_*.md`)
on your host machine (mounted volume) — these are your execution-log deliverables.

## 3. Shut down

```bash
docker compose down          # stop containers, keep the pulled model (named volume)
docker compose down -v       # also delete the model volume (re-pull needed next time)
```

## Project layout

```
model_local_agent/
├── docker-compose.yml
├── .env                     # OLLAMA_MODEL=llama3.2:3b
├── agent/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── ml_tools.py          # 6 tools: 3 baseline (Task 1) + 3 advanced (Task 2)
│   ├── react_agent.py       # ReAct loop + self-healing (Task 3)
│   ├── benchmark_runner.py  # 3-algorithm x 2-dataset benchmark (Task 3)
│   └── logs/                # execution traces land here
├── step.md                  # assignment breakdown / checklist
└── README.md
```

## Tool registry (`ml_tools.py`)

| Tool | Task | Purpose |
|---|---|---|
| `load_dataset_summary` | 1 | Dataset shape/features/classes summary |
| `train_sklearn_model` | 1 | decision_tree / logistic_regression / random_forest |
| `train_pytorch_mlp` | 1 | Simple PyTorch MLP baseline |
| `tune_hyperparameters` | 2 | GridSearchCV / RandomizedSearchCV over SVC or DecisionTree |
| `reduce_dimensionality` | 2 | PCA or Sequential Feature Selection |
| `train_deep_classifier` | 2 | Configurable MLP w/ Dropout, BatchNorm, LR scheduler |

## Self-correction (Task 3)

`react_agent.py` catches every tool exception and every explicit `{"error": ...}`
payload (bad dataset/model name, invalid `n_components`, NaN loss, etc.) and
feeds it back to the LLM as an `Observation` that names the problem and hints
at a valid fix, instead of a bare stack trace — so the next turn is a genuine
corrected retry, not a repeat of the same failing call.

## Notes for the technical report

- Model/prompt config lives in `react_agent.py` (`SYSTEM_PROMPT`, `MODEL_NAME`, `OLLAMA_URL`).
- Latency: timestamps are written at the start of every `logs/run_*.log`; time
  between steps in the same log gives you per-turn LLM latency for the report.
- Architecture diagram: see `step.md` §2 for the base diagram to extend.
