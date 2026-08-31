# CSE445 Assignment #3 — Step-by-Step Bible
**Building an Autonomous Local LLM ML Agent in Windows WSL**

> Source: `CSE445 Assignment#3 Building an Autonomous Local LLM ML Agent in Windows WSL.pdf`
> Course: CSE445 – Machine Learning | Section 6 | Instructor: Dr. Mohammad Abdul Qayum (MAQm)
> Total Marks: **100** | Type: Assignment / Project Component

---

## 0. TL;DR — What You're Building

A **fully local, privacy-preserving ReAct agent** that runs inside **WSL2 (Ubuntu)**, talks to a
**quantized LLM via Ollama** (e.g. `llama3.2:3b`), and uses that LLM to reason about which
**Scikit-Learn / PyTorch tool** to call in order to solve ML tasks (train models, tune
hyperparameters, reduce dimensionality, self-correct on errors, and benchmark results).

No paid APIs. No cloud. Everything runs on your machine inside WSL.

Final deliverables (see §6): GitHub repo + execution logs + a 3–5 page technical report.

---

## 1. Learning Outcomes (why each part matters)

- **Local LLM Inference & Architecture** — run a quantized LLM inside Linux (WSL2), no external APIs.
- **Agentic Reasoning (ReAct)** — implement Thought → Action → Action Input → Observation → Final Answer from scratch (no LangChain-style black boxes).
- **ML Workflow Automation** — give the LLM "hands": Python tools wrapping Scikit-Learn + PyTorch.
- **Robust Error Handling** — catch hallucinated tool calls / bad params, parse structured output, self-correct.

---

## 2. System Architecture (mental model)

```
User Query
   │
   ▼
┌─────────────────────────────────────────────┐
│  react_agent.py  (Python 3.10+ ReAct loop)   │
│  - builds prompt w/ SYSTEM_PROMPT + history  │
│  - sends prompt to Ollama REST API           │
│  - parses "Action:" / "Action Input:"        │
│  - calls matching tool from ml_tools.py      │
│  - appends "Observation:" and loops          │
│  - stops when "Final Answer:" appears        │
└───────────────┬───────────────────────────────┘
                │ HTTP POST :11434/api/generate
                ▼
┌─────────────────────────────────────────────┐
│  Ollama (WSL2, port 11434)                   │
│  Quantized model: llama3.2:3b / mistral:7b   │
└─────────────────┬─────────────────────────────┘
                  │ tool_name + kwargs (JSON)
                  ▼
┌─────────────────────────────────────────────┐
│  ml_tools.py                                 │
│  Scikit-Learn + PyTorch + Pandas + NumPy     │
│  load_dataset_summary / train_sklearn_model  │
│  train_pytorch_mlp / (+3 new tools, Task 2)  │
└─────────────────────────────────────────────┘
```

| Component | Technology | Role |
|---|---|---|
| OS / Subsystem | Windows WSL2 (Ubuntu 22.04/24.04) | Native Linux kernel, GPU passthrough (CUDA) |
| Inference Engine | Ollama, REST API on port `11434` | Runs quantized 4-bit/8-bit models locally |
| Agent Logic | Python 3.10+ custom ReAct controller | Maintains state, prompts LLM, parses actions, executes tools |
| ML Frameworks | PyTorch, Scikit-Learn, Pandas, NumPy | Preprocessing, model fitting, metrics, predictions |

---

## 3. Hands-On Build Steps

### Step 3.1 — Environment Setup (Windows PowerShell → WSL2)

In **PowerShell (Run as Administrator)**:
```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

Inside the **Ubuntu WSL terminal**:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv curl build-essential git

# Dedicated workspace
mkdir -p ~/cse445_agent && cd ~/cse445_agent
python3 -m venv venv
source venv/bin/activate
```

### Step 3.2 — Install & Serve Local LLM via Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service (background or separate terminal tab)
ollama serve &

# Pull a lightweight instruction-tuned model
ollama pull llama3.2:3b
```
> Alternative models mentioned: `mistral:7b`, `phi3`. Stick with `llama3.2:3b` unless you have GPU headroom — smaller = faster iteration.

### Step 3.3 — Install Python Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn pandas numpy requests pydantic
```
(Swap the `--index-url` for a CUDA build if you have GPU passthrough configured.)

### Step 3.4 — Build `ml_tools.py` (baseline tools)
Create `ml_tools.py` with:
- `DATASETS` registry → `iris`, `wine`, `breast_cancer` (from `sklearn.datasets`)
- `load_dataset_summary(dataset_name)` → JSON summary (shape, features, classes, missing values)
- `train_sklearn_model(dataset_name, model_type, test_size=0.2)` → supports `decision_tree`, `logistic_regression`, `random_forest`; returns test accuracy + 5-fold CV mean/std
- `train_pytorch_mlp(dataset_name, hidden_dim=32, epochs=50, lr=0.01)` → standardizes features, trains a `Linear → ReLU → Linear` MLP with Adam + CrossEntropyLoss, returns final loss + test accuracy
- `AVAILABLE_TOOLS` dict mapping tool name string → callable

> Full reference code is in the PDF §3.4 / p.3–6 — copy the structure, don't need to retype from scratch, but understand every function (you'll extend this in Task 2).

### Step 3.5 — Build `react_agent.py` (the ReAct controller)
Create `react_agent.py` with:
- `OLLAMA_URL = "http://127.0.0.1:11434/api/generate"`, `MODEL_NAME = "llama3.2:3b"`
- `SYSTEM_PROMPT` describing available tools + the strict output format:
  ```
  Thought: ...
  Action: <tool_name>
  Action Input: {"param": "value"}
  ```
  and the terminal format:
  ```
  Thought: I have gathered all necessary experimental data.
  Final Answer: ...
  ```
- `query_local_llm(prompt)` → POSTs to Ollama with `temperature=0.1`, `stop=["Observation:"]`
- `run_agent_loop(user_query, max_iterations=6)`:
  1. Build initial prompt = `SYSTEM_PROMPT + user query`
  2. Loop up to `max_iterations`:
     - Query LLM, print raw output, append to prompt
     - If `"Final Answer:"` present → done, break
     - Else regex-parse `Action:` and `Action Input: {...}`
     - `json.loads` the action input; on failure, feed back a parse-error Observation and `continue`
     - If tool name is in `AVAILABLE_TOOLS`, call it with `**kwargs`, wrap result/exception as `Observation:`
     - Append Observation to prompt, loop again
- Test with:
  ```python
  test_task = "Analyze the breast_cancer dataset, train a Random Forest and a PyTorch MLP on it, compare their accuracies, and recommend the best model for clinical screening."
  run_agent_loop(test_task)
  ```

**Checkpoint:** run `python react_agent.py` and confirm you see multiple `--- Step N ---` blocks with real `Thought/Action/Action Input/Observation` cycles ending in a `Final Answer:`.

---

## 4. Graded Tasks — Your Actual To-Do List

### ✅ Task 1 — Environment & Baseline ReAct Engine (25 marks)
- [ ] WSL2 + Ubuntu installed and working
- [ ] Ollama installed, `ollama serve` running, `llama3.2:3b` (or chosen model) pulled
- [ ] Python venv with PyTorch (CPU or CUDA) + Scikit-Learn installed
- [ ] Baseline ReAct loop implemented per §3.5
- [ ] Verify **single-tool** execution trace (e.g. just `load_dataset_summary`)
- [ ] Verify **multi-tool** execution trace (e.g. summary → train_sklearn_model → train_pytorch_mlp)
- [ ] Save these traces — you'll need them for deliverables (§5)

### ✅ Task 2 — Advanced ML Tool Expansion (40 marks)
Add **3 new tools** to `ml_tools.py` and register them in `AVAILABLE_TOOLS`, and describe them in `SYSTEM_PROMPT` so the agent knows they exist:

1. **Hyperparameter Tuning Tool**
   - Use `GridSearchCV` or `RandomizedSearchCV`
   - Targets: `SVC` (Kernel SVM) and `DecisionTreeClassifier`
   - Return best params + best CV score as JSON

2. **Feature Selection & Dimensionality Reduction Tool**
   - Implement `PCA` (Principal Component Analysis)
   - Implement a sequential feature selection pipeline (`SequentialFeatureSelector` from `sklearn.feature_selection`)
   - Return explained variance ratio / selected feature indices as JSON

3. **Deep PyTorch Classifier with Regularization**
   - Configurable PyTorch module with `Dropout`, `BatchNorm1d`, and a learning-rate scheduler (e.g. `StepLR` / `ReduceLROnPlateau`)
   - Parameterize depth/width/dropout rate so the agent can tune it
   - Return training curve summary + final test accuracy as JSON

> Design note: keep every new tool's signature simple (JSON-serializable args, string return) so the existing regex-based Action Input parser keeps working without changes.

### ✅ Task 3 — Self-Correction & Model Comparison Benchmark (35 marks)
- [ ] **Self-healing logic** in `react_agent.py`: when a tool raises (shape mismatch, bad param, NaN loss), catch it, feed a descriptive `Observation:` back to the LLM, and let it retry with corrected `Action Input` — verify this actually happens in a captured trace (don't just handle exceptions, prove the agent *reasons* about the fix).
- [ ] Build `benchmark_runner.py`: a script/prompt where the agent autonomously:
  - Evaluates **3 algorithms** (e.g. decision tree, random forest, PyTorch MLP/deep classifier)
  - Across **2 datasets** (e.g. `wine` + `breast_cancer`)
  - Performs cross-validation for each
  - Writes a **Markdown experimental summary table** (model × dataset × accuracy × CV mean/std) as the `Final Answer`

---

## 5. Submission Deliverables Checklist

- [ ] **GitHub repo** containing:
  - [ ] `ml_tools.py`
  - [ ] `react_agent.py`
  - [ ] `benchmark_runner.py`
  - [ ] `requirements.txt`
  - [ ] Well-documented code (docstrings + comments)
- [ ] **Execution logs**: recorded terminal output (or exported `.txt`) showing **at least 3 multi-step agent reasoning traces** with real tool execution (Task 1 + Task 3 traces work well here)
- [ ] **Technical Report** (PDF or Markdown, 3–5 pages) covering:
  - [ ] Local LLM architecture, prompt engineering techniques used, latency benchmarks measured in WSL2
  - [ ] Mathematical/statistical comparison of models evaluated by the agent (ties to CO1 + CO2 — discuss bias/variance, CV mean/std, accuracy trade-offs)
  - [ ] Architectural diagram of the agent controller loop + tool registry (can reuse/expand the diagram in §2 above)

---

## 6. Grading Rubric (100 marks total)

| Criteria | Marks | CO | What's being checked |
|---|---|---|---|
| WSL & Local LLM Setup | 15 | CO3 | Quantized model runs successfully in WSL via Ollama |
| ReAct Loop & Action Parsing | 25 | CO3 | Accurate Thought/Action/Input parsing, robust execution, no manual intervention |
| PyTorch & Scikit-Learn Tool Implementation | 30 | CO1, CO3 | Correct estimators, NN architectures, validation splits, metric summaries |
| Statistical Analysis & Agent Reasoning Quality | 20 | CO2 | Sound interpretation of CV results, variance, trade-offs in Final Answers |
| Report & Engineering Quality | 10 | CO3 | Documentation clarity, repo structure, reproducibility |
| **Total** | **100** | — | |

---

## 7. Suggested Execution Order (fastest path to done)

1. Get WSL2 + Ollama + venv working, confirm `ollama run llama3.2:3b` responds in terminal.
2. Write `ml_tools.py` baseline (3 tools) → sanity test each function directly in a Python shell (no agent yet).
3. Write `react_agent.py` baseline loop → run the sample multi-tool query → capture trace #1 (Task 1 done).
4. Add the 3 Task 2 tools to `ml_tools.py`, update `SYSTEM_PROMPT` tool list, sanity test directly.
5. Run agent queries that exercise the new tools → capture trace #2.
6. Add self-healing (try/except → corrective Observation) to `react_agent.py`; deliberately trigger an error (e.g. bad `model_type`) to prove retry works → capture trace #3.
7. Write `benchmark_runner.py` with the 3-algorithms × 2-datasets prompt → capture the Markdown summary table output.
8. Write the technical report using your captured logs + a diagram.
9. Push everything to GitHub, double-check `requirements.txt` is accurate (`pip freeze > requirements.txt`), write a clear `README.md` with setup/run instructions.

---

## 8. Common Pitfalls to Avoid

- **Ollama not running** → agent gets connection errors. Always `ollama serve &` before running the agent.
- **Model not pulled** → `ollama pull llama3.2:3b` first; check with `ollama list`.
- **LLM ignoring the strict format** → keep `temperature` low (0.1), consider few-shot examples in `SYSTEM_PROMPT` if parsing keeps failing.
- **Regex parser breaks on multi-line/nested JSON** → keep tool `Action Input` JSON flat and simple.
- **Forgetting to update `SYSTEM_PROMPT`** after adding Task 2 tools — the LLM won't know they exist otherwise.
- **No real self-correction trace** — an empty `try/except` that just prints an error won't satisfy Task 3; the log must show the LLM receiving the failure Observation and changing its next Action Input.
- **CV scores without interpretation** — Task 3 and the report explicitly need statistical reasoning (CO2), not just raw numbers.
