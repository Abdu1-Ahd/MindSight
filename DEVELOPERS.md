# MindSight — Developer Guide

Internal engineering reference for local setup, execution, and architectural context.

---

## Dependencies

| Package | Version Range | Role |
|:---|:---|:---|
| Python | >= 3.10 | Runtime |
| pip | >= 23.0 | Package manager |
| pandas | >= 2.0 | Data loading and preprocessing |
| numpy | >= 1.26 | All ML and numerical computation |
| matplotlib | >= 3.8 | Plot generation (non-interactive `Agg` backend) |
| jupyter | >= 1.0 | Notebook server |
| nbformat | >= 5.9 | Notebook serialization in generator scripts |
| nbconvert | >= 7.14 | Notebook execution via `ExecutePreprocessor` |

**Dev dependencies** (linting, testing): see `requirements-dev.txt`.

> **No external ML frameworks.** All model implementations use NumPy only. The train/test split is implemented as a NumPy permutation — no scikit-learn dependency.

---

## Local Setup

**1. Clone the repository:**

```bash
git clone https://github.com/Abdu1-Ahd/MindSight.git
cd MindSight
```

**2. Install dependencies:**

```bash
# Windows
scripts\setup_env.bat

# Linux / macOS
pip install -r requirements.txt
```

**3. Install dev tools (optional):**

```bash
pip install -r requirements-dev.txt
```

**4. Verify dataset files are present:**

```bash
ls data/Track_D_Mental_Health/
# Expected: Mental_Health_Survey_2016.csv through Mental_Health_Survey_2022.csv
```

---

## Running the Pipeline

**Full pipeline (regenerate + execute all notebooks):**

```bash
python scripts/run_all.py
```

**Individual phase execution:**

```bash
jupyter nbconvert --to notebook --execute phase1/phase1_foundation.ipynb --inplace
```

**Output verification:**

```bash
python scripts/check_outputs.py
```

**Run tests:**

```bash
pytest tests/ -v
```

**Lint:**

```bash
ruff check scripts/ tests/
```

---

## Directory → Architectural Function Matrix

| Directory / File | Architectural Role |
|:---|:---|
| `data/Track_D_Mental_Health/` | Raw source data. Read-only during all phases. Never modified by pipeline. |
| `phase1/` | Data foundation: load, merge 7 CSVs, EDA, label encode, construct bipartite graph. |
| `phase2/` | Search layer: implement and benchmark 10 algorithm variants on the phase1 graph. |
| `phase3/` | CSP layer: model workplace constraints; apply AC-3, backtracking, min-conflicts. |
| `phase4/` | Logic layer: derive propositional rules; run forward chaining and CNF theorem prover. |
| `phase5/` | ML layer: train Perceptron, Delta Rule, MLP, K-Means, K-Medoid from scratch in NumPy. |
| `scripts/run_all.py` | Master orchestrator. Sets `WindowsSelectorEventLoopPolicy` then executes all phases sequentially. |
| `scripts/check_outputs.py` | Post-execution verifier. Scans notebook output cells for expected keywords. |
| `docs/` | Research deliverables, architecture docs, dataset card, and evaluation methodology. |
| `tests/` | Pytest suite covering data loading, preprocessing, and model validation. |

For detailed architecture with data flow diagrams, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Windows-Specific Notes

The notebook execution engine (Jupyter `ExecutePreprocessor`) uses Tornado + ZMQ for IPC.
On Windows, Python 3.10+ defaults to `ProactorEventLoop`, which is incompatible with ZMQ socket operations.

`scripts/run_all.py` sets the following at startup to prevent this:

```python
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

This line must appear before any `nbconvert` import or execution call.

---

## GitHub Setup — Manual Steps

The following repository settings require GitHub admin access and cannot be automated via file commits:

1. **Branch protection on `master`:** Require PR, 1 CODEOWNERS approval, strict status checks, linear history, signed commits, no admin bypass.
2. **Custom label taxonomy:** Delete default labels. Create scoped labels per `.github/LABELS.md`.
3. **Semantic Release:** Configure via `.releaserc.json` or GitHub Action once a CI workflow is added.
