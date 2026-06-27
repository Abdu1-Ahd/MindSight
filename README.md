# MindSight

[![CI](https://github.com/Abdu1-Ahd/MindSight/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdu1-Ahd/MindSight/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

MindSight is a five-phase AI pipeline that predicts mental health treatment-seeking behavior among technology workers using the [OSMI Mental Health in Tech Survey](https://osmihelp.org/research) (2016–2022). The pipeline implements search algorithms, constraint satisfaction, propositional logic, and supervised/unsupervised machine learning **entirely from scratch using NumPy** — no scikit-learn, TensorFlow, or PyTorch.

---

## Results

### Supervised Models

| Model | Test Accuracy | vs. Majority Baseline (+58.05%) | Epochs | Learning Rate |
|:---|---:|:---|---:|:---|
| **Delta Rule** | **84.13%** | **+26.08 pp** | 100 | 0.01 |
| MLP (16 → 8 → 1) | 71.32% | +13.27 pp | 200 | 0.01 |
| Perceptron | 66.67% | +8.62 pp | 50 | 0.1 |

### Unsupervised Models

| Model | Cluster Purity | Convergence | WCSD |
|:---|---:|:---|---:|
| K-Means (K=2) | 58.05% | 28 iterations | 10,941 |
| K-Medoid (K=2) | 58.05% | 2 iterations | 12,554 |

> **Note:** Both clustering methods produce purity equal to the majority-class frequency, indicating the feature space lacks cluster-separable structure for the treatment target.

### Search Algorithms (Phase 2)

| Algorithm | Nodes Explored | Path Length |
|:---|---:|---:|
| A* | 2 | 2 |
| Best-First | 2 | 2 |
| BFS | 3 | 2 |
| DFS | 2 | 2 |
| Alpha-Beta pruning | 45 nodes (vs. Minimax 133 — 66% reduction) | — |

Full evaluation methodology, hyperparameter justifications, and ablation discussion: [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## Dataset

**Source:** [OSMI Mental Health in Tech Survey](https://osmihelp.org/research) (CC BY-SA 4.0)

| Property | Value |
|:---|:---|
| Responses | 3,433 across 7 annual waves (2016–2022) |
| Raw columns | 193 (union across survey years) |
| Features retained | 18 (17 categorical + 1 numerical) |
| Target | `treatment` — binary (1 = sought professional help) |
| Class balance | 58.05% positive / 41.95% negative |
| Train / Test | 2,746 / 687 (80/20 split, `random_state=42`) |

Full dataset documentation: [`docs/DATASET.md`](docs/DATASET.md).

---

## Architecture

```
CSV files (7 years) → merge → prune sparse cols → drop free-text →
label-encode → z-score normalize → train/test split → models
```

Each phase is a standard Jupyter notebook. The master runner executes all phases sequentially:

```
phase*/phase*_*.ipynb  →  executed outputs
      (source)              (run_all.py)
```

| Phase | Module | Purpose |
|:---|:---|:---|
| 1 | Foundation | Data loading, merging, EDA, bipartite graph construction |
| 2 | Search | 10 search algorithm implementations (BFS, DFS, UCS, IDDFS, A*, Greedy, Hill Climbing, SA, Minimax, Alpha-Beta) |
| 3 | CSP | Constraint satisfaction: AC-3 arc consistency, backtracking (MRV, forward checking), min-conflicts |
| 4 | Logic | Propositional rules, forward chaining inference engine, CNF theorem proving |
| 5 | ML | Perceptron, Delta Rule, MLP (manual backprop), K-Means, K-Medoid — all from scratch in NumPy |

Full architecture documentation with data flow diagrams: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quickstart

### Prerequisites

- Python >= 3.10
- pip >= 23.0

### Install

```bash
git clone https://github.com/Abdu1-Ahd/MindSight.git
cd MindSight
pip install -r requirements.txt
```

### Run

```bash
python scripts/run_all.py
```

This executes all 5 notebooks sequentially. Phase 5 (MLP training, 200 epochs) takes approximately 3–5 minutes.

### Verify

```bash
python scripts/check_outputs.py
```

### Test

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

### Docker

```bash
docker build -t mindsight .
docker run mindsight
```

---

## Project Structure

```
MindSight/
├── data/Track_D_Mental_Health/   # 7 OSMI CSVs (2016–2022) + SQLite DB
├── docs/
│   ├── ARCHITECTURE.md           # System design, data flow diagrams, module matrix
│   ├── DATASET.md                # Dataset card: source, biases, preprocessing
│   ├── EVALUATION.md             # Benchmarks, baselines, hyperparameter justification
│   └── MindSight_Report.docx     # Formal research report
├── phase1/                       # Foundation: data loading, EDA, graph construction
├── phase2/                       # Search: 10 algorithm implementations
├── phase3/                       # CSP: AC-3, backtracking, min-conflicts
├── phase4/                       # Logic: propositional rules, forward chaining, CNF
├── phase5/                       # ML: Perceptron, Delta Rule, MLP, K-Means, K-Medoid
├── scripts/
│   ├── run_all.py                # Pipeline orchestrator
│   └── check_outputs.py          # Output verification
├── tests/
│   └── test_pipeline.py          # Data loading, preprocessing, model validation
├── .github/
│   ├── workflows/ci.yml          # Lint → Test → Full Pipeline
│   ├── CONTRIBUTING.md           # Branch strategy, commit conventions
│   └── SECURITY.md               # Vulnerability disclosure
├── Dockerfile                    # Containerized reproducibility
├── requirements.txt              # Pinned production dependencies
├── requirements-dev.txt          # Linting and testing tools
├── pyproject.toml                # Ruff + pytest configuration
├── DEVELOPERS.md                 # Engineering onboarding guide
├── CHANGELOG.md                  # Release history (Keep a Changelog)
└── LICENSE                       # MIT
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|:---|:---|:---|
| ML framework | NumPy only | Demonstrates raw algorithmic mechanics without framework abstraction |
| Missing data | Drop columns > 50% sparse | Prevents synthetic signal injection; preserves data integrity |
| Free-text fields | Excluded | NLP is out of scope; avoids noise from unstructured survey responses |
| Encoding | Label encoding | Sufficient for linear classifiers on ordinal survey data |
| MLP architecture | 2 hidden layers (16 → 8) | 2,746 training samples with 18 features does not justify deeper networks |
| Train/test split | Numpy permutation (no sklearn) | Maintains "no external ML frameworks" constraint end-to-end |
| Normalization | Z-score | Standardizes features to zero mean and unit variance for gradient stability |

---

## Limitations

- **No cross-validation.** Results depend on a single 80/20 split. Variance across splits is unknown.
- **No regularization in MLP.** Likely contributes to the MLP underperforming the Delta Rule.
- **Purity as clustering metric.** Purity ignores cluster homogeneity. Adjusted Rand Index would be more informative.
- **Self-selection bias.** Only tech workers who voluntarily participated in the OSMI survey are represented.
- **No feature importance analysis.** Which of the 18 features drive the Delta Rule's 84% accuracy is not investigated.

---

## Contributing

See [`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md) for branch strategy, commit conventions, and PR protocol.

---

## Authors

* **Abdu1-Ahd** — [GitHub Profile](https://github.com/Abdu1-Ahd)
* **Abdul Rahim Subh** — [GitHub Profile](https://github.com/abdulrahim-subh)

---

## License

[MIT](LICENSE)










<!-- dev-sync: 1e15804b | ts: 2026-06-27T13:11:34+0500 -->
