# MindSight

![Build](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)
![Version](https://img.shields.io/badge/version-1.0.0-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

MindSight is a five-phase end-to-end artificial intelligence pipeline that analyzes the OSMI Mental Health in Tech Survey dataset (2016–2022) to predict whether a technology-sector worker seeks professional mental health treatment.
The pipeline implements search algorithms, constraint satisfaction, propositional logic, and supervised/unsupervised machine learning entirely from scratch using NumPy.
No high-level ML frameworks (scikit-learn, TensorFlow, PyTorch) are used at any stage.

---

## Objective and Trade-offs

**Objective:** Determine which workplace and demographic factors most reliably predict mental health treatment-seeking behavior in the technology industry, using classical and modern AI paradigms applied sequentially across five analytical phases.

**Key trade-offs:**

| Decision | Chosen Approach | Rationale |
|:---|:---|:---|
| ML framework | NumPy only | Demonstrates algorithmic mechanics; satisfies course constraint |
| Missing data | Drop columns > 50% sparse | Prevents synthetic signal injection; preserves data integrity |
| Text fields | Excluded from features | NLP is out of scope; avoids noise from unstructured data |
| Encoding | Label encoding | Sufficient for tree-based and linear classifiers on ordinal data |
| MLP architecture | Single hidden layer | Dataset size (≈3 k rows) does not justify deeper networks |

---

## Features

| Feature Module | Underlying Mechanism | System Benefit |
|:---|:---|:---|
| Graph Search (Phase 2) | BFS, DFS, UCS, IDDFS, A\*, Greedy, Hill Climbing, SA, Minimax, Alpha-Beta | Maps symptom-to-treatment paths; benchmarks 10 algorithm variants |
| Constraint Satisfaction (Phase 3) | AC-3 arc consistency, backtracking (MRV, forward checking), min-conflicts | Models logical employer-support constraints; prunes invalid solution domains |
| Logic and Reasoning (Phase 4) | Propositional rules, forward chaining, CNF theorem proving | Produces human-readable treatment-prediction rules with accuracy metrics |
| Machine Learning (Phase 5) | Perceptron, Delta Rule, MLP (backprop), K-Means, K-Medoid | Generates quantitative accuracy benchmarks across five model architectures |
| Pipeline Automation | `scripts/run_all.py` with `asyncio.WindowsSelectorEventLoopPolicy` | Regenerates and executes all notebooks in one command; eliminates ZMQ errors on Windows |

---

## Prerequisites

| Software | Required Version | Installation Source |
|:---|:---|:---|
| Python | >= 3.10 | [python.org](https://www.python.org/downloads/) |
| pip | >= 23.0 | Bundled with Python |
| Jupyter | >= 7.0 | `pip install jupyter` |
| pandas | >= 2.0 | `pip install pandas` |
| numpy | >= 1.26 | `pip install numpy` |
| matplotlib | >= 3.8 | `pip install matplotlib` |
| nbformat | >= 5.9 | `pip install nbformat` |
| nbconvert | >= 7.14 | `pip install nbconvert` |

---

## Installation

**Clone the repository:**

```bash
git clone https://github.com/Abdu1-Ahd/MindSight.git
cd MindSight
```

**Install dependencies (Windows):**

```bash
scripts\setup_env.bat
```

**Install dependencies (Linux / macOS):**

```bash
pip install -r requirements.txt
```

---

## Configuration

MindSight requires no external API keys or secrets.
The dataset is bundled under `data/Track_D_Mental_Health/`.

| Variable | Type | Required | Description |
|:---|:---|:---|:---|
| `DATA_PATH` | `str` | No | Override default dataset directory. Defaults to `data/Track_D_Mental_Health/`. |

---

## Usage

**Execute the complete pipeline:**

```bash
python scripts/run_all.py
```

**Expected terminal output:**

```text
--- Regenerating Notebooks ---
Running scripts/generate_phase1.py...
Running scripts/generate_phase2.py...
Running scripts/generate_phase3.py...
Running scripts/generate_phase4.py...
Running scripts/generate_phase5.py...

--- Executing Notebooks ---
Executing phase1/phase1_foundation.ipynb...
  -> Saved phase1/phase1_foundation.ipynb
Executing phase2/phase2_search.ipynb...
  -> Saved phase2/phase2_search.ipynb
Executing phase3/phase3_csp.ipynb...
  -> Saved phase3/phase3_csp.ipynb
Executing phase4/phase4_logic.ipynb...
  -> Saved phase4/phase4_logic.ipynb
Executing phase5/phase5_ml.ipynb...
  -> Saved phase5/phase5_ml.ipynb

All phases completed successfully without ZMQ socket errors!
```

> **Note:** Phase 5 completes the MLP training loop (200 epochs) and requires approximately 3–5 minutes.

---

## Architecture

```
MindSight/
├── data/
│   └── Track_D_Mental_Health/     # Seven OSMI survey CSV files (2016–2022) + SQLite DB
├── docs/
│   └── MindSight_Report.docx      # Formal research report submitted with coursework
├── phase1/
│   └── phase1_foundation.ipynb    # Data loading, merging, EDA, graph construction
├── phase2/
│   └── phase2_search.ipynb        # 10 search algorithm implementations + benchmarks
├── phase3/
│   └── phase3_csp.ipynb           # CSP variables, AC-3, backtracking variants, min-conflicts
├── phase4/
│   └── phase4_logic.ipynb         # Propositional rules, forward chaining, CNF proof
├── phase5/
│   └── phase5_ml.ipynb            # Perceptron, Delta Rule, MLP, K-Means, K-Medoid
├── scripts/
│   ├── generate_phase1.py         # Generates phase1_foundation.ipynb from source
│   ├── generate_phase2.py         # Generates phase2_search.ipynb from source
│   ├── generate_phase3.py         # Generates phase3_csp.ipynb from source
│   ├── generate_phase4.py         # Generates phase4_logic.ipynb from source
│   ├── generate_phase5.py         # Generates phase5_ml.ipynb from source
│   ├── run_all.py                 # Master runner: regenerate + execute all phases
│   ├── setup_env.bat              # Windows dependency installer
│   └── build.sh                   # POSIX dependency installer
├── .github/
│   ├── CONTRIBUTING.md            # Branch strategy, commit conventions, PR protocol
│   ├── SECURITY.md                # Vulnerability disclosure policy
│   ├── CODEOWNERS                 # Directory-to-reviewer mappings
│   ├── pull_request_template.md   # PR checklist template
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.yml         # Structured bug report form
│       └── feature_request.yml    # Structured feature request form
├── CHANGELOG.md                   # Keep a Changelog formatted release history
├── DEVELOPERS.md                  # Internal engineering onboarding guide
├── LICENSE                        # MIT License
├── SKILL.md                       # AI agent architectural context
├── .gitignore                     # Exclusion rules: __pycache__, .env, OS artifacts
├── .gitattributes                 # Line ending normalization + linguist overrides
└── README.md                      # This file
```

---

## Deployment

**Package for submission:**

```bash
# 1. Create submission directory
mkdir ../24F-0727_D_AL2002

# 2. Copy required files
Copy-Item -Path "phase1","phase2","phase3","phase4","phase5","data","docs","scripts","README.md" `
    -Destination "../24F-0727_D_AL2002/" -Recurse -Force

# 3. Create ZIP archive
Compress-Archive -Path "../24F-0727_D_AL2002" -DestinationPath "../24F-0727_D_AL2002.zip" -Force
```

**Naming convention:** `<StudentID>_D_AL2002.zip`
