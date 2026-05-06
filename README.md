# MindSight

Project workspace for Mental Health Data Analysis — AL2002 Track D.

## Setup

Run `scripts/setup_env.bat` (Windows) or install requirements via `pip install -r requirements.txt`.

## Data

Data is located in `data/Track_D_Mental_Health/`.

---

## How to Run

Execute each phase from the project root. Each command generates the notebook then executes it in-place.

**Phase 1 — Data Exploration:**
```bash
cd MindSight && python scripts/generate_phase1.py && \
jupyter nbconvert --to notebook --execute phase1/phase1_exploration.ipynb --inplace --ExecutePreprocessor.timeout=120
```

**Phase 2 — Search Algorithms:**
```bash
cd MindSight && python scripts/generate_phase2.py && \
jupyter nbconvert --to notebook --execute phase2/phase2_search.ipynb --inplace --ExecutePreprocessor.timeout=120
```

**Phase 3 — Constraint Satisfaction:**
```bash
cd MindSight && python scripts/generate_phase3.py && \
jupyter nbconvert --to notebook --execute phase3/phase3_csp.ipynb --inplace --ExecutePreprocessor.timeout=120
```

**Phase 4 — Logic and Reasoning:**
```bash
cd MindSight && python scripts/generate_phase4.py && \
jupyter nbconvert --to notebook --execute phase4/phase4_logic.ipynb --inplace --ExecutePreprocessor.timeout=120
```

**Phase 5 — Machine Learning:**
> ⚠️ Phase 5 requires a longer timeout due to iterative training loops.

```bash
cd MindSight && python scripts/generate_phase5.py && \
jupyter nbconvert --to notebook --execute phase5/phase5_ml.ipynb --inplace --ExecutePreprocessor.timeout=300
```

---

## Submission

Pack the following contents into a ZIP archive before submitting:

```
phase1/
phase2/
phase3/
phase4/
phase5/
data/
docs/
scripts/
README.md
```

**ZIP naming convention:** `StudentID_D_AL2002.zip`

Example: `20231234_D_AL2002.zip`
