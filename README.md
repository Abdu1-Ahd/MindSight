# MindSight 🧠

Welcome to MindSight! This is my project workspace for the AL2002 Artificial Intelligence course (Track D: Mental Health in Tech). 

I built an end-to-end AI pipeline from scratch—no scikit-learn allowed!—to figure out what actually drives tech workers to seek professional mental health treatment. It covers everything from graph search algorithms to neural networks.

## Getting Started

1. First, make sure you have the dependencies installed:
   * **Windows:** Just run `scripts/setup_env.bat`
   * **Anywhere else:** Run `pip install -r requirements.txt`

2. The raw OSMI survey data is already hanging out in `data/Track_D_Mental_Health/`. You don't need to download anything else.

---

## How to Run the Whole Thing

I wrote a unified runner script that handles everything for you. It automatically regenerates all five Jupyter notebooks from their clean source scripts, bypasses those annoying Windows Jupyter/ZMQ socket errors, and executes the entire pipeline in-place.

Just open your terminal and run:

```bash
cd MindSight
python scripts/run_all.py
```

*A quick heads-up: Phase 5 takes a few minutes to finish. The Multi-Layer Perceptron needs time to grind through all 200 of its training epochs.*

---

## Prepping for Submission

When it's time to turn this in, just zip up these specific folders and files:

```text
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

**Make sure you name the ZIP file correctly:** `StudentID_D_AL2002.zip` (e.g., `20231234_D_AL2002.zip`).
