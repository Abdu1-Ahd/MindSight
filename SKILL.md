# MindSight — Agent Context (SKILL.md)

Architectural context for AI coding agents working on this repository.

---

## Project Type

Academic AI pipeline (Python, Jupyter). Five sequential analytical phases applied to a tabular survey dataset.

---

## Architecture Patterns

- **Generator pattern:** Each phase has a `scripts/generate_phase*.py` that writes its `.ipynb` file using `nbformat`. Modifying a phase means modifying its generator, then re-running `python scripts/run_all.py`.
- **No runtime modification:** Notebooks are fully regenerated from source on each run. Do not hand-edit `.ipynb` files directly — changes are overwritten.
- **Sequential dependency:** Phase 2 imports the graph built in Phase 1. Phase 5 uses the feature set established in Phase 1. Maintain consistent variable names across generators.

---

## Constraints

| Constraint | Rule |
|:---|:---|
| ML frameworks | NumPy only. No scikit-learn, TensorFlow, PyTorch, or any ML library. Train/test split uses `np.random.RandomState(42).permutation`. |
| Visualization | Use `matplotlib` with `Agg` backend (non-interactive). Call `matplotlib.use('Agg')` before any other matplotlib import. |
| Data path | Always construct paths with `os.path.join('..', 'data', 'Track_D_Mental_Health')` relative to the phase directory. |
| Event loop | Set `asyncio.WindowsSelectorEventLoopPolicy()` in `run_all.py` before any nbconvert calls. |
| Notebook execution | Use `nbconvert.preprocessors.ExecutePreprocessor` with `timeout=600`. |

---

## Key Variables (Phase 1 → downstream)

| Variable | Type | Description |
|:---|:---|:---|
| `df` | `pd.DataFrame` | Master merged dataset. Shape (3433, 193) before preprocessing. |
| `treatment` | Column (int) | Target variable. 1 = sought treatment, 0 = did not. ~58% positive. |
| `work_interfere` | Column (str) | Primary predictor. Values: Often, Sometimes, Rarely, Never. |
| `adj` | dict | Bipartite adjacency graph linking `work_interfere` levels to `treatment` values. |

---

## Data Pipeline Summary

1. Load 7 CSVs → tag with `year` → `pd.concat` → raw shape (3433, 193).
2. Drop columns with > 50% missing values.
3. Drop free-text columns.
4. Label-encode all remaining categorical columns.
5. Z-score normalize all features: `(x - mean) / (std + 1e-8)`.
6. Train/test split 80/20 via numpy permutation → (2746, 18) train, (687, 18) test.

---

## Model Architectures (Phase 5)

| Model | Key Implementation Details |
|:---|:---|
| Perceptron | Hard step activation; online weight update on misclassification only. |
| Delta Rule | Sigmoid activation; MSE loss; batch gradient descent; lr=0.01. |
| MLP | 2 hidden layers (16 → 8 units), ReLU + sigmoid; He initialization; backpropagation; 200 epochs. |
| K-Means | Euclidean distance; random centroid init; convergence on centroid stability. |
| K-Medoid | Actual data points as medoids; Manhattan distance; swap-based optimization. |
