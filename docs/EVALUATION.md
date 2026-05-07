# Evaluation Methodology

Quantitative results, baseline comparisons, and hyperparameter justifications for MindSight.

---

## Baseline

**Majority-class classifier:** Always predict the most frequent class (`treatment = 1`).

| Metric | Value |
|:---|:---|
| Baseline accuracy | 58.05% |
| Reasoning | 58.05% of the training set sought treatment. Predicting "yes" for every sample achieves this score with zero learning. |

Any model that fails to exceed 58.05% test accuracy provides no predictive value beyond random guessing biased toward the majority class.

---

## Supervised Model Results

| Model | Test Accuracy | vs. Baseline | Training Epochs | Learning Rate | Architecture |
|:---|---:|:---|---:|:---|:---|
| Perceptron | 66.67% | +8.62 pp | 50 | 0.1 | Linear, step activation |
| Delta Rule | **84.13%** | **+26.08 pp** | 100 | 0.01 | Linear, continuous sigmoid output |
| MLP | 71.32% | +13.27 pp | 200 | 0.01 | 2 hidden layers (16 → 8), ReLU + sigmoid |

**Best performer:** Delta Rule at 84.13% test accuracy.

---

## Unsupervised Model Results

| Model | Cluster Purity | vs. Baseline | Convergence Iterations | K |
|:---|---:|:---|---:|---:|
| K-Means | 58.05% | +0.00 pp | 28 | 2 |
| K-Medoid | 58.05% | +0.00 pp | 2 | 2 |

**Finding:** Both clustering methods produce purity scores exactly equal to the majority-class frequency. This means the discovered clusters do not align with the treatment variable — the feature space lacks cluster-separable structure for this target.

| Metric | K-Means | K-Medoid |
|:---|---:|---:|
| Within-Cluster Sum of Distances (WCSD) | 10,941 | 12,554 |
| Winner | **K-Means** | — |

K-Means achieves lower WCSD because it optimizes centroids as mean vectors (unconstrained), while K-Medoid restricts centers to actual data points.

---

## Search Algorithm Results (Phase 2)

| Algorithm | Path Length | Nodes Explored | Cost |
|:---|---:|---:|---:|
| BFS | 2 | 3 | 1 |
| DFS | 2 | 2 | 1 |
| IDDFS | 2 | 3 | 1 |
| UCS | 2 | 3 | 1 |
| Best-First | 2 | 2 | — |
| A* | 2 | 2 | 1 |

**Adversarial search:**

| Metric | Minimax | Alpha-Beta |
|:---|---:|---:|
| Nodes evaluated | 133 | 45 |
| Nodes pruned | — | 88 (66.2%) |

**Local search:**

| Algorithm | Success Rate |
|:---|:---|
| Hill Climbing | 70% (7/10 runs) |
| Simulated Annealing | 100% (1/1 run) |

---

## Hyperparameter Justification

| Parameter | Value | Rationale |
|:---|:---|:---|
| Perceptron LR | 0.1 | Standard starting point for step-function classifiers. Higher LR acceptable since updates only occur on misclassification. |
| Delta Rule LR | 0.01 | Conservative to prevent overshooting with continuous gradient descent on 18 features. |
| MLP LR | 0.01 | Matched to Delta Rule for fair comparison. Lower rates tested but converged too slowly within 200 epochs. |
| MLP hidden layers | 16 → 8 | Dataset has 2,746 training samples and 18 features. A wider or deeper network risks memorizing noise. Single hidden layer variants were tested; two layers provided marginal improvement. |
| MLP epochs | 200 | Loss curve plateaus around epoch 150. Additional epochs yield < 0.5% accuracy gain. |
| K (clustering) | 2 | Matches the binary target variable (treatment/no treatment). |
| Perceptron epochs | 50 | Accuracy stabilizes after ~30 epochs on this dataset. 50 provides convergence margin. |
| Delta Rule epochs | 100 | MSE reaches 0.1462 at epoch 100. Further training shows diminishing returns (< 0.001 MSE reduction per epoch). |
| Weight initialization | He (MLP), zeros (Perceptron), N(0, 0.01) (Delta) | He initialization prevents dead ReLU neurons. Zero init works for single-layer models with step/linear activation. |
| Train/test split | 80/20 | Standard ratio. Fixed `random_state=42` for reproducibility. |
| Missing data strategy | Median (numeric), "Unknown" (categorical) | Avoids mean imputation bias on skewed distributions. "Unknown" preserves information that data was missing. |

---

## Why Delta Rule Outperforms MLP

The Delta Rule achieves 84.13% while the MLP reaches only 71.32%. This is counterintuitive — deeper models typically outperform linear ones.

**Explanation:** The feature space after label encoding and z-score normalization is approximately linearly separable for the `treatment` target. The MLP's nonlinear capacity provides no benefit and likely overfits to training noise, particularly given:

- Only 2,746 training samples
- 18 features (many ordinal-encoded with limited value ranges)
- No regularization (dropout, weight decay) implemented
- No validation set for early stopping

The Delta Rule, being a simple linear model with continuous output, finds the optimal hyperplane without overfitting.

---

## Limitations

| Limitation | Impact |
|:---|:---|
| No cross-validation | Results depend on a single 80/20 split. Variance across splits is unknown. |
| No regularization in MLP | Likely contributes to the MLP underperforming the Delta Rule. |
| No feature importance analysis | Which of the 18 features drive the Delta Rule's accuracy is not investigated. |
| Purity as clustering metric | Purity is a weak metric that ignores cluster homogeneity. Adjusted Rand Index or Normalized Mutual Information would be more informative. |
| No confidence intervals | Single-run accuracy numbers without error bars limit statistical claims. |
