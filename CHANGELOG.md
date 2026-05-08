# Changelog

All notable changes to MindSight are documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Project partner **Abdul Rahim Subh** ([@abdulrahim-subh](https://github.com/abdulrahim-subh)) added as a collaborator and co-author.

### Removed
- `SKILL.md` — internal AI agent context file removed from public repository.
- `scripts/generate_phase*.py` — removed generator scripts. The `.ipynb` files are now the primary source of truth, removing the "AI-generated" look of the repository.

---

## [1.1.0] - 2026-05-07

### Added
- `docs/ARCHITECTURE.md` — system design documentation with Mermaid data flow diagrams, module responsibility matrix, and phase dependency graph.
- `docs/DATASET.md` — formal dataset card documenting OSMI source, biases, preprocessing pipeline, and ethical considerations.
- `docs/EVALUATION.md` — evaluation methodology with benchmark results, baseline comparisons, hyperparameter justifications, and ablation analysis.
- `tests/test_pipeline.py` — pytest suite covering data loading (7 CSVs), preprocessing (no NaN, binary target), and model validation (above-baseline accuracy, convergence).
- `.github/workflows/ci.yml` — three-stage CI pipeline: lint (ruff) → test (pytest) → full pipeline execution.
- `Dockerfile` — containerized reproducibility with Python 3.10-slim.
- `pyproject.toml` — ruff linter and pytest configuration.
- `requirements-dev.txt` — dev dependencies (pytest, ruff).
- `.python-version` — pin Python 3.10 for reproducibility.

### Changed
- **README.md:** Complete rewrite. Surfaces actual benchmark numbers, dataset description, architecture overview, design decisions, limitations, and reproducibility steps.
- **DEVELOPERS.md:** Updated with references to new documentation, test suite, and linting.
- **requirements.txt:** Pinned dependency version ranges. Removed `scikit-learn` and `notebook`.

### Removed
- `sklearn.model_selection.train_test_split` dependency in Phase 5. Replaced with pure NumPy permutation-based split.

### Fixed
- `requirements.txt` listed `scikit-learn` while README claimed "no ML frameworks" — resolved by replacing with NumPy implementation.

---

## [1.0.0] - 2026-05-07

### Added
- Phase 1: Data foundation — seven-year OSMI CSV merge, EDA, bipartite graph construction.
- Phase 2: Search algorithms — BFS, DFS, UCS, IDDFS, A\*, Greedy, Hill Climbing, Simulated Annealing, Beam Search, Minimax, Alpha-Beta pruning.
- Phase 3: Constraint satisfaction — CSP modeling, AC-3 arc consistency, backtracking variants (standard, MRV, forward checking), min-conflicts.
- Phase 4: Logic and reasoning — five propositional rules, forward chaining inference engine, CNF conversion and theorem proving.
- Phase 5: Machine learning — Perceptron, Delta Rule, MLP (manual backpropagation), K-Means, K-Medoid; all implemented from scratch in NumPy.
- `scripts/run_all.py` — unified pipeline runner with `asyncio.WindowsSelectorEventLoopPolicy` to eliminate Windows ZMQ socket errors.
- `scripts/generate_phase*.py` — deterministic notebook generators for all five phases.
- `scripts/check_outputs.py` — automated output verification across all notebooks.
- `docs/MindSight_Report.docx` — formal research report with quantitative results and cross-track comparison.
- Professional repository scaffolding: `DEVELOPERS.md`, `CHANGELOG.md`, `LICENSE`, `SKILL.md`, `.gitattributes`, `.github/CONTRIBUTING.md`, `.github/SECURITY.md`, `.github/CODEOWNERS`, issue templates, PR template.

### Fixed
- Phase 1 data path bug: incorrect relative path caused `FileNotFoundError` during notebook execution.
- ZMQ "not a socket" errors during bulk notebook execution on Windows resolved via event loop policy override.

---

[Unreleased]: https://github.com/Abdu1-Ahd/MindSight/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/Abdu1-Ahd/MindSight/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Abdu1-Ahd/MindSight/releases/tag/v1.0.0
