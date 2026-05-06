# Changelog

All notable changes to MindSight are documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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

[Unreleased]: https://github.com/<username>/MindSight/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/<username>/MindSight/releases/tag/v1.0.0
