# Contributing to MindSight

---

## Branch Strategy

| Branch Pattern | Purpose |
|:---|:---|
| `main` | Stable, production-ready state. Direct pushes blocked. |
| `feat/<short-description>` | New phase implementations or feature additions. |
| `fix/<short-description>` | Bug fixes in pipeline scripts or notebook generators. |
| `chore/<short-description>` | Dependency updates, formatting, internal refactoring. |
| `docs/<short-description>` | Documentation-only changes. |

---

## Commit Convention

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short imperative description>

[optional body]

[optional footer: BREAKING CHANGE: <description>]
```

**Allowed types:** `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `perf`

**Scope examples:** `phase1`, `phase5`, `scripts`, `docs`, `ci`

**Examples:**

```
feat(phase5): add dropout regularization to MLP hidden layer
fix(scripts): correct relative data path in generate_phase1.py
docs(readme): add architecture directory tree annotation
```

**Mapping to version bumps:**
- `feat:` → MINOR
- `fix:` → PATCH
- `BREAKING CHANGE:` footer → MAJOR

---

## Pull Request Protocol

1. Branch from `main` using the naming convention above.
2. Keep PRs scoped to a single concern. Split unrelated changes into separate PRs.
3. Fill in the PR template completely. Empty checkboxes block merge.
4. Request review from the relevant `CODEOWNERS` team.
5. Resolve all review conversations before merging.
6. Squash-merge only — no merge commits allowed on `main`.

---

## Pre-Flight Checks

Run the following locally before opening a PR. All checks must pass:

```bash
# 1. Full pipeline must execute without errors
python scripts/run_all.py

# 2. Output verification must report all phases passing
python scripts/check_outputs.py

# 3. No syntax errors in generator scripts
python -m py_compile scripts/generate_phase1.py
python -m py_compile scripts/generate_phase2.py
python -m py_compile scripts/generate_phase3.py
python -m py_compile scripts/generate_phase4.py
python -m py_compile scripts/generate_phase5.py
```
