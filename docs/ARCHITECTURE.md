# Architecture

System design documentation for MindSight.

---

## Pipeline Overview

MindSight is a five-phase sequential pipeline. Each phase is a standard Jupyter notebook executed in-process by the master runner.

```mermaid
graph TD
    subgraph Data Layer
        A["7 OSMI CSVs<br/>(2016–2022)"]
        B["pd.concat merge<br/>(3,433 × 193)"]
    end

    subgraph Preprocessing
        C["Drop cols > 50% null"]
        D["Drop free-text cols<br/>(>50 uniques)"]
        E["Label-encode categoricals"]
        F["Z-score normalize"]
        G["Train/Test split<br/>80/20 (2,746 / 687)"]
    end

    subgraph Phase Outputs
        H["Bipartite Graph<br/>work_interfere ↔ treatment"]
        I["Search Benchmarks<br/>10 algorithms"]
        J["CSP Solutions<br/>AC-3, backtracking, min-conflicts"]
        K["Logic Rules<br/>forward chaining, CNF proof"]
        L["ML Models<br/>Perceptron, Delta, MLP, K-Means, K-Medoid"]
    end

    A --> B
    B --> C --> D --> E --> F --> G
    B --> H
    H --> I
    B --> J
    B --> K
    G --> L
```

---

## Data Flow

```
CSV files (raw)
    │
    ▼
pd.concat ──────────────────────────┐
    │                               │
    ▼                               ▼
Column pruning              Bipartite graph construction
    │                               │
    ▼                               ▼
Label encoding              Phase 2: Search algorithms
    │                               │
    ▼                               ▼
Z-score normalization       Phase 3: CSP modeling
    │                               │
    ▼                               ▼
Train/Test split            Phase 4: Propositional logic
    │
    ▼
Phase 5: ML models
    │
    ▼
Comparison table (stdout)
```

---

## Module Responsibility Matrix

| Module | Responsibility | Input | Output |
|:---|:---|:---|:---|
| `phase1_foundation.ipynb` | Data loading, EDA, graph construction | 7 CSVs | Loaded data |
| `phase2_search.ipynb` | 10 search algorithm implementations | Bipartite graph (rebuilt from CSVs) | Search benchmarks |
| `phase3_csp.ipynb` | CSP variables, AC-3, backtracking, min-conflicts | Survey data (rebuilt from CSVs) | CSP Solutions |
| `phase4_logic.ipynb` | Propositional rules, forward chaining, CNF | Survey data (rebuilt from CSVs) | Logic derivations |
| `phase5_ml.ipynb` | 5 ML model implementations + comparison | Survey data (rebuilt from CSVs) | ML Models |
| `scripts/run_all.py` | Pipeline orchestration | All notebooks | All notebooks (executed) |
| `scripts/check_outputs.py` | Output verification | Executed notebooks | Pass/fail report |

---

## Phase Dependency Graph

```mermaid
graph LR
    P1["Phase 1<br/>Foundation"] -.->|"conceptual: graph structure"| P2["Phase 2<br/>Search"]
    P1 -.->|"conceptual: feature set"| P5["Phase 5<br/>ML"]
    P1 -.->|"conceptual: data schema"| P3["Phase 3<br/>CSP"]
    P1 -.->|"conceptual: column semantics"| P4["Phase 4<br/>Logic"]
```

**Note:** Dependencies are conceptual, not code-level imports. Each phase rebuilds its own data from the raw CSVs independently. This is intentional: it allows any phase to execute in isolation without requiring Phase 1 to have run first, at the cost of duplicated loading logic.

---

## Preprocessing Stages (Column Counts)

| Stage | Columns | Rows | Operation |
|:---|---:|---:|:---|
| Raw merge | 193 | 3,433 | `pd.concat` across 7 CSVs |
| Drop sparse | ~40 | 3,433 | Remove columns with > 50% null values |
| Drop text | ~19 | 3,433 | Remove columns with > 50 unique values |
| Drop null target | 19 | 3,433 | Remove rows where `treatment` is null |
| After encoding | 19 | 3,433 | All columns now numeric (label-encoded) |
| After split | 18 features | 2,746 train / 687 test | `treatment` separated as target |

---

## Windows Compatibility

The pipeline uses Jupyter's `ExecutePreprocessor` which depends on Tornado + ZMQ for IPC. On Windows, Python 3.10+ defaults to `ProactorEventLoop`, which is incompatible with ZMQ socket operations.

`scripts/run_all.py` sets the event loop policy at startup:

```python
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

This line must execute before any `nbconvert` import or execution call.
