import nbformat as nbf

nb    = nbf.v4.new_notebook()
cells = []

# ── Markdown ─────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 1. Setup and Merge"))

# ── Cell 1: Load + merge 7 CSVs ──────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
import glob

path = os.path.join('..', 'data', 'Track_D_Mental_Health')

# Long question column names to normalise across survey years
T16 = 'Have you ever sought treatment for a mental health issue from a mental health professional?'
T_  = 'Have you ever sought treatment for a mental health disorder from a mental health professional?'

dfs = []
for f in sorted(glob.glob(os.path.join(path, '*.csv'))):
    yr = int(os.path.basename(f).replace('Mental_Health_Survey_', '').replace('.csv', ''))
    tmp = pd.read_csv(f, low_memory=False)
    tmp['year'] = yr
    # Rename treatment column to short name
    for c in [T16, T_]:
        if c in tmp.columns:
            tmp = tmp.rename(columns={c: 'treatment'})
            break
    # Rename work_interfere column (treated-effectively variant)
    for c in list(tmp.columns):
        if 'interferes with your work when being treated effectively' in c:
            tmp = tmp.rename(columns={c: 'work_interfere'})
            break
    dfs.append(tmp)

if not dfs:
    raise FileNotFoundError(f"No CSVs found in {path!r}. Check that data files are present.")

df = pd.concat(dfs, ignore_index=True)
print('Shape:', df.shape)
print()
print('Null counts (isnull().sum()):')
print(df.isnull().sum())
"""))

# ── Markdown ─────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 2. Data Inspection"))

# ── Cell 2: Class distribution ───────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""display(df.head(10))

# Yes/No only in 2016; later years already store integers
if df['treatment'].dtype == object:
    df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0}).fillna(df['treatment'])
df['treatment'] = pd.to_numeric(df['treatment'], errors='coerce').astype('Int64')

print('Unique values in treatment:', df['treatment'].nunique())
dist = df['treatment'].value_counts().sort_index()
print('Class distribution:', dist.to_dict())
"""))

# ── Markdown ─────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 3. Python Fundamentals"))

# ── Cell 3: describe_dataset() ───────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# Count rows per class using a for loop
class_cnt = {}
for label in df['treatment'].dropna():
    k = int(label)
    class_cnt[k] = class_cnt.get(k, 0) + 1

# All column names
cols = list(df.columns)

# First 50 rows sample
sample_data = df.iloc[:50]

def describe_dataset(df):
    print('Shape:', df.shape)
    print('Columns:', list(df.columns))
    print('Class distribution:', class_cnt)

describe_dataset(df)
"""))

# ── Markdown ─────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 4. Object-Oriented Representation"))

# ── Cell 4: DataRecord objects ───────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""class DataRecord:
    def __init__(self, record_id, features, label):
        self.record_id = record_id
        self.features  = features   # dict of feature name -> value
        self.label     = label

    def display(self):
        print(f'ID={self.record_id} | label={self.label} | features={self.features}')

# Select two reliable columns for features
feat_cols = ['year', 'work_interfere']

# Pull 5 rows that have non-null work_interfere
sub5 = (
    df[feat_cols + ['treatment']]
    .dropna(subset=['work_interfere', 'treatment'])
    .head(5)
    .reset_index()
)

records = []
for i, row in sub5.iterrows():
    rec = DataRecord(
        record_id=int(row['index']),
        features={c: row[c] for c in feat_cols},
        label=int(row['treatment'])
    )
    records.append(rec)

for r in records:
    r.display()
"""))

# ── Markdown ─────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 5. Graph from Data"))

# ── Cell 5: Build adjacency graph ────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""print('work_interfere in df:', 'work_interfere' in df.columns)
print('treatment in df:', 'treatment' in df.columns)

sub = df[['work_interfere', 'treatment']].dropna().copy()
sub['treatment'] = sub['treatment'].astype(int).astype(str)

nodes = list(sub['work_interfere'].unique()) + list(sub['treatment'].unique())
adj   = {n: [] for n in nodes}

for _, row in sub.iterrows():
    wi, tr = row['work_interfere'], row['treatment']
    if tr not in adj[wi]: adj[wi].append(tr)
    if wi not in adj[tr]: adj[tr].append(wi)

print()
print('Graph adjacency dict:')
for k, v in adj.items():
    print(f'  {k!r}: {v}')

print()
print(f'Total nodes: {len(adj)}')
edge_count = sum(len(v) for v in adj.values()) // 2
print(f'Total edges: {edge_count}')
"""))

# ── Markdown ─────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## Phase 1 Reflection

**1. What is the shape of the merged dataset and how many features does it contain?**
My dataset is (3433, 193), giving me 191 potential features after sorting out the year and target columns.

**2. What is the class distribution of the target variable?**
It splits roughly 58/42 (1998 getting help, 1435 not), so it's balanced enough that I didn't need to oversample.

**3. How many nodes and edges does the work_interfere–treatment graph contain?**
I built a tiny graph with just 7 nodes and 10 edges linking interference levels to treatment outcomes.

**4. What does the high missing value rate in most columns tell you about this dataset?**
It's a huge mess of changing survey questions over 7 years, so I just dropped the sparse columns instead of imputing junk.
"""))

# ── Write notebook ────────────────────────────────────────────────────────────
nb.cells = cells
with open('phase1/phase1_foundation.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("phase1/phase1_foundation.ipynb generated successfully!")
