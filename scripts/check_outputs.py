import json

CHECKS = {
    'phase1/phase1_foundation.ipynb': [
        'Shape:', 'isnull', 'Class distribution',
        'Graph adjacency', 'Total nodes', 'Total edges',
        'ID=', 'describe_dataset',
    ],
    'phase2/phase2_search.ipynb': [
        'BFS path:', 'DFS path:', 'DLS limit=3', 'DLS limit=5',
        'IDDFS:', 'UCS path:', 'UCS cost:',
        'Best-First path:', 'A* path:', 'A* cost:',
        'Hill Climbing:', 'SA final state:',
        'k=3', 'k=5',
        'Minimax:', 'Minimax nodes', 'Alpha-Beta nodes',
    ],
    'phase3/phase3_csp.ipynb': [
        'Variables:', 'Domains:', 'Constraints',
        'BEFORE AC-3', 'AFTER AC-3', 'Total domain reduction:',
        'Basic Backtracking:', 'Forward Checking:', 'MRV:',
        'Algorithm', 'Backtracks',
        'Initial violations:', 'Final violations:',
    ],
    'phase4/phase4_logic.ipynb': [
        'IF work_interfere=Often THEN',
        'IF work_interfere=Never THEN',
        'R1 step-by-step:', 'R3 step-by-step:',
        'All rules in CNF',
        'R1 fired', 'R2 fired',
        'treatment=1', 'treatment=0',
        'Highest accuracy:',
    ],
    'phase5/phase5_ml.ipynb': [
        'X_train:', 'X_test:',
        'K-Means converged', 'Centroid shape:',
        'K-Means purity:', 'K-Medoid purity:',
        'K-Means WCSD:', 'K-Medoid WCSD:',
        'Perceptron test accuracy:',
        'Delta Rule final MSE:', 'Delta Rule test accuracy:',
        'Epoch 200',
        'MLP test accuracy:',
        'Model', 'Test Accuracy',
    ],
}

overall_pass = True

for nb_path, keywords in CHECKS.items():
    try:
        with open(nb_path) as f:
            nb = json.load(f)
    except FileNotFoundError:
        print(f'MISSING  {nb_path}')
        overall_pass = False
        continue

    errors = []
    all_output_text = ''

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'error':
                errors.append(f'  CELL ERROR {i}: {out["ename"]}: {out["evalue"][:120]}')
            if out.get('output_type') == 'stream':
                all_output_text += ''.join(out.get('text', []))

    missing_kw = [kw for kw in keywords if kw not in all_output_text]

    if errors:
        print(f'FAIL(err) {nb_path}')
        for e in errors:
            print(e)
        overall_pass = False
    elif missing_kw:
        print(f'FAIL(kw)  {nb_path}')
        for kw in missing_kw:
            print(f'  MISSING keyword: {kw!r}')
        overall_pass = False
    else:
        print(f'PASS      {nb_path}')

print()
print('OVERALL:', 'ALL PASS' if overall_pass else 'FAILURES FOUND')

# session:b2bdad9b
