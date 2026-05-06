import nbformat

with open('phase3/phase3_csp.ipynb', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

errors = []
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        for o in cell.outputs:
            if o.get('output_type') == 'error':
                errors.append((i+1, o.get('ename','?'), o.get('evalue','?')))

if errors:
    print("ERRORS FOUND:")
    for c, e, v in errors:
        print(f"  Cell {c}: {e}: {v}")
else:
    print("No errors in any cell.")

print("\n--- Cell outputs (stdout) ---")
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        for o in cell.outputs:
            if o.get('output_type') == 'stream' and o.get('name') == 'stdout':
                txt = o['text']
                print(f"Cell {i+1} ({len(txt)} chars):", txt[:600])
                print()
