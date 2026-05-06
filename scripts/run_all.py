import os
import sys
import asyncio
import subprocess
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# FIX for Windows ZMQ / Tornado Asyncio errors (Proactor vs Selector)
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def main():
    notebooks = [
        'phase1/phase1_foundation.ipynb',
        'phase2/phase2_search.ipynb',
        'phase3/phase3_csp.ipynb',
        'phase4/phase4_logic.ipynb',
        'phase5/phase5_ml.ipynb'
    ]

    generators = [
        'scripts/generate_phase1.py',
        'scripts/generate_phase2.py',
        'scripts/generate_phase3.py',
        'scripts/generate_phase4.py',
        'scripts/generate_phase5.py'
    ]

    # 1. Run all generators
    print("--- Regenerating Notebooks ---")
    for gen in generators:
        print(f"Running {gen}...")
        subprocess.run([sys.executable, gen], check=True)

    # 2. Execute all notebooks
    print("\n--- Executing Notebooks ---")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    for nb_path in notebooks:
        print(f"Executing {nb_path}...")
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Execute the notebook, simulating running it from its own directory
        nb_dir = os.path.dirname(nb_path)
        ep.preprocess(nb, {'metadata': {'path': nb_dir}})
        
        # Save the executed notebook
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"  -> Saved {nb_path}")

    print("\nAll phases completed successfully without ZMQ socket errors!")

if __name__ == '__main__':
    main()
