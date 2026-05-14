"""
Convert entrypoint scripts to Jupyter notebooks using jupytext.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = ["run_training.py", "evaluate.py", "run_experiments.py", "real_data_demo.py"]

Path("notebooks").mkdir(exist_ok=True)

for script in SCRIPTS:
    output = f"notebooks/{Path(script).stem}.ipynb"
    print(f"Converting {script} -> {output}")
    subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "notebook", "--output", output, script],
        check=True,
    )

print("Done.")
