"""
Run the full pipeline: tests, training, evaluation, experiments, and notebook generation.
"""

import os
import subprocess
import sys

STEPS = [
    ("Tests", None, [sys.executable, "-m", "pytest", "tests/", "-v"]),
    ("Analytic benchmark plots", "configs/analytic.toml", [sys.executable, "evaluate.py"]),
    ("Train PINN", "configs/pinn.toml", [sys.executable, "run_training.py"]),
    ("Train GatedPINN", "configs/gated.toml", [sys.executable, "run_training.py"]),
    ("Evaluate PINN", "configs/pinn.toml", [sys.executable, "evaluate.py"]),
    ("Evaluate GatedPINN", "configs/gated.toml", [sys.executable, "evaluate.py"]),
    ("Hyperparameter experiments (PINN)", "configs/pinn.toml", [sys.executable, "run_experiments.py"]),
    ("Hyperparameter experiments (GatedPINN)", "configs/gated.toml", [sys.executable, "run_experiments.py"]),
    ("Real data demo", "configs/real_data.toml", [sys.executable, "real_data_demo.py"]),
    ("Generate notebooks", None, [sys.executable, "generate_notebooks.py"]),
]

for title, config, cmd in STEPS:
    print(f"\n=== {title} ===")
    env = os.environ.copy()
    if config:
        env["CONFIG"] = config
    subprocess.run(cmd, env=env, check=True)

print("\nDone.")
