#!/usr/bin/env bash
set -euo pipefail

mkdir -p notebooks

for script in run_training.py evaluate.py run_experiments.py real_data_demo.py; do
    echo "Converting $script -> notebooks/${script%.py}.ipynb"
    jupytext --to notebook --output "notebooks/${script%.py}.ipynb" "$script"
done

echo "Done."
