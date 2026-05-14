#!/usr/bin/env bash
set -euo pipefail

echo "=== Tests ==="
python -m pytest tests/ -v

echo ""
echo "=== Analytic benchmark plots ==="
CONFIG=configs/analytic.toml python evaluate.py

echo ""
echo "=== Train PINN ==="
CONFIG=configs/pinn.toml python run_training.py

echo ""
echo "=== Train GatedPINN ==="
CONFIG=configs/gated.toml python run_training.py

echo ""
echo "=== Evaluate PINN ==="
CONFIG=configs/pinn.toml python evaluate.py

echo ""
echo "=== Evaluate GatedPINN ==="
CONFIG=configs/gated.toml python evaluate.py

echo ""
echo "=== Hyperparameter experiments (PINN) ==="
CONFIG=configs/pinn.toml python run_experiments.py

echo ""
echo "=== Hyperparameter experiments (GatedPINN) ==="
CONFIG=configs/gated.toml python run_experiments.py

echo ""
echo "=== Real data demo ==="
CONFIG=configs/real_data.toml python real_data_demo.py

echo ""
echo "Done."
