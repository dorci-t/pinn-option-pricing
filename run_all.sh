#!/usr/bin/env bash
set -euo pipefail

echo "=== Tests ==="
python -m pytest tests/ -v

echo ""
echo "=== Analytic benchmark plots ==="
python evaluate.py --analytic-only

echo ""
echo "=== Train PINN ==="
python run_training.py --model pinn

echo ""
echo "=== Train GatedPINN ==="
python run_training.py --model gated

echo ""
echo "=== Evaluate PINN ==="
python evaluate.py --model pinn

echo ""
echo "=== Evaluate GatedPINN ==="
python evaluate.py --model gated

echo ""
echo "=== Hyperparameter experiments ==="
python run_experiments.py

echo ""
echo "=== Real data demo ==="
python real_data_demo.py

echo ""
echo "Done."
