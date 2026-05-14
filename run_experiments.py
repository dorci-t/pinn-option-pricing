"""
Small hyperparameter experiments for the PINN option-pricing model.

This script runs a few short training jobs with different settings and compares
their final MSE/MAE against the analytic Black-Scholes benchmark.
"""

import argparse
from pathlib import Path
import csv

import torch

from src.pinn_model import MODELS
from src.plotting import plot_lines
from src.train import evaluate_vs_analytic, train_model

DEFAULTS = {
    "hidden_dim": 32,
    "hidden_layers": 2,
    "lr": 1e-3,
    "beta": 1.0,
    "epochs": 500,
    "n_interior": 300,
    "n_terminal": 100,
    "n_boundary": 100,
}

EXPERIMENTS = [
    {"name": "baseline"},
    {"name": "larger_hidden_dim", "hidden_dim": 64},
    {"name": "lower_lr", "lr": 5e-4},
    {"name": "higher_pde_weight", "beta": 10.0},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=MODELS.keys(),
        default="gated",
    )
    args = parser.parse_args()

    model_cls = MODELS[args.model]

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    rows = []
    histories = {}

    for experiment in EXPERIMENTS:
        config = {**DEFAULTS, **experiment}
        print(f"Running experiment: {config['name']}")

        torch.manual_seed(0)

        model = model_cls(
            hidden_dim=config["hidden_dim"],
            hidden_layers=config["hidden_layers"],
            T=1.0,
            S_max=160.0,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        loss_history = train_model(
            model,
            optimizer,
            n_epochs=config["epochs"],
            print_every=0,
            n_interior=config["n_interior"],
            n_terminal=config["n_terminal"],
            n_boundary=config["n_boundary"],
            beta=config["beta"],
        )

        mse, mae, *_ = evaluate_vs_analytic(model, grid_size=80)

        histories[config["name"]] = loss_history

        rows.append(
            {
                "name": config["name"],
                "hidden_dim": config["hidden_dim"],
                "hidden_layers": config["hidden_layers"],
                "lr": config["lr"],
                "beta": config["beta"],
                "epochs": config["epochs"],
                "mse": mse,
                "mae": mae,
                "final_loss": loss_history[-1],
            }
        )

        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")

    prefix = "gated_pinn" if args.model == "gated" else "pinn"
    model_name = model_cls.__name__

    csv_path = results_dir / f"{prefix}_hyperparameter_experiments.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    loss_plot = output_dir / f"{prefix}_hyperparameter_loss_curves.png"

    plot_lines(
        range(max(len(h) for h in histories.values())),
        histories,
        f"{model_name} hyperparameter experiment losses",
        "Epoch",
        "Total loss",
        loss_plot,
    )

    print("Saved:")
    print(f"- {csv_path}")
    print(f"- {loss_plot}")


if __name__ == "__main__":
    main()
