"""
Small hyperparameter experiments for the PINN option-pricing model.

This script runs a few short training jobs with different settings and compares
their final MSE/MAE against the analytic Black-Scholes benchmark.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from src.pinn_model import PINN
from src.train import evaluate_vs_analytic, train_model


def plot_experiment_losses(histories, output_path):
    plt.figure(figsize=(8, 5))

    for name, loss_history in histories.items():
        plt.plot(loss_history, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title("PINN hyperparameter experiment losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    configs = [
        {
            "name": "baseline",
            "hidden_dim": 32,
            "hidden_layers": 2,
            "lr": 1e-3,
            "beta": 1.0,
            "epochs": 500,
            "n_interior": 300,
            "n_terminal": 100,
            "n_boundary": 100,
        },
        {
            "name": "larger_hidden_dim",
            "hidden_dim": 64,
            "hidden_layers": 2,
            "lr": 1e-3,
            "beta": 1.0,
            "epochs": 500,
            "n_interior": 300,
            "n_terminal": 100,
            "n_boundary": 100,
        },
        {
            "name": "lower_lr",
            "hidden_dim": 32,
            "hidden_layers": 2,
            "lr": 5e-4,
            "beta": 1.0,
            "epochs": 500,
            "n_interior": 300,
            "n_terminal": 100,
            "n_boundary": 100,
        },
        {
            "name": "higher_pde_weight",
            "hidden_dim": 32,
            "hidden_layers": 2,
            "lr": 1e-3,
            "beta": 10.0,
            "epochs": 500,
            "n_interior": 300,
            "n_terminal": 100,
            "n_boundary": 100,
        },
    ]

    rows = []
    histories = {}

    for config in configs:
        print(f"Running experiment: {config['name']}")

        torch.manual_seed(0)

        model = PINN(
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

    csv_path = results_dir / "hyperparameter_experiments.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plot_experiment_losses(
        histories,
        output_dir / "hyperparameter_loss_curves.png",
    )

    print("Saved:")
    print(f"- {csv_path}")
    print("- figures/hyperparameter_loss_curves.png")


if __name__ == "__main__":
    main()
