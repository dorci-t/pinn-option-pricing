"""
Train a PINN model on the Black-Scholes benchmark.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.pinn_model import PINN, GatedPINN
from src.train import train_model

MODELS = {
    "pinn": PINN,
    "gated": GatedPINN,
}


def plot_loss_curve(loss_history, title, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=MODELS.keys(),
        default="gated",
    )
    args = parser.parse_args()

    torch.manual_seed(0)

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    device = "cpu"

    model_cls = MODELS[args.model]
    model = model_cls(
        hidden_dim=32,
        hidden_layers=2,
        T=1.0,
        S_max=160.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_history = train_model(
        model,
        optimizer,
        n_epochs=1000,
        n_interior=500,
        n_terminal=200,
        n_boundary=200,
        beta=1.0,
        device=device,
    )

    prefix = "gated_pinn" if args.model == "gated" else "pinn"

    loss_plot = output_dir / f"{prefix}_training_loss.png"
    model_file = f"{prefix}_model.pt"

    plot_loss_curve(loss_history, f"{model_cls.__name__} training loss", loss_plot)
    torch.save(model.state_dict(), model_file)

    print("Training finished.")
    print("Saved:")
    print(f"- {loss_plot}")
    print(f"- {model_file}")


if __name__ == "__main__":
    main()
