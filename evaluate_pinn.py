"""
Evaluate a trained PINN against the analytic Black-Scholes benchmark.

Loads saved weights, predicts option prices on a regular (t, S) grid, compares
them to the analytic Black-Scholes prices, and saves error plots.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.black_scholes import european_call_price
from src.pinn_model import PINN, GatedPINN

MODELS = {
    "pinn": PINN,
    "gated": GatedPINN,
}



def plot_error_surface(SS, TT, error, title, output_path):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(SS, TT, error, linewidth=0, antialiased=True)

    ax.set_xlabel("Stock price S")
    ax.set_ylabel("Time t")
    ax.set_zlabel("Prediction error")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_price_slices(S_values, true_slices, pred_slices, model_name, output_path):
    plt.figure(figsize=(8, 5))

    for label in true_slices:
        plt.plot(S_values, true_slices[label], label=f"{label} analytic")
        plt.plot(
            S_values,
            pred_slices[label],
            linestyle="--",
            label=f"{label} {model_name}",
        )

    plt.xlabel("Stock price S")
    plt.ylabel("Option price V(t, S)")
    plt.title(f"Analytic Black-Scholes vs {model_name} prediction")
    plt.legend()
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

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    prefix = "gated_pinn" if args.model == "gated" else "pinn"
    model_cls = MODELS[args.model]
    model_name = model_cls.__name__

    model_path = Path(f"{prefix}_model.pt")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. Run `python train_pinn.py --model {args.model}` first."
        )

    K = 40.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    S_max = 160.0

    model = model_cls(hidden_dim=32, hidden_layers=2, T=T, S_max=S_max)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    S_grid = np.linspace(1.0, S_max, 100)
    t_grid = np.linspace(0.0, T, 100)
    SS, TT = np.meshgrid(S_grid, t_grid)

    V_true = european_call_price(SS, TT, K=K, r=r, sigma=sigma, T=T)

    t_tensor = torch.tensor(TT.reshape(-1, 1), dtype=torch.float32)
    S_tensor = torch.tensor(SS.reshape(-1, 1), dtype=torch.float32)

    with torch.no_grad():
        V_pred = model(t_tensor, S_tensor).numpy().reshape(SS.shape)

    error = V_pred - V_true

    mse = np.mean((V_pred - V_true) ** 2)
    mae = np.mean(np.abs(V_pred - V_true))

    print(f"{model_name} vs analytic Black-Scholes benchmark:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    error_plot = output_dir / f"{prefix}_error_surface.png"
    slices_plot = output_dir / f"{prefix}_vs_analytic_slices.png"

    plot_error_surface(
        SS,
        TT,
        error,
        f"{model_name} prediction error",
        error_plot,
    )

    slice_times = [0.0, 0.5, 1.0]
    true_slices = {}
    pred_slices = {}

    for t in slice_times:
        true_slices[f"t = {t}"] = european_call_price(
            S_grid,
            t,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
        )

        t_slice = torch.full((len(S_grid), 1), t, dtype=torch.float32)
        S_slice = torch.tensor(S_grid.reshape(-1, 1), dtype=torch.float32)

        with torch.no_grad():
            pred_slices[f"t = {t}"] = model(t_slice, S_slice).numpy().reshape(-1)

    plot_price_slices(
        S_grid,
        true_slices,
        pred_slices,
        model_name,
        slices_plot,
    )

    print("Saved:")
    print(f"- {error_plot}")
    print(f"- {slices_plot}")


if __name__ == "__main__":
    main()
