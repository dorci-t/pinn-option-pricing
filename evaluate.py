# %% [markdown]
# # Evaluate PINN against the analytic Black-Scholes benchmark
#
# Set `analytic_only = true` in the config to generate standalone analytic plots.

# %%
import os
import sys
from pathlib import Path

import numpy as np
import torch

# When run as a notebook from notebooks/, the working directory and sys.path
# point at notebooks/, not the project root. We detect this by checking whether
# __file__ exists (it does in scripts, not in notebooks) and adjust accordingly.
# The src/ check makes os.chdir idempotent - it only moves if we're not already
# at the project root.
_project_root = str(Path(__file__).resolve().parent) if "__file__" in dir() else ".."
sys.path.insert(0, _project_root)
if not Path("src").is_dir():
    os.chdir(_project_root)

from src.black_scholes import european_call_price
from src.config import load_config
from src.pinn_model import MODELS
from src.plotting import plot_comparison_slices, plot_lines, plot_surface
from src.train import evaluate_vs_analytic

CONFIG_PATH = os.environ.get("CONFIG", "configs/gated.toml")
config = load_config(CONFIG_PATH)

# %%
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

K = config["K"]
r = config["r"]
sigma = config["sigma"]
T = config["T"]
S_max = config["S_max"]

# %% [markdown]
# ## Analytic benchmark plots

# %%
if config.get("analytic_only", False):
    S_values = np.linspace(1.0, S_max, 300)

    slice_data = {}
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        slice_data[f"t = {t}"] = european_call_price(S_values, t, K=K, r=r, sigma=sigma, T=T)

    plot_lines(
        S_values, slice_data,
        "Analytic Black-Scholes European Call Price",
        "Stock price S", "Option price V(t, S)",
        output_dir / "analytic_price_slices.png",
    )

    S_grid = np.linspace(1.0, S_max, 100)
    t_grid = np.linspace(0.0, T, 100)
    SS, TT = np.meshgrid(S_grid, t_grid)
    VV = european_call_price(SS, TT, K=K, r=r, sigma=sigma, T=T)

    plot_surface(
        SS, TT, VV,
        "Analytic Black-Scholes European Call Price",
        "Option price V(t, S)",
        output_dir / "analytic_price_surface.png",
    )

    print("Generated figures:")
    print("- figures/analytic_price_slices.png")
    print("- figures/analytic_price_surface.png")

# %% [markdown]
# ## Load trained model and evaluate

# %%
if not config.get("analytic_only", False):
    prefix = "gated_pinn" if config["model"] == "gated" else "pinn"
    model_cls = MODELS[config["model"]]
    model_name = model_cls.__name__

    model_path = Path(f"{prefix}_model.pt")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. Train the model first with run_training.py."
        )

    model = model_cls(
        hidden_dim=config["hidden_dim"],
        hidden_layers=config["hidden_layers"],
        T=T,
        S_max=S_max,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    mse, mae, V_pred, V_true, SS, TT = evaluate_vs_analytic(
        model, K=K, r=r, sigma=sigma, T=T, S_max=S_max,
    )

    print(f"{model_name} vs analytic Black-Scholes benchmark:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

# %% [markdown]
# ## Error surface

# %%
if not config.get("analytic_only", False):
    error_plot = output_dir / f"{prefix}_error_surface.png"

    plot_surface(
        SS, TT, V_pred - V_true,
        f"{model_name} prediction error",
        "Prediction error",
        error_plot,
    )

# %% [markdown]
# ## Comparison slices

# %%
if not config.get("analytic_only", False):
    slices_plot = output_dir / f"{prefix}_vs_analytic_slices.png"

    S_grid = np.linspace(1.0, S_max, 100)
    slice_times = [0.0, 0.5, 1.0]
    true_slices = {}
    pred_slices = {}

    for t in slice_times:
        true_slices[f"t = {t}"] = european_call_price(
            S_grid, t, K=K, r=r, sigma=sigma, T=T,
        )

        t_slice = torch.full((len(S_grid), 1), t, dtype=torch.float32)
        S_slice = torch.tensor(S_grid.reshape(-1, 1), dtype=torch.float32)

        with torch.no_grad():
            pred_slices[f"t = {t}"] = model(t_slice, S_slice).numpy().reshape(-1)

    plot_comparison_slices(
        S_grid, true_slices, pred_slices, model_name, slices_plot,
    )

    print("Saved:")
    print(f"- {error_plot}")
    print(f"- {slices_plot}")
