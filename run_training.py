# %% [markdown]
# # Train a PINN model on the Black-Scholes benchmark

# %%
import os
import sys
from pathlib import Path

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

from src.config import load_config
from src.pinn_model import MODELS
from src.plotting import plot_lines
from src.train import train_model

CONFIG_PATH = os.environ.get("CONFIG", "configs/gated.toml")
config = load_config(CONFIG_PATH)

# %% [markdown]
# ## Model setup

# %%
torch.manual_seed(0)

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

model_cls = MODELS[config["model"]]
model = model_cls(
    hidden_dim=config["hidden_dim"],
    hidden_layers=config["hidden_layers"],
    T=config["T"],
    S_max=config["S_max"],
)

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

# %% [markdown]
# ## Training

# %%
loss_history = train_model(
    model,
    optimizer,
    n_epochs=config["epochs"],
    n_interior=config["n_interior"],
    n_terminal=config["n_terminal"],
    n_boundary=config["n_boundary"],
    beta=config["beta"],
)

# %% [markdown]
# ## Save model and plot loss curve

# %%
prefix = "gated_pinn" if config["model"] == "gated" else "pinn"

loss_plot = output_dir / f"{prefix}_training_loss.png"
model_file = f"{prefix}_model.pt"

plot_lines(
    range(len(loss_history)),
    {"loss": loss_history},
    f"{model_cls.__name__} training loss",
    "Epoch",
    "Total loss",
    loss_plot,
)
torch.save(model.state_dict(), model_file)

print("Training finished.")
print("Saved:")
print(f"- {loss_plot}")
print(f"- {model_file}")
