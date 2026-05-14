"""
Shared training loop and evaluation utilities.
"""

import numpy as np
import torch
from tqdm import trange

from src.black_scholes import european_call_price
from src.losses import compute_pinn_loss


def train_model(model, optimizer, n_epochs, print_every=100, **loss_kwargs):
    loss_history = []

    for epoch in trange(n_epochs):
        optimizer.zero_grad()

        loss, components = compute_pinn_loss(model, **loss_kwargs)

        loss.backward()
        optimizer.step()

        loss_history.append(components["total"])

        if print_every and epoch % print_every == 0:
            print(
                f"epoch {epoch:4d} | "
                f"total={components['total']:.4f} | "
                f"pde={components['pde']:.4f} | "
                f"terminal={components['terminal']:.4f} | "
                f"boundary={components['boundary']:.4f}"
            )

    return loss_history


def evaluate_vs_analytic(model, K=40.0, r=0.05, sigma=0.2, T=1.0, S_max=160.0, grid_size=100):
    S_grid = np.linspace(1.0, S_max, grid_size)
    t_grid = np.linspace(0.0, T, grid_size)
    SS, TT = np.meshgrid(S_grid, t_grid)

    V_true = european_call_price(SS, TT, K=K, r=r, sigma=sigma, T=T)

    t_tensor = torch.tensor(TT.reshape(-1, 1), dtype=torch.float32)
    S_tensor = torch.tensor(SS.reshape(-1, 1), dtype=torch.float32)

    with torch.no_grad():
        V_pred = model(t_tensor, S_tensor).numpy().reshape(SS.shape)

    mse = np.mean((V_pred - V_true) ** 2)
    mae = np.mean(np.abs(V_pred - V_true))

    return mse, mae, V_pred, V_true, SS, TT
