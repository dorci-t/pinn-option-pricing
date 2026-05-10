"""
Loss function for the parametric Black-Scholes PINN.

This is the synthetic benchmark version of the parametric model. A market data
loss term can later be added to the same model.
"""

import torch

from src.parametric_pde import parametric_black_scholes_residual
from src.parametric_sampling import (
    sample_parametric_boundary_points,
    sample_parametric_interior_points,
    sample_parametric_terminal_points,
)


def compute_parametric_pinn_loss(
    model,
    r=0.05,
    sigma=0.2,
    T=1.0,
    S_max=160.0,
    K_min=20.0,
    K_max=120.0,
    n_interior=1000,
    n_terminal=300,
    n_boundary=300,
    beta=1.0,
    device="cpu",
):
    """
    Compute one loss value for the parametric PINN.
    """
    t_int, S_int, K_int = sample_parametric_interior_points(
        n_interior,
        T=T,
        S_max=S_max,
        K_min=K_min,
        K_max=K_max,
        device=device,
    )

    residual = parametric_black_scholes_residual(
        model,
        t_int,
        S_int,
        K_int,
        r=r,
        sigma=sigma,
    )

    loss_pde = torch.mean(residual**2)

    t_T, S_T, K_T, payoff = sample_parametric_terminal_points(
        n_terminal,
        T=T,
        S_max=S_max,
        K_min=K_min,
        K_max=K_max,
        device=device,
    )

    V_T_pred = model(t_T, S_T, K_T)
    loss_terminal = torch.mean((V_T_pred - payoff) ** 2)

    t_b, S_left, V_left, S_right, V_right, K_b = sample_parametric_boundary_points(
        n_boundary,
        r=r,
        T=T,
        S_max=S_max,
        K_min=K_min,
        K_max=K_max,
        device=device,
    )

    V_left_pred = model(t_b, S_left, K_b)
    V_right_pred = model(t_b, S_right, K_b)

    loss_boundary = torch.mean((V_left_pred - V_left) ** 2)
    loss_boundary += torch.mean((V_right_pred - V_right) ** 2)

    total_loss = loss_terminal + loss_boundary + beta * loss_pde

    components = {
        "total": total_loss.item(),
        "pde": loss_pde.item(),
        "terminal": loss_terminal.item(),
        "boundary": loss_boundary.item(),
    }

    return total_loss, components