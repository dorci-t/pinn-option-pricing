"""
PINN loss components for the Black-Scholes benchmark.

The total loss combines three terms: the Black-Scholes PDE residual on
interior points, the terminal payoff condition at maturity, and the boundary
conditions at the edges of the stock price domain.
"""

import torch

from src.pde import black_scholes_residual
from src.sampling import (
    sample_boundary_points,
    sample_interior_points,
    sample_terminal_points,
)


def compute_pinn_loss(
    model,
    K=40.0,
    r=0.05,
    sigma=0.2,
    T=1.0,
    S_max=160.0,
    K_min=None,
    K_max=None,
    n_interior=1000,
    n_terminal=300,
    n_boundary=300,
    beta=1.0,
    device="cpu",
):
    parametric = K_min is not None

    interior = sample_interior_points(
        n_interior,
        T=T,
        S_max=S_max,
        K_min=K_min,
        K_max=K_max,
        device=device,
    )
    t_int, S_int = interior[0], interior[1]
    extra_int = interior[2:]

    residual = black_scholes_residual(
        model,
        t_int,
        S_int,
        *extra_int,
        r=r,
        sigma=sigma,
    )
    loss_pde = torch.mean(residual**2)

    terminal = sample_terminal_points(
        n_terminal,
        K=K,
        T=T,
        S_max=S_max,
        K_min=K_min,
        K_max=K_max,
        device=device,
    )
    if parametric:
        t_T, S_T, K_T, payoff = terminal
        V_T_pred = model(t_T, S_T, K_T)
    else:
        t_T, S_T, payoff = terminal
        V_T_pred = model(t_T, S_T)
    loss_terminal = torch.mean((V_T_pred - payoff) ** 2)

    boundary = sample_boundary_points(
        n_boundary,
        K=K,
        r=r,
        T=T,
        S_max=S_max,
        K_min=K_min,
        K_max=K_max,
        device=device,
    )
    if parametric:
        t_b, S_left, V_left, S_right, V_right, K_b = boundary
        V_left_pred = model(t_b, S_left, K_b)
        V_right_pred = model(t_b, S_right, K_b)
    else:
        t_b, S_left, V_left, S_right, V_right = boundary
        V_left_pred = model(t_b, S_left)
        V_right_pred = model(t_b, S_right)

    loss_boundary = (V_left_pred - V_left) ** 2
    loss_boundary += (V_right_pred - V_right) ** 2
    loss_boundary = torch.mean(loss_boundary) / 2

    total_loss = loss_terminal + loss_boundary + beta * loss_pde

    components = {
        "total": total_loss.item(),
        "pde": loss_pde.item(),
        "terminal": loss_terminal.item(),
        "boundary": loss_boundary.item(),
    }

    return total_loss, components
