"""
Black-Scholes PDE residual for the parametric PINN.

The model approximates V(t, S, K). The Black-Scholes PDE is still evaluated
with respect to t and S; the strike K is treated as an input parameter.
"""

import torch


def parametric_black_scholes_residual(model, t, S, K, r=0.05, sigma=0.2):
    """
    Compute the Black-Scholes PDE residual for V(t, S, K).
    """
    V = model(t, S, K)

    V_t = torch.autograd.grad(
        V,
        t,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
    )[0]

    V_S = torch.autograd.grad(
        V,
        S,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
    )[0]

    V_SS = torch.autograd.grad(
        V_S,
        S,
        grad_outputs=torch.ones_like(V_S),
        create_graph=True,
    )[0]

    residual = V_t + 0.5 * sigma**2 * S**2 * V_SS + r * S * V_S - r * V

    return residual