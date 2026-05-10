"""
Sampling functions for the parametric PINN.

The parametric model also receives the strike price K as input. This is needed
before adding real option-chain data, where several strikes occur for the same
underlying asset and expiration date.
"""

import torch


def sample_parametric_interior_points(
    n,
    T=1.0,
    S_max=160.0,
    K_min=20.0,
    K_max=120.0,
    device="cpu",
):
    """
    Sample interior points from the PDE domain.

    These points are used for the Black-Scholes PDE residual. The strike K is
    sampled as an additional parameter.
    """
    t = torch.rand(n, 1, device=device) * T

    eps = 1e-6
    S = eps + torch.rand(n, 1, device=device) * (S_max - eps)

    K = K_min + torch.rand(n, 1, device=device) * (K_max - K_min)

    t.requires_grad_(True)
    S.requires_grad_(True)

    return t, S, K


def sample_parametric_terminal_points(
    n,
    T=1.0,
    S_max=160.0,
    K_min=20.0,
    K_max=120.0,
    device="cpu",
):
    """
    Sample terminal points at maturity.

    For a European call option, the terminal condition is max(S-K, 0).
    """
    S = torch.rand(n, 1, device=device) * S_max
    K = K_min + torch.rand(n, 1, device=device) * (K_max - K_min)
    t = torch.full_like(S, T)

    payoff = torch.clamp(S - K, min=0.0)

    return t, S, K, payoff


def sample_parametric_boundary_points(
    n,
    r=0.05,
    T=1.0,
    S_max=160.0,
    K_min=20.0,
    K_max=120.0,
    device="cpu",
):
    """
    Sample boundary points at S = 0 and S = S_max.
    """
    t = torch.rand(n, 1, device=device) * T
    K = K_min + torch.rand(n, 1, device=device) * (K_max - K_min)

    S_left = torch.zeros_like(t)
    V_left = torch.zeros_like(t)

    S_right = torch.full_like(t, S_max)
    V_right = S_max - K * torch.exp(-r * (T - t))

    return t, S_left, V_left, S_right, V_right, K