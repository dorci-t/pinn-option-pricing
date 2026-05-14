"""
Sampling utilities for PINN training points.

Each function samples points from the Black-Scholes PDE domain. When K_min
and K_max are provided, the strike price K is also sampled — this supports
the parametric model V(t, S, K).
"""

import torch


def sample_interior_points(n, T=1.0, S_max=160.0, K_min=None, K_max=None, device="cpu"):
    """
    Sample interior points from the PDE domain.
    These points are used for the Black-Scholes residual loss.
    """
    t = torch.rand(n, 1, device=device) * T

    eps = 1e-6
    S = eps + torch.rand(n, 1, device=device) * (S_max - eps)

    t.requires_grad_(True)
    S.requires_grad_(True)

    if K_min is not None:
        K = K_min + torch.rand(n, 1, device=device) * (K_max - K_min)
        return t, S, K

    return t, S


def sample_terminal_points(n, K=40.0, T=1.0, S_max=160.0, K_min=None, K_max=None, device="cpu"):
    """
    Sample points at maturity t = T.
    For a European call option, the target is max(S-K, 0).
    """
    S = torch.rand(n, 1, device=device) * S_max
    t = torch.full_like(S, T)

    if K_min is not None:
        K = K_min + torch.rand(n, 1, device=device) * (K_max - K_min)
        payoff = torch.clamp(S - K, min=0.0)
        return t, S, K, payoff

    payoff = torch.clamp(S - K, min=0.0)
    return t, S, payoff


def sample_boundary_points(n, K=40.0, r=0.05, T=1.0, S_max=160.0, K_min=None, K_max=None, device="cpu"):
    """
    Sample boundary points at S = 0 and S = S_max.
    """
    t = torch.rand(n, 1, device=device) * T

    S_left = torch.zeros_like(t)
    V_left = torch.zeros_like(t)

    S_right = torch.full_like(t, S_max)

    if K_min is not None:
        K = K_min + torch.rand(n, 1, device=device) * (K_max - K_min)
        V_right = S_max - K * torch.exp(-r * (T - t))
        return t, S_left, V_left, S_right, V_right, K

    V_right = S_max - K * torch.exp(-r * (T - t))
    return t, S_left, V_left, S_right, V_right
