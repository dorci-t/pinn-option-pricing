"""
Initial PINN model skeleton.

This file is not complete yet. The next project step is to implement the PDE residual
and the training loop.
"""

import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Simple fully connected neural network.

    Input:
        t, S

    Output:
        V(t, S)
    """

    def __init__(self, hidden_dim=64, hidden_layers=4, T=1.0, S_max=160.0):
        super().__init__()

        self.T = T
        self.S_max = S_max

        layers = [
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
        ]

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, t, S):
        t_scaled = t / self.T
        S_scaled = S / self.S_max

        x = torch.cat([t_scaled, S_scaled], dim=1)
        return self.net(x)
    
class ParametricPINN(nn.Module):
    """
    PINN model with strike price as an additional input.

    This version approximates V(t, S, K), so it can be used with several
    strikes instead of only one fixed strike.
    """

    def __init__(
        self,
        hidden_dim=64,
        hidden_layers=4,
        T=1.0,
        S_max=160.0,
        K_scale=160.0,
    ):
        super().__init__()

        self.T = T
        self.S_max = S_max
        self.K_scale = K_scale

        layers = [
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
        ]

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, t, S, K):
        t_scaled = t / self.T
        S_scaled = S / self.S_max
        K_scaled = K / self.K_scale

        x = torch.cat([t_scaled, S_scaled, K_scaled], dim=1)
        return self.net(x)