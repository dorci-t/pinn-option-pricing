"""
PINN model architectures for Black-Scholes option pricing.
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
    
class ResidualBlocks(nn.Module):
    def __init__(self, hidden_dim, n_layers, act_class):
        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.acts = nn.ModuleList([act_class() for _ in range(n_layers)])
    
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.acts[i](x) + x

        return x

class GatedPINN(nn.Module):
    """
    Dual-path gated network (Figure 3 in the paper).

    A simple linear path captures the ramp-shaped payoff, while a deeper
    residual path with a learned gate captures nonlinearities.

    Input:
        t, S

    Output:
        V(t, S)
    """

    def __init__(self, hidden_dim=64, hidden_layers=4, T=1.0, S_max=160.0):
        super().__init__()

        self.T = T
        self.S_max = S_max

        self.fc1_in = nn.Linear(2, hidden_dim)
        self.act1_in = nn.Tanh()

        self.fc1_res = ResidualBlocks(hidden_dim, hidden_layers, nn.Tanh)

        self.fce = nn.Linear(hidden_dim, 1)

        self.fcw = nn.Linear(3, 1)

        self.fc2 = nn.Linear(2, 1)

    def forward(self, t, S):
        t_scaled = t / self.T
        S_scaled = S / self.S_max

        x = torch.cat([t_scaled, S_scaled], dim=1)

        yx = self.fc2(x)

        yh = self.fc1_in(x)
        yh = self.act1_in(yh)
        yh = self.fc1_res(yh)
        yh = self.fce(yh)

        wh = torch.cat([x, yh], dim=1)
        wh = self.fcw(wh)

        return yx + wh * yh
    
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


MODELS = {
    "pinn": PINN,
    "gated": GatedPINN,
}
