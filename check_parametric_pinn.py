"""
Quick smoke test for the parametric PINN.

This checks that the strike-aware model, sampling functions, PDE residual and
loss calculation work together before adding market data loss.
"""

import torch

from src.losses import compute_pinn_loss
from src.pinn_model import ParametricPINN


def main():
    torch.manual_seed(0)

    model = ParametricPINN(
        hidden_dim=32,
        hidden_layers=2,
        T=1.0,
        S_max=160.0,
        K_scale=160.0,
    )

    loss, components = compute_pinn_loss(
        model,
        K_min=20.0,
        K_max=120.0,
        n_interior=200,
        n_terminal=100,
        n_boundary=100,
    )

    print("One parametric PINN loss evaluation on an untrained model:")
    print(f"total loss:    {components['total']:.6f}")
    print(f"PDE loss:      {components['pde']:.6f}")
    print(f"terminal loss: {components['terminal']:.6f}")
    print(f"boundary loss: {components['boundary']:.6f}")

    loss.backward()
    print("Backward pass completed successfully.")


if __name__ == "__main__":
    main()