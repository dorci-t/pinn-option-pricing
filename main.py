"""
Computes & visualises the analytic Black-Scholes price of a European call option. Later,
the PINN result will be compared against this benchmark.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.black_scholes import european_call_price


def main():
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    # Parameters for a simple European call example.
    K = 40.0        # strike price (kötési árfolyam)
    r = 0.05        # risk-free interest rate (kockázatmentes kamatláb)
    sigma = 0.2     # volatility (volatilitás)
    T = 1.0         # maturity time (lejárati idő)
    S_max = 160.0   # maximum stock price considered (legnagyobb vizsgált
                    # részvényárfolyam)

    # Plot a few fixed-time slices first.
    # We start from S=1 instead of S=0 because the formula contains log(S/K).
    S_values = np.linspace(1.0, S_max, 300)

    slice_data = {}
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        prices = european_call_price(S_values, t, K=K, r=r, sigma=sigma, T=T)
        slice_data[f"t = {t}"] = prices

    plt.figure(figsize=(8, 5))
    for label, prices in slice_data.items():
        plt.plot(S_values, prices, label=label)
    plt.xlabel("Stock price S")
    plt.ylabel("Option price V(t, S)")
    plt.title("Analytic Black-Scholes European Call Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "analytic_price_slices.png", dpi=200)
    plt.close()

    # Now compute the same formula on a full (t, S) grid for a surface plot.
    S_grid = np.linspace(1.0, S_max, 100)
    t_grid = np.linspace(0.0, T, 100)
    SS, TT = np.meshgrid(S_grid, t_grid)

    VV = european_call_price(SS, TT, K=K, r=r, sigma=sigma, T=T)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(SS, TT, VV, linewidth=0, antialiased=True)
    ax.set_xlabel("Stock price S")
    ax.set_ylabel("Time t")
    ax.set_zlabel("Option price V(t, S)")
    ax.set_title("Analytic Black-Scholes European Call Price")
    plt.tight_layout()
    plt.savefig(output_dir / "analytic_price_surface.png", dpi=200)
    plt.close()

    print("Done.")
    print("Generated figures:")
    print("- figures/analytic_price_slices.png")
    print("- figures/analytic_price_surface.png")


if __name__ == "__main__":
    main()
