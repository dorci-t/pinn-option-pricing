# %% [markdown]
# # Real data demo
#
# Downloads one call option chain with yfinance, computes market mid prices
# from bid/ask quotes, and compares them with analytic Black-Scholes prices.
#
# Data is cached to `data/` — delete the cached CSV to re-download.

# %%
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# When run as a notebook from notebooks/, the working directory and sys.path
# point at notebooks/, not the project root. We detect this by checking whether
# __file__ exists (it does in scripts, not in notebooks) and adjust accordingly.
# The src/ check makes os.chdir idempotent - it only moves if we're not already
# at the project root.
_project_root = str(Path(__file__).resolve().parent) if "__file__" in dir() else ".."
sys.path.insert(0, _project_root)
if not Path("src").is_dir():
    os.chdir(_project_root)

from src.black_scholes import european_call_price
from src.config import load_config

CONFIG_PATH = os.environ.get("CONFIG", "configs/real_data.toml")
config = load_config(CONFIG_PATH)


def annualised_volatility(close_prices):
    """
    Estimate annualised volatility from daily close prices.
    """
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)


def plot_market_vs_black_scholes(calls, output_path):
    """
    Plot market mid prices and Black-Scholes prices against strike.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(
        calls["strike"],
        calls["mid_price"],
        marker="o",
        linestyle="",
        label="Market mid price",
    )

    plt.plot(
        calls["strike"],
        calls["bs_price"],
        marker="o",
        linestyle="-",
        label="Black-Scholes price",
    )

    plt.xlabel("Strike price")
    plt.ylabel("Call option price")
    plt.title("Market option prices vs Black-Scholes benchmark")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# %% [markdown]
# ## Load option chain (download + cache, or read from cache)

# %%
data_dir = Path("data")
figures_dir = Path("figures")

data_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

ticker_symbol = config["ticker"]
risk_free_rate = config["risk_free_rate"]
expiration = config["expiration"]

cache_path = data_dir / f"{ticker_symbol}_calls_{expiration}.csv"

if cache_path.exists():
    print(f"Using cached data from {cache_path}")
    calls = pd.read_csv(cache_path)
    spot_price = float(calls["spot_price"].iloc[0])
    sigma = float(calls["sigma_estimate"].iloc[0])
    days_to_expiry = int(calls["days_to_expiry"].iloc[0])
else:
    print(f"Downloading option chain for {ticker_symbol}, expiration {expiration}")
    ticker = yf.Ticker(ticker_symbol)

    option_chain = ticker.option_chain(expiration)
    calls = option_chain.calls.copy()

    calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)].copy()
    calls["mid_price"] = (calls["bid"] + calls["ask"]) / 2

    spot_history = ticker.history(
        start=config["spot_history_start"],
        end=config["spot_history_end"],
    )
    if spot_history.empty:
        raise RuntimeError(f"No historical price data found for {ticker_symbol}.")

    spot_price = float(spot_history["Close"].iloc[-1])
    sigma = float(annualised_volatility(spot_history["Close"]))

    today = pd.Timestamp(config["spot_history_end"])
    expiry_date = pd.Timestamp(expiration)
    days_to_expiry = (expiry_date - today).days

    calls["ticker"] = ticker_symbol
    calls["spot_price"] = spot_price
    calls["expiration"] = expiration
    calls["days_to_expiry"] = days_to_expiry
    calls["sigma_estimate"] = sigma
    calls["risk_free_rate"] = risk_free_rate

    calls.to_csv(cache_path, index=False)
    print(f"Cached data to {cache_path}")

T = days_to_expiry / 365

print(f"Ticker: {ticker_symbol}")
print(f"Expiration: {expiration}")
print(f"Spot price: {spot_price:.2f}")
print(f"Estimated annual volatility: {sigma:.4f}")
print(f"Days to expiry: {days_to_expiry}")

# %% [markdown]
# ## Compute Black-Scholes prices and compare

# %%
calls["bs_price"] = european_call_price(
    S=spot_price,
    t=0.0,
    K=calls["strike"].to_numpy(),
    r=risk_free_rate,
    sigma=sigma,
    T=T,
)

lower = 0.7 * spot_price
upper = 1.3 * spot_price
plot_calls = calls[(calls["strike"] >= lower) & (calls["strike"] <= upper)].copy()

plot_market_vs_black_scholes(
    plot_calls,
    figures_dir / "market_vs_black_scholes.png",
)

print("Saved figure to: figures/market_vs_black_scholes.png")
