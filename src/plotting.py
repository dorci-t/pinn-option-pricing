"""
Shared plotting utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_surface(SS, TT, ZZ, title, zlabel, output_path):
    """Plot a 3D surface of ZZ over (SS, TT) grid."""
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(SS, TT, ZZ, linewidth=0, antialiased=True)
    ax.set_xlabel("Stock price S")
    ax.set_ylabel("Time t")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(Path(output_path), dpi=200)
    plt.close()


def plot_lines(x, line_data, title, xlabel, ylabel, output_path):
    """
    Plot one or more lines on a shared axes.

    line_data: dict of {label: y_values} or {label: (y_values, style_kwargs)}.
    """
    plt.figure(figsize=(8, 5))
    for label, value in line_data.items():
        if isinstance(value, tuple):
            y, kwargs = value
            plt.plot(x, y, label=label, **kwargs)
        else:
            plt.plot(x, value, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(line_data) > 1:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(output_path), dpi=200)
    plt.close()


def plot_comparison_slices(S_values, true_slices, pred_slices, model_name, output_path):
    """Plot analytic vs predicted price slices at several time values."""
    plt.figure(figsize=(8, 5))
    for label in true_slices:
        plt.plot(S_values, true_slices[label], label=f"{label} analytic")
        plt.plot(S_values, pred_slices[label], linestyle="--", label=f"{label} {model_name}")
    plt.xlabel("Stock price S")
    plt.ylabel("Option price V(t, S)")
    plt.title(f"Analytic Black-Scholes vs {model_name} prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(output_path), dpi=200)
    plt.close()
