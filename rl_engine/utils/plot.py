"""
utils/plot.py
Plotting helpers for training curves and episode visualisation.

All functions return matplotlib Figure objects so callers can
save or display as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def plot_training_curve(
    log_dir: str | Path,
    save_path: str | Path | None = None,
    window: int = 50,
):
    """
    Parse SB3 monitor CSV files and plot mean episode rewards.

    Parameters
    ----------
    log_dir    : directory containing Monitor .csv files
    save_path  : if given, saves the figure instead of showing it
    window     : rolling average window size
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    log_dir = Path(log_dir)
    dfs = []
    for csv_file in log_dir.glob("**/*.monitor.csv"):
        df = pd.read_csv(csv_file, skiprows=1)
        dfs.append(df[["t", "r", "l"]])

    if not dfs:
        print(f"No monitor CSV files found in {log_dir}")
        return None

    data = pd.concat(dfs).sort_values("t").reset_index(drop=True)
    rewards = data["r"].values
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("PPO Training Progress — Data Center Cooling", fontsize=13)

    # Reward
    axes[0].plot(rewards,  alpha=0.3, color="#4A90E2", label="raw")
    axes[0].plot(np.arange(window - 1, len(rewards)), smoothed,
                 color="#4A90E2", linewidth=2, label=f"{window}-ep avg")
    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Episode Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Episode length
    lengths = data["l"].values
    axes[1].plot(lengths, alpha=0.5, color="#E24B4A")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Episode Length")
    axes[1].axhline(200, color="gray", linewidth=0.5, linestyle="--", label="max steps")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    return fig


def plot_episode(
    temperatures: Sequence[float],
    workloads:    Sequence[float],
    actions:      Sequence[int],
    rewards:      Sequence[float],
    save_path: str | Path | None = None,
):
    """
    Visualise a single episode: temperature, workload, cooling actions, reward.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    steps = list(range(len(temperatures)))
    cooling_colors = {0: "#3B8BD4", 1: "#EF9F27", 2: "#E24B4A"}
    action_labels  = {0: "Low", 1: "Medium", 2: "High"}

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Episode Playback — Data Center Cooling", fontsize=13)

    # Temperature
    axes[0].plot(steps, temperatures, color="#E24B4A", linewidth=1.5)
    axes[0].axhline(75, color="#E24B4A", linestyle="--", linewidth=0.8,
                    alpha=0.7, label="safe limit (75°C)")
    axes[0].axhline(35, color="#3B8BD4", linestyle="--", linewidth=0.8,
                    alpha=0.7, label="ideal (35°C)")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Workload + cooling action (as bar colours)
    axes[1].plot(steps, workloads, color="#4A90E2", linewidth=1.5, label="workload")
    bar_colors = [cooling_colors[a] for a in actions]
    axes[1].bar(steps, actions, color=bar_colors, alpha=0.4, width=1.0)
    axes[1].set_ylabel("Workload / Cooling")
    patches = [mpatches.Patch(color=v, label=f"{action_labels[k]} cooling")
               for k, v in cooling_colors.items()]
    axes[1].legend(handles=patches, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Reward
    axes[2].fill_between(steps, rewards, alpha=0.4,
                          color="#1D9E75" if np.mean(rewards) >= 0 else "#E24B4A")
    axes[2].plot(steps, rewards, color="#1D9E75", linewidth=1.0)
    axes[2].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[2].set_ylabel("Reward")
    axes[2].set_xlabel("Step")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    return fig
