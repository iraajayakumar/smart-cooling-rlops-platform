"""
utils/logging.py
Structured logging helpers.
"""

from __future__ import annotations

import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_logger(name: str = "rl_agent", level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a clean console handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger            # already configured
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


class EpisodeLogger:
    """
    Collects per-step data during an episode and writes a CSV summary.

    Usage
    ─────
        logger = EpisodeLogger(out_dir="logs/episodes")
        logger.begin_episode()
        for step:
            logger.log_step(temp, workload, cooling, reward)
        logger.end_episode()
        logger.save()
    """

    def __init__(self, out_dir: str | Path = "logs/episodes"):
        self._dir   = Path(out_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._episodes: list[dict[str, Any]] = []
        self._current: dict[str, list] = defaultdict(list)
        self._ep_idx = 0

    def begin_episode(self):
        self._current = defaultdict(list)

    def log_step(
        self,
        temperature: float,
        workload:    float,
        cooling:     int,
        reward:      float,
    ):
        self._current["temperature"].append(temperature)
        self._current["workload"].append(workload)
        self._current["cooling"].append(cooling)
        self._current["reward"].append(reward)

    def end_episode(self):
        import numpy as np
        temps   = self._current["temperature"]
        rewards = self._current["reward"]
        self._episodes.append({
            "episode":       self._ep_idx,
            "steps":         len(temps),
            "mean_temp":     round(float(np.mean(temps)),   2),
            "max_temp":      round(float(np.max(temps)),    2),
            "min_temp":      round(float(np.min(temps)),    2),
            "total_reward":  round(float(sum(rewards)),     2),
            "mean_reward":   round(float(np.mean(rewards)), 4),
            "mean_cooling":  round(float(np.mean(self._current["cooling"])), 2),
        })
        self._ep_idx += 1

    def save(self, filename: str | None = None) -> Path:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = filename or f"episodes_{ts}.csv"
        path = self._dir / name

        if not self._episodes:
            return path

        fieldnames = list(self._episodes[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._episodes)

        return path
