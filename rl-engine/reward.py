"""
reward.py - Rebalanced reward function
"""
from __future__ import annotations
import numpy as np

ENERGY_COEFF         = 0.2
TEMP_VIOLATION_COEFF = 5.0
STABILITY_COEFF      = 2.0
CRITICAL_THRESHOLD   = 85.0
CATASTROPHIC_PENALTY = -30.0


def compute_reward(
    temperature: float,
    energy: float,
    cooling: int,
    temp_safe: float  = 75.0,
    temp_ideal: float = 35.0,
) -> float:

    # 1. Energy cost (small penalty)
    energy_penalty = -ENERGY_COEFF * energy

    # 2. Temperature violation (only above safe limit)
    if temperature >= CRITICAL_THRESHOLD:
        temp_penalty = CATASTROPHIC_PENALTY
    elif temperature > temp_safe:
        overshoot = temperature - temp_safe
        temp_penalty = -TEMP_VIOLATION_COEFF * (overshoot ** 2)
    else:
        temp_penalty = 0.0

    # 3. Stability bonus (strong pull toward ideal temp)
    sigma = 15.0
    deviation = temperature - temp_ideal
    stability_bonus = STABILITY_COEFF * np.exp(
        -(deviation ** 2) / (2 * sigma ** 2)
    )

    return float(energy_penalty + temp_penalty + stability_bonus)


def reward_breakdown(
    temperature: float,
    energy: float,
    cooling: int,
    temp_safe: float  = 75.0,
    temp_ideal: float = 35.0,
) -> dict:
    energy_penalty  = -ENERGY_COEFF * energy
    sigma           = 15.0
    stability_bonus = STABILITY_COEFF * np.exp(
        -((temperature - temp_ideal) ** 2) / (2 * sigma ** 2)
    )
    if temperature >= CRITICAL_THRESHOLD:
        temp_penalty = CATASTROPHIC_PENALTY
    elif temperature > temp_safe:
        temp_penalty = -TEMP_VIOLATION_COEFF * ((temperature - temp_safe) ** 2)
    else:
        temp_penalty = 0.0

    total = energy_penalty + temp_penalty + stability_bonus
    return {
        "reward/total":           round(total, 4),
        "reward/energy_penalty":  round(energy_penalty, 4),
        "reward/temp_penalty":    round(temp_penalty, 4),
        "reward/stability_bonus": round(stability_bonus, 4),
    }


def _temp_penalty(temperature: float, temp_safe: float) -> float:
    if temperature >= CRITICAL_THRESHOLD:
        return CATASTROPHIC_PENALTY
    if temperature > temp_safe:
        return -TEMP_VIOLATION_COEFF * ((temperature - temp_safe) ** 2)
    return 0.0