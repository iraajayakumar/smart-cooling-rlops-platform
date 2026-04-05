"""
reward.py
Reward function for the data center cooling RL agent.

Design philosophy
─────────────────
reward = −energy_cost − temperature_penalty + stability_bonus

We want the agent to:
  1. Use as little cooling energy as possible         (−energy_cost)
  2. Never let temperature exceed TEMP_SAFE (75 °C)  (−temperature_penalty)
  3. Converge on stable temperature near TEMP_IDEAL  (+stability_bonus)

All coefficients were determined empirically during reward engineering.
"""

from __future__ import annotations

import numpy as np


# ── Reward shaping coefficients ─────────────────────────────────────────────
ENERGY_COEFF        = 1.0    # weight on energy penalty  (higher → be stingy)
TEMP_VIOLATION_COEFF = 3.0   # weight on overheating penalty  (higher → safer)
STABILITY_COEFF     = 0.5    # weight on stability bonus
CATASTROPHIC_PENALTY = -50.0 # one-shot penalty if temperature is critical (≥85°C)
CRITICAL_THRESHOLD  = 85.0   # °C — triggers catastrophic penalty

# ── Temperature zones ────────────────────────────────────────────────────────
# reward shaping is progressive:
#   [TEMP_MIN, TEMP_IDEAL]     → ideal range, full stability bonus
#   [TEMP_IDEAL, TEMP_SAFE]    → linear penalty gradient
#   [TEMP_SAFE,  TEMP_CRITICAL]→ heavy quadratic penalty
#   [TEMP_CRITICAL, TEMP_MAX]  → catastrophic penalty + episode ends


def compute_reward(
    temperature: float,
    energy: float,
    cooling: int,
    temp_safe: float  = 75.0,
    temp_ideal: float = 35.0,
) -> float:
    """
    Compute the scalar reward for one environment step.

    Parameters
    ----------
    temperature : current temperature in °C
    energy      : energy consumed this step (cooling * ENERGY_PER_LEVEL)
    cooling     : cooling action applied (0, 1, 2)
    temp_safe   : soft upper temperature limit
    temp_ideal  : target operating temperature

    Returns
    -------
    float : scalar reward (can be negative)
    """
    # ── 1. Energy cost (always negative) ─────────────────────────────────
    energy_penalty = -ENERGY_COEFF * energy

    # ── 2. Temperature violation penalty ─────────────────────────────────
    if temperature >= CRITICAL_THRESHOLD:
        # Catastrophic zone: heavy flat penalty
        temp_penalty = CATASTROPHIC_PENALTY
    elif temperature > temp_safe:
        # Soft violation zone: quadratic penalty
        overshoot = temperature - temp_safe
        temp_penalty = -TEMP_VIOLATION_COEFF * (overshoot ** 2)
    else:
        temp_penalty = 0.0

    # ── 3. Stability bonus ────────────────────────────────────────────────
    # Positive reward for keeping temperature near the ideal set point.
    # Shape: Gaussian bell centred at temp_ideal with σ=10°C.
    sigma = 10.0
    deviation = temperature - temp_ideal
    stability_bonus = STABILITY_COEFF * np.exp(
        -(deviation ** 2) / (2 * sigma ** 2)
    )

    reward = energy_penalty + temp_penalty + stability_bonus
    return float(reward)


# ── Convenience: describe reward components (used in logging / debugging) ────

def reward_breakdown(
    temperature: float,
    energy: float,
    cooling: int,
    temp_safe: float  = 75.0,
    temp_ideal: float = 35.0,
) -> dict:
    """Return a dict of each reward component for logging / TensorBoard."""
    energy_penalty  = -ENERGY_COEFF * energy
    temp_penalty    = _temp_penalty(temperature, temp_safe)
    sigma           = 10.0
    stability_bonus = STABILITY_COEFF * np.exp(
        -((temperature - temp_ideal) ** 2) / (2 * sigma ** 2)
    )
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
