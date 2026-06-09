"""
inference.py
Inference interface for the trained PPO cooling agent.

This module is the ONLY public API the backend needs to consume.
It exposes a single function: predict_action(state) → int

Integration contract
────────────────────
Input  (dict or CoolingState):
    {
        "temperature": float,   # °C
        "workload":    float,   # 0–10
        "cooling":     int      # 0=Low, 1=Medium, 2=High
    }

Output (int): 0 | 1 | 2  — next cooling action

Usage
─────
    from inference import CoolingAgent

    agent = CoolingAgent()               # loads from default models/ path
    action = agent.predict(state_dict)   # deterministic inference
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from environment import DataCenterEnv

logger = logging.getLogger(__name__)

# ── Default paths ─────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).parent
_MODEL_PATH    = _ROOT / "models" / "best_model.zip"
_NORMALIZER    = _ROOT / "models" / "vec_normalize.pkl"


# ── Typed input for IDE support ───────────────────────────────────────────────
class CoolingState(TypedDict):
    temperature: float
    workload:    float
    cooling:     int


# ─────────────────────────────────────────────────────────────────────────────
# CoolingAgent
# ─────────────────────────────────────────────────────────────────────────────

class CoolingAgent:
    """
    Singleton-style inference wrapper.

    Loads the PPO model once on construction and exposes `predict()`.
    Thread-safe for concurrent read-only inference (no shared mutable state
    after __init__).
    """

    ACTION_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

    def __init__(
        self,
        model_path: str | Path = _MODEL_PATH,
        normalizer_path: str | Path = _NORMALIZER,
    ):
        model_path       = Path(model_path)
        normalizer_path  = Path(normalizer_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                "Run train.py first."
            )

        logger.info("Loading PPO model from %s", model_path)
        self._model: PPO = PPO.load(str(model_path))

        # Load observation normalizer if available
        self._vec_norm: VecNormalize | None = None
        if normalizer_path.exists():
            dummy_env = DummyVecEnv([lambda: DataCenterEnv()])
            self._vec_norm = VecNormalize.load(
                str(normalizer_path), dummy_env
            )
            self._vec_norm.training   = False   # disable updates
            self._vec_norm.norm_reward = False
            logger.info("Observation normalizer loaded.")
        else:
            logger.warning(
                "No normalizer found at %s — using raw observations.",
                normalizer_path,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, state: CoolingState | dict) -> int:
        """
        Run deterministic inference.

        Parameters
        ----------
        state : dict with keys temperature, workload, cooling

        Returns
        -------
        int : 0 (Low) | 1 (Medium) | 2 (High)
        """
        obs = self._state_to_obs(state)
        obs = self._normalize(obs)
        action, _states = self._model.predict(obs, deterministic=True)
        action = np.asarray(action).squeeze()
        return int(action.item())

    def predict_with_confidence(self, state: CoolingState | dict) -> dict:
        """
        Return action + per-action probability distribution.

        Useful for monitoring dashboards or explaining decisions.
        """
        obs     = self._normalize(self._state_to_obs(state))
        action  = int(self._model.predict(obs, deterministic=True)[0])

        # Extract per-action probabilities from policy distribution
        import torch
        with torch.no_grad():
            obs_tensor = self._model.policy.obs_to_tensor(obs)[0]
            dist       = self._model.policy.get_distribution(obs_tensor)
            probs      = dist.distribution.probs.cpu().numpy().flatten()

        return {
            "action":        action,
            "action_label":  self.ACTION_LABELS[action],
            "probabilities": {
                "LOW":    round(float(probs[0]), 4),
                "MEDIUM": round(float(probs[1]), 4),
                "HIGH":   round(float(probs[2]), 4),
            },
        }

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _state_to_obs(state: dict) -> np.ndarray:
        return np.array(
            [
                float(state["temperature"]),
                float(state["workload"]),
                float(state["cooling"]),
            ],
            dtype=np.float32,
        ).reshape(1, -1)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self._vec_norm is not None:
            return self._vec_norm.normalize_obs(obs)
        return obs


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience function (matches integration contract)
# ─────────────────────────────────────────────────────────────────────────────

_agent: CoolingAgent | None = None


def predict_action(state: dict) -> int:
    """
    Module-level function for backend integration.

    Lazily initialises a singleton CoolingAgent on first call.

    Parameters
    ----------
    state : { "temperature": float, "workload": float, "cooling": int }

    Returns
    -------
    int : 0 | 1 | 2
    """
    global _agent
    if _agent is None:
        _agent = CoolingAgent()
    return _agent.predict(state)


# ─────────────────────────────────────────────────────────────────────────────
# CLI / smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_states: list[CoolingState] = [
        {"temperature": 30.0, "workload": 2.0, "cooling": 1},   # cool, low load
        {"temperature": 50.0, "workload": 5.0, "cooling": 1},   # warm, medium
        {"temperature": 70.0, "workload": 8.0, "cooling": 1},   # hot, high load
        {"temperature": 80.0, "workload": 9.5, "cooling": 2},   # critical
    ]

    agent = CoolingAgent()
    print("\n── Inference smoke test ─────────────────────────")
    print(f"{'Temp':>6} {'Load':>6} {'Prev':>6} → {'Action':>8}")
    print("─" * 40)
    for s in test_states:
        action = agent.predict(s)
        label  = CoolingAgent.ACTION_LABELS[action]
        print(
            f"{s['temperature']:>6.1f}°C "
            f"{s['workload']:>6.1f} "
            f"{s['cooling']:>6}  → "
            f"{label:>8} ({action})"
        )
    print("─────────────────────────────────────────────────\n")
