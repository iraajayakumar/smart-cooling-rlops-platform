"""
environment.py
Gymnasium environment for Smart Data Center Cooling Optimization.

State  : [temperature, workload, cooling]
Actions: 0=Low, 1=Medium, 2=High cooling
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .reward import compute_reward


# ── Physics constants (must match backend physics.py) ──────────────────────
WORKLOAD_HEAT_FACTOR = 0.3   # temperature rise per workload unit per step
COOLING_FACTOR       = 1.0   # temperature drop per cooling level per step
ENERGY_PER_LEVEL     = 2.0   # energy units per cooling level per step

# ── Safety / operating limits ───────────────────────────────────────────────
TEMP_MIN   = 15.0   # °C — absolute cold floor
TEMP_MAX   = 90.0   # °C — absolute hot ceiling (episode terminates beyond this)
TEMP_SAFE  = 75.0   # °C — soft upper limit used in reward shaping
TEMP_IDEAL = 35.0   # °C — target operating temperature

WORKLOAD_MIN = 0.0
WORKLOAD_MAX = 10.0

# ── Episode config ──────────────────────────────────────────────────────────
MAX_STEPS = 500


class DataCenterEnv(gym.Env):
    """
    A single-zone data center cooling simulation environment.

    Observation space:
        Box([temperature, workload, cooling])
        temperature : float in [TEMP_MIN, TEMP_MAX]
        workload    : float in [0, 10]
        cooling     : int   in {0, 1, 2}  (as float)

    Action space:
        Discrete(3)  →  0=Low, 1=Medium, 2=High

    Reward:
        See reward.py for full definition.
        Roughly: −energy − penalty_for_overheating + stability_bonus
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        render_mode: str | None = None,
        workload_pattern: str = "random",   # "random" | "sine" | "spike"
        seed: int | None = None,
    ):
        super().__init__()

        self.render_mode    = render_mode
        self.workload_pattern = workload_pattern

        # ── Spaces ────────────────────────────────────────────────────────
        low  = np.array([TEMP_MIN,    WORKLOAD_MIN, 0.0], dtype=np.float32)
        high = np.array([TEMP_MAX,    WORKLOAD_MAX, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space      = spaces.Discrete(3)

        # ── Internal state ─────────────────────────────────────────────────
        self._temperature  = TEMP_IDEAL
        self._workload     = 0.0
        self._cooling      = 1          # start at medium
        self._step_count   = 0
        self._rng          = np.random.default_rng(seed)

        # For sine workload pattern
        self._sine_phase   = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._temperature = float(
            self._rng.uniform(TEMP_IDEAL - 5, TEMP_IDEAL + 5)
        )
        self._workload   = float(self._rng.uniform(1.0, 4.0))
        self._cooling    = 1
        self._step_count = 0
        self._sine_phase = 0.0

        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        self._cooling = int(action)

        # ── Physics update ────────────────────────────────────────────────
        self._workload   = self._next_workload()
        self._temperature = self._update_temperature(
            self._temperature, self._workload, self._cooling
        )
        self._temperature = float(np.clip(self._temperature, TEMP_MIN, TEMP_MAX))

        # ── Reward ────────────────────────────────────────────────────────
        energy = self._cooling * ENERGY_PER_LEVEL
        reward = compute_reward(
            temperature=self._temperature,
            energy=energy,
            cooling=self._cooling,
            temp_safe=TEMP_SAFE,
            temp_ideal=TEMP_IDEAL,
        )

        # ── Termination conditions ─────────────────────────────────────────
        self._step_count += 1
        terminated = self._temperature >= TEMP_MAX
        truncated  = self._step_count >= MAX_STEPS

        if self.render_mode == "human":
            self._render_human()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "human":
            self._render_human()

    def close(self):
        pass

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [self._temperature, self._workload, float(self._cooling)],
            dtype=np.float32,
        )

    def _get_info(self) -> dict:
        return {
            "temperature": self._temperature,
            "workload":    self._workload,
            "cooling":     self._cooling,
            "step":        self._step_count,
        }

    def _update_temperature(
        self, temp: float, workload: float, cooling: int
    ) -> float:
        """
        Shared physics equation (identical to backend physics.py):
            T_new = T + workload * 0.4 − cooling * 0.6
        """
        temp += workload * WORKLOAD_HEAT_FACTOR
        temp -= cooling  * COOLING_FACTOR
        noise = float(self._rng.normal(0.0, 0.3))   # thermal noise
        return temp + noise

    def _next_workload(self) -> float:
        """Generate next workload based on selected pattern."""
        if self.workload_pattern == "sine":
            self._sine_phase += 0.05
            base = 5.0 + 4.0 * np.sin(self._sine_phase)
            noise = float(self._rng.normal(0, 0.5))
            return float(np.clip(base + noise, WORKLOAD_MIN, WORKLOAD_MAX))

        if self.workload_pattern == "spike":
            # Random spikes simulating burst traffic
            if self._rng.random() < 0.1:  # 10% chance of spike
                return float(self._rng.uniform(7.0, WORKLOAD_MAX))
            return float(
                np.clip(self._workload + self._rng.normal(0, 0.5), 0.5, 7.0)
            )

        # Default: random walk
        delta = float(self._rng.normal(0.0, 0.8))
        return float(np.clip(self._workload + delta, WORKLOAD_MIN, WORKLOAD_MAX))

    def _render_human(self):
        bar  = "█" * self._cooling + "░" * (2 - self._cooling)
        cool = ["LOW", "MED", "HI "][self._cooling]
        print(
            f"Step {self._step_count:3d} | "
            f"Temp={self._temperature:5.1f}°C | "
            f"Load={self._workload:4.1f} | "
            f"Cool={cool} [{bar}]"
        )

    def _render_ansi(self) -> str:
        cool = ["LOW", "MED", "HI "][self._cooling]
        return (
            f"Step={self._step_count} | "
            f"T={self._temperature:.1f}°C | "
            f"W={self._workload:.1f} | "
            f"C={cool}"
        )
