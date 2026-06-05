import random
from typing import Dict

from backend.physics import update_temperature, update_energy


TEMP_MIN = 15.0
TEMP_MAX = 90.0
TEMP_IDEAL = 35.0  # for reference; reward uses this in rl-engine
WORKLOAD_MIN = 0.0
WORKLOAD_MAX = 10.0


class Simulator:
    """
    Simple data center simulator aligned with rl-engine/DataCenterEnv.

    State:
        temperature : float in [TEMP_MIN, TEMP_MAX]
        workload    : float in [0, 10]
        cooling     : int in {0, 1, 2}
        energy      : cumulative energy usage
    """

    def __init__(self):
        # Initial system state (roughly similar to rl-engine reset())
        self.temperature: float = random.uniform(TEMP_IDEAL - 5, TEMP_IDEAL + 5)
        self.workload: float = random.uniform(1.0, 4.0)
        self.cooling: int = 1  # start at medium cooling
        self.energy: float = 0.0

        self.step_count: int = 0

    def get_state(self) -> Dict[str, float]:
        return {
            "temperature": self.temperature,
            "workload": self.workload,
            "cooling": self.cooling,
        }

    def _next_workload(self) -> float:
        """
        Simple random-walk workload in [0, 10].

        This is a simplified version of rl-engine's _next_workload().
        """
        delta = random.gauss(0.0, 0.8)
        new_workload = self.workload + delta
        return max(WORKLOAD_MIN, min(WORKLOAD_MAX, new_workload))

    def step(self, action: int) -> Dict[str, float]:
        """
        Apply action and advance the simulator by one step.

        Action:
            0 -> Low cooling
            1 -> Medium cooling
            2 -> High cooling
        """
        # Apply action
        self.cooling = int(action)

        # Update workload
        self.workload = self._next_workload()

        # Apply physics model
        self.temperature = update_temperature(
            self.temperature,
            self.workload,
            self.cooling,
        )

        # Clamp temperature to safety range
        self.temperature = max(TEMP_MIN, min(TEMP_MAX, self.temperature))

        # Update energy
        self.energy = update_energy(self.energy, self.cooling)

        # Increment step counter
        self.step_count += 1

        return self.get_state()

    def get_energy(self) -> float:
        return self.energy