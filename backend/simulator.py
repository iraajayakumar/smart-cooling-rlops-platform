import random
from physics import update_temperature, update_energy


class Simulator:
    def __init__(self):
        # Initial system state
        self.temperature = 25.0   # in °C
        self.workload = 0.5       # range: 0 to 1
        self.cooling = 1          # discrete levels: 0, 1, 2
        self.energy = 0.0         # total energy consumed

    def get_state(self):
        return {
            "temperature": self.temperature,
            "workload": self.workload,
            "cooling": self.cooling
        }

    def step(self, action):
        """
        Action:
        0 -> Low cooling
        1 -> Medium cooling
        2 -> High cooling
        """

        # Apply action
        self.cooling = action

        # 🔄 Simulate changing workload (realistic behavior)
        self.workload = random.uniform(0.3, 0.9)

        # ⚙️ Apply physics model
        self.temperature = update_temperature(
            self.temperature,
            self.workload,
            self.cooling
        )

        self.energy = update_energy(
            self.energy,
            self.cooling
        )

        # Return updated state
        return self.get_state()

    def get_energy(self):
        return self.energy
