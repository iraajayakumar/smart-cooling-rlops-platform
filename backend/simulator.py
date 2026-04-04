class DataCenterSimulator:
    def __init__(self):
        self.temperature = 30.0   # starting temperature
        self.workload = 0.5       # 0 to 1
        self.cooling = 1          # 0,1,2
        self.energy = 0

    def step(self, cooling_action):
        self.cooling = cooling_action

        # Physics model
        self.temperature += self.workload * 0.4
        self.temperature -= self.cooling * 0.6

        self.energy += self.cooling * 2

        return self.get_state()

    def get_state(self):
        return {
            "temperature": self.temperature,
            "workload": self.workload,
            "cooling": self.cooling
        }
