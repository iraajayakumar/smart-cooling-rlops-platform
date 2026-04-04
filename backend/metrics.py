class Metrics:
    def __init__(self):
        self.history = []

    def log(self, state, energy):
        self.history.append({
            "temperature": state["temperature"],
            "cooling": state["cooling"],
            "energy": energy
        })

    def get_metrics(self):
        return self.history
