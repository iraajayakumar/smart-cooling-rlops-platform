import time


class Metrics:
    def __init__(self, model_version="unknown", episode_id="default"):
        self.history = []
        self.step_index = 0
        self.model_version = model_version
        self.episode_id = episode_id

    def log(self, state, energy, action=None, reward=None, done=False, temperature_before=None, reward_breakdown=None):
        temperature = state["temperature"]
        temperature_delta = None
        if temperature_before is not None:
            temperature_delta = temperature - temperature_before

        record = {
            "timestamp": time.time(),
            "step": self.step_index,
            "episode_id": self.episode_id,
            "model_version": self.model_version,
            "temperature": temperature,
            "workload": state["workload"],
            "cooling": state["cooling"],
            "action": action,
            "energy": energy,
            "reward": reward,
            "done": done,
            "temperature_delta": temperature_delta,
        }

        if reward_breakdown:
            record.update(reward_breakdown)

        self.history.append(record)
        self.step_index += 1

    def get_metrics(self):
        return self.history