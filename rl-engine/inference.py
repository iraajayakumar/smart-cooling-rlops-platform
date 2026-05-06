from __future__ import annotations

import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


class CoolingState(TypedDict):
    temperature: float
    workload: float
    cooling: int


class CoolingAgent:
    ACTION_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

    def __init__(self):
        self._demo_mode = True
        logger.warning("Running in DEMO MODE with deterministic heuristic policy.")

    def predict(self, state: CoolingState | dict) -> int:
        temp = float(state.get("temperature", 25.0))
        load = float(state.get("workload", 0.5))
        cooling = int(state.get("cooling", 1))

        if temp >= 32 or load >= 8:
            return 2
        if temp >= 27 or load >= 5:
            return 1
        if cooling == 2 and temp < 24:
            return 1
        return 0

    def predict_with_confidence(self, state: CoolingState | dict) -> dict:
        action = self.predict(state)

        if action == 2:
            probs = {"LOW": 0.05, "MEDIUM": 0.15, "HIGH": 0.80}
        elif action == 1:
            probs = {"LOW": 0.20, "MEDIUM": 0.65, "HIGH": 0.15}
        else:
            probs = {"LOW": 0.80, "MEDIUM": 0.15, "HIGH": 0.05}

        return {
            "action": action,
            "action_label": self.ACTION_LABELS[action],
            "probabilities": probs,
            "mode": "demo",
        }


_agent: CoolingAgent | None = None


def predict_action(state: dict) -> int:
    global _agent
    if _agent is None:
        _agent = CoolingAgent()
    return _agent.predict(state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = CoolingAgent()
    samples = [
        {"temperature": 23.0, "workload": 2.0, "cooling": 0},
        {"temperature": 28.0, "workload": 5.5, "cooling": 1},
        {"temperature": 34.0, "workload": 8.5, "cooling": 1},
    ]
    for s in samples:
        print(s, "->", agent.predict_with_confidence(s))