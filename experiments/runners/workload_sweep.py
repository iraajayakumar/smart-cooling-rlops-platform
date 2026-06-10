from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from backend.simulator import Simulator
from rl_engine.inference import CoolingAgent
from rl_engine.reward import compute_reward, reward_breakdown

from experiments.utils.loader import resolve_path
from experiments.utils.writer import write_csv, write_json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "experiments" / "outputs"


PATTERNS = ["random", "sine", "spike"]


def _choose_pattern(rng: random.Random, current: str | None = None) -> str:
    weights = {
        "random": 0.50,
        "sine": 0.30,
        "spike": 0.20,
    }
    choices = [p for p in PATTERNS if p != current]
    w = [weights[p] for p in choices]
    return rng.choices(choices, weights=w, k=1)[0]


def _set_workload_pattern(sim: Simulator, pattern: str) -> None:
    sim.workload_pattern = pattern
    if hasattr(sim, "_sine_phase") and pattern != "sine":
        sim._sine_phase = 0.0


def run_sweep(
    steps: int,
    seed: int,
    model_path: str | Path = "rl_engine/models/best_model.zip",
    normalizer_path: str | Path = "rl_engine/models/vec_normalize.pkl",
) -> list[dict]:
    rng = random.Random(seed)
    agent = CoolingAgent(
        model_path=resolve_path(model_path),
        normalizer_path=resolve_path(normalizer_path),
    )

    sim = Simulator()
    rows: list[dict] = []

    current_pattern = rng.choice(PATTERNS)
    _set_workload_pattern(sim, current_pattern)

    switch_interval = rng.randint(8, 20)
    next_switch_at = switch_interval

    for step in range(steps):
        if step == next_switch_at:
            current_pattern = _choose_pattern(rng, current=current_pattern)
            _set_workload_pattern(sim, current_pattern)
            switch_interval = rng.randint(8, 20)
            next_switch_at = step + switch_interval

        state = sim.get_state()
        temperature_before = state["temperature"]
        action = agent.predict(state)
        new_state = sim.step(action)

        reward = compute_reward(
            temperature=new_state["temperature"],
            energy=sim.energy,
            cooling=new_state["cooling"],
        )
        breakdown = reward_breakdown(
            temperature=new_state["temperature"],
            energy=sim.energy,
            cooling=new_state["cooling"],
        )

        rows.append(
            {
                "session_step": step,
                "pattern": current_pattern,
                "temperature_before": temperature_before,
                "temperature": new_state["temperature"],
                "workload": new_state["workload"],
                "cooling": new_state["cooling"],
                "action": action,
                "energy": sim.energy,
                "reward": reward,
                **breakdown,
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="rl_engine/models/best_model.zip")
    parser.add_argument("--normalizer-path", default="rl_engine/models/vec_normalize.pkl")
    args = parser.parse_args()

    rows = run_sweep(
        steps=args.steps,
        seed=args.seed,
        model_path=args.model_path,
        normalizer_path=args.normalizer_path,
    )

    csv_path = OUTPUT_DIR / "workload_sweep.csv"
    json_path = OUTPUT_DIR / "workload_sweep.json"

    write_csv(rows, csv_path)
    write_json(rows, json_path)

    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "rows": len(rows)}))


if __name__ == "__main__":
    main()