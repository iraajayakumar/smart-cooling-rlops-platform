from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.simulator import Simulator
from rl_engine.inference import CoolingAgent
from rl_engine.reward import compute_reward, reward_breakdown

from experiments.utils.loader import load_config
from experiments.utils.writer import write_csv, write_json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "experiments" / "outputs"


def run_experiment(config: dict) -> list[dict]:
    steps = int(config["steps"])
    seed = int(config.get("seed", 42))
    workload_pattern = config.get("workload_pattern", "random")
    model_path = config.get("model_path", "rl_engine/models/best_model.zip")
    normalizer_path = config.get("normalizer_path", "rl_engine/models/vec_normalize.pkl")

    sim = Simulator()
    sim.workload_pattern = workload_pattern

    try:
        import random
        random.seed(seed)
    except Exception:
        pass

    agent = CoolingAgent(
        model_path=ROOT / model_path if not Path(model_path).is_absolute() else model_path,
        normalizer_path=ROOT / normalizer_path if not Path(normalizer_path).is_absolute() else normalizer_path,
    )

    rows: list[dict] = []
    for step in range(steps):
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
                "step": step,
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
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    rows = run_experiment(config)

    prefix = config.get("output_prefix", "experiment")
    csv_path = OUTPUT_DIR / f"{prefix}.csv"
    json_path = OUTPUT_DIR / f"{prefix}.json"

    write_csv(rows, csv_path)
    write_json(rows, json_path)

    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "rows": len(rows)}))


if __name__ == "__main__":
    main()