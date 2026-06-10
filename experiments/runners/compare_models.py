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


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def run_comparison(config: dict) -> list[dict]:
    steps = int(config["steps"])
    seed = int(config.get("seed", 42))
    workload_pattern = config.get("workload_pattern", "random")

    model_a_path = _resolve(config["model_a_path"])
    model_b_path = _resolve(config["model_b_path"])
    normalizer_path = _resolve(config.get("normalizer_path", "rl_engine/models/vec_normalize.pkl"))

    try:
        import random
        random.seed(seed)
    except Exception:
        pass

    agent_a = CoolingAgent(model_path=model_a_path, normalizer_path=normalizer_path)
    agent_b = CoolingAgent(model_path=model_b_path, normalizer_path=normalizer_path)

    def run_agent(agent: CoolingAgent, model_name: str) -> list[dict]:
        sim = Simulator()
        sim.workload_pattern = workload_pattern
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
                    "model": model_name,
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

    rows = []
    rows.extend(run_agent(agent_a, "model_a"))
    rows.extend(run_agent(agent_b, "model_b"))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    rows = run_comparison(config)

    prefix = config.get("output_prefix", "compare_models")
    csv_path = OUTPUT_DIR / f"{prefix}.csv"
    json_path = OUTPUT_DIR / f"{prefix}.json"

    write_csv(rows, csv_path)
    write_json(rows, json_path)

    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "rows": len(rows)}))


if __name__ == "__main__":
    main()