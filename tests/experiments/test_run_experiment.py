import importlib
from unittest import mock


def test_run_experiment_output_schema():
    mod = importlib.import_module("experiments.runners.run_experiment")

    fake_agent = mock.MagicMock()
    fake_agent.predict.return_value = 1

    fake_sim = mock.MagicMock()
    fake_sim.get_state.side_effect = [
        {"temperature": 35.0},
        {"temperature": 34.5},
        {"temperature": 34.0},
    ]
    fake_sim.step.side_effect = [
        {"temperature": 34.8, "workload": 0.4, "cooling": 0.2},
        {"temperature": 34.3, "workload": 0.5, "cooling": 0.3},
        {"temperature": 34.1, "workload": 0.6, "cooling": 0.3},
    ]
    fake_sim.energy = 0.0

    with mock.patch.object(mod, "Simulator", return_value=fake_sim), \
         mock.patch.object(mod, "CoolingAgent", return_value=fake_agent), \
         mock.patch.object(mod, "compute_reward", return_value=1.0), \
         mock.patch.object(mod, "reward_breakdown", return_value={
             "reward_total": 1.0,
             "reward_energy_penalty": 0.0,
             "reward_temp_penalty": 0.0,
             "reward_stability_bonus": 0.0,
         }):
        rows = mod.run_experiment(
            {
                "steps": 3,
                "seed": 42,
                "workload_pattern": "random",
                "model_path": "rl_engine/models/best_model.zip",
                "normalizer_path": "rl_engine/models/vec_normalize.pkl",
            }
        )

    assert len(rows) == 3
    assert rows[0]["step"] == 0
    assert "reward" in rows[0]