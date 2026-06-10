import importlib
from unittest import mock


def test_run_sweep_continuous_session():
    mod = importlib.import_module("experiments.runners.workload_sweep")

    fake_agent = mock.MagicMock()
    fake_agent.predict.return_value = 1

    fake_sim = mock.MagicMock()
    fake_sim.energy = 0.0
    fake_sim.get_state.side_effect = [{"temperature": 35.0}] * 30
    fake_sim.step.side_effect = [
        {"temperature": 34.9, "workload": 0.4, "cooling": 0.2} for _ in range(30)
    ]

    with mock.patch.object(mod, "CoolingAgent", return_value=fake_agent), \
         mock.patch.object(mod, "Simulator", return_value=fake_sim), \
         mock.patch.object(mod, "compute_reward", return_value=1.0), \
         mock.patch.object(mod, "reward_breakdown", return_value={
             "reward_total": 1.0,
             "reward_energy_penalty": 0.0,
             "reward_temp_penalty": 0.0,
             "reward_stability_bonus": 0.0,
         }):
        rows = mod.run_sweep(
            steps=30,
            seed=42,
            model_path="rl_engine/models/best_model.zip",
            normalizer_path="rl_engine/models/vec_normalize.pkl",
        )

    assert len(rows) == 30
    assert rows[0]["session_step"] == 0
    assert rows[-1]["session_step"] == 29