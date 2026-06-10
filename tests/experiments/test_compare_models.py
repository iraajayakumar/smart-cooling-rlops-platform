import importlib
from unittest import mock


def test_run_comparison_schema():
    mod = importlib.import_module("experiments.runners.compare_models")

    fake_agent_a = mock.MagicMock()
    fake_agent_a.predict.return_value = 1
    fake_agent_b = mock.MagicMock()
    fake_agent_b.predict.return_value = 0

    fake_sim_a = mock.MagicMock()
    fake_sim_a.get_state.side_effect = [
        {"temperature": 35.0},
        {"temperature": 34.8},
    ]
    fake_sim_a.step.side_effect = [
        {"temperature": 34.8, "workload": 0.4, "cooling": 0.2},
        {"temperature": 34.5, "workload": 0.5, "cooling": 0.3},
    ]
    fake_sim_a.energy = 0.0

    fake_sim_b = mock.MagicMock()
    fake_sim_b.get_state.side_effect = [
        {"temperature": 35.0},
        {"temperature": 34.7},
    ]
    fake_sim_b.step.side_effect = [
        {"temperature": 34.7, "workload": 0.4, "cooling": 0.1},
        {"temperature": 34.4, "workload": 0.5, "cooling": 0.2},
    ]
    fake_sim_b.energy = 0.0

    def sim_factory():
        return fake_sim_a if not hasattr(sim_factory, "used") else fake_sim_b
    sim_factory.used = False

    def sim_side_effect():
        if not sim_factory.used:
            sim_factory.used = True
            return fake_sim_a
        return fake_sim_b

    with mock.patch.object(mod, "Simulator", side_effect=sim_side_effect), \
         mock.patch.object(mod, "CoolingAgent", side_effect=[fake_agent_a, fake_agent_b]), \
         mock.patch.object(mod, "compute_reward", return_value=1.0), \
         mock.patch.object(mod, "reward_breakdown", return_value={
             "reward_total": 1.0,
             "reward_energy_penalty": 0.0,
             "reward_temp_penalty": 0.0,
             "reward_stability_bonus": 0.0,
         }):
        rows = mod.run_comparison(
            {
                "steps": 2,
                "seed": 42,
                "workload_pattern": "random",
                "model_a_path": "rl_engine/models/model.zip",
                "model_b_path": "rl_engine/models/best_model.zip",
                "normalizer_path": "rl_engine/models/vec_normalize.pkl",
            }
        )

    assert len(rows) == 4
    assert {r["model"] for r in rows} == {"model_a", "model_b"}