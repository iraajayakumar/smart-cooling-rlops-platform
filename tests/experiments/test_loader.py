import json
from pathlib import Path

from experiments.utils.loader import load_config, resolve_path


def test_load_config(tmp_path):
    cfg = {
        "name": "test",
        "steps": 3,
        "seed": 1,
        "workload_pattern": "random",
        "model_path": "rl_engine/models/best_model.zip",
        "normalizer_path": "rl_engine/models/vec_normalize.pkl",
        "output_prefix": "test_run",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    loaded = load_config(path)
    assert loaded["name"] == "test"
    assert loaded["steps"] == 3
    assert loaded["workload_pattern"] == "random"


def test_resolve_path_relative():
    p = resolve_path("rl_engine/models/best_model.zip")
    # Use as_posix() to normalize path separators for cross-platform compatibility
    assert p.as_posix().endswith("rl_engine/models/best_model.zip")