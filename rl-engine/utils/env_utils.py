"""
utils/env_utils.py
Environment registration and factory helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.registration import register

# Make sure parent directory is on path when imported from subdirectory
_RL_ROOT = Path(__file__).parent.parent
if str(_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RL_ROOT))


def register_env(force: bool = False) -> str:
    """
    Register DataCenterEnv with Gymnasium's registry.

    Returns the env_id string.
    """
    env_id = "DataCenter-v1"

    if force and env_id in gym.envs.registration.registry:
        del gym.envs.registration.registry[env_id]

    if env_id not in gym.envs.registration.registry:
        register(
            id=env_id,
            entry_point="environment:DataCenterEnv",
            max_episode_steps=200,
        )

    return env_id


def make_single_env(
    workload_pattern: str = "random",
    render_mode: str | None = None,
    seed: int = 0,
) -> gym.Env:
    """
    Create a single (unwrapped) DataCenterEnv instance.
    Useful for evaluation and visualisation.
    """
    from environment import DataCenterEnv
    from stable_baselines3.common.monitor import Monitor

    env = DataCenterEnv(
        render_mode=render_mode,
        workload_pattern=workload_pattern,
        seed=seed,
    )
    return Monitor(env)
