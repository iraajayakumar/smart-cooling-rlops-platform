"""
train.py
PPO training pipeline for the data center cooling RL agent.

Usage
─────
    python train.py                        # default config
    python train.py --timesteps 500000     # longer run
    python train.py --pattern sine         # sine workload
    python train.py --config config.yaml   # load YAML config

TensorBoard
───────────
    tensorboard --logdir logs/
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from environment import DataCenterEnv
from reward import reward_breakdown

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
MODELS   = ROOT / "models"
LOGS     = ROOT / "logs"
MODELS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

# ── Default hyper-parameters ──────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "timesteps":     300_000,
    "n_envs":        4,
    "workload":      "random",
    "learning_rate": 3e-4,
    "n_steps":       2048,
    "batch_size":    64,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_range":    0.2,
    "ent_coef":      0.01,
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
    "net_arch":      [128, 128],
    "reward_threshold": 150.0,   # stop early if eval reward exceeds this
    "eval_freq":     10_000,
    "seed":          42,
}


# ─────────────────────────────────────────────────────────────────────────────
# Custom callback: logs reward breakdown components to TensorBoard
# ─────────────────────────────────────────────────────────────────────────────
class RewardComponentCallback(BaseCallback):
    """Log individual reward components (energy, temp penalty, stability)."""

    def __init__(self, eval_env: DataCenterEnv, eval_every: int = 5_000):
        super().__init__(verbose=0)
        self.eval_env   = eval_env
        self.eval_every = eval_every

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_every != 0:
            return True

        obs, _ = self.eval_env.reset()
        components_accum: dict[str, list[float]] = {
            "reward/total": [],
            "reward/energy_penalty": [],
            "reward/temp_penalty": [],
            "reward/stability_bonus": [],
        }

        for _ in range(200):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = self.eval_env.step(int(action))
            energy = info["cooling"] * 2.0
            bd = reward_breakdown(
                temperature=info["temperature"],
                energy=energy,
                cooling=info["cooling"],
            )
            for k, v in bd.items():
                components_accum[k].append(v)
            if terminated or truncated:
                obs, _ = self.eval_env.reset()

        for k, vals in components_accum.items():
            self.logger.record(k, float(np.mean(vals)))

        return True


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    print("\n╔══════════════════════════════════════════════╗")
    print("║  Data Center RL — PPO Training               ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Timesteps   : {cfg['timesteps']:,}")
    print(f"  Workload    : {cfg['workload']}")
    print(f"  Envs        : {cfg['n_envs']}")
    print(f"  Seed        : {cfg['seed']}\n")

    # ── Training environments (vectorised + normalised) ───────────────────
    def make_env():
        env = DataCenterEnv(workload_pattern=cfg["workload"], seed=cfg["seed"])
        return Monitor(env)

    train_env = make_vec_env(make_env, n_envs=cfg["n_envs"], seed=cfg["seed"])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # ── Evaluation environment (wrapped to match training env) ────────────
    eval_env = DataCenterEnv(workload_pattern=cfg["workload"], seed=cfg["seed"] + 1)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: DataCenterEnv(workload_pattern=cfg["workload"], seed=cfg["seed"] + 1)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # ── Callbacks ─────────────────────────────────────────────────────────
    stop_cb = StopTrainingOnRewardThreshold(
        reward_threshold=cfg["reward_threshold"], verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS),
        log_path=str(LOGS),
        eval_freq=max(cfg["eval_freq"] // cfg["n_envs"], 1),
        deterministic=True,
        render=False,
        callback_on_new_best=stop_cb,
    )
    component_cb = RewardComponentCallback(
        eval_env=DataCenterEnv(workload_pattern=cfg["workload"], seed=42),
        eval_every=5_000,
    )

    # ── PPO Model ─────────────────────────────────────────────────────────
    policy_kwargs = dict(net_arch=cfg["net_arch"])

    model = PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        learning_rate = cfg["learning_rate"],
        n_steps       = cfg["n_steps"],
        batch_size    = cfg["batch_size"],
        n_epochs      = cfg["n_epochs"],
        gamma         = cfg["gamma"],
        gae_lambda    = cfg["gae_lambda"],
        clip_range    = cfg["clip_range"],
        ent_coef      = cfg["ent_coef"],
        vf_coef       = cfg["vf_coef"],
        max_grad_norm = cfg["max_grad_norm"],
        policy_kwargs = policy_kwargs,
        tensorboard_log= str(LOGS),
        verbose       = 1,
        seed          = cfg["seed"],
    )

    # ── Train ─────────────────────────────────────────────────────────────
    start = time.time()
    model.learn(
        total_timesteps   = cfg["timesteps"],
        callback          = [eval_cb, component_cb],
        tb_log_name       = "PPO_DataCenter",
        reset_num_timesteps= True,
    )
    elapsed = time.time() - start
    print(f"\n  Training finished in {elapsed/60:.1f} min")

    # ── Save artefacts ────────────────────────────────────────────────────
    model_path     = MODELS / "model.zip"
    normalizer_path= MODELS / "vec_normalize.pkl"

    model.save(str(model_path))
    train_env.save(str(normalizer_path))

    print(f"  Model saved     → {model_path}")
    print(f"  Normalizer saved→ {normalizer_path}")

    # ── Quick evaluation ──────────────────────────────────────────────────
    _quick_eval(model, train_env, cfg)

    return model, train_env


def _quick_eval(model, vec_env, cfg, n_episodes: int = 5):
    """Print a quick summary of trained-model performance."""
    print("\n── Quick evaluation ───────────────────────────────")
    eval_env = Monitor(
        DataCenterEnv(workload_pattern=cfg["workload"], seed=99)
    )

    ep_rewards, ep_temps = [], []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_r, temps = 0.0, []
        while not done:
            obs_norm = vec_env.normalize_obs(np.array([obs]))[0]
            action, _ = model.predict(obs_norm, deterministic=True)
            obs, r, terminated, truncated, info = eval_env.step(int(action))
            done = terminated or truncated
            ep_r += r
            temps.append(info["temperature"])
        ep_rewards.append(ep_r)
        ep_temps.append(np.mean(temps))

    print(f"  Mean episode reward : {np.mean(ep_rewards):.2f}")
    print(f"  Mean temperature    : {np.mean(ep_temps):.1f} °C")
    print(f"  Max temperature     : {max(np.max(t) for t in [ep_temps]):.1f} °C")
    print("──────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path and Path(path).exists():
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        cfg.update(overrides)
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO agent for data center cooling")
    p.add_argument("--config",     type=str,   default=None,       help="Path to config.yaml")
    p.add_argument("--timesteps",  type=int,   default=None)
    p.add_argument("--pattern",    type=str,   default=None,       choices=["random", "sine", "spike"])
    p.add_argument("--seed",       type=int,   default=None)
    p.add_argument("--n-envs",     type=int,   default=None,       dest="n_envs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI overrides
    if args.timesteps: cfg["timesteps"]   = args.timesteps
    if args.pattern:   cfg["workload"]    = args.pattern
    if args.seed:      cfg["seed"]        = args.seed
    if args.n_envs:    cfg["n_envs"]      = args.n_envs

    train(cfg)
