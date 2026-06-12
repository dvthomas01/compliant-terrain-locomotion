"""
Train Policy A: rigid terrain baseline.

Runs PPO with 128 parallel environments on flat rigid ground.
Policy A has no terrain curriculum — it specialises for rigid terrain only,
giving it the highest possible rigid-ground performance for comparison.

Usage:
    # Full training (~3-4 hours on MacBook)
    python training/train_policy_a.py

    # Watch GIF snapshots saved to renders/ every 500k steps
    python training/train_policy_a.py --render-freq 500000

    # Pop up the live MuJoCo viewer every 500k steps (pauses training ~5s each time)
    python training/train_policy_a.py --live-viewer

    # Quick smoke-test (~2 minutes)
    python training/train_policy_a.py --n-envs 8 --total-steps 6144

    # Monitor live in browser (separate terminal)
    tensorboard --logdir runs/

Outputs:
    checkpoints/policy_a_v2/model_5M.zip   (+ vecnorm_5M.pkl)
    checkpoints/policy_a_v2/model_10M.zip  (+ vecnorm_10M.pkl)
    checkpoints/policy_a_v2/model_15M.zip  (+ vecnorm_15M.pkl)
    checkpoints/policy_a_v2/policy_a_final.zip  (+ vecnorm_final.pkl)
    renders/policy_0.5M.gif, policy_1M.gif, ...
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CallbackList

from environments.rigid_env import RigidTerrainEnv
from training.ppo_config import PPOConfig
from training.timeout_bootstrap import (
    PenaltyCurriculum,
    PenaltyCurriculumCallback,
    AdaptiveLRCallback,
    VideoRenderCallback,
    CheckpointAtStepsCallback,
)


def make_env(rank: int, seed: int = 0):
    def _init():
        env = RigidTerrainEnv()
        env.reset(seed=seed + rank)
        return env
    return _init


def make_render_env():
    """Single env with rgb_array rendering for the VideoRenderCallback."""
    return RigidTerrainEnv(render_mode="rgb_array")


def build_vec_env(cfg: PPOConfig) -> VecNormalize:
    """
    Build the normalised vectorised environment stack:
        SubprocVecEnv → VecMonitor → VecNormalize

    VecNormalize tracks a running mean/std for every observation dimension
    and for the reward, re-scaling them to a consistent range the network
    can learn from regardless of the raw physics units.
    """
    factories = [make_env(i) for i in range(cfg.n_envs)]
    vec_cls   = SubprocVecEnv if cfg.n_envs > 4 else DummyVecEnv
    vec_env   = vec_cls(factories)
    vec_env   = VecMonitor(vec_env)
    vec_env   = VecNormalize(
        vec_env,
        norm_obs=True,       # normalise each of the 47 obs dims
        norm_reward=True,    # normalise reward to zero mean, unit variance
        clip_obs=10.0,       # clip normalised obs to ±10 (avoids outlier blow-up)
        gamma=cfg.gamma,
    )
    return vec_env


class VecNormCheckpointCallback(CheckpointAtStepsCallback):
    """Checkpoint both the model weights AND the VecNormalize statistics."""

    def __init__(self, vec_env: VecNormalize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vec_env = vec_env

    def _on_step(self) -> bool:
        while (
            self._next_idx < len(self._steps)
            and self.num_timesteps >= self._steps[self._next_idx]
        ):
            milestone = self._steps[self._next_idx]
            label     = f"{milestone // 1_000_000}M"
            model_path  = os.path.join(self._save_dir, f"{self._prefix}_{label}")
            vecnorm_path = os.path.join(self._save_dir, f"vecnorm_{label}.pkl")
            self.model.save(model_path)
            self._vec_env.save(vecnorm_path)
            print(f"\n[checkpoint] {model_path}.zip + {vecnorm_path}  ({self.num_timesteps:,} steps)")
            self._next_idx += 1
        return True


def train(cfg: PPOConfig, run_name: str = "policy_a_v2",
          render_freq: int = 0, live_viewer: bool = False) -> None:
    checkpoint_dir = f"checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Launching {cfg.n_envs} envs ({'SubprocVecEnv' if cfg.n_envs > 4 else 'DummyVecEnv'})...")
    vec_env = build_vec_env(cfg)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        learning_rate=cfg.learning_rate,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=nn.ELU,
        ),
        tensorboard_log=f"{cfg.tensorboard_log}{run_name}",
        verbose=1,
    )

    curriculum = PenaltyCurriculum(k0=cfg.penalty_k0, decay=cfg.penalty_decay)

    callback_list = [
        PenaltyCurriculumCallback(curriculum),
        AdaptiveLRCallback(kl_target=cfg.kl_target, lr_min=cfg.lr_min, lr_max=cfg.lr_max),
        VecNormCheckpointCallback(
            vec_env=vec_env,
            steps=cfg.checkpoint_steps,
            save_dir=checkpoint_dir,
            prefix="model",
        ),
    ]

    if render_freq > 0:
        callback_list.append(
            VideoRenderCallback(
                eval_env_factory=make_render_env,
                render_freq=render_freq,
                n_steps=250,
                save_dir="renders",
                live_viewer=live_viewer,
            )
        )

    print(f"\nPolicy A v2 training — {cfg.total_timesteps:,} total steps")
    print(f"  Batch:        {cfg.n_envs} envs × {cfg.n_steps} steps = {cfg.batch_size} samples/update")
    print(f"  Network:      MLP [256, 128] ELU — separate actor/critic")
    print(f"  VecNormalize: ON (norm_obs + norm_reward)")
    print(f"  Penalty k₀:   {cfg.penalty_k0} → 1.0 at ~10.8M steps (decay={cfg.penalty_decay})")
    print(f"  Render freq:  {'every ' + str(render_freq // 1000) + 'k steps → renders/' if render_freq > 0 else 'OFF (use --render-freq N to enable)'}")
    print(f"  TensorBoard:  tensorboard --logdir {cfg.tensorboard_log}\n")

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList(callback_list),
        progress_bar=True,
    )

    # Save final weights + normalisation stats
    final_model  = os.path.join(checkpoint_dir, "policy_a_final")
    final_vecnorm = os.path.join(checkpoint_dir, "vecnorm_final.pkl")
    model.save(final_model)
    vec_env.save(final_vecnorm)
    print(f"\nTraining complete.")
    print(f"  Model:    {final_model}.zip")
    print(f"  VecNorm:  {final_vecnorm}")

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Policy A v2 (rigid terrain baseline)")
    parser.add_argument("--n-envs",      type=int,  default=128)
    parser.add_argument("--total-steps", type=int,  default=15_000_000)
    parser.add_argument("--run-name",    type=str,  default="policy_a_v2")
    parser.add_argument(
        "--render-freq", type=int, default=500_000,
        help="Save a GIF snapshot every N steps (0 to disable). Default 500000."
    )
    parser.add_argument(
        "--live-viewer", action="store_true",
        help="Open the MuJoCo interactive window at each render checkpoint (~5s pause)."
    )
    args = parser.parse_args()

    cfg = PPOConfig(n_envs=args.n_envs, total_timesteps=args.total_steps)
    train(cfg, run_name=args.run_name,
          render_freq=args.render_freq, live_viewer=args.live_viewer)
