"""
Train Policy A v3: flat-ground walking debug run.

Goal: prove the policy can discover forward walking on flat rigid ground
before adding any terrain complexity.

Key differences from v2:
  - Fixed velocity command: 0.3 m/s forward only (no random targets)
  - Base height penalty: penalises the low-crouch local minimum
  - Feet air time weight raised 2.0 → 5.0: stronger gait-cycle incentive
  - Linear velocity tracking weight raised 1.0 → 1.5
  - Alive bonus: constant per-step reward for surviving
  - Per-term reward logging to TensorBoard (rewards/ and locomotion/ namespaces)
  - Checkpoint milestones at 1M, 3M, 6M, 10M, 15M steps

Usage:
    python training/train_policy_v3.py

    # Quick smoke-test (~1 minute)
    python training/train_policy_v3.py --n-envs 8 --total-steps 6144

    # Monitor live
    tensorboard --logdir runs/

Success criteria (abort at 6M if none of these appear):
  3M  — legs start cycling (not identical crouch across all GIF checkpoints)
  6M  — robot moves forward at least briefly
  10M — ep_len_mean clearly > 25 steps (the v2 ceiling)
  15M — recognisable forward walk or trot on flat ground
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
    RewardLoggingCallback,
    VideoRenderCallback,
    CheckpointAtStepsCallback,
)

# Fixed slow forward walk — consistent target lets the policy learn one thing at a time
_CMD_LIN_VEL = (0.3, 0.0)   # (forward m/s, lateral m/s)
_CMD_ANG_VEL = 0.0           # yaw rad/s

V3_CHECKPOINTS = [1_000_000, 3_000_000, 6_000_000, 10_000_000, 15_000_000]


def make_env(rank: int, seed: int = 0):
    def _init():
        env = RigidTerrainEnv(
            target_lin_vel=_CMD_LIN_VEL,
            target_ang_vel=_CMD_ANG_VEL,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def make_render_env():
    return RigidTerrainEnv(
        render_mode="rgb_array",
        target_lin_vel=_CMD_LIN_VEL,
        target_ang_vel=_CMD_ANG_VEL,
    )


def build_vec_env(cfg: PPOConfig) -> VecNormalize:
    factories = [make_env(i) for i in range(cfg.n_envs)]
    vec_cls   = SubprocVecEnv if cfg.n_envs > 4 else DummyVecEnv
    vec_env   = vec_cls(factories)
    vec_env   = VecMonitor(vec_env)
    vec_env   = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=cfg.gamma,
    )
    return vec_env


class VecNormCheckpointCallback(CheckpointAtStepsCallback):
    """Saves model weights AND VecNormalize statistics at each milestone."""

    def __init__(self, vec_env: VecNormalize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vec_env = vec_env

    def _on_step(self) -> bool:
        while (
            self._next_idx < len(self._steps)
            and self.num_timesteps >= self._steps[self._next_idx]
        ):
            milestone    = self._steps[self._next_idx]
            label        = f"{milestone // 1_000_000}M"
            model_path   = os.path.join(self._save_dir, f"{self._prefix}_{label}")
            vecnorm_path = os.path.join(self._save_dir, f"vecnorm_{label}.pkl")
            self.model.save(model_path)
            self._vec_env.save(vecnorm_path)
            print(f"\n[checkpoint] {model_path}.zip + {vecnorm_path}  ({self.num_timesteps:,} steps)")
            self._next_idx += 1
        return True


def train(cfg: PPOConfig, run_name: str = "policy_a_v3",
          render_freq: int = 500_000, live_viewer: bool = False) -> None:

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
        RewardLoggingCallback(),
        VecNormCheckpointCallback(
            vec_env=vec_env,
            steps=V3_CHECKPOINTS,
            save_dir=checkpoint_dir,
            prefix="model",
        ),
    ]

    if render_freq > 0:
        callback_list.append(
            VideoRenderCallback(
                eval_env_factory=make_render_env,
                render_freq=render_freq,
                n_episodes=3,
                save_dir=f"renders/{run_name}",
                live_viewer=live_viewer,
            )
        )

    print(f"\nPolicy A v3 — flat-ground walk debug run — {cfg.total_timesteps:,} steps")
    print(f"  Command:      {_CMD_LIN_VEL[0]} m/s forward only (fixed)")
    print(f"  Reward mods:  lin_vel_w=1.5, feet_air_w=5.0, base_height_w=2.0, alive_w=0.5")
    print(f"  Checkpoints:  {[f'{s//1_000_000}M' for s in V3_CHECKPOINTS]}")
    print(f"  TensorBoard:  tensorboard --logdir {cfg.tensorboard_log}\n")

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList(callback_list),
        progress_bar=True,
    )

    final_model   = os.path.join(checkpoint_dir, "policy_v3_final")
    final_vecnorm = os.path.join(checkpoint_dir, "vecnorm_final.pkl")
    model.save(final_model)
    vec_env.save(final_vecnorm)
    print(f"\nTraining complete.")
    print(f"  Model:   {final_model}.zip")
    print(f"  VecNorm: {final_vecnorm}")

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy A v3: flat-ground walking debug")
    parser.add_argument("--n-envs",      type=int,  default=128)
    parser.add_argument("--total-steps", type=int,  default=15_000_000)
    parser.add_argument("--run-name",    type=str,  default="policy_a_v3")
    parser.add_argument(
        "--render-freq", type=int, default=500_000,
        help="Save GIF snapshot every N steps (0 to disable)."
    )
    parser.add_argument(
        "--live-viewer", action="store_true",
        help="Open MuJoCo interactive window at each render checkpoint (~5s pause)."
    )
    args = parser.parse_args()

    cfg = PPOConfig(n_envs=args.n_envs, total_timesteps=args.total_steps)
    train(cfg, run_name=args.run_name,
          render_freq=args.render_freq, live_viewer=args.live_viewer)
