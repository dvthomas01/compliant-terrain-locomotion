"""
Train Policy A v8: reachability run (command-velocity curriculum + overspeed penalty).

Across v5–v7 the robot NEVER sustained balance: peak ep_len ever was 52 (v5), 40 (v6),
38 (v7), and every run decayed to ~10–14 (falls in ~0.25 s), always via forward pitch.
Velocity-reward tuning only swapped the degenerate behaviour (v5/v7 sprint, v6 freeze)
without producing a stable gait. The pre-flight proves a stable walk (+5) out-returns
lunge-and-die (−17), so this is an EXPLORATION failure: the good basin is unreachable
from random init. v8 makes it reachable, with two changes (everything else held from v7):

  1. COMMAND-VELOCITY CURRICULUM (success-gated). The commanded forward velocity starts at
     0.0 (learn to STAND/balance — statically stable, easy to reach) and advances toward
     0.2 by 0.02 ONLY when rolling ep_len clears 400 (robot stable at current command);
     retreats if ep_len drops below 150. Each speed increase is a small perturbation of an
     already-balanced policy instead of a from-scratch gait discovery. The command is part
     of the observation, so the policy learns a velocity-conditioned gait.
     DIAGNOSTIC: if it can't even hold ep_len≥400 at command 0, the blocker is fundamental
     (control gains / action scale), not the velocity reward — and v9 targets that instead.

  2. OVERSPEED PENALTY (base_env: _W_OVERSPEED=3.0, outside k). −3·max(0, vel_x−command)·dt.
     v7's clipped progress reward was flat above command and the σ²=0.04 Gaussian was dead
     far above it, so nothing decelerated the sprint. This adds a constant decel gradient at
     any speed above the live command.

Held from v7: penalty cap k_max=0.3, forward-progress W=4.0, tracking σ²=0.04, pitch penalty
W=25, pitch termination 0.40, termination penalty −20, PPO config, network, VecNormalize, flat terrain.

Usage:
    python training/train_policy_v8.py
    tensorboard --logdir runs/

Watch:
  train/command_vel    — should climb 0.0→0.2 as the robot proves stability at each step
  rollout/ep_len_mean  — should reach the horizon (1000) at low command, then HOLD as command rises
  locomotion/base_lin_vel_x vs train/command_vel — tracking, not sprinting
  rewards/r_overspeed  — bites only when overshooting; near 0 once tracking
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CallbackList

from environments.rigid_env import RigidTerrainEnv
from environments import base_env as B
from training.ppo_config import PPOConfig
from training.timeout_bootstrap import (
    PenaltyCurriculum,
    PenaltyCurriculumCallback,
    CommandCurriculum,
    CommandCurriculumCallback,
    AdaptiveLRCallback,
    RewardLoggingCallback,
    DistributionHistogramCallback,
    VideoRenderCallback,
    CheckpointAtStepsCallback,
)

_CMD_START   = 0.0    # start commanding zero velocity (learn to stand first)
_CMD_TARGET  = 0.2    # final forward command
_CMD_ANG_VEL = 0.0

V7_K_MAX = 0.3
V8_CHECKPOINTS = [1_000_000, 3_000_000, 6_000_000, 10_000_000, 15_000_000]


def _core_per_step_reward(vx: float, pitch: float, target: float = 0.2) -> float:
    dt   = 0.02
    stab = math.exp(-2.0 * pitch ** 2)
    lin_err_sq = (target - vx) ** 2
    r_lin    = math.exp(-lin_err_sq / B._SIGMA_SQ_VEL) * B._W_LIN_VEL * dt * stab
    r_alive  = B._W_ALIVE * dt
    excess   = max(0.0, abs(pitch) - B._PITCH_FREE_ZONE)
    r_pitch  = -B._W_PITCH * excess ** 2 * dt
    r_orient = -(math.sin(pitch) ** 2) * B._W_ORIENTATION * dt
    r_fwd    = B._W_FWD_PROGRESS * min(max(0.0, vx), target) * dt
    r_over   = -B._W_OVERSPEED * max(0.0, vx - target) * dt
    return r_lin + r_alive + r_pitch + r_orient + r_fwd + r_over


def preflight_termination_economics(gamma: float = 0.99) -> None:
    die_steps = 10
    sprint = sum(gamma ** t * _core_per_step_reward(0.58, 0.30) for t in range(die_steps))
    sprint += gamma ** die_steps * B._TERMINATION_PENALTY
    walk = _core_per_step_reward(0.20, 0.24) * (1.0 - gamma ** B._MAX_STEPS) / (1.0 - gamma)
    print("\n[pre-flight] discounted return (γ=%.2f):" % gamma)
    print(f"  (a) sprint 10 steps @0.58, pitch 0.30, then die = {sprint:+.3f}")
    print(f"  (b) walk full horizon @0.20, pitch 0.24          = {walk:+.3f}")
    if walk < 2.0 * sprint:
        raise SystemExit(f"[pre-flight] ABORT: walk ({walk:+.3f}) < 2× sprint ({sprint:+.3f}).")
    print(f"  PASS: walking dominates dying (margin {walk - sprint:+.3f}).\n")


def make_env(rank: int, seed: int = 0):
    def _init():
        env = RigidTerrainEnv(target_lin_vel=(_CMD_START, 0.0), target_ang_vel=_CMD_ANG_VEL)
        env.reset(seed=seed + rank)
        return env
    return _init


def build_vec_env(cfg: PPOConfig) -> VecNormalize:
    factories = [make_env(i) for i in range(cfg.n_envs)]
    vec_cls   = SubprocVecEnv if cfg.n_envs > 4 else DummyVecEnv
    vec_env   = vec_cls(factories)
    vec_env   = VecMonitor(vec_env)
    vec_env   = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg.gamma)
    return vec_env


class VecNormCheckpointCallback(CheckpointAtStepsCallback):
    def __init__(self, vec_env: VecNormalize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vec_env = vec_env

    def _on_step(self) -> bool:
        while (self._next_idx < len(self._steps)
               and self.num_timesteps >= self._steps[self._next_idx]):
            milestone    = self._steps[self._next_idx]
            label        = f"{milestone // 1_000_000}M"
            model_path   = os.path.join(self._save_dir, f"{self._prefix}_{label}")
            vecnorm_path = os.path.join(self._save_dir, f"vecnorm_{label}.pkl")
            self.model.save(model_path)
            self._vec_env.save(vecnorm_path)
            print(f"\n[checkpoint] {model_path}.zip + {vecnorm_path}  ({self.num_timesteps:,} steps)")
            self._next_idx += 1
        return True


def train(cfg: PPOConfig, run_name: str = "policy_a_v8",
          render_freq: int = 500_000, live_viewer: bool = False) -> None:

    preflight_termination_economics(gamma=cfg.gamma)

    checkpoint_dir = f"checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Launching {cfg.n_envs} envs...")
    vec_env = build_vec_env(cfg)

    model = PPO(
        policy="MlpPolicy", env=vec_env,
        n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
        gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef, learning_rate=cfg.learning_rate, max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128]), activation_fn=nn.ELU),
        tensorboard_log=f"{cfg.tensorboard_log}{run_name}", verbose=1,
    )

    penalty_curr = PenaltyCurriculum(k0=cfg.penalty_k0, decay=cfg.penalty_decay, k_max=V7_K_MAX)
    command_curr = CommandCurriculum(start=_CMD_START, target=_CMD_TARGET,
                                     step=0.02, advance=400.0, retreat=150.0)

    # Render env tracks the live command so GIFs show the policy at its current trained speed.
    def make_render_env():
        return RigidTerrainEnv(render_mode="rgb_array",
                               target_lin_vel=(command_curr.cmd, 0.0), target_ang_vel=_CMD_ANG_VEL)

    callback_list = [
        PenaltyCurriculumCallback(penalty_curr),
        CommandCurriculumCallback(command_curr),
        AdaptiveLRCallback(kl_target=cfg.kl_target, lr_min=cfg.lr_min, lr_max=cfg.lr_max),
        RewardLoggingCallback(),
        DistributionHistogramCallback(),
        VecNormCheckpointCallback(vec_env=vec_env, steps=V8_CHECKPOINTS,
                                  save_dir=checkpoint_dir, prefix="model"),
    ]
    if render_freq > 0:
        callback_list.append(VideoRenderCallback(
            eval_env_factory=make_render_env, render_freq=render_freq,
            n_episodes=3, save_dir=f"renders/{run_name}", live_viewer=live_viewer))

    print(f"\nPolicy A v8 — reachability (command curriculum + overspeed) — {cfg.total_timesteps:,} steps")
    print(f"  New in v8:    command-velocity curriculum {_CMD_START}→{_CMD_TARGET} (advance@ep_len 400)")
    print(f"  New in v8:    overspeed penalty _W_OVERSPEED={B._W_OVERSPEED}")
    print(f"  Held from v7: penalty cap k_max={V7_K_MAX}, fwd-progress W={B._W_FWD_PROGRESS}, σ²={B._SIGMA_SQ_VEL}")
    print(f"  Watch for:    train/command_vel climbing; ep_len reaching horizon\n")

    model.learn(total_timesteps=cfg.total_timesteps,
                callback=CallbackList(callback_list), progress_bar=True)

    final_model   = os.path.join(checkpoint_dir, "policy_v8_final")
    final_vecnorm = os.path.join(checkpoint_dir, "vecnorm_final.pkl")
    model.save(final_model)
    vec_env.save(final_vecnorm)
    print(f"\nTraining complete.\n  Model:   {final_model}.zip\n  VecNorm: {final_vecnorm}")
    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy A v8: reachability run")
    parser.add_argument("--n-envs",      type=int, default=128)
    parser.add_argument("--total-steps", type=int, default=15_000_000)
    parser.add_argument("--run-name",    type=str, default="policy_a_v8")
    parser.add_argument("--render-freq", type=int, default=500_000)
    parser.add_argument("--live-viewer", action="store_true")
    args = parser.parse_args()
    cfg = PPOConfig(n_envs=args.n_envs, total_timesteps=args.total_steps)
    train(cfg, run_name=args.run_name, render_freq=args.render_freq, live_viewer=args.live_viewer)
