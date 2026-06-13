"""
Train Policy A v19: ROBUSTNESS PASS — straight heading, consistent speed, less crouch.

v18 hit the 4/4 success gate (ep_len 937, vel_x 0.233, term_pitch 0, feet_in_contact 2.38): the robot
trots forward. But for a robust baseline (so the compliance ablation isn't confounded by a flaky gait)
three qualities need tightening, each addressed by ONE targeted change:

  1. VEERS off diagonally → wants a straight heading. v18 still ran at roll −0.16 (an asymmetric,
     sideways-pushing stance). Fix: (a) tighten the roll free zone 0.10→0.05 (force a symmetric upright
     trot), and (b) add a lateral-velocity penalty `_W_LATERAL=8 · vy²` (a straight walk has ~0 sideways
     velocity; the tracking Gaussian penalises vy too softly to hold a heading).
  2. CROUCHED (~0.20 m vs 0.27 nominal). Fix: `_W_BASE_HEIGHT 2→6` to push the stance up.
  3. INCONSISTENT speed (drifted to 0.233 vs 0.2). Fix: `_W_OVERSPEED 3→5` — a sharper upper bound.

New diagnostics logged for the veer: locomotion/base_lin_vel_y (lateral drift) and locomotion/yaw_vel
(turning rate). These join the gate so robustness is measurable.

Held (the winning v18 recipe): PMTG TG (use_tg=True, 49D, lift −0.55), ACTION_SCALE 0.25, dense roll
penalty W=25, feet-air thr 0.2 s, ent_coef=0.0, command curriculum 0→0.2 (track_frac 0.3), k_max=0.3,
tracking σ²=0.04, pitch W=25, fwd-progress W=8, termination −20, log_std_init=−1.5.

ROBUSTNESS GATE (final 2M) = the 4 core metrics PLUS:
  |roll| < 0.08 · base_height ∈ [0.24,0.30] · |base_lin_vel_y| < 0.03 · |yaw_vel| < 0.06.
Watch: roll → ~0; base_lin_vel_y / yaw_vel → ~0 (straight); base_height → ~0.26; vel_x tight at ~0.2;
ep_len stay >800. Risk: the tighter roll zone + lateral penalty could dip ep_len briefly — if it
collapses <600, loosen roll free zone to 0.07 or _W_LATERAL to 4.

Usage:
    python training/train_policy_v19.py
    tensorboard --logdir runs/
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
    PenaltyCurriculum, PenaltyCurriculumCallback,
    CommandCurriculum, CommandCurriculumCallback,
    AdaptiveLRCallback, RewardLoggingCallback, DistributionHistogramCallback,
    VideoRenderCallback, CheckpointAtStepsCallback,
)

_CMD_START   = 0.0
_CMD_TARGET  = 0.2
_CMD_ANG_VEL = 0.0

V7_K_MAX        = 0.3
V9_ENT_COEF     = 0.0
V14_LOG_STD     = -1.5
V14_TRACK_FRAC  = 0.3
V19_CHECKPOINTS = [1_000_000, 3_000_000, 6_000_000, 10_000_000, 15_000_000]


def _core_per_step_reward(vx, pitch, target=0.2):
    dt = 0.02
    stab = math.exp(-2.0 * pitch ** 2)
    r_lin  = math.exp(-((target - vx) ** 2) / B._SIGMA_SQ_VEL) * B._W_LIN_VEL * dt * stab
    r_alive = B._W_ALIVE * dt
    r_pitch = -B._W_PITCH * max(0.0, abs(pitch) - B._PITCH_FREE_ZONE) ** 2 * dt
    r_orient = -(math.sin(pitch) ** 2) * B._W_ORIENTATION * dt
    r_fwd = B._W_FWD_PROGRESS * min(max(0.0, vx), target) * dt
    r_over = -B._W_OVERSPEED * max(0.0, vx - target) * dt
    return r_lin + r_alive + r_pitch + r_orient + r_fwd + r_over


def preflight(gamma=0.99):
    sprint = sum(gamma ** t * _core_per_step_reward(0.58, 0.30) for t in range(10)) + gamma ** 10 * B._TERMINATION_PENALTY
    walk = _core_per_step_reward(0.20, 0.24) * (1.0 - gamma ** B._MAX_STEPS) / (1.0 - gamma)
    print(f"\n[pre-flight] sprint-die={sprint:+.3f}  walk={walk:+.3f}")
    if walk < 2.0 * sprint:
        raise SystemExit("[pre-flight] ABORT")
    print(f"  PASS (margin {walk - sprint:+.3f}).\n")


def make_env(rank, seed=0):
    def _init():
        env = RigidTerrainEnv(target_lin_vel=(_CMD_START, 0.0), target_ang_vel=_CMD_ANG_VEL, use_tg=True)
        env.reset(seed=seed + rank)
        return env
    return _init


def build_vec_env(cfg):
    factories = [make_env(i) for i in range(cfg.n_envs)]
    vec_cls = SubprocVecEnv if cfg.n_envs > 4 else DummyVecEnv
    vec = VecMonitor(vec_cls(factories))
    return VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg.gamma)


class VecNormCheckpointCallback(CheckpointAtStepsCallback):
    def __init__(self, vec_env, *a, **k):
        super().__init__(*a, **k)
        self._vec_env = vec_env

    def _on_step(self):
        while self._next_idx < len(self._steps) and self.num_timesteps >= self._steps[self._next_idx]:
            label = f"{self._steps[self._next_idx] // 1_000_000}M"
            mp = os.path.join(self._save_dir, f"{self._prefix}_{label}")
            self.model.save(mp); self._vec_env.save(os.path.join(self._save_dir, f"vecnorm_{label}.pkl"))
            print(f"\n[checkpoint] {mp}.zip  ({self.num_timesteps:,} steps)")
            self._next_idx += 1
        return True


def train(cfg, run_name="policy_a_v19", render_freq=500_000, live_viewer=False):
    preflight(cfg.gamma)
    checkpoint_dir = f"checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Launching {cfg.n_envs} envs...")
    vec_env = build_vec_env(cfg)

    model = PPO(
        policy="MlpPolicy", env=vec_env,
        n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
        gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
        ent_coef=V9_ENT_COEF, learning_rate=cfg.learning_rate, max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128]), activation_fn=nn.ELU,
                           log_std_init=V14_LOG_STD),
        tensorboard_log=f"{cfg.tensorboard_log}{run_name}", verbose=1,
    )

    penalty_curr = PenaltyCurriculum(k0=cfg.penalty_k0, decay=cfg.penalty_decay, k_max=V7_K_MAX)
    command_curr = CommandCurriculum(start=_CMD_START, target=_CMD_TARGET, step=0.02,
                                     advance=400.0, retreat=150.0, track_frac=V14_TRACK_FRAC)

    def make_render_env():
        return RigidTerrainEnv(render_mode="rgb_array", target_lin_vel=(command_curr.cmd, 0.0),
                               target_ang_vel=_CMD_ANG_VEL, use_tg=True)

    callbacks = [
        PenaltyCurriculumCallback(penalty_curr),
        CommandCurriculumCallback(command_curr),
        AdaptiveLRCallback(kl_target=cfg.kl_target, lr_min=cfg.lr_min, lr_max=cfg.lr_max),
        RewardLoggingCallback(),
        DistributionHistogramCallback(),
        VecNormCheckpointCallback(vec_env=vec_env, steps=V19_CHECKPOINTS, save_dir=checkpoint_dir, prefix="model"),
    ]
    if render_freq > 0:
        callbacks.append(VideoRenderCallback(eval_env_factory=make_render_env, render_freq=render_freq,
                                             n_episodes=3, save_dir=f"renders/{run_name}",
                                             live_viewer=live_viewer, vecnorm=vec_env))

    print(f"\nPolicy A v19 — ROBUSTNESS PASS (straight heading, consistent speed, less crouch) — {cfg.total_timesteps:,} steps")
    print(f"  New in v19:   roll free-zone 0.10→{B._ROLL_FREE_ZONE}, +lateral penalty W={B._W_LATERAL}, base_height W 2→{B._W_BASE_HEIGHT}, overspeed W 3→{B._W_OVERSPEED}")
    print(f"  Held: v18 recipe (PMTG lift −0.55, ACTION_SCALE 0.25, roll pen W=25, feet-air thr 0.2s, ent_coef=0)")
    print(f"  Watch: roll→0, base_lin_vel_y/yaw_vel→0 (straight), base_height→~0.26, vel_x tight ~0.2, ep_len>800\n")
    model.learn(total_timesteps=cfg.total_timesteps, callback=CallbackList(callbacks), progress_bar=True)

    model.save(os.path.join(checkpoint_dir, "policy_v19_final"))
    vec_env.save(os.path.join(checkpoint_dir, "vecnorm_final.pkl"))
    print("\nTraining complete.")
    vec_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=128)
    p.add_argument("--total-steps", type=int, default=15_000_000)
    p.add_argument("--run-name", type=str, default="policy_a_v19")
    p.add_argument("--render-freq", type=int, default=500_000)
    p.add_argument("--live-viewer", action="store_true")
    a = p.parse_args()
    cfg = PPOConfig(n_envs=a.n_envs, total_timesteps=a.total_steps)
    train(cfg, run_name=a.run_name, render_freq=a.render_freq, live_viewer=a.live_viewer)
