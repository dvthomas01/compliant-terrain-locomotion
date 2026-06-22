"""
Train Policy A v20: ROBUSTNESS SPRINT iter 2 — kill the banked roll (and, by coupling, the veer).

v19 (bundled robustness pass) fixed speed (vel_x 0.228 ± 0.005), crouch (height 0.20→0.25) and lateral
slip (vy 0.003), but LEFT two coupled failures:
    roll −0.201  (gate <0.08)  and  yaw_vel −0.124  (gate <0.06).
Diagnosis: the diagonal veer is yaw DRIFT, not lateral slip (vy≈0). The robot holds a persistent banked
stance (roll −0.20) and leans into a continuous turn (~−142°/episode). The v19 lateral-velocity penalty
targeted a non-existent problem. Root cause is the entrenched roll (attractor since v11), and a banked
angle mechanically drives the yaw turn.

v20 change (SINGLE variable, in base_env): `_W_ROLL 25 → 70`. The roll penalty bites only above
|roll|>0.05, so a symmetric upright trot pays nothing. At roll 0.20 the per-step penalty goes from a
toothless −0.011 to ~−0.031 (≈ the fwd-progress reward) — real pressure to straighten up.
PREDICTION: roll → toward gate AND yaw_vel falls out with it (confirms the coupling). If roll drops but
yaw_vel persists → independent → v21 adds a dense quadratic yaw-rate penalty (yaw_vel is already in the
obs, so no obs change / no ablation impact).

Held (v19 recipe): base_height W=6, overspeed W=5, lateral W=8, roll-free-zone 0.05, PMTG TG (use_tg,
49D, lift −0.55), ACTION_SCALE 0.25, feet-air thr 0.2s, ent_coef=0, command 0→0.2 (track_frac 0.3),
k_max=0.3, tracking σ²=0.04, pitch W=25, fwd-progress W=8, termination −20, log_std_init=−1.5.

ROBUSTNESS GATE (final 2M) = 4 core (ep_len>800, base_lin_vel_x≈0.2, term_pitch 0, feet_in_contact)
PLUS |roll|<0.08 · base_height∈[0.24,0.30] · |base_lin_vel_y|<0.03 · |yaw_vel|<0.06.
Watch: roll→0, yaw_vel→0, ep_len stay >800, vel_x [0.15,0.25], feet_in_contact not worsening.

Usage:
    python training/train_policy_v20.py
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

from environments.clean_transition_env import CleanTransitionEnv
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
_MAX_LEVEL   = 4        # Level-1 compliance curriculum only (Level 2 deferred; see compliance_env)

V7_K_MAX        = 0.3
V9_ENT_COEF     = 0.0
V14_LOG_STD     = -1.5
V14_TRACK_FRAC  = 0.3
TRANS_CK = [1_000_000, 3_000_000, 6_000_000, 10_000_000, 15_000_000]


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
        env = CleanTransitionEnv(target_lin_vel=(_CMD_START, 0.0), target_ang_vel=_CMD_ANG_VEL, use_tg=True, max_level=_MAX_LEVEL)
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


def train(cfg, run_name="clean_trans", render_freq=500_000, live_viewer=False):
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
        return CleanTransitionEnv(render_mode="rgb_array", target_lin_vel=(command_curr.cmd, 0.0),
                                   target_ang_vel=_CMD_ANG_VEL, use_tg=True, max_level=_MAX_LEVEL)

    callbacks = [
        PenaltyCurriculumCallback(penalty_curr),
        CommandCurriculumCallback(command_curr),
        AdaptiveLRCallback(kl_target=cfg.kl_target, lr_min=cfg.lr_min, lr_max=cfg.lr_max),
        RewardLoggingCallback(),
        DistributionHistogramCallback(),
        VecNormCheckpointCallback(vec_env=vec_env, steps=TRANS_CK, save_dir=checkpoint_dir, prefix="model"),
    ]
    if render_freq > 0:
        callbacks.append(VideoRenderCallback(eval_env_factory=make_render_env, render_freq=render_freq,
                                             n_episodes=3, save_dir=f"renders/{run_name}",
                                             live_viewer=live_viewer, vecnorm=vec_env))

    print(f"\nPolicy B_trans (89D history) - TRANSITION terrain — COMPLIANCE-RANDOMIZED (89D obs, Level-1 curriculum max_level={_MAX_LEVEL}) — {cfg.total_timesteps:,} steps")
    print(f"  Identical to Policy A v24 recipe EXCEPT: 89D obs (+foot history) and compliance-randomized terrain")
    print(f"  Recipe (shared w/ A): roll W={B._W_ROLL}, yaw-rate W={B._W_YAW_RATE}, base-height LINEAR W={B._W_BASE_HEIGHT}, lift {B._TG_LIFT_AMP}, ACTION_SCALE 0.25")
    print(f"  Watch: ep_len>800 across compliance; terrain_level climbs (curriculum); gait holds as floor softens\n")
    model.learn(total_timesteps=cfg.total_timesteps, callback=CallbackList(callbacks), progress_bar=True)

    model.save(os.path.join(checkpoint_dir, "clean_trans_final"))
    vec_env.save(os.path.join(checkpoint_dir, "vecnorm_final.pkl"))
    print("\nTraining complete.")
    vec_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=128)
    p.add_argument("--total-steps", type=int, default=15_000_000)
    p.add_argument("--run-name", type=str, default="clean_trans")
    p.add_argument("--render-freq", type=int, default=500_000)
    p.add_argument("--live-viewer", action="store_true")
    a = p.parse_args()
    cfg = PPOConfig(n_envs=a.n_envs, total_timesteps=a.total_steps)
    train(cfg, run_name=a.run_name, render_freq=a.render_freq, live_viewer=a.live_viewer)
