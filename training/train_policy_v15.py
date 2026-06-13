"""
Train Policy A v15: kill the tilt-brace with a dense roll penalty.

v14 (PMTG gait prior) result (final 2M): ep_len 921 ✓, term_pitch 0 ✓, but base_lin_vel_x −0.009
✗ and feet_in_contact 2.99 ✗ → 2/4 (same as v13). The TG let the command curriculum climb to 0.1
(vs v13's 0.04) and the feet step, but the robot still won't propel forward. Two diagnostics:
  • feet_in_contact 2.99 (a real trot is ~2.0) → the policy SUPPRESSES the TG, holding ~3 feet down.
  • roll −0.27 vs pitch −0.035 → it braces in a persistent ~15° side-tilt. Pitch is tightly
    controlled by the dense W=25 pitch penalty; there was NO equivalent roll term, so roll is the
    free exploit the robot uses to hold a stable sprawl and survive without trotting.

v15 makes ONE change (in base_env): add a dense ROLL penalty mirroring the pitch penalty
(`r_roll_pen = −_W_ROLL·max(0,|roll|−0.10)²·dt`, W=25, free-zone 0.10 rad, outside the
k-curriculum). A straight forward trot needs a pitch lean but ZERO roll, so the free zone is
small. Forcing roll → 0 removes the tilt-brace, so the (symmetric, verified-propulsive) TG must
provide BOTH stability and propulsion → the robot should finally walk. Lower-risk than cutting
residual authority: it doesn't touch the TG or the ±0.5 rad balance authority.

Held from v14: PMTG TG (use_tg=True, 49D obs), ent_coef=0.0, command curriculum 0→0.2 from
standing (track_frac=0.3), k_max=0.3, tracking σ²=0.04, pitch W=25, feet-air gated, overspeed
W=3, fwd-progress W=8, termination −20, log_std_init=−1.5.

Watch: locomotion/roll should collapse toward 0 (was −0.27); base_lin_vel_x should TRACK
command upward; feet_in_contact drop toward ~2; command_vel climb past 0.1 toward 0.2.

Usage:
    python training/train_policy_v15.py
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
V15_CHECKPOINTS = [1_000_000, 3_000_000, 6_000_000, 10_000_000, 15_000_000]


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
    # roll = 0 at the target walk → r_roll_pen = 0, so the core reward is unchanged by v15.
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


def train(cfg, run_name="policy_a_v15", render_freq=500_000, live_viewer=False):
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
        VecNormCheckpointCallback(vec_env=vec_env, steps=V15_CHECKPOINTS, save_dir=checkpoint_dir, prefix="model"),
    ]
    if render_freq > 0:
        callbacks.append(VideoRenderCallback(eval_env_factory=make_render_env, render_freq=render_freq,
                                             n_episodes=3, save_dir=f"renders/{run_name}",
                                             live_viewer=live_viewer, vecnorm=vec_env))

    print(f"\nPolicy A v15 — kill the tilt-brace with a dense roll penalty — {cfg.total_timesteps:,} steps")
    print(f"  New in v15:   dense roll penalty W={B._W_ROLL}, free-zone {B._ROLL_FREE_ZONE} rad (mirror of the pitch penalty)")
    print(f"  Held: PMTG TG (49D), ent_coef=0.0, command curriculum 0→0.2, feet-air gated, k_max=0.3, log_std_init={V14_LOG_STD}")
    print(f"  Watch: locomotion/roll → 0 (was -0.27); base_lin_vel_x track command up; feet_in_contact → ~2\n")
    model.learn(total_timesteps=cfg.total_timesteps, callback=CallbackList(callbacks), progress_bar=True)

    model.save(os.path.join(checkpoint_dir, "policy_v15_final"))
    vec_env.save(os.path.join(checkpoint_dir, "vecnorm_final.pkl"))
    print("\nTraining complete.")
    vec_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=128)
    p.add_argument("--total-steps", type=int, default=15_000_000)
    p.add_argument("--run-name", type=str, default="policy_a_v15")
    p.add_argument("--render-freq", type=int, default=500_000)
    p.add_argument("--live-viewer", action="store_true")
    a = p.parse_args()
    cfg = PPOConfig(n_envs=a.n_envs, total_timesteps=a.total_steps)
    train(cfg, run_name=a.run_name, render_freq=a.render_freq, live_viewer=a.live_viewer)
