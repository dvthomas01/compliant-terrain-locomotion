"""
Train Policy A v18: raise the TG foot-lift so the trot duty factor drops to ~50%.

State: gate has been 3/4 since v16 — ep_len ~990 ✓, base_lin_vel_x ~0.2 ✓ (on target), term_pitch 0 ✓.
The ONLY miss is feet_in_contact (v16 2.66, v17 3.33) > the 2.5 trot ceiling. v17's feet-air-time
threshold tweak (0.5→0.2 s) did NOT help — feet_in_contact actually rose and r_feet_air barely moved,
proving feet_in_contact is NOT driven by the feet-air reward. It needs a STRUCTURAL lever.

Mechanism: with ACTION_SCALE 0.25 the policy residual can cancel most of the TG's 0.35 rad foot-lift,
so swing feet barely clear the ground → duty factor >50% → feet_in_contact >2.5.

v18 makes ONE change (base_env): `_TG_LIFT_AMP −0.35 → −0.55`. After maximal residual cancellation
(0.25) the swing foot still gets ≥0.30 rad net knee flex → it clearly clears the ground (verify_tg:
kinematic lift 6.6→10.7 cm, phasing + forward-propulsion intact) → each leg is airborne for its swing
half → duty ~50% → feet_in_contact → ~2. Keeps the ±0.25 balance authority (unlike cutting it further).

Held from v17: ACTION_SCALE 0.25, feet-air threshold 0.2 s, roll penalty W=25, PMTG TG (49D),
ent_coef=0.0, command curriculum 0→0.2 (track_frac 0.3), k_max=0.3, tracking σ²=0.04, pitch W=25,
feet-air gated, overspeed W=3, fwd-progress W=8, termination −20, log_std_init=−1.5.

Watch: feet_in_contact → ~2 (was 3.33); ep_len stay >800; vel_x stay in [0.15,0.25]. If all 4 gate
metrics hold over the final 2M → SUCCESS, stop the loop. Risk: a higher step could add base bounce —
watch base_height / r_lin_vel_z; if ep_len dips, the lift is too aggressive (try −0.45).

Usage:
    python training/train_policy_v18.py
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
V18_CHECKPOINTS = [1_000_000, 3_000_000, 6_000_000, 10_000_000, 15_000_000]


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


def train(cfg, run_name="policy_a_v18", render_freq=500_000, live_viewer=False):
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
        VecNormCheckpointCallback(vec_env=vec_env, steps=V18_CHECKPOINTS, save_dir=checkpoint_dir, prefix="model"),
    ]
    if render_freq > 0:
        callbacks.append(VideoRenderCallback(eval_env_factory=make_render_env, render_freq=render_freq,
                                             n_episodes=3, save_dir=f"renders/{run_name}",
                                             live_viewer=live_viewer, vecnorm=vec_env))

    print(f"\nPolicy A v18 — raise TG foot-lift 0.35→{abs(B._TG_LIFT_AMP)} to drop duty factor to ~50% — {cfg.total_timesteps:,} steps")
    print(f"  New in v18:   _TG_LIFT_AMP −0.35→{B._TG_LIFT_AMP} (swing feet clear the ground even after residual cancellation)")
    print(f"  Held: ACTION_SCALE 0.25, feet-air thr 0.2s, roll penalty W=25, PMTG TG (49D), ent_coef=0.0, command 0→0.2")
    print(f"  Watch: feet_in_contact → ~2 (was 3.33); ep_len stay >800; vel_x stay in [0.15,0.25]; if all 4 hold → SUCCESS\n")
    model.learn(total_timesteps=cfg.total_timesteps, callback=CallbackList(callbacks), progress_bar=True)

    model.save(os.path.join(checkpoint_dir, "policy_v18_final"))
    vec_env.save(os.path.join(checkpoint_dir, "vecnorm_final.pkl"))
    print("\nTraining complete.")
    vec_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=128)
    p.add_argument("--total-steps", type=int, default=15_000_000)
    p.add_argument("--run-name", type=str, default="policy_a_v18")
    p.add_argument("--render-freq", type=int, default=500_000)
    p.add_argument("--live-viewer", action="store_true")
    a = p.parse_args()
    cfg = PPOConfig(n_envs=a.n_envs, total_timesteps=a.total_steps)
    train(cfg, run_name=a.run_name, render_freq=a.render_freq, live_viewer=a.live_viewer)
