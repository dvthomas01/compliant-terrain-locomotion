"""
Train Policy A v6: velocity-tracking + pitch-stability fix.

v5 produced genuine forward trotting (~0.58 m/s, feet_in_contact ≈ 2.3, zero roll
terminations) but overshot the 0.2 m/s target ~3× and nose-dived (pitch > 0.4) within
~10 steps. Root cause: the tracking Gaussian exp(-err²/0.25) (σ≈0.5) was nearly flat
across the achievable range at a 0.2 m/s target, so it gave almost no deceleration
gradient — and pitch was bounded only by the multiplicative stability gate, which goes
toothless once tracking → 0 (the exact overshoot+pitched regime).

v6 makes TWO load-bearing reward changes (both in environments/base_env.py):

  1. Tighten the tracking Gaussian: σ² 0.25 → 0.04 (σ≈0.2). Restores a real gradient
     pulling toward 0.2 m/s from both sides; 0.58 m/s now scores 0.027× instead of 0.56×.

  2. Add a dense, absolute pitch penalty OUTSIDE the k-curriculum:
         r_pitch_pen = -_W_PITCH * max(0, |pitch| - 0.26)² * dt,  _W_PITCH = 25
     Sized for the narrow 0.26→0.40 band (termination held at 0.40): ~33% of max tracking
     at the cliff, ~3% at 0.30, zero inside the healthy ~0.24 rad trot lean. Unlike the
     multiplicative gate, it bites even when tracking reward is already ~0.

Deliberately NOT changed from v5 (clean attribution; decide from logs):
  - Termination thresholds (pitch held at 0.40, NOT widened to 0.50)
  - Termination penalty (-20). The pre-flight below shows the v5 "choose to die" framing
    does not survive the real numbers: with -20 and γ=0.99, dying is already ~-18 vs a
    walk's ~+positive return. The failure was a bad PPO basin, not death-positive economics.
  - PPO hyperparameters, network, VecNormalize, all other reward weights, backward penalty,
    feet-air term, k-curriculum contents, target velocity (0.2 m/s), flat terrain.

Usage:
    python training/train_policy_v6.py
    python training/train_policy_v6.py --n-envs 8 --total-steps 6144   # smoke-test
    tensorboard --logdir runs/

Watch (do not react before 2M — reward reshaping shifts VecNormalize stats):
  hist/base_lin_vel_x  — unimodal ~0.2 (bimodal ⇒ incentives still broken)
  rewards/r_lin_vel vs rewards/r_pitch_pen  — must see them trade off
  locomotion/term_pitch → < 0.02 ;  rollout/ep_len_mean → past v4's 32, toward horizon
  locomotion/feet_in_contact ~2.3 (drifting toward 4 ⇒ _W_PITCH too strong; halve it)
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
    AdaptiveLRCallback,
    RewardLoggingCallback,
    DistributionHistogramCallback,
    VideoRenderCallback,
    CheckpointAtStepsCallback,
)

_CMD_LIN_VEL = (0.2, 0.0)
_CMD_ANG_VEL = 0.0

V6_CHECKPOINTS = [1_000_000, 3_000_000, 6_000_000, 10_000_000, 15_000_000]


def _core_per_step_reward(vx: float, pitch: float) -> float:
    """Dominant policy-relevant reward terms at a held (vx, pitch), roll=0.

    Covers tracking (stability-gated), survival bonus, pitch penalty, and the
    orientation penalty — the terms that decide the die-fast vs walk trade-off.
    Omits feet-air/torque/action terms (similar both sides, and feet-air favours
    the sprint), so this is a conservative lower bound on the walk's advantage.
    """
    dt   = 0.02
    stab = math.exp(-2.0 * pitch ** 2)
    lin_err_sq = (_CMD_LIN_VEL[0] - vx) ** 2 + _CMD_LIN_VEL[1] ** 2
    r_lin   = math.exp(-lin_err_sq / B._SIGMA_SQ_VEL) * B._W_LIN_VEL * dt * stab
    r_alive = B._W_ALIVE * dt
    excess  = max(0.0, abs(pitch) - B._PITCH_FREE_ZONE)
    r_pitch = -B._W_PITCH * excess ** 2 * dt
    r_orient = -(math.sin(pitch) ** 2) * B._W_ORIENTATION * dt
    return r_lin + r_alive + r_pitch + r_orient


def preflight_termination_economics(gamma: float = 0.99) -> None:
    """Print the discounted die-fast vs full-horizon-walk return and abort if walking
    does not strictly dominate by a 2× margin.  (Expected to pass comfortably — the
    point is to confirm the failure is a bad basin, not death-positive economics.)"""
    die_steps = 10
    sprint = sum(gamma ** t * _core_per_step_reward(0.58, 0.30) for t in range(die_steps))
    sprint += gamma ** die_steps * B._TERMINATION_PENALTY

    # γ-discounted full horizon ≈ geometric sum; effective horizon ≈ 1/(1-γ) steps
    walk_step = _core_per_step_reward(0.20, 0.24)
    walk = walk_step * (1.0 - gamma ** B._MAX_STEPS) / (1.0 - gamma)

    print("\n[pre-flight] discounted return (γ=%.2f):" % gamma)
    print(f"  (a) sprint {die_steps} steps @0.58 m/s, pitch 0.30, then die  = {sprint:+.3f}")
    print(f"  (b) walk full horizon @0.20 m/s, pitch 0.24 (eff ~{1/(1-gamma):.0f} steps) = {walk:+.3f}")
    if walk < 2.0 * sprint:
        raise SystemExit(
            f"[pre-flight] ABORT: walk ({walk:+.3f}) does not dominate sprint "
            f"({sprint:+.3f}) by 2×. Raise _TERMINATION_PENALTY before launching."
        )
    print(f"  PASS: walking dominates dying (margin {walk - sprint:+.3f}); "
          f"failure mode is basin/exploration, not economics.\n")


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


def train(cfg: PPOConfig, run_name: str = "policy_a_v6",
          render_freq: int = 500_000, live_viewer: bool = False) -> None:

    preflight_termination_economics(gamma=cfg.gamma)

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
        DistributionHistogramCallback(),
        VecNormCheckpointCallback(
            vec_env=vec_env,
            steps=V6_CHECKPOINTS,
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

    print(f"\nPolicy A v6 — tracking + pitch fix — {cfg.total_timesteps:,} steps")
    print(f"  Command:         {_CMD_LIN_VEL[0]} m/s forward (fixed)")
    print(f"  New in v6:       tracking σ² 0.25 → {B._SIGMA_SQ_VEL} (σ≈{B._SIGMA_SQ_VEL**0.5:.2f})")
    print(f"  New in v6:       dense pitch penalty _W_PITCH={B._W_PITCH}, free zone {B._PITCH_FREE_ZONE} rad")
    print(f"  Held from v5:    pitch termination {B._MAX_PITCH} rad, termination penalty {B._TERMINATION_PENALTY}")
    print(f"  Watch for:       hist/base_lin_vel_x → unimodal ~0.2 ; locomotion/term_pitch → <0.02")
    print(f"  TensorBoard:     tensorboard --logdir {cfg.tensorboard_log}\n")

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList(callback_list),
        progress_bar=True,
    )

    final_model   = os.path.join(checkpoint_dir, "policy_v6_final")
    final_vecnorm = os.path.join(checkpoint_dir, "vecnorm_final.pkl")
    model.save(final_model)
    vec_env.save(final_vecnorm)
    print(f"\nTraining complete.")
    print(f"  Model:   {final_model}.zip")
    print(f"  VecNorm: {final_vecnorm}")

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy A v6: tracking + pitch fix")
    parser.add_argument("--n-envs",      type=int,  default=128)
    parser.add_argument("--total-steps", type=int,  default=15_000_000)
    parser.add_argument("--run-name",    type=str,  default="policy_a_v6")
    parser.add_argument("--render-freq", type=int,  default=500_000)
    parser.add_argument("--live-viewer", action="store_true")
    args = parser.parse_args()

    cfg = PPOConfig(n_envs=args.n_envs, total_timesteps=args.total_steps)
    train(cfg, run_name=args.run_name,
          render_freq=args.render_freq, live_viewer=args.live_viewer)
