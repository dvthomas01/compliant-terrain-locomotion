"""
Train Policy A v7: anti-freeze run (penalty cap + forward-progress reward).

v6 fixed v5's nose-dive-and-sprint (mean pitch settled at 0.18 rad, overshoot gone) but
converged to the OPPOSITE failure: a near-standing shuffle. The TensorBoard story was
unambiguous — performance peaked at ~3M steps (ep_len 37, moving) when the penalty scale
k was still low (~0.08), then collapsed monotonically to ep_len ~14, vel_x ~0.02 m/s as
k ramped to 1.0. The final velocity histogram was multimodal around zero (30% backward,
36% standing, only 6% near the 0.2 target) and action std diverged to ~2.0. Final
r_lin_vel (0.0101) sat almost exactly at the analytic standing-still optimum of the
σ²=0.04 Gaussian (0.0099) — the policy parked itself at "stand still."

Two root causes, addressed by exactly TWO changes (everything else held from v6):

  1. PENALTY CURRICULUM CAP (the recurring villain since v1). Runs v2–v6 all peaked at
     low k and collapsed as k→1.0; the efficiency penalties (torque/joint-motion/action-
     rate) crushed locomotion before it stabilised. In v6, ep_len already fell 37→16 across
     k=0.08→0.35, so v7 caps k at 0.3 (PenaltyCurriculum k_max=0.3) — inside v6's functional band.

  2. FORWARD-PROGRESS REWARD (base_env: _W_FWD_PROGRESS=4.0, OUTSIDE the k-curriculum).
     A dense, monotonic "always move forward" term the tracking Gaussian cannot provide —
     the Gaussian gives 37% reward for standing still at v=0. Clipped at the 0.2 m/s target
     so it adds NO overshoot incentive (the Gaussian still penalises going faster).

Deliberately held from v6 (clean attribution): tracking σ²=0.04, dense pitch penalty
(_W_PITCH=25, free zone 0.26), pitch termination 0.40, termination penalty -20, backward
penalty, feet-air term, PPO hyperparameters, network, VecNormalize, target 0.2 m/s, flat terrain.

Usage:
    python training/train_policy_v7.py
    python training/train_policy_v7.py --n-envs 8 --total-steps 6144   # smoke-test
    tensorboard --logdir runs/

Watch (do not react before 2M):
  rollout/ep_len_mean  — must hold/grow PAST v6's 3M peak of 37 instead of collapsing
  hist/base_lin_vel_x  — should shift to unimodal near 0.2 (v6 was multimodal near 0)
  rewards/r_fwd_progress vs rewards/r_lin_vel  — progress term should be clearly positive
  train/penalty_scale  — plateaus at 0.3 (not 1.0) ;  train/std should NOT diverge
  locomotion/term_pitch → <0.02 ;  locomotion/feet_in_contact ~2.3 (trot)
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

V7_K_MAX = 0.3   # cap the penalty curriculum. v6's ep_len already collapsed 37→16 across
                 # k=0.08→0.35, so 0.3 keeps penalties inside v6's still-functional band.

V7_CHECKPOINTS = [1_000_000, 3_000_000, 6_000_000, 10_000_000, 15_000_000]


def _core_per_step_reward(vx: float, pitch: float) -> float:
    """Dominant policy-relevant reward terms at a held (vx, pitch), roll=0:
    tracking (stability-gated), survival bonus, pitch penalty, orientation penalty,
    and the v7 forward-progress term. Omits feet-air/efficiency terms (similar both
    sides), so this is a conservative lower bound on the walk's advantage."""
    dt   = 0.02
    stab = math.exp(-2.0 * pitch ** 2)
    lin_err_sq = (_CMD_LIN_VEL[0] - vx) ** 2 + _CMD_LIN_VEL[1] ** 2
    r_lin    = math.exp(-lin_err_sq / B._SIGMA_SQ_VEL) * B._W_LIN_VEL * dt * stab
    r_alive  = B._W_ALIVE * dt
    excess   = max(0.0, abs(pitch) - B._PITCH_FREE_ZONE)
    r_pitch  = -B._W_PITCH * excess ** 2 * dt
    r_orient = -(math.sin(pitch) ** 2) * B._W_ORIENTATION * dt
    r_fwd    = B._W_FWD_PROGRESS * min(max(0.0, vx), _CMD_LIN_VEL[0]) * dt
    return r_lin + r_alive + r_pitch + r_orient + r_fwd


def preflight_termination_economics(gamma: float = 0.99) -> None:
    """Discounted die-fast vs full-horizon-walk return; abort if walking does not
    dominate by 2×. Expected to pass — confirms the failure is basin/exploration,
    not death-positive economics."""
    die_steps = 10
    sprint = sum(gamma ** t * _core_per_step_reward(0.58, 0.30) for t in range(die_steps))
    sprint += gamma ** die_steps * B._TERMINATION_PENALTY
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


def train(cfg: PPOConfig, run_name: str = "policy_a_v7",
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

    curriculum = PenaltyCurriculum(k0=cfg.penalty_k0, decay=cfg.penalty_decay, k_max=V7_K_MAX)

    callback_list = [
        PenaltyCurriculumCallback(curriculum),
        AdaptiveLRCallback(kl_target=cfg.kl_target, lr_min=cfg.lr_min, lr_max=cfg.lr_max),
        RewardLoggingCallback(),
        DistributionHistogramCallback(),
        VecNormCheckpointCallback(
            vec_env=vec_env,
            steps=V7_CHECKPOINTS,
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

    print(f"\nPolicy A v7 — anti-freeze (penalty cap + forward progress) — {cfg.total_timesteps:,} steps")
    print(f"  Command:         {_CMD_LIN_VEL[0]} m/s forward (fixed)")
    print(f"  New in v7:       penalty curriculum capped at k_max={V7_K_MAX} (was 1.0)")
    print(f"  New in v7:       forward-progress reward _W_FWD_PROGRESS={B._W_FWD_PROGRESS} (outside k)")
    print(f"  Held from v6:    tracking σ²={B._SIGMA_SQ_VEL}, pitch penalty W={B._W_PITCH}, term {B._MAX_PITCH} rad")
    print(f"  Watch for:       ep_len past v6's 3M peak (37); hist/base_lin_vel_x → unimodal ~0.2")
    print(f"  TensorBoard:     tensorboard --logdir {cfg.tensorboard_log}\n")

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList(callback_list),
        progress_bar=True,
    )

    final_model   = os.path.join(checkpoint_dir, "policy_v7_final")
    final_vecnorm = os.path.join(checkpoint_dir, "vecnorm_final.pkl")
    model.save(final_model)
    vec_env.save(final_vecnorm)
    print(f"\nTraining complete.")
    print(f"  Model:   {final_model}.zip")
    print(f"  VecNorm: {final_vecnorm}")

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy A v7: anti-freeze run")
    parser.add_argument("--n-envs",      type=int,  default=128)
    parser.add_argument("--total-steps", type=int,  default=15_000_000)
    parser.add_argument("--run-name",    type=str,  default="policy_a_v7")
    parser.add_argument("--render-freq", type=int,  default=500_000)
    parser.add_argument("--live-viewer", action="store_true")
    args = parser.parse_args()

    cfg = PPOConfig(n_envs=args.n_envs, total_timesteps=args.total_steps)
    train(cfg, run_name=args.run_name,
          render_freq=args.render_freq, live_viewer=args.live_viewer)
