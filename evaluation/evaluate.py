"""A-vs-B ablation: evaluate both policies on the held-out terrain suite T0-T9.

Loads each policy with its OWN saved VecNormalize obs stats (training=False), runs
N episodes per terrain, and reports the 5 ablation metrics:
  forward_velocity (mean, std) · fall_rate · cost_of_transport (Σ|τ·q̇|/(m·g·d)) ·
  foot_contact_variance · base_height_variance.

Both policies see identical physics per terrain (EvalCompliantEnv); only the obs dim
differs (49D A / 89D B). Usage:
    python evaluation/evaluate.py --n 100
    python evaluation/evaluate.py --n 3   # quick smoke
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evaluation.terrain_suite import TERRAINS, EvalCompliantEnv

_DT        = 0.02
_G         = 9.81
_MAX_STEPS = 1000

_SEEDS = [0, 1, 2]   # seed 0 = original dir; seeds 1,2 = *_s1/_s2 multiseed dirs

# name -> (obs_mode, checkpoint_dir_base, final_zip_name, seeds)
# All three conditions use 3 seeds. Policy A's compliant-terrain failure is
# categorical, but multi-seeding it confirms the rigid-terrain behaviour is
# consistent and removes the "single-seed baseline" objection.
POLICY_SPECS = {
    "A_rigid":      ("A", "policy_a_v24", "policy_v24_final.zip",  _SEEDS),
    "B_compliance": ("B", "policy_b",     "policy_b_final.zip",     _SEEDS),
    # obs-ablation: compliance-trained but 49D (no foot history)
    "Bp_noh":       ("A", "policy_b_noh", "policy_b_noh_final.zip", _SEEDS),
}


def _ckpt_paths(base_dir, final_name, seed):
    d = base_dir if seed == 0 else f"{base_dir}_s{seed}"
    return f"checkpoints/{d}/{final_name}", f"checkpoints/{d}/vecnorm_final.pkl"


def iter_runs():
    """Yield (pol_name, seed, obs_mode, model_path, vecnorm_path) for every seed."""
    for pol_name, (obs_mode, base_dir, final_name, seeds) in POLICY_SPECS.items():
        for seed in seeds:
            model_path, vecnorm_path = _ckpt_paths(base_dir, final_name, seed)
            yield pol_name, seed, obs_mode, model_path, vecnorm_path


def make_venv(terrain, obs_mode, vecnorm_path):
    venv = DummyVecEnv([lambda: EvalCompliantEnv(terrain=terrain, obs_mode=obs_mode,
                                                 target_lin_vel=(0.2, 0.0), use_tg=True)])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False
    return venv


_PROGRESS_M = 2.0   # min forward distance (m) for a "real traverse" (gates out frozen-but-upright)
_STALL_M    = 1.0   # below this distance the robot effectively froze


def run_terrain(model, venv, n_episodes, seed):
    mass = float(venv.get_attr("robot_mass")[0])
    ep_vels, ep_fell, ep_cot, ep_contact_var, ep_height_var, ep_dist = [], [], [], [], [], []
    steps = 0

    def fresh():
        return [], [], [], []

    venv.seed(seed)
    obs = venv.reset()
    vels, powers, contacts, heights = fresh()
    while len(ep_vels) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        info = infos[0]
        vels.append(info["base_lin_vel_x"]); powers.append(info["power"])
        contacts.append(info["feet_in_contact"]); heights.append(info["base_height"])
        steps += 1
        if dones[0]:
            dist = abs(info["x_pos"])                           # forward distance travelled
            fell = steps < _MAX_STEPS                           # ended early == fell
            energy = float(np.sum(powers)) * _DT
            ep_cot.append(energy / (mass * _G * max(1e-3, dist)))
            ep_vels.append(float(np.mean(vels)))
            ep_contact_var.append(float(np.var(contacts)))
            ep_height_var.append(float(np.var(heights)))
            ep_fell.append(1.0 if fell else 0.0)
            ep_dist.append(dist)
            vels, powers, contacts, heights = fresh(); steps = 0
    fell = np.array(ep_fell); dist = np.array(ep_dist); cot = np.array(ep_cot)
    cot_clean = cot[fell == 0.0]                                # COT only over non-fallen episodes
    return {
        "forward_velocity_mean": float(np.mean(ep_vels)),
        "forward_velocity_std":  float(np.std(ep_vels)),
        "fall_rate":             float(np.mean(fell)),
        # progress-gated: a "success" must NOT fall AND actually move (freezing != survival)
        "success_rate":          float(np.mean((fell == 0.0) & (dist > _PROGRESS_M))),
        "stall_rate":            float(np.mean((fell == 0.0) & (dist < _STALL_M))),
        "cost_of_transport":     float(np.median(cot_clean)) if cot_clean.size else float("nan"),
        "foot_contact_variance": float(np.mean(ep_contact_var)),
        "base_height_variance":  float(np.mean(ep_height_var)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="episodes per terrain")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=str, default="evaluation/ablation_results.csv")
    args = ap.parse_args()

    rows = []
    for pol_name, seed, obs_mode, model_path, vecnorm_path in iter_runs():
        model = PPO.load(model_path, device="cpu")
        for ti, terrain in enumerate(TERRAINS):
            venv = make_venv(terrain, obs_mode, vecnorm_path)
            m = run_terrain(model, venv, args.n, seed=args.seed + ti)
            venv.close()
            m.update({"policy": pol_name, "seed": seed, "terrain": terrain["name"]})
            rows.append(m)
            print(f"[{pol_name:13s} s{seed} {terrain['name']:16s}] "
                  f"vel={m['forward_velocity_mean']:.3f}±{m['forward_velocity_std']:.3f} "
                  f"fall={m['fall_rate']:.2f} COT={m['cost_of_transport']:.2f} "
                  f"contactVar={m['foot_contact_variance']:.3f} hVar={m['base_height_variance']:.5f}")

    df = pd.DataFrame(rows)[["policy", "seed", "terrain", "forward_velocity_mean", "forward_velocity_std",
                             "fall_rate", "success_rate", "stall_rate", "cost_of_transport",
                             "foot_contact_variance", "base_height_variance"]]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")
    # side-by-side fall_rate and velocity for the headline comparison (mean across seeds)
    agg = df.groupby(["policy", "terrain"], as_index=False).agg(
        forward_velocity_mean=("forward_velocity_mean", "mean"), fall_rate=("fall_rate", "mean"))
    piv = agg.pivot(index="terrain", columns="policy", values=["forward_velocity_mean", "fall_rate"])
    print("\n=== A vs B (seed-mean velocity_mean / fall_rate) ===")
    print(piv.to_string())


if __name__ == "__main__":
    main()
