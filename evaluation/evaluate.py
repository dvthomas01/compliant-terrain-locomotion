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

POLICIES = {
    "A_rigid":      ("A", "checkpoints/policy_a_v24/policy_v24_final.zip",
                          "checkpoints/policy_a_v24/vecnorm_final.pkl"),
    "B_compliance": ("B", "checkpoints/policy_b/policy_b_final.zip",
                          "checkpoints/policy_b/vecnorm_final.pkl"),
    # obs-ablation: compliance-trained but 49D (no foot history)
    "Bp_noh":       ("A", "checkpoints/policy_b_noh/policy_b_noh_final.zip",
                          "checkpoints/policy_b_noh/vecnorm_final.pkl"),
}


def make_venv(terrain, obs_mode, vecnorm_path):
    venv = DummyVecEnv([lambda: EvalCompliantEnv(terrain=terrain, obs_mode=obs_mode,
                                                 target_lin_vel=(0.2, 0.0), use_tg=True)])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def run_terrain(model, venv, n_episodes, seed):
    mass = float(venv.get_attr("robot_mass")[0])
    ep_vels, ep_fell, ep_cot, ep_contact_var, ep_height_var = [], [], [], [], []
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
            dist = max(1e-3, abs(info["x_pos"]))               # forward distance travelled
            energy = float(np.sum(powers)) * _DT
            ep_cot.append(energy / (mass * _G * dist))
            ep_vels.append(float(np.mean(vels)))
            ep_contact_var.append(float(np.var(contacts)))
            ep_height_var.append(float(np.var(heights)))
            ep_fell.append(1.0 if steps < _MAX_STEPS else 0.0)  # ended early == fell
            vels, powers, contacts, heights = fresh(); steps = 0
    return {
        "forward_velocity_mean": float(np.mean(ep_vels)),
        "forward_velocity_std":  float(np.std(ep_vels)),
        "fall_rate":             float(np.mean(ep_fell)),
        "cost_of_transport":     float(np.median(ep_cot)),  # median: robust to fallen-episode spikes
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
    for pol_name, (obs_mode, model_path, vecnorm_path) in POLICIES.items():
        model = PPO.load(model_path, device="cpu")
        for ti, terrain in enumerate(TERRAINS):
            venv = make_venv(terrain, obs_mode, vecnorm_path)
            m = run_terrain(model, venv, args.n, seed=args.seed + ti)
            venv.close()
            m.update({"policy": pol_name, "terrain": terrain["name"]})
            rows.append(m)
            print(f"[{pol_name:13s} {terrain['name']:16s}] "
                  f"vel={m['forward_velocity_mean']:.3f}±{m['forward_velocity_std']:.3f} "
                  f"fall={m['fall_rate']:.2f} COT={m['cost_of_transport']:.2f} "
                  f"contactVar={m['foot_contact_variance']:.3f} hVar={m['base_height_variance']:.5f}")

    df = pd.DataFrame(rows)[["policy", "terrain", "forward_velocity_mean", "forward_velocity_std",
                             "fall_rate", "cost_of_transport", "foot_contact_variance",
                             "base_height_variance"]]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")
    # side-by-side fall_rate and velocity for the headline comparison
    piv = df.pivot(index="terrain", columns="policy", values=["forward_velocity_mean", "fall_rate"])
    print("\n=== A vs B (velocity_mean / fall_rate) ===")
    print(piv.to_string())


if __name__ == "__main__":
    main()
