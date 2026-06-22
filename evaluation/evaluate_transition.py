"""Held-out transition eval: does foot-history enable ONLINE compliance adaptation?

Evaluates B_trans (89D, history) vs B_trans_noh (49D, no history) on rigid->soft->rigid
terrain with FIXED soft-zone tile stiffness (no curriculum). Tiles range from firm through
the trained-soft edge into EXTRAPOLATION (softer than trained). Key metrics: cross_rate
(did the robot make it past the soft zone, x>3.3), fall_rate, velocity, cost of transport.

Usage: python evaluation/evaluate_transition.py --n 100
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd, mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.transition_env import TransitionEnv, _TILE_DAMPING, _SOFT_ZONE_END
from environments.compliance_env import CompliantTerrainEnv

_DT, _G, _MAX_STEPS = 0.02, 9.81, 1000
_CROSS_X = _SOFT_ZONE_END + 0.3   # x past which the soft zone is crossed

# Tile stiffness (N/m). Training curriculum: firm 20000 -> soft 1500. Below 1500 = extrapolation.
TT = [
    {"name": "TT0_firm",        "k": 20000},
    {"name": "TT1",             "k": 8000},
    {"name": "TT2",             "k": 4000},
    {"name": "TT3",             "k": 2500},
    {"name": "TT4_train_soft",  "k": 1500},
    {"name": "TT5_extrap",      "k": 1000},
    {"name": "TT6_extrap",      "k": 600},
    {"name": "TT7_extrap_soft", "k": 400},
]

_SEEDS = [0, 1, 2]   # seed 0 = original dir; seeds 1,2 = *_s1/_s2 multiseed dirs

# name -> (obs_mode, checkpoint_dir_base, final_zip_name, seeds)
POLICY_SPECS = {
    "Btrans_hist":   ("B", "policy_b_trans",     "policy_b_trans_final.zip",     _SEEDS),
    "Btrans_nohist": ("A", "policy_b_trans_noh", "policy_b_trans_noh_final.zip", _SEEDS),
}


def _ckpt_paths(base_dir, final_name, seed):
    d = base_dir if seed == 0 else f"{base_dir}_s{seed}"
    return f"checkpoints/{d}/{final_name}", f"checkpoints/{d}/vecnorm_final.pkl"


def iter_runs():
    """Yield (pol_name, seed, obs_mode, model_path, vecnorm_path) for every seed."""
    for pol_name, (obs_mode, base_dir, final_name, seeds) in POLICY_SPECS.items():
        for seed in seeds:
            mp, vp = _ckpt_paths(base_dir, final_name, seed)
            yield pol_name, seed, obs_mode, mp, vp


class EvalTransitionEnv(TransitionEnv):
    def __init__(self, stiffness, obs_mode, **kwargs):
        self._eval_k = float(stiffness)
        super().__init__(obs_mode=obs_mode, max_level=0, **kwargs)
        tile_bodies = [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, f"tile{i}") for i in range(4)]
        self._robot_mass = float(self._model.body_mass.sum() - sum(self._model.body_mass[b] for b in tile_bodies))

    def _reset_terrain(self):
        self._model.jnt_stiffness[self._tile_jids] = self._eval_k
        self._model.dof_damping[self._tile_dofs] = _TILE_DAMPING
        self._data.qpos[self._tile_qadr] = 0.0
        self._data.qvel[self._tile_dofs] = 0.0
        mujoco.mj_forward(self._model, self._data)
        foot_xy = self._foot_pos_base()[:, :2]
        self._foot_hist[:] = foot_xy
        self._prev_foot_world = self._foot_world()

    def reset(self, *, seed=None, options=None):       # no curriculum
        return super(CompliantTerrainEnv, self).reset(seed=seed, options=options)

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        info["power"] = float(np.sum(np.abs(self._data.actuator_force * self._data.qvel[6:18])))
        return obs, r, term, trunc, info

    @property
    def robot_mass(self): return self._robot_mass


def make_venv(stiffness, obs_mode, vn_path):
    venv = DummyVecEnv([lambda: EvalTransitionEnv(stiffness=stiffness, obs_mode=obs_mode,
                                                  target_lin_vel=(0.2, 0.0), use_tg=True)])
    venv = VecNormalize.load(vn_path, venv); venv.training = False; venv.norm_reward = False
    return venv


def run(model, venv, n, seed):
    mass = float(venv.get_attr("robot_mass")[0])
    ep_v, ep_fell, ep_cross, ep_cot = [], [], [], []
    venv.seed(seed); obs = venv.reset()
    vels, powers, steps, max_x = [], [], 0, 0.0
    while len(ep_v) < n:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(a); info = infos[0]
        vels.append(info["base_lin_vel_x"]); powers.append(info["power"])
        max_x = max(max_x, info["x_pos"]); steps += 1
        if dones[0]:
            dist = abs(info["x_pos"]); fell = steps < _MAX_STEPS
            ep_cot.append((float(np.sum(powers)) * _DT / (mass * _G * max(1e-3, dist)), fell))
            ep_v.append(float(np.mean(vels)))
            ep_fell.append(1.0 if fell else 0.0)
            ep_cross.append(1.0 if max_x > _CROSS_X else 0.0)
            vels, powers, steps, max_x = [], [], 0, 0.0
    cot_clean = [c for c, fell in ep_cot if not fell]          # COT over non-fallen episodes only
    return {"forward_velocity_mean": float(np.mean(ep_v)), "fall_rate": float(np.mean(ep_fell)),
            "cross_rate": float(np.mean(ep_cross)),
            "cost_of_transport": float(np.median(cot_clean)) if cot_clean else float("nan")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=777); ap.add_argument("--out", type=str, default="evaluation/transition_results.csv")
    args = ap.parse_args()
    rows = []
    for pol, seed, mode, mp, vp in iter_runs():
        model = PPO.load(mp, device="cpu")
        for ti, t in enumerate(TT):
            venv = make_venv(t["k"], mode, vp); m = run(model, venv, args.n, args.seed + ti); venv.close()
            m.update({"policy": pol, "seed": seed, "terrain": t["name"], "stiffness": t["k"]}); rows.append(m)
            print(f"[{pol:14s} s{seed} {t['name']:16s} k={t['k']:5d}] cross={m['cross_rate']:.2f} fall={m['fall_rate']:.2f} "
                  f"vel={m['forward_velocity_mean']:.3f} COT={m['cost_of_transport']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False); print(f"\nSaved {args.out}")
    agg = df.groupby(["policy", "terrain"], as_index=False).agg(cross_rate=("cross_rate", "mean"))
    print("\n=== cross_rate (seed-mean; history vs no-history) ===")
    print(agg.pivot(index="terrain", columns="policy", values="cross_rate").to_string())


if __name__ == "__main__":
    main()
