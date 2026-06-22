"""Clean transition eval: history vs no-history on the CONFOUND-FREE rigid->soft->rigid terrain.

Flush tiles (no step), constant damping ratio (c=ALPHA*sqrt(k)), deep range (no saturation).
Sweeps soft-zone stiffness firm->soft->extrapolation. Reports progress-gated cross_rate, fall_rate,
non-fallen COT, velocity, and mean tile_sink (to confirm difficulty tracks sinkage, not bounce).

Usage: python evaluation/evaluate_clean_transition.py --n 100
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd, mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.clean_transition_env import CleanTransitionEnv, _ALPHA, _K_RIGID, _SOFT_ZONE_END

_DT, _G, _MAX_STEPS = 0.02, 9.81, 1000
_CROSS_X = _SOFT_ZONE_END + 0.3

# soft-zone stiffness (N/m). Trained curriculum firm 80000 -> soft 3000; below 3000 = extrapolation.
CT = [
    {"name": "CT0_firm",        "k": 80000},
    {"name": "CT1",             "k": 40000},
    {"name": "CT2",             "k": 20000},
    {"name": "CT3",             "k": 10000},
    {"name": "CT4",             "k": 5000},
    {"name": "CT5_train_soft",  "k": 3000},
    {"name": "CT6_extrap",      "k": 2000},
    {"name": "CT7_extrap_soft", "k": 1000},
]

_SEEDS = [0]   # pilot: seed 0 only; extend to [0,1,2] if it discriminates
POLICY_SPECS = {
    "B2_hist":   ("B", "clean_trans",     "clean_trans_final.zip",     _SEEDS),
    "B2_nohist": ("A", "clean_trans_noh", "clean_trans_noh_final.zip", _SEEDS),
}


def _ckpt(base, final, seed):
    d = base if seed == 0 else f"{base}_s{seed}"
    return f"checkpoints/{d}/{final}", f"checkpoints/{d}/vecnorm_final.pkl"


class CleanEvalTransitionEnv(CleanTransitionEnv):
    def __init__(self, stiffness, obs_mode, **kwargs):
        self._eval_k = float(stiffness)
        super().__init__(obs_mode=obs_mode, max_level=0, **kwargs)
        tile_bodies = [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, b)
                       for b in ["tile_soft_0","tile_soft_1","tile_soft_2","tile_soft_3","tile_soft_4",
                                 "tile_rigid_approach","tile_rigid_exit"]]
        self._robot_mass = float(self._model.body_mass.sum() - sum(self._model.body_mass[b] for b in tile_bodies))

    def _reset_terrain(self):
        for j in self._soft_j:
            self._model.jnt_stiffness[j] = self._eval_k
            self._model.dof_damping[self._model.jnt_dofadr[j]] = _ALPHA * np.sqrt(self._eval_k)
        for j in self._rigid_j:
            self._model.jnt_stiffness[j] = _K_RIGID
            self._model.dof_damping[self._model.jnt_dofadr[j]] = _ALPHA * np.sqrt(_K_RIGID)
        self._data.qpos[self._all_qadr] = 0.0
        self._data.qvel[self._all_dof] = 0.0
        mujoco.mj_forward(self._model, self._data)
        foot_xy = self._foot_pos_base()[:, :2]
        self._foot_hist[:] = foot_xy
        self._prev_foot_world = self._foot_world()

    def reset(self, *, seed=None, options=None):
        from environments.compliance_env import CompliantTerrainEnv
        return super(CompliantTerrainEnv, self).reset(seed=seed, options=options)

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        info["power"] = float(np.sum(np.abs(self._data.actuator_force * self._data.qvel[6:18])))
        return obs, r, term, trunc, info

    @property
    def robot_mass(self): return self._robot_mass


def make_venv(k, mode, vp):
    venv = DummyVecEnv([lambda: CleanEvalTransitionEnv(stiffness=k, obs_mode=mode,
                                                       target_lin_vel=(0.2, 0.0), use_tg=True)])
    venv = VecNormalize.load(vp, venv); venv.training = False; venv.norm_reward = False
    return venv


def run(model, venv, n, seed):
    mass = float(venv.get_attr("robot_mass")[0])
    ep_v, ep_fell, ep_cross, ep_cot, ep_sink = [], [], [], [], []
    venv.seed(seed); obs = venv.reset()
    vels, powers, steps, max_x, max_sink = [], [], 0, 0.0, 0.0
    while len(ep_v) < n:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(a); info = infos[0]
        vels.append(info["base_lin_vel_x"]); powers.append(info["power"])
        max_x = max(max_x, info["x_pos"]); steps += 1
        if 1.8 <= info["x_pos"] <= 3.05:
            max_sink = max(max_sink, info["tile_sink"])
        if dones[0]:
            dist = abs(info["x_pos"]); fell = steps < _MAX_STEPS
            ep_cot.append((float(np.sum(powers)) * _DT / (mass * _G * max(1e-3, dist)), fell))
            ep_v.append(float(np.mean(vels))); ep_fell.append(1.0 if fell else 0.0)
            ep_cross.append(1.0 if max_x > _CROSS_X else 0.0); ep_sink.append(max_sink)
            vels, powers, steps, max_x, max_sink = [], [], 0, 0.0, 0.0
    cot_clean = [c for c, f in ep_cot if not f]
    return {"forward_velocity_mean": float(np.mean(ep_v)), "fall_rate": float(np.mean(ep_fell)),
            "cross_rate": float(np.mean(ep_cross)), "tile_sink_cm": float(np.mean(ep_sink)) * 100,
            "cost_of_transport": float(np.median(cot_clean)) if cot_clean else float("nan")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2024); ap.add_argument("--out", type=str, default="evaluation/clean_transition_results.csv")
    args = ap.parse_args()
    rows = []
    for pol, (mode, base, final, seeds) in POLICY_SPECS.items():
        for sd in seeds:
            mp, vp = _ckpt(base, final, sd)
            model = PPO.load(mp, device="cpu")
            for ti, t in enumerate(CT):
                venv = make_venv(t["k"], mode, vp); m = run(model, venv, args.n, args.seed + ti); venv.close()
                m.update({"policy": pol, "seed": sd, "terrain": t["name"], "stiffness": t["k"]}); rows.append(m)
                print(f"[{pol:10s} s{sd} {t['name']:16s} k={t['k']:5d}] cross={m['cross_rate']:.2f} fall={m['fall_rate']:.2f} "
                      f"sink={m['tile_sink_cm']:.1f}cm vel={m['forward_velocity_mean']:.3f} COT={m['cost_of_transport']:.2f}")
    df = pd.DataFrame(rows); df.to_csv(args.out, index=False); print(f"\nSaved {args.out}")
    print("\n=== cross_rate (history vs no-history) ===")
    print(df.pivot_table(index="terrain", columns="policy", values="cross_rate").to_string())


if __name__ == "__main__":
    main()
