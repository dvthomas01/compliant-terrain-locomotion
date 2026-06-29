"""Causal test of the gait-speed hypothesis (§5.6).

Correlation (across seeds) says slower gaits are more robust on compliant terrain.
This intervenes on speed *within each seed* by commanding lower forward velocity
(the PMTG amplitude + tracking target scale with command), and asks: does slowing a
brittle seed down reduce its fall rate on the transition terrains?

If speed CAUSES brittleness: lowering the command lowers fall rate (esp. for the
fast/brittle seeds). If it's seed identity: command barely matters.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from evaluation.terrain_suite import TERRAINS, EvalCompliantEnv
from evaluation.evaluate import run_terrain, _ckpt_paths

TERRAIN_BY_NAME = {t["name"]: t for t in TERRAINS}
COMMANDS = [0.10, 0.15, 0.20]
TERRAINS_TO_TEST = ["T5", "T6_train_edge"]
N = 100


def make_venv_cmd(terrain, obs_mode, vecnorm_path, cmd):
    venv = DummyVecEnv([lambda: EvalCompliantEnv(terrain=terrain, obs_mode=obs_mode,
                                                 target_lin_vel=(cmd, 0.0), use_tg=True)])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False; venv.norm_reward = False
    return venv


def main():
    rows = []
    for seed in range(8):
        model_path, vecnorm_path = _ckpt_paths("policy_b", "policy_b_final.zip", seed)
        model = PPO.load(model_path, device="cpu")
        for tname in TERRAINS_TO_TEST:
            terrain = TERRAIN_BY_NAME[tname]
            for cmd in COMMANDS:
                venv = make_venv_cmd(terrain, "B", vecnorm_path, cmd)
                m = run_terrain(model, venv, N, seed=12345)
                venv.close()
                rows.append({"seed": seed, "terrain": tname, "command": cmd,
                             "fall_rate": m["fall_rate"],
                             "velocity": m["forward_velocity_mean"]})
                print(f"  B s{seed} {tname} cmd={cmd:.2f}: "
                      f"fall={m['fall_rate']:.2f} vel={m['forward_velocity_mean']:.3f}")
    df = pd.DataFrame(rows)
    out = "evaluation/speed_matched_results.csv"
    df.to_csv(out, index=False)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
