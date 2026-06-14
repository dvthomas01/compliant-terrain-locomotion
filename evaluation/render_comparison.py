"""Side-by-side A|B comparison GIFs, one per held-out terrain T0-T9.

For each terrain, rolls one deterministic episode of Policy A (left) and Policy B (right)
from the SAME seed/physics, labels each panel (policy, terrain, FELL@step), and writes a
compact GIF. Kept small (downscaled, subsampled, step-capped) vs the 330MB training GIFs.

Usage:
    python evaluation/render_comparison.py                 # all terrains
    python evaluation/render_comparison.py --terrains T5_train_edge_test  # not used; see --only
    python evaluation/render_comparison.py --only T5 --steps 300          # one terrain, quick
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evaluation.terrain_suite import TERRAINS, EvalCompliantEnv

PANEL_W = 320          # downscaled panel width (from 640); height scales proportionally
SUBSAMPLE = 2          # keep every Nth frame
FRAME_MS = 60          # gif frame duration

POLICIES = {
    "A": ("A", "checkpoints/policy_a_v24/policy_v24_final.zip", "checkpoints/policy_a_v24/vecnorm_final.pkl"),
    "B": ("B", "checkpoints/policy_b/policy_b_final.zip", "checkpoints/policy_b/vecnorm_final.pkl"),
}


def load(obs_mode, model_path, vn_path, terrain):
    model = PPO.load(model_path, device="cpu")
    vn = VecNormalize.load(vn_path, DummyVecEnv([lambda: EvalCompliantEnv(terrain=terrain, obs_mode=obs_mode)]))
    vn.training = False
    return model, vn


def rollout(terrain, obs_mode, model_path, vn_path, max_steps, seed):
    """Return (list of downscaled RGB frames, fell_step or None)."""
    model, vn = load(obs_mode, model_path, vn_path, terrain)
    env = EvalCompliantEnv(terrain=terrain, obs_mode=obs_mode, render_mode="rgb_array",
                           target_lin_vel=(0.2, 0.0), use_tg=True)
    obs, _ = env.reset(seed=seed)
    frames, fell = [], None
    for t in range(max_steps):
        action, _ = model.predict(vn.normalize_obs(obs), deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if t % SUBSAMPLE == 0:
            f = env.render()
            if f is not None:
                im = Image.fromarray(f)
                im = im.resize((PANEL_W, int(PANEL_W * im.height / im.width)))
                frames.append(im)
        if terminated or truncated:
            if terminated:
                fell = t
            break
    env.close()
    return frames, fell


def label(im, policy, terrain_name, fell, total):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
    status = f"FELL@{fell}" if fell is not None else "upright"
    color = (255, 90, 90) if fell is not None else (120, 230, 120)
    d.text((3, 3), f"Policy {policy}  {terrain_name}", fill=(255, 255, 255))
    d.text((im.width - 70, 3), status, fill=color)
    return im


def make_comparison(terrain, max_steps, seed, out_dir):
    name = terrain["name"]
    fa, fell_a = rollout(terrain, *POLICIES["A"], max_steps, seed)
    fb, fell_b = rollout(terrain, *POLICIES["B"], max_steps, seed)
    if not fa or not fb:
        print(f"  {name}: no frames"); return
    n = max(len(fa), len(fb))
    fa += [fa[-1]] * (n - len(fa))          # freeze the fallen one so panels stay aligned
    fb += [fb[-1]] * (n - len(fb))
    combined = []
    for a, b in zip(fa, fb):
        a = label(a.copy(), "A (rigid)", name, fell_a, max_steps)
        b = label(b.copy(), "B (compliance)", name, fell_b, max_steps)
        h = max(a.height, b.height)
        canvas = Image.new("RGB", (a.width + b.width + 4, h), (30, 30, 30))
        canvas.paste(a, (0, 0)); canvas.paste(b, (a.width + 4, 0))
        combined.append(canvas)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.gif")
    combined[0].save(path, save_all=True, append_images=combined[1:], duration=FRAME_MS, loop=0)
    sz = os.path.getsize(path) / 1e6
    print(f"  {name}: A {'FELL@'+str(fell_a) if fell_a is not None else 'upright'} | "
          f"B {'FELL@'+str(fell_b) if fell_b is not None else 'upright'}  -> {path} ({sz:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None, help="substring of terrain name to render just one")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="renders/comparison")
    a = ap.parse_args()
    terrains = [t for t in TERRAINS if (a.only is None or a.only in t["name"])]
    print(f"rendering {len(terrains)} comparison GIF(s) -> {a.out}")
    for t in terrains:
        make_comparison(t, a.steps, a.seed, a.out)
    print("done")


if __name__ == "__main__":
    main()
