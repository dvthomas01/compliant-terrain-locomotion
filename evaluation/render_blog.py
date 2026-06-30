"""Two comparison GIFs for the blog, built on the render_comparison helpers.

  blog/renders/A_t0_vs_t6.gif      -- Policy A: confident on T0 (left) vs faceplant on T6 (right)
  blog/renders/Bprime_vs_B_t5.gif  -- B' robust seed (left) vs B brittle seed (right), both on T5

A physics seed is scanned so each single deterministic rollout actually shows the
intended contrast, then the two panels are composited side by side.

    source .venv/bin/activate && PYTHONPATH="$PWD" python evaluation/render_blog.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evaluation.terrain_suite import TERRAINS, EvalCompliantEnv
from evaluation.render_comparison import label

# Web-sized overrides: smaller panels, fewer frames, quantized palette so the
# GIFs land at a few MB instead of ~16 MB.
PANEL_W = 300
SUBSAMPLE = 4
FRAME_MS = 80
GIF_COLORS = 96

OUT = "blog/renders"
os.makedirs(OUT, exist_ok=True)
T_BY_NAME = {t["name"]: t for t in TERRAINS}

# (obs_mode, model_path, vecnorm_path)
A_RIGID = ("A", "checkpoints/policy_a_v24/policy_v24_final.zip",
           "checkpoints/policy_a_v24/vecnorm_final.pkl")
BP_ROBUST = ("A", "checkpoints/policy_b_noh_s7/policy_b_noh_final.zip",
             "checkpoints/policy_b_noh_s7/vecnorm_final.pkl")
B_BRITTLE = ("B", "checkpoints/policy_b_s1/policy_b_final.zip",
             "checkpoints/policy_b_s1/vecnorm_final.pkl")


def make_env(terrain_name, obs_mode):
    return EvalCompliantEnv(terrain=T_BY_NAME[terrain_name], obs_mode=obs_mode,
                            render_mode="rgb_array", target_lin_vel=(0.2, 0.0),
                            use_tg=True)


def load(obs_mode, model_path, vn_path):
    model = PPO.load(model_path, device="cpu")
    # the vecnorm only supplies running obs stats; its inner env terrain is irrelevant
    vn = VecNormalize.load(vn_path, DummyVecEnv([lambda: make_env("T0_rigid", obs_mode)]))
    vn.training = False
    return model, vn


def rollout(model, vn, terrain_name, obs_mode, max_steps, seed):
    """Return (frames, fell_step_or_None) for one deterministic episode."""
    env = make_env(terrain_name, obs_mode)
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


def compose(left, right, out_name, max_steps):
    """left/right are dicts: frames, fell, policy_label, terrain_label."""
    n = max(len(left["frames"]), len(right["frames"]))
    for side in (left, right):
        side["frames"] += [side["frames"][-1]] * (n - len(side["frames"]))
    combined = []
    for a, b in zip(left["frames"], right["frames"]):
        a = label(a.copy(), left["policy"], left["terrain"], left["fell"], max_steps)
        b = label(b.copy(), right["policy"], right["terrain"], right["fell"], max_steps)
        h = max(a.height, b.height)
        canvas = Image.new("RGB", (a.width + b.width + 4, h), (30, 30, 30))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width + 4, 0))
        combined.append(canvas.quantize(colors=GIF_COLORS, method=Image.MEDIANCUT))
    path = os.path.join(OUT, out_name)
    combined[0].save(path, save_all=True, append_images=combined[1:],
                     duration=FRAME_MS, loop=0, optimize=True, disposal=2)
    sz = os.path.getsize(path) / 1e6
    print(f"  wrote {path}  ({sz:.1f} MB, {len(combined)} frames)")


def scan_seed(model, vn, terrain, obs_mode, want_fall, steps, seeds):
    """Find a seed whose rollout fell (want_fall=True) or survived (False)."""
    for s in seeds:
        _, fell = rollout(model, vn, terrain, obs_mode, steps, s)
        if (fell is not None) == want_fall:
            return s
    return seeds[0]


def render_a_t0_vs_t6(steps=340, seeds=range(8)):
    print("A_t0_vs_t6: scanning for a seed where A walks T0 but falls on T6")
    model, vn = load(*A_RIGID)
    seed = None
    for s in seeds:
        _, fell_t0 = rollout(model, vn, "T0_rigid", "A", steps, s)
        _, fell_t6 = rollout(model, vn, "T6_train_edge", "A", steps, s)
        if fell_t0 is None and fell_t6 is not None:
            seed = s
            break
    if seed is None:
        seed = 0
    print(f"  using seed {seed}")
    fa0, fell0 = rollout(model, vn, "T0_rigid", "A", steps, seed)
    fa6, fell6 = rollout(model, vn, "T6_train_edge", "A", steps, seed)
    compose(
        {"frames": fa0, "fell": fell0, "policy": "A on T0", "terrain": "rigid"},
        {"frames": fa6, "fell": fell6, "policy": "A on T6", "terrain": "soft edge"},
        "A_t0_vs_t6.gif", steps,
    )


def render_bprime_vs_b(scan_steps=400, seeds=range(12)):
    print("Bprime_vs_B_t5: rolling robust B' (s7) and brittle B (s1) on T5")
    mp, vp = load(*BP_ROBUST)
    mb, vb = load(*B_BRITTLE)
    # pick the seed where B' survives T5 and B stays up the LONGEST before falling,
    # so the side-by-side shows real walking from both before B goes down
    best = None
    for s in seeds:
        _, f_bp = rollout(mp, vp, "T5", "A", scan_steps, s)
        _, f_b = rollout(mb, vb, "T5", "B", scan_steps, s)
        if f_bp is None and f_b is not None:
            if best is None or f_b > best[1]:
                best = (s, f_b)
    seed, fall_b = best if best else (0, 120)
    steps = min(scan_steps, fall_b + 90)   # end shortly after B falls, B' still upright
    print(f"  using seed {seed}, B falls at {fall_b}, clip length {steps}")
    fbp, fell_bp = rollout(mp, vp, "T5", "A", steps, seed)
    fb, fell_b = rollout(mb, vb, "T5", "B", steps, seed)
    compose(
        {"frames": fbp, "fell": fell_bp, "policy": "B' no history", "terrain": "T5"},
        {"frames": fb, "fell": fell_b, "policy": "B  + history", "terrain": "T5"},
        "Bprime_vs_B_t5.gif", steps,
    )


if __name__ == "__main__":
    render_a_t0_vs_t6()
    render_bprime_vs_b()
    print("done")
