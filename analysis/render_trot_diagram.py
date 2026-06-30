"""Trot diagram for the blog: real Go1 top and side renders plus the phase clock.

Poses the Go1 at gait phase pi/2 (the FR+RL diagonal in swing, FL+RR in stance),
renders a top down and a side view, and pairs them with the phase signal that drives
the diagonal trot. Replaces the earlier hand drawn schematic.

    source .venv/bin/activate && PYTHONPATH="$PWD" python analysis/render_trot_diagram.py
"""
import os

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from environments.base_env import NOMINAL_JOINT_POS
from evaluation.terrain_suite import TERRAINS, EvalCompliantEnv

OUT = "blog/figures"
PAIR1 = "#2563a8"   # FR + RL
PAIR2 = "#dd5b00"   # FL + RR
INK = "#31302e"
FAINT = "#a39e98"


def render_views():
    t0 = [t for t in TERRAINS if t["name"] == "T0_rigid"][0]
    env = EvalCompliantEnv(terrain=t0, obs_mode="A", render_mode="rgb_array",
                           target_lin_vel=(0.2, 0.0), use_tg=True)
    env.reset(seed=0)
    env._phase = np.pi / 2.0           # FR+RL swing, FL+RR stance
    env._data.qpos[7:19] = NOMINAL_JOINT_POS + env._tg_offsets()
    env._data.qpos[2] = 0.34           # lift trunk so swung feet read clearly
    mujoco.mj_forward(env._model, env._data)
    r = mujoco.Renderer(env._model, height=480, width=640)
    trunk = env._data.xpos[env._trunk_id].copy()

    def shot(az, el, dist, box):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = trunk
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        r.update_scene(env._data, camera=cam)
        return Image.fromarray(r.render()).crop(box)

    top = shot(90, -89, 1.3, (150, 120, 510, 370))
    side = shot(90, -12, 1.7, (150, 170, 520, 360))
    env.close()
    return top, side


def forward_arrow(ax):
    ax.annotate("forward", xy=(0.93, 0.5), xytext=(0.72, 0.5),
                xycoords="axes fraction", ha="center", va="center",
                fontsize=10, color="#ffffff",
                arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=2.0))


def main():
    top, side = render_views()
    fig = plt.figure(figsize=(9.2, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.28, wspace=0.12)

    axt = fig.add_subplot(gs[0, 0])
    axt.imshow(top)
    axt.set_title("Go1, top view", fontsize=12, color="#000", fontweight="bold")
    axt.axis("off")
    forward_arrow(axt)

    axs = fig.add_subplot(gs[0, 1])
    axs.imshow(side)
    axs.set_title("Go1, side view", fontsize=12, color="#000", fontweight="bold")
    axs.axis("off")
    forward_arrow(axs)

    # phase clock: the two diagonal pairs are exactly antiphase
    axp = fig.add_subplot(gs[1, :])
    phi = np.linspace(0, 4 * np.pi, 400)
    lift1 = np.maximum(0.0, np.sin(phi))            # FR + RL  (theta = phi)
    lift2 = np.maximum(0.0, np.sin(phi + np.pi))    # FL + RR  (theta = phi + pi)
    axp.fill_between(phi, 0, lift1, color=PAIR1, alpha=0.18)
    axp.fill_between(phi, 0, lift2, color=PAIR2, alpha=0.18)
    axp.plot(phi, lift1, color=PAIR1, lw=2.4, label="FR + RL swing  (θ = φ)")
    axp.plot(phi, lift2, color=PAIR2, lw=2.4, label="FL + RR swing  (θ = φ + π)")
    axp.set_xlim(0, 4 * np.pi)
    axp.set_ylim(-0.05, 1.15)
    axp.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
    axp.set_xticklabels(["0", "π", "2π", "3π", "4π"])
    axp.set_xlabel("gait phase  φ")
    axp.set_ylabel("foot lift,  max(0, sin θ)")
    axp.set_title("One pair lifts while the other carries the body. That alternation is the trot.",
                  fontsize=11.5, color=INK)
    axp.legend(loc="upper right", frameon=False, fontsize=10.5, ncol=2)
    axp.spines["top"].set_visible(False)
    axp.spines["right"].set_visible(False)
    axp.grid(True, color="#ececec")
    axp.set_axisbelow(True)

    fig.savefig(os.path.join(OUT, "trot_diagram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.join(OUT, "trot_diagram.png"))


if __name__ == "__main__":
    main()
