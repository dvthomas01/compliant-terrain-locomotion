"""Tier 2 blog schematics, no data, pure drawing.

  blog/figures/pmtg_trot_schematic.png  -- diagonal trot pairing + foot-path ellipse
  blog/figures/stiffness_gauge.png      -- the T0..T9 softness axis the floors hide

    source .venv/bin/activate && PYTHONPATH="$PWD" python analysis/plot_schematics.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch

OUT = "blog/figures"
os.makedirs(OUT, exist_ok=True)

INK = "#31302e"
FAINT = "#a39e98"
PAIR1 = "#2563a8"   # FR + RL
PAIR2 = "#dd5b00"   # FL + RR

plt.rcParams.update({
    "figure.facecolor": "white",
    "font.family": "sans-serif",
    "text.color": INK,
})


def pmtg_trot():
    fig, (axr, axe) = plt.subplots(1, 2, figsize=(9.2, 4.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- left: top-down robot, diagonal pairing ----
    axr.set_xlim(0, 10)
    axr.set_ylim(0, 10)
    axr.set_aspect("equal")
    axr.axis("off")
    axr.set_title("Diagonal trot, by construction", fontsize=13, color="#000",
                  fontweight="bold", pad=6)

    # body
    body = FancyBboxPatch((3.1, 3.0), 3.8, 4.0,
                          boxstyle="round,pad=0.1,rounding_size=0.6",
                          facecolor="#f0eeec", edgecolor="#cfcdca", linewidth=1.5)
    axr.add_patch(body)
    axr.annotate("", xy=(5.0, 8.4), xytext=(5.0, 7.2),
                 arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=1.6))
    axr.text(5.0, 8.7, "forward", ha="center", fontsize=10, color=FAINT)

    # foot positions: FL, FR (front), RL, RR (rear)
    feet = {
        "FL": (3.0, 7.0, PAIR2), "FR": (7.0, 7.0, PAIR1),
        "RL": (3.0, 3.0, PAIR1), "RR": (7.0, 3.0, PAIR2),
    }
    for name, (x, y, color) in feet.items():
        axr.plot([5.0 if "F" in name else 5.0], [y], alpha=0)  # keep limits
        # leg link from body corner to foot
        bx = 3.1 if "L" in name else 6.9
        by = 6.6 if "F" in name else 3.4
        axr.plot([bx, x], [by, y], color=color, lw=2.2, zorder=1)
        axr.scatter([x], [y], s=420, color=color, edgecolor="white",
                    linewidth=2.0, zorder=3)
        axr.text(x, y, name, ha="center", va="center", color="white",
                 fontsize=10.5, fontweight="bold", zorder=4)

    # diagonal phase links
    axr.plot([3.0, 7.0], [3.0, 7.0], color=PAIR1, ls="--", lw=1.6, alpha=0.6, zorder=0)
    axr.plot([3.0, 7.0], [7.0, 3.0], color=PAIR2, ls="--", lw=1.6, alpha=0.6, zorder=0)

    axr.text(1.0, 1.2, "Pair 1  (FR + RL)", color=PAIR1, fontsize=10.5,
             fontweight="bold")
    axr.text(6.0, 1.2, "Pair 2  (FL + RR)", color=PAIR2, fontsize=10.5,
             fontweight="bold")
    axr.text(5.0, 0.2, "the two pairs cycle 180° apart", ha="center",
             fontsize=10, color=FAINT)

    # ---- right: foot-path ellipse ----
    axe.set_xlim(-1.5, 1.5)
    axe.set_ylim(-0.8, 1.4)
    axe.set_aspect("equal")
    axe.axis("off")
    axe.set_title("Each foot traces an ellipse", fontsize=13, color="#000",
                  fontweight="bold", pad=6)

    # ground line
    axe.plot([-1.4, 1.4], [0, 0], color="#cfcdca", lw=2.0)
    axe.text(1.4, -0.18, "ground", ha="right", color=FAINT, fontsize=9.5)

    # ellipse path: stance along ground, swing arc above
    th = np.linspace(0, 2 * np.pi, 200)
    ex, ey = 1.05 * np.cos(th), 0.55 * np.sin(th) + 0.55
    axe.plot(ex, ey, color=INK, lw=2.2)
    # direction arrows on the loop
    for frac, lab in [(0.25, None), (0.75, None)]:
        i = int(frac * len(th))
        axe.annotate("", xy=(ex[i + 3], ey[i + 3]), xytext=(ex[i], ey[i]),
                     arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
    axe.text(0, 1.25, "swing: knee flexes, foot clears", ha="center",
             fontsize=9.8, color=PAIR1)
    axe.text(0, -0.5, "stance: foot pushes back → forward thrust", ha="center",
             fontsize=9.8, color=PAIR2)

    fig.tight_layout()
    path = os.path.join(OUT, "pmtg_trot_schematic.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def stiffness_gauge():
    fig, ax = plt.subplots(figsize=(9.2, 2.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    labels = [f"T{i}" for i in range(10)]
    xs = np.linspace(0.6, 9.4, 10)

    # gradient bar firm -> soft
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, extent=(0.4, 9.6, 1.5, 2.1), aspect="auto",
              cmap="YlOrBr", alpha=0.85, zorder=1)
    ax.add_patch(plt.Rectangle((0.4, 1.5), 9.2, 0.6, fill=False,
                               edgecolor="#cfcdca", lw=1.2, zorder=2))

    for x, lab in zip(xs, labels):
        ax.plot([x, x], [1.5, 2.1], color="white", lw=1.0, alpha=0.6, zorder=2)
        ax.text(x, 1.25, lab, ha="center", va="top", fontsize=11, color=INK)

    # training edge marker at T6
    xe = xs[6]
    ax.annotate("", xy=(xe, 2.1), xytext=(xe, 2.75),
                arrowprops=dict(arrowstyle="-|>", color="#b3261e", lw=2.0), zorder=4)
    ax.text(xe, 2.85, "training edge (T6)", ha="center", fontsize=10.5,
            color="#b3261e", fontweight="bold")

    ax.text(0.4, 2.45, "firm", ha="left", fontsize=11, color="#7a4a00",
            fontweight="bold")
    ax.text(9.6, 2.45, "soft", ha="right", fontsize=11, color="#dd5b00",
            fontweight="bold")
    ax.text(5.0, 0.55,
            "Every floor renders identically. Only the contact-solver stiffness "
            "changes across T0–T9.",
            ha="center", fontsize=10.5, color=FAINT)

    fig.tight_layout()
    path = os.path.join(OUT, "stiffness_gauge.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    pmtg_trot()
    stiffness_gauge()
