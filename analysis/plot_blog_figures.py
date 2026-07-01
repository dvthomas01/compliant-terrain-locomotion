"""Tier 1 blog figures, generated from the committed eval CSVs.

Produces seven clean, Notion-styled PNGs into blog/figures/. Every quantity is read
straight from the CSVs so the figures reconcile with results.md by construction. Run
prints the survival reconciliation so a reviewer can eyeball it against the doc.

    source .venv/bin/activate && PYTHONPATH="$PWD" python analysis/plot_blog_figures.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "blog/figures"
os.makedirs(OUT, exist_ok=True)

ORDER = ["T0_rigid", "T1_firm", "T2", "T3", "T4", "T5",
         "T6_train_edge", "T7_extrap", "T8_extrap", "T9_extrap_soft"]
TLAB = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
TRANS = ["T4", "T5", "T6_train_edge", "T7_extrap"]
EDGE = 6  # T6 is the training edge

# Policy identity, kept consistent with the existing research figures.
C_A, C_B, C_BP = "#c0392b", "#2563a8", "#1f9d57"
POLS = [
    ("A_rigid", C_A, "A  rigid", "o"),
    ("B_compliance", C_B, "B  compliance + history", "s"),
    ("Bp_noh", C_BP, "B'  compliance, no history", "^"),
]

# ---- shared clean style -----------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 12.5,
    "font.family": "sans-serif",
    "axes.edgecolor": "#cfcdca",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "#ececec",
    "grid.linewidth": 1.0,
    "xtick.color": "#615d59",
    "ytick.color": "#615d59",
    "axes.labelcolor": "#31302e",
    "text.color": "#31302e",
})


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def load():
    ev = pd.read_csv("evaluation/ablation_results_multiseed.csv")
    sm = pd.read_csv("evaluation/speed_matched_results.csv")
    cv = pd.read_csv("evaluation/convergence_variance.csv")
    return ev, sm, cv


def per_seed(ev):
    rows = []
    for (pol, seed), g in ev.groupby(["policy", "seed"]):
        gt = g[g.terrain.isin(TRANS)]
        rows.append({
            "policy": pol, "seed": seed,
            "nsurv": int((g.fall_rate < 0.5).sum()),
            "vel_trans": gt.forward_velocity_mean.mean(),
            "cvar_trans": gt.foot_contact_variance.mean(),
        })
    return pd.DataFrame(rows)


# ============================================================================
# 1 + 2. hero variance, plain and annotated
# ============================================================================
def fall_curve(ax, ev):
    x = np.arange(len(ORDER))
    for pol, color, label, mk in POLS:
        sub = ev[ev.policy == pol]
        mean = np.array([sub[sub.terrain == t].fall_rate.mean() for t in ORDER])
        std = np.array([sub[sub.terrain == t].fall_rate.std() for t in ORDER])
        ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                        color=color, alpha=0.13, linewidth=0)
        ax.plot(x, mean, mk + "-", color=color, lw=2.6, ms=7, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(TLAB)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("terrain softness, firm → soft")
    ax.set_ylabel("fall rate")
    clean(ax)


def hero_variance(ev):
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fall_curve(ax, ev)
    ax.legend(loc="upper left", frameon=False, fontsize=11.5)
    save(fig, "hero_variance.png")


def hero_variance_annotated(ev):
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.axvspan(-0.5, EDGE + 0.5, color="#eaf3ec", zorder=0)
    ax.axvspan(EDGE + 0.5, len(ORDER) - 0.5, color="#f3ece9", zorder=0)
    fall_curve(ax, ev)
    ax.axvline(EDGE + 0.5, color="#a39e98", lw=1.2, ls="--", zorder=1)
    ax.text(EDGE / 2, 1.0, "trained on this range", ha="center", fontsize=10.5,
            color="#1f9d57")
    ax.text((EDGE + 1 + len(ORDER) - 1) / 2, 1.0, "extrapolation", ha="center",
            fontsize=10.5, color="#b3261e")
    ax.legend(loc="center left", frameon=False, fontsize=11.5)
    save(fig, "hero_variance_annotated.png")


# ============================================================================
# 3. T0 home-turf performance: velocity and cost of transport
# ============================================================================
def t0_performance(ev):
    t0 = ev[ev.terrain == "T0_rigid"].groupby("policy")
    vel = t0.forward_velocity_mean.mean()
    cot = t0.cost_of_transport.mean()
    labels = ["A\nrigid", "B\ncompliance", "B'\nno history"]
    keys = ["A_rigid", "B_compliance", "Bp_noh"]
    colors = [C_A, C_B, C_BP]
    fig, (axv, axc) = plt.subplots(1, 2, figsize=(8.6, 4.2))
    for ax, series, title, unit in [
        (axv, vel, "forward velocity", "m/s"),
        (axc, cot, "cost of transport", "dimensionless"),
    ]:
        vals = [series[k] for k in keys]
        bars = ax.bar(labels, vals, color=colors, width=0.62)
        ax.set_title(title, fontsize=13, color="#31302e", pad=8)
        ax.set_ylabel(unit)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=11, color="#31302e")
        ax.margins(y=0.18)
        clean(ax)
    fig.suptitle("On rigid ground, the rigid policy A wins on both",
                 fontsize=13.5, fontweight="bold", color="#000")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "t0_performance.png")


# ============================================================================
# 4 + 5. per-seed scatters: speed and contact-variance vs robustness
# ============================================================================
def scatter_vs_robustness(ps, xcol, xlabel, name):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    rng = np.random.default_rng(0)
    for pol, color, label, mk in POLS:
        sub = ps[ps.policy == pol]
        jit = rng.uniform(-0.12, 0.12, size=len(sub))
        ax.scatter(sub[xcol], sub.nsurv + jit, s=95, marker=mk,
                   facecolor=color, edgecolor="white", linewidth=1.0,
                   alpha=0.9, label=label, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("terrains survived (of 10)")
    ax.set_ylim(3.2, 8.8)
    ax.legend(loc="best", frameon=False, fontsize=11)
    clean(ax)
    save(fig, name)


# ============================================================================
# 6. speed-matched intervention: brittle vs robust at matched command
# ============================================================================
def speed_matched(sm):
    brittle = [1, 2]
    robust = [0, 3, 5, 6, 7]
    t5 = sm[sm.terrain == "T5"]
    cmds = [0.10, 0.15, 0.20]
    bvals = [t5[(t5.command == c) & (t5.seed.isin(brittle))].fall_rate.mean() for c in cmds]
    rvals = [t5[(t5.command == c) & (t5.seed.isin(robust))].fall_rate.mean() for c in cmds]
    x = np.arange(len(cmds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(x - w / 2, bvals, w, color=C_A, label="brittle seeds")
    ax.bar(x + w / 2, rvals, w, color=C_BP, label="robust seeds")
    for i, (bv, rv) in enumerate(zip(bvals, rvals)):
        ax.text(i - w / 2, bv, f"{bv:.2f}", ha="center", va="bottom", fontsize=10.5)
        ax.text(i + w / 2, rv, f"{rv:.2f}", ha="center", va="bottom", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"command\n{c:.2f} m/s" for c in cmds])
    ax.set_ylabel("fall rate on T5")
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("driven to the same speed")
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    clean(ax)
    save(fig, "speed_matched.png")


# ============================================================================
# 7. convergence vs robustness orthogonality
# ============================================================================
def convergence_orthogonality(ev, cv, ps):
    recipe_map = {"B_baseline": "B_compliance", "Bp_noh": "Bp_noh"}
    nsurv = {(r.policy, r.seed): r.nsurv for r in ps.itertuples()}
    cats = ["B_baseline", "Bp_noh"]
    catlab = {"B_baseline": "B\ncompliance + history", "Bp_noh": "B'\nno history"}
    robust_c, brittle_c = "#1f9d57", "#c0392b"
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    rng = np.random.default_rng(1)
    for i, cat in enumerate(cats):
        sub = cv[cv.recipe == cat]
        for row in sub.itertuples():
            ns = nsurv.get((recipe_map[cat], row.seed))
            if ns is None:
                continue
            color = robust_c if ns >= 6 else brittle_c
            jit = rng.uniform(-0.09, 0.09)
            ax.scatter(i + jit, row.terrain_level_final2M, s=110, color=color,
                       edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([catlab[c] for c in cats], fontsize=11.5)
    ax.set_ylabel("final curriculum level reached")
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0.0, 3.15)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="white", markerfacecolor=robust_c,
                  markersize=11, label="robust at evaluation, ≥6 survived"),
           Line2D([0], [0], marker="o", color="white", markerfacecolor=brittle_c,
                  markersize=11, label="brittle at evaluation, <6 survived")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              frameon=False, fontsize=11, ncol=2)
    clean(ax)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "convergence_orthogonality.png")


def reconcile(ev):
    print("\n=== survival reconciliation (fall_rate<0.5), compare to results.md ===")
    piv = (ev.assign(s=(ev.fall_rate < 0.5))
           .pivot_table(index="policy", columns="terrain", values="s", aggfunc="sum")
           .reindex(columns=ORDER)
           .loc[["A_rigid", "B_compliance", "Bp_noh"]])
    print(piv.to_string())
    print("expected T5  A=3 B=6 B'=8 ;  T6  A=1 B=5 B'=7 ;  T7  A=0 B=1 B'=3")


def main():
    ev, sm, cv = load()
    ps = per_seed(ev)
    hero_variance(ev)
    hero_variance_annotated(ev)
    t0_performance(ev)
    scatter_vs_robustness(ps, "vel_trans", "mean velocity across T4–T7 (m/s)",
                          "velocity_vs_robustness.png")
    scatter_vs_robustness(ps, "cvar_trans", "foot-contact variance across T4–T7",
                          "contactvar_vs_robustness.png")
    speed_matched(sm)
    convergence_orthogonality(ev, cv, ps)
    reconcile(ev)


if __name__ == "__main__":
    main()
