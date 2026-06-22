"""Option-B figure: the honest variance story.

Panel A (eval): fall-rate vs terrain compliance, PER SEED. Policy A fails
  categorically on compliant terrain; Policy B / B' are robust but seed-variable.
Panel B (training): final terrain_level by recipe — the bimodal "robust vs
  stalled" convergence that survives two reliability interventions.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["T0_rigid","T1_firm","T2","T3","T4","T5","T6_train_edge","T7_extrap","T8_extrap","T9_extrap_soft"]
EXTRAP = 7  # T7+ is beyond the training compliance range
C_A, C_B, C_BP = "#c0392b", "#2980b9", "#27ae60"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

# ---------- Panel A: per-seed fall rate (eval) ----------
ev = pd.read_csv("evaluation/ablation_results_multiseed.csv")
ev["terrain"] = pd.Categorical(ev["terrain"], ORDER, ordered=True)
x = np.arange(len(ORDER))

def seed_curves(policy, color, label, marker):
    sub = ev[ev.policy == policy]
    seeds = sorted(sub.seed.unique())
    mat = []
    for s in seeds:
        d = sub[sub.seed == s].set_index("terrain").sort_index()
        y = d["fall_rate"].reindex(ORDER).values.astype(float)
        mat.append(y)
        ax1.plot(x, y, color=color, lw=0.9, alpha=0.35)
    mat = np.array(mat)
    ax1.plot(x, mat.mean(0), marker + "-", color=color, lw=2.4,
             label=f"{label} (n={len(seeds)} seed{'s' if len(seeds)>1 else ''})")

ax1.axvspan(EXTRAP - 0.5, len(ORDER) - 0.5, color="0.92")
seed_curves("A_rigid", C_A, "A (rigid-trained)", "o")
seed_curves("Bp_noh", C_BP, "B' (compliance, no history)", "^")
seed_curves("B_compliance", C_B, "B (compliance + history)", "s")
ax1.set_xticks(x); ax1.set_xticklabels(ORDER, rotation=45, ha="right")
ax1.set_ylim(-0.05, 1.08); ax1.set_ylabel("fall rate (N=100)")
ax1.set_title("Held-out robustness: A fails categorically;\nB / B' robust but seed-variable")
ax1.legend(loc="center left", fontsize=9)
ax1.annotate("training range", (0.1, 1.02), fontsize=8, color="0.4")
ax1.annotate("extrapolation", (EXTRAP + 0.05, 1.02), fontsize=8, color="0.4")

# ---------- Panel B: terrain_level convergence by recipe ----------
cv = pd.read_csv("evaluation/convergence_variance.csv")
RECIPES = ["B_baseline", "Bp_noh", "Cur_gated", "Entropy"]
RLABEL = {"B_baseline": "B\nbaseline", "Bp_noh": "B'\nno-history",
          "Cur_gated": "competence\ngated", "Entropy": "entropy\n0.005"}
ROBUST, STALL = 2.0, 1.4
ax2.axhspan(ROBUST, 3.0, color=C_B, alpha=0.07)
ax2.axhspan(0, STALL, color=C_A, alpha=0.08)
ax2.text(3.45, (ROBUST + 3.0) / 2, "robust basin", color=C_B, fontsize=8, va="center", rotation=90)
ax2.text(3.45, STALL / 2, "stalled", color=C_A, fontsize=8, va="center", rotation=90)

rng = np.random.default_rng(0)
for i, r in enumerate(RECIPES):
    vals = cv[cv.recipe == r]["terrain_level_final2M"].values
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    colors = [C_A if v < STALL else C_B for v in vals]
    ax2.scatter(np.full(len(vals), i) + jitter, vals, c=colors, s=90,
                edgecolor="k", linewidth=0.6, zorder=3)
ax2.set_xticks(range(len(RECIPES)))
ax2.set_xticklabels([RLABEL[r] for r in RECIPES], fontsize=9)
ax2.set_ylim(0, 3.0); ax2.set_ylabel("final terrain_level (3 = softest curriculum)")
ax2.set_title("Convergence is bimodal across seeds —\ntwo reliability fixes do not close it")
ax2.margins(x=0.12)

fig.suptitle("Compliance training achieves robustness, but acquisition is high-variance",
             fontweight="bold", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = "analysis/variance_plot.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
