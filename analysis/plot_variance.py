"""Hero figure: the whole story in two panels.

Panel A (eval, 8 seeds): fall rate vs terrain softness for A, B, B' with shaded
  seed-variance bands. Monotonic B' > B > A robustness; wide bands at T4-T7 = the
  bimodal seed variance.
Panel B (training, 8 seeds): final terrain_level by recipe — both B and B' have a
  stalled seed; reliability fixes did not help.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["T0_rigid","T1_firm","T2","T3","T4","T5","T6_train_edge","T7_extrap","T8_extrap","T9_extrap_soft"]
EXTRAP = 7
C_A, C_B, C_BP = "#c0392b", "#2980b9", "#27ae60"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

# ---------- Panel A: fall rate vs softness, mean ± shaded seed-std ----------
ev = pd.read_csv("evaluation/ablation_results_multiseed.csv")
x = np.arange(len(ORDER))
ax1.axvspan(EXTRAP - 0.5, len(ORDER) - 0.5, color="0.92")

def band(policy, color, label, marker):
    sub = ev[ev.policy == policy]
    n = sub.seed.nunique()
    mean = np.array([sub[sub.terrain == t]["fall_rate"].mean() for t in ORDER])
    std  = np.array([sub[sub.terrain == t]["fall_rate"].std()  for t in ORDER])
    ax1.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.15)
    ax1.plot(x, mean, marker + "-", color=color, lw=2.4, label=f"{label} (n={n})")

band("A_rigid", C_A, "A  rigid-trained", "o")
band("B_compliance", C_B, "B  compliance + history", "s")
band("Bp_noh", C_BP, "B'  compliance, no history", "^")
ax1.set_xticks(x); ax1.set_xticklabels(ORDER, rotation=45, ha="right")
ax1.set_ylim(-0.05, 1.08); ax1.set_ylabel("fall rate (N=100, mean ± seed std)")
ax1.set_title("Robustness vs terrain softness (8 seeds):\nB' > B > A; wide bands at T4-T7 = bimodal seed variance")
ax1.legend(loc="upper left", fontsize=9)
ax1.annotate("training range", (0.1, 1.02), fontsize=8, color="0.4")
ax1.annotate("extrapolation", (EXTRAP + 0.05, 1.02), fontsize=8, color="0.4")

# ---------- Panel B: 8-seed terrain_level convergence ----------
cv = pd.read_csv("evaluation/convergence_variance.csv")
RECIPES = ["B_baseline", "Bp_noh", "Cur_gated", "Entropy"]
RLABEL = {"B_baseline": "B\n(8 seeds)", "Bp_noh": "B'\n(8 seeds)",
          "Cur_gated": "competence\ngated (3)", "Entropy": "entropy\n0.005 (3)"}
ROBUST, STALL = 2.0, 1.4
ax2.axhspan(ROBUST, 3.0, color=C_B, alpha=0.07)
ax2.axhspan(0, STALL, color=C_A, alpha=0.08)
ax2.text(3.45, 2.5, "robust basin", color=C_B, fontsize=8, va="center", rotation=90)
ax2.text(3.45, 0.7, "stalled", color=C_A, fontsize=8, va="center", rotation=90)

rng = np.random.default_rng(0)
for i, r in enumerate(RECIPES):
    vals = cv[cv.recipe == r]["terrain_level_final2M"].values
    jitter = rng.uniform(-0.09, 0.09, size=len(vals))
    colors = [C_A if v < STALL else C_B for v in vals]
    ax2.scatter(np.full(len(vals), i) + jitter, vals, c=colors, s=80,
                edgecolor="k", linewidth=0.6, zorder=3)
ax2.set_xticks(range(len(RECIPES)))
ax2.set_xticklabels([RLABEL[r] for r in RECIPES], fontsize=9)
ax2.set_ylim(0, 3.0); ax2.set_ylabel("final terrain_level (3 = softest curriculum)")
ax2.set_title("Training convergence: both B and B' have a stalled seed;\nreliability fixes did not close it")
ax2.margins(x=0.12)

fig.suptitle("Rigid-trained A is fast but brittle; compliance training extends survivable softness "
             "(B' > B > A) — but acquisition is seed-variable",
             fontweight="bold", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = "analysis/variance_plot.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
