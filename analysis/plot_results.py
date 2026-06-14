"""Plot the A-vs-B compliance ablation: fall-rate and forward-velocity vs terrain."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["T0_rigid","T1_firm","T2","T3","T4","T5","T6_train_edge","T7_extrap","T8_extrap","T9_extrap_soft"]
df = pd.read_csv("evaluation/ablation_results_3way.csv")
df["terrain"] = pd.Categorical(df["terrain"], ORDER, ordered=True)
A = df[df.policy=="A_rigid"].set_index("terrain").sort_index()
B = df[df.policy=="B_compliance"].set_index("terrain").sort_index()
Bp = df[df.policy=="Bp_noh"].set_index("terrain").sort_index()
x = np.arange(len(ORDER))
EXTRAP = 7  # T7+ is beyond training range

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for ax in (ax1, ax2):
    ax.axvspan(EXTRAP-0.5, len(ORDER)-0.5, color="0.92", label="_")
    ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=45, ha="right")
    ax.text(EXTRAP+0.9, ax.get_ylim()[1] if False else 0, "")

ax1.plot(x, A.fall_rate, "o-", color="#c0392b", label="A (rigid-trained)", lw=2)
ax1.plot(x, Bp.fall_rate, "^--", color="#27ae60", label="B′ (compliance, no history)", lw=2)
ax1.plot(x, B.fall_rate, "s-", color="#2980b9", label="B (compliance + history)", lw=2)
ax1.set_title("Fall rate vs terrain compliance (N=100)"); ax1.set_ylabel("fall rate")
ax1.set_ylim(-0.05, 1.05); ax1.legend(loc="center left")
ax1.annotate("training range", (3, 1.02)); ax1.annotate("extrapolation", (EXTRAP+0.1, 1.02))

ax2.errorbar(x, A.forward_velocity_mean, yerr=A.forward_velocity_std, fmt="o-", color="#c0392b", label="A (rigid)", lw=2, capsize=3)
ax2.errorbar(x, Bp.forward_velocity_mean, yerr=Bp.forward_velocity_std, fmt="^--", color="#27ae60", label="B′ (compliance, no history)", lw=2, capsize=3)
ax2.errorbar(x, B.forward_velocity_mean, yerr=B.forward_velocity_std, fmt="s-", color="#2980b9", label="B (compliance + history)", lw=2, capsize=3)
ax2.axhline(0.2, ls="--", color="0.5", label="commanded 0.2 m/s")
ax2.set_title("Forward velocity vs terrain compliance"); ax2.set_ylabel("forward velocity (m/s)"); ax2.legend()

fig.suptitle("Ablation: terrain (A vs B′/B) and observation (B′ no-history vs B +history)", fontweight="bold")
fig.tight_layout()
out = "analysis/ablation_plot_3way.png"; fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
