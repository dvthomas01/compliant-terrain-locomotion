"""Plot the transition obs-ablation: history (89D) vs no-history (49D) on rigid->soft->rigid."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ORDER=["TT0_firm","TT1","TT2","TT3","TT4_train_soft","TT5_extrap","TT6_extrap","TT7_extrap_soft"]
df=pd.read_csv("evaluation/transition_results.csv"); df["terrain"]=pd.Categorical(df["terrain"],ORDER,ordered=True)
H=df[df.policy=="Btrans_hist"].set_index("terrain").sort_index()
N=df[df.policy=="Btrans_nohist"].set_index("terrain").sort_index()
x=np.arange(len(ORDER)); EXTRAP=5
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
for ax in (ax1,ax2):
    ax.axvspan(EXTRAP-0.5,len(ORDER)-0.5,color="0.92"); ax.set_xticks(x); ax.set_xticklabels(ORDER,rotation=45,ha="right")
ax1.plot(x,H.cross_rate,"s-",color="#2980b9",lw=2,label="B_trans (89D, history)")
ax1.plot(x,N.cross_rate,"^--",color="#e67e22",lw=2,label="B_trans_noh (49D, no history)")
ax1.set_title("Soft-zone crossing rate (robustness): history ≈ no-history"); ax1.set_ylabel("cross rate"); ax1.set_ylim(-0.05,1.05); ax1.legend(loc="lower right")
ax1.text(EXTRAP+0.1,1.02,"extrapolation")
ax2.plot(x,H.cost_of_transport,"s-",color="#2980b9",lw=2,label="B_trans (89D, history)")
ax2.plot(x,N.cost_of_transport,"^--",color="#e67e22",lw=2,label="B_trans_noh (49D, no history)")
ax2.set_title("Cost of transport (efficiency): history ~25% cheaper"); ax2.set_ylabel("cost of transport"); ax2.legend()
ax2.text(EXTRAP+0.1,ax2.get_ylim()[1]*0.97,"extrapolation")
fig.suptitle("Transition obs-ablation: foot-history buys EFFICIENCY, not survival (N=100)",fontweight="bold")
fig.tight_layout(); fig.savefig("analysis/transition_plot.png",dpi=130,bbox_inches="tight"); print("saved analysis/transition_plot.png")
