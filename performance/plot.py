#!/usr/bin/env python
"""
Read bench.csv and produce PDF figures:
  fig_runtime.pdf, fig_bias.pdf, fig_flops_vs_time.pdf, fig_heatmap_lam_vs_n.pdf
"""
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import numpy as np


df = pd.read_csv("bench.csv")

# --- Fig 1: runtime vs n ----------------------------------------------------
fig, ax = plt.subplots()
for algo, g in df.groupby("algo"):
    ax.loglog(g["size"], g["time_ms"], "o", label=algo, alpha=.7)
sizes = sorted(df["size"].unique())
ax.loglog(sizes, np.array(sizes)**2 * 0.01, "k--", alpha=.3,
          label="$O(n^2)$")
ax.set_xlabel("support size n"); ax.set_ylabel("runtime [ms]"); ax.legend()
fig.savefig("fig_runtime.pdf", bbox_inches="tight")

# --- Fig 2: bias vs λ -------------------------------------------------------
sub = df.query("algo=='Sinkhorn' and p==2 and size==100")
bias = abs(sub.cost - sub.exact_cost) / sub.exact_cost
plt.figure(); plt.semilogx(sub.lam, bias, "o-")
plt.xlabel("λ"); plt.ylabel("relative bias")
plt.savefig("fig_bias.pdf", bbox_inches="tight")

# --- Fig 3: FLOPs vs time ---------------------------------------------------
sink = df[df.algo=="Sinkhorn"].dropna()
plt.figure(); plt.loglog(sink.flops, sink.time_ms, "o")
plt.xlabel("FLOPs (est.)"); plt.ylabel("runtime [ms]")
plt.savefig("fig_flops_vs_time.pdf", bbox_inches="tight")

# --- Fig 4: heat-map --------------------------------------------------------
plt.figure()   
heat = sink.query("p==2").pivot_table(index="lam", columns="size",
                                      values="time_ms")
sns.heatmap(heat, annot=True, fmt=".0f", cmap="rocket_r")
plt.title("Sinkhorn runtime [ms]")
plt.savefig("fig_heatmap_lam_vs_n.pdf", bbox_inches="tight")

