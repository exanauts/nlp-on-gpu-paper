
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = sys.argv[1]
plt.rcParams['axes.linewidth'] = 0.5
# plt.rcParams['font.family'] = "sans-serif"
# plt.rcParams['font.sans-serif'] = "Helvetica"

colors = plt.cm.twilight(np.linspace(0, .8, 6))

diagnostic = pd.read_csv(f"{RESULTS_DIR}/benchmark_kkt.txt", index_col=0, header=None, sep="\t")

diagnostic.columns = ["build", "factorize", "backsolve", "accuracy"]

fig, axs = plt.subplots(
    sharey=True,
    figsize=(5, 3),
)

shift = 1
x = np.array([0, 4, 8, 12, 16])
axs.bar(x, diagnostic["build"].values, label="Build KKT", color=colors[0])
axs.bar(x + shift, diagnostic["factorize"].values, label="Factorize KKT", color=colors[1])
axs.bar(x + 2*shift, diagnostic["backsolve"].values, label="Backsolve", color=colors[2])

axs.set_xticks(
    x + shift,
    diagnostic.index,
    rotation=20,
    bbox={'pad': 1.5, 'fc': '1.0'},
    style="italic",
    # verticalalignment='center'
)

axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.legend(fontsize="x-small", loc="upper right", fancybox=True) #, bbox_to_anchor=(0.7, 0.5))
axs.set_ylabel("Time (s)")
axs.grid(ls=":", which="both", alpha=.8, axis="y", lw=.5)
plt.tight_layout()
plt.savefig("figures/breakdown.pdf")

print(diagnostic.to_latex(float_format="%.2e"))
