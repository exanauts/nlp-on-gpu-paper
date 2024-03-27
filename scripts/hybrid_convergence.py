
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

RESULTS_DIR = sys.argv[1]
GAMMA = [4, 5, 6, 7, 8]
nexp = len(GAMMA)

colors2 = plt.cm.Greens(np.linspace(0.4, 1, nexp))

fig, axs = plt.subplots(figsize=(10, 4), ncols=2)

# Plotting
for (k, gamma) in enumerate(GAMMA):
    results = np.loadtxt(f"{RESULTS_DIR}/hybrid-convergence-{gamma}.txt")
    n_iters = np.size(results, 0)
    iters = range(n_iters)
    axs[0].plot(iters, results[:, 0], label=f"$\gamma=10^{gamma}$", ls="-", lw=2, c=colors2[k])
    axs[1].plot(iters, results[:, 1], label=f"$\gamma=10^{gamma}$", ls="-", lw=2, c=colors2[k])

# Formatting
axs[0].set_title("Number of CG iterations")
axs[0].set_ylabel("# CG iters")
axs[1].set_title("Relative accuracy along the iterations")
axs[1].set_ylabel("Relative residual")
axs[1].set_yscale("log")

# Global formatting
for ax in axs:
    ax.grid(ls=":", which="both", alpha=.8, axis="y", lw=.5)
    ax.set_xlabel("#IPM it")
    ax.legend(fontsize="x-small")

plt.tight_layout()
plt.savefig("hybrid-gamma.pdf")


# Analyze statistics
results_cpu = np.loadtxt(f"{RESULTS_DIR}/hybrid-stats-cpu.txt")
results_cuda = np.loadtxt(f"{RESULTS_DIR}/hybrid-stats-cuda.txt")

# Check consistency
assert np.size(results_cpu, 0) == np.size(results_cuda, 0)

df = pd.DataFrame()
df["gamma"] = results_cpu[:, 0]
df["it_cpu"] = results_cpu[:, 1]
df["t_condensation_cpu"] = results_cpu[:, 2]
df["t_cg_cpu"] = results_cpu[:, 3]
df["t_linsol_cpu"] = results_cpu[:, 4]
df["t_total_cpu"] = results_cpu[:, 5]

df["it_cuda"] = results_cuda[:, 1]
df["t_condensation_cuda"] = results_cuda[:, 2]
df["t_cg_cuda"] = results_cuda[:, 3]
df["t_linsol_cuda"] = results_cuda[:, 4]
df["t_total_cuda"] = results_cuda[:, 5]

df = df.astype({'it_cpu': int, 'it_cuda': int})

print(df.to_latex(index=False, float_format="%.2f"))


