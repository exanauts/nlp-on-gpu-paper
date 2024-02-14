
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "Helvetica"

FULL = 0
LINRED = 1
REDLIN = 2

NROWS = 6
colors = plt.cm.twilight(np.linspace(0, .8, NROWS))
kn_colors = plt.cm.PuOr(np.linspace(0, 1, 2))

CASES = [
    "case1354pegase",
    "case9241pegase",
    "case_ACTIVSg25K",
]
NCASES = len(CASES)

KN_MATPOWER = {
    "case1354pegase": [0.93, 1.15, 22],
    "case9241pegase": [18.8, 31.2, 102],
    "case_ACTIVSg25K": [23.5, 35.7, 47],
}
KN_AMPL = {
    "case1354pegase": [0.34, 0.85, 26],
    "case9241pegase": [9.4, 19.0, 60],
    "case_ACTIVSg25K": [10.5, 23.5, 49],
}

fig, axs = plt.subplots(
    ncols=NCASES, sharey=True,
    figsize=(10, 4),
)

for (k, case) in enumerate(CASES):
    diagnostic = np.loadtxt(f"numerics/diagnostic_{case}.txt")

    t_func = np.sum(diagnostic[0:1, :], axis=0)
    t_grad = np.sum(diagnostic[2:3, :], axis=0)
    t_hess = diagnostic[4, :]

    t_kkt_build = diagnostic[5, :]
    t_kkt_factorize = diagnostic[6, :]
    t_kkt_solve = diagnostic[7, :]

    t_pf = diagnostic[8, :]

    print(diagnostic[-1, :])
    # t_h_full = diagnostic[4, FULL]

    # t_fac_full = diagnostic[7]
    # t_solve_full = diagnostic[7]

    y = [2, 4, 2]

    # KNITRO `
    # nit = KN_AMPL[case][2]
    # t_eval = KN_AMPL[case][0] / nit
    # t_ls = KN_AMPL[case][1] / nit - t_eval
    # axs[k].barh(17, t_eval, color=kn_colors[0], label="KN: callbacks")
    # axs[k].barh(16, t_ls, color=kn_colors[1], label="KN: Ma27")

    nit = KN_MATPOWER[case][2]
    t_eval = KN_MATPOWER[case][0] / nit
    t_ls = KN_MATPOWER[case][1] / nit - t_eval
    axs[k].barh(14, t_eval, color=kn_colors[0], label="KN: callbacks")
    axs[k].barh(13, t_ls, color=kn_colors[1], label="KN: Ma27+internals")

    # Callbacks
    x = [10, 6, 2]
    axs[k].barh(x, t_func, color=colors[0])
    bot_ = t_func
    axs[k].barh(x, t_grad, left=bot_[:], label="AD: Jacobian", color=colors[0])
    bot_ = t_func + t_grad
    axs[k].barh(x, t_hess, left=bot_[:], label="AD: Hessian", color=colors[1])

    # Linear solver
    x = [9, 5, 1]
    axs[k].barh(x, t_kkt_build, label="LS: Reduction", color=colors[2])
    bot_ = t_kkt_build
    axs[k].barh(x, t_kkt_factorize, left=bot_, label="LS: Factorization", color=colors[3])
    bot_ += t_kkt_factorize
    axs[k].barh(x, t_kkt_solve, left=bot_, label="LS: backsolve", color=colors[4])

    # Power flow
    x = [4, 1, 0]
    axs[k].barh(x, t_pf, label="Power flow", color=colors[5])
    axs[k].set_xlabel("Time (s)")
    axs[k].set_title(case)

    axs[k].set_yticks(
        # [14, 13, 10, 9, 6, 5, 2, 1, 0],
        # ["eval", "LS", "AD", "LS", "AD", "LS", "AD", "LS", "PF"],
        [13.5, 9.5, 5.5, 1],
        ["Knitro+Mat.", "Full-space", "LinRed", "RedLin"],
        rotation=60,
        bbox={'pad': 1.5, 'fc': '1.0'},
        style="italic",
        # verticalalignment='center'
    )

    axs[k].spines['right'].set_visible(False)
    axs[k].spines['top'].set_visible(False)

xl = axs[1].get_xlim()
xpos = 0.5 * (xl[1] - xl[0])
# xpos = -1
# bbox = {'pad': 1.5, 'fc': '1.0'}
# axs[1].text(xpos, 3, "RedLin", bbox=bbox, horizontalalignment='center')
# axs[1].text(xpos, 7, "LinRed", bbox=bbox, horizontalalignment='center')
# axs[1].text(xpos, 11, "Full-space", bbox=bbox, horizontalalignment='center')
# axs[1].text(xpos, 14.5, "Knitro", bbox=bbox, horizontalalignment='center')

axs[-1].legend(fontsize="small", loc="center left", fancybox=True, bbox_to_anchor=(0.7, 0.5))


plt.tight_layout()

plt.savefig("pprof.pdf")
