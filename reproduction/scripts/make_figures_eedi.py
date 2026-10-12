"""Figures of the Eedi and OULAD studies, same style as make_figures_v2.py.

    python make_figures_eedi.py --eedi <out_eedi> --dest ../Revised_manuscript
    python make_figures_eedi.py --eedi <out_oulad>/imd --prefix imd_student_split --stem oulad \
        --groups "Less deprived areas,Most deprived areas (IMD 0-30%)" --gaplabel "Gap in mean score (most deprived - others)" \
        --unit "course-presentation" --splitname "student splits" --ylim -0.08,0.02

Writes fig_<stem>_replay.png and fig_<stem>_baselines.png from the results.json
files (aggregate outputs only).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figures_v2 import AQUA, BLUE, H, INK, INK2, MAGENTA, ORANGE, YELLOW, panel

L = lambda p: json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eedi", required=True)
    ap.add_argument("--dest", default="../Revised_manuscript")
    ap.add_argument("--prefix", default="class_disjoint")
    ap.add_argument("--stem", default="eedi")
    ap.add_argument("--groups", default="Other pupils,Disadvantaged pupils")
    ap.add_argument("--gaplabel", default="Final gap in $k_{25}$ (disadvantaged − others)")
    ap.add_argument("--unit", default="class")
    ap.add_argument("--splitname", default="class-disjoint splits")
    ap.add_argument("--ylim", default="-0.135,0.04")
    a = ap.parse_args()
    runs = [L(f"{a.eedi}/{a.prefix}_{j}/results.json") for j in range(5)]
    g0, g1 = a.groups.split(",")

    # --- figure 1: replay of the gap
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axs[0]
    x = np.arange(5)
    obs = np.array([r["replay"]["terminal_gap_observed"] for r in runs])
    ci = np.array([r["replay"]["terminal_gap_observed_school_ci"] for r in runs])
    ax.errorbar(x - 0.21, obs, yerr=[obs - ci[:, 0], ci[:, 1] - obs], fmt="o", color=INK, ms=6, capsize=3,
                lw=1.2, label=f"Observed ({a.unit} bootstrap 95%)")
    for off, m, c, lab in ((-0.07, "edmd_raw", BLUE, "Original EDMD"), (0.07, "edmd_groupcal", AQUA, "Recalibrated per group"),
                           (0.21, "edmd_proj", ORANGE, "Projected EDMD")):
        ax.plot(x + off, [r["replay"]["models"][m]["terminal_gap_pred"] for r in runs], "D", color=c, ms=6,
                mec="white", mew=1, label=lab, ls="none")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, [f"Split {j + 1}" for j in x])
    ax.set_ylabel(a.gaplabel)
    ax.set_ylim(*map(float, a.ylim.split(",")))
    ax.legend(fontsize=7.5, loc="lower center", ncol=2)
    panel(ax, f"(a) One-step replay, five {a.splitname}")
    ax = axs[1]
    r = runs[0]["replay"]["per_step_gap"]
    t = np.arange(1, len(r["observed"]) + 1)
    ends = []
    for key, c, lab in (("observed", INK, "Observed"), ("edmd_raw", BLUE, "Original"), ("edmd_proj", ORANGE, "Projected")):
        ax.plot(t, r[key], color=c, label=lab)
        ends.append([r[key][-1], lab])
    span = max(max(r[k]) for k in ("observed", "edmd_raw", "edmd_proj")) - min(min(r[k]) for k in ("observed", "edmd_raw", "edmd_proj"))
    ends.sort()
    for i in range(1, len(ends)):                       # keep end labels at least 6% of the range apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.06 * span)
    for y, lab in ends:
        ax.annotate(lab, (t[-1], y), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=7.5, va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(1, t[-1] + 5)
    ax.set_xlabel("Step $t$")
    ax.set_ylabel("Gap in $k_t$")
    panel(ax, "(b) Gap at every step, split 1")
    fig.tight_layout()
    fig.savefig(f"{a.dest}/fig_{a.stem}_replay.png")
    plt.close(fig)

    # --- figure 2: baselines by horizon and calibration per group
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axs[0]
    for m, lab, c in (("persistence", "Persistence", INK2), ("gbm", "Gradient boosting", MAGENTA),
                      ("affine_ls", "Linear (affine LS)", YELLOW), ("edmd_raw", "Original EDMD", BLUE),
                      ("edmd_proj", "Projected EDMD", ORANGE)):
        v = np.array([[rr["replay"]["models"][m]["mse"][str(h)]["k_t"] for h in H] for rr in runs])
        ax.plot(H, v.mean(0), "-o", color=c, ms=5, mec="white", label=lab)
        ax.fill_between(H, v.min(0), v.max(0), color=c, alpha=0.12, lw=0)
    ax.set_yscale("log")
    ax.set_xticks(H)
    ax.set_xlabel("Horizon $h$")
    ax.set_ylabel("MSE of $k$ (log scale)")
    ax.legend(fontsize=7, ncol=2)
    panel(ax, "(a) Held-out error by horizon")
    for ax, g, lab in ((axs[1], "0", f"(b) {g0}"), (axs[2], "1", f"(c) {g1}")):
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
        for m, c, name in (("edmd_raw", BLUE, "Original"), ("edmd_groupcal", AQUA, "Recalibrated"), ("edmd_proj", ORANGE, "Projected")):
            for j, rr in enumerate(runs):
                rel = np.array(rr["calibration"][m][g]["reliability_deciles"])
                ax.plot(rel[:, 0], rel[:, 1], "-o", color=c, ms=2.5, lw=1.1, alpha=0.75, label=name if j == 0 else None)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted $k_{t+1}$ (decile)")
        ax.set_ylabel("Mean observed $k_{t+1}$")
        panel(ax, lab)
    axs[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{a.dest}/fig_{a.stem}_baselines.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
