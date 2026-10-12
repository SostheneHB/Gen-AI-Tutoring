"""Figures of the second revision, regenerated from the JSON outputs.

    python make_figures_v2.py --real out_time --synthetic out_synthetic \
        --ablation ../../TMLR_submission_package/Journal_manuscript/ablation_frozen_results.json \
        --dest ../Revised_manuscript
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
L = lambda p: json.load(open(p))
H = [1, 5, 10, 25]


def panel(ax, s):
    ax.set_title(s, loc="left", fontsize=9, color=INK)


def real(real_dir, dest):
    runs = [L(f"{real_dir}/school_disjoint_{j}/results.json") for j in range(5)]
    # --- replay gaps per split
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axs[0]
    x = np.arange(5)
    obs = [r["replay"]["terminal_gap_observed"] for r in runs]
    ci = np.array([r["replay"]["terminal_gap_observed_school_ci"] for r in runs])
    ax.errorbar(x - 0.21, obs, yerr=[np.array(obs) - ci[:, 0], ci[:, 1] - np.array(obs)], fmt="o",
                color=INK, ms=6, capsize=3, lw=1.2, label="Observed (school bootstrap 95%)")
    for off, m, c, lab in ((-0.07, "edmd_raw", BLUE, "Raw EDMD"), (0.07, "edmd_groupcal", AQUA, "Group-calibrated"),
                           (0.21, "edmd_proj", ORANGE, "Projected EDMD")):
        ax.plot(x + off, [r["replay"]["models"][m]["terminal_gap_pred"] for r in runs], "D", color=c,
                ms=6, mec="white", mew=1, label=lab, ls="none")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, [f"Split {j + 1}" for j in x])
    ax.set_ylabel("Terminal gap in $k_{25}$ (stratum 1 − 0)")
    ax.set_ylim(-0.03, 0.235)
    ax.legend(fontsize=7.5, loc="upper center", ncol=2)
    panel(ax, "(a) One-step replay, five school-disjoint splits")
    ax = axs[1]
    r = runs[0]["replay"]["per_step_gap"]
    t = np.arange(1, 26)
    ax.plot(t, r["observed"], color=INK, label="Observed")
    ax.plot(t, r["edmd_raw"], color=BLUE, label="Raw EDMD")
    ax.plot(t, r["edmd_proj"], color=ORANGE, label="Projected EDMD")
    for series, c, lab in ((r["observed"], INK, "Observed"), (r["edmd_raw"], BLUE, "Raw"),
                           (r["edmd_proj"], ORANGE, "Projected")):
        ax.annotate(lab, (25, series[-1]), xytext=(4, 0), textcoords="offset points", color=INK,
                    fontsize=7.5, va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(1, 29)
    ax.set_xlabel("Step $t$")
    ax.set_ylabel("Gap in $k_t$")
    panel(ax, "(b) Per-step gap, split 1")
    fig.tight_layout()
    fig.savefig(f"{dest}/fig_real_replay.png")
    plt.close(fig)

    # --- baselines: MSE of k vs horizon (mean over splits, range as band)
    models = [("persistence", "Persistence", INK2), ("gbm", "Gradient boosting", MAGENTA),
              ("affine_ls", "Affine LS", YELLOW), ("edmd_raw", "Raw EDMD", BLUE),
              ("edmd_proj", "Projected EDMD", ORANGE)]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axs[0]
    for m, lab, c in models:
        v = np.array([[rr["replay"]["models"][m]["mse"][str(h)]["k_t"] for h in H] for rr in runs])
        ax.plot(H, v.mean(0), "-o", color=c, ms=5, mec="white", label=lab)
        ax.fill_between(H, v.min(0), v.max(0), color=c, alpha=0.12, lw=0)
    ax.set_yscale("log")
    ax.set_xticks(H)
    ax.set_xlabel("Horizon $h$ (steps ahead of an observed state)")
    ax.set_ylabel("MSE of $k$ (log scale)")
    ax.legend(fontsize=7.5, ncol=2)
    panel(ax, "(a) Held-out error by horizon, mean and range over splits")
    # soft-projection frontier
    ax = axs[1]
    names = ["edmd_raw", "edmd_soft_0.25", "edmd_soft_0.5", "edmd_soft_0.75", "edmd_proj"]
    alphas = [0, 0.25, 0.5, 0.75, 1]
    for j, rr in enumerate(runs):
        mse = [rr["replay"]["models"][m]["mse"]["1"]["k_t"] for m in names]
        g = [rr["replay"]["models"][m]["terminal_gap_pred"] for m in names]
        ax.plot(mse, g, "-o", color=BLUE, alpha=0.35 + 0.13 * j, ms=4, lw=1.2,
                label="Soft projection $\\alpha=0\\to1$" if j == 0 else None)
        gc = rr["replay"]["models"]["edmd_groupcal"]
        ax.plot(gc["mse"]["1"]["k_t"], gc["terminal_gap_pred"], "s", color=AQUA, ms=6, mec="white",
                label="Group-calibrated" if j == 0 else None)
        ax.plot(gc["mse"]["1"]["k_t"] * 0.93, rr["replay"]["terminal_gap_observed"], "_", color=INK, ms=12,
                mew=2, label="Observed gap" if j == 0 else None)
    for a, m in zip(alphas, names):
        rr = runs[0]["replay"]["models"][m]
        ax.annotate(f"$\\alpha$={a:g}", (rr["mse"]["1"]["k_t"], rr["terminal_gap_pred"]),
                    xytext=(4, 5), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("One-step MSE of $k$ (log scale)")
    ax.set_ylabel("Predicted terminal gap")
    ax.legend(fontsize=7.5, loc="upper right")
    panel(ax, "(b) Gap suppression versus predictive error")
    fig.tight_layout()
    fig.savefig(f"{dest}/fig_real_baselines.png")
    plt.close(fig)

    # --- TRS
    trs = np.concatenate([np.load(f"{real_dir}/school_disjoint_{j}/arrays.npz")["trs"] for j in range(5)])
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axs[0]
    ax.hist(trs, bins=40, range=(0, 2), color=BLUE, edgecolor="white", lw=0.5)
    ax.axvline(trs.mean(), color=INK, ls="--", lw=1.2)
    ax.annotate(f"mean {trs.mean():.2f}", (trs.mean(), ax.get_ylim()[1] * 0.9), xytext=(8, 0),
                textcoords="offset points", fontsize=7.5)
    ax.axvline(1, color=INK2, ls=":", lw=1)
    ax.set_xlabel("TRS$_i$ (separate centring)")
    ax.set_ylabel("Students")
    panel(ax, f"(a) TRS, {len(trs):,} students, 5 splits")
    ax = axs[1]
    keys = [("separate", "Separate\ncentring"), ("common_centring", "Common\ncentring"),
            ("standardised", "Standardised"), ("no_centring", "No\ncentring")]
    for j, rr in enumerate(runs):
        v = [rr["trs"]["mean"]] + [rr["trs"]["sensitivity_mean"][k] for k, _ in keys[1:]]
        ax.plot(range(4), v, "o", color=BLUE, ms=5, alpha=0.8, mec="white")
    ax.set_xticks(range(4), [l for _, l in keys], fontsize=7.5)
    ax.axhline(1, color=INK2, ls=":", lw=1)
    ax.set_ylim(-0.05, 2)
    ax.set_ylabel("Mean TRS")
    panel(ax, "(b) Dependence on centring, per split")
    ax = axs[2]
    n0 = np.concatenate([np.load(f"{real_dir}/school_disjoint_{j}/arrays.npz")["norm_raw"] for j in range(5)])
    n1 = np.concatenate([np.load(f"{real_dir}/school_disjoint_{j}/arrays.npz")["norm_proj"] for j in range(5)])
    bins = np.logspace(np.log10(min(n0.min(), n1.min())), np.log10(max(n0.max(), n1.max())), 40)
    ax.hist(n0, bins=bins, color=BLUE, alpha=0.75, label="Raw", edgecolor="white", lw=0.4)
    ax.hist(n1, bins=bins, color=ORANGE, alpha=0.75, label="Projected", edgecolor="white", lw=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Norm of centred trajectory (log scale)")
    ax.legend(fontsize=7.5)
    panel(ax, "(c) Magnitudes of the compared vectors")
    fig.tight_layout()
    fig.savefig(f"{dest}/fig_real_trs.png")
    plt.close(fig)


def synthetic(path, dest):
    R = L(path)
    arms = [("mpc_proj", "Projected MPC", ORANGE), ("gap_penalty", "Gap-penalty MPC", MAGENTA),
            ("gap_constraint", "Gap-constrained MPC", AQUA)]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    rows = sorted(R["one_factor"]["gamma"], key=lambda r: r["config"]["gamma"])
    g = [r["config"]["gamma"] for r in rows]
    for key, ax, lab in (("abs_gap", axs[0], "Change in $|\\delta_T|$ vs raw MPC"),
                         ("knowledge", axs[1], "Change in $\\overline{k}_T$ vs raw MPC")):
        for off, (a, name, c) in zip((-0.04, 0, 0.04), arms):
            e = [r[f"effect_{a}_vs_raw"][key] for r in rows]
            m = np.array([x["mean"] for x in e])
            lo = np.array([x["ci"][0] for x in e])
            hi = np.array([x["ci"][1] for x in e])
            ax.errorbar(np.array(g) + off, m, yerr=[m - lo, hi - m], fmt="-o", color=c, ms=5,
                        mec="white", capsize=2, lw=1.5, label=name)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("Group strength $\\gamma$ in the simulator")
        ax.set_ylabel(lab)
    axs[0].legend(fontsize=7.5)
    panel(axs[0], "(a) Terminal disparity gap (positive = wider)")
    panel(axs[1], "(b) Pooled terminal knowledge")
    fig.tight_layout()
    fig.savefig(f"{dest}/fig_synthetic_regimes.png")
    plt.close(fig)


def ablation(path, dest):
    A = L(path)
    runs = [r for r in A["runs"] if r["status"] == "ok"]
    ep = [0] + list(range(5, 41, 5))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key, title in ((axs[0], "D1", "(a) One-step state MSE, unprojected"),
                           (axs[1], "P1", "(b) One-step state MSE, projected")):
        V = np.array([r["frozen"][key]["trace"] for r in runs])
        for v in V:
            ax.plot(ep, v, color=INK2, alpha=0.18, lw=0.8)
        ax.plot(ep, np.median(V, 0), color=BLUE, lw=2.2, label="Median of 80 runs")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch (checkpoint)")
        ax.set_ylabel("Common frozen diagnostic (log)")
        ax.legend(fontsize=7.5)
        panel(ax, title)
    fig.tight_layout()
    fig.savefig(f"{dest}/fig_ablation_frozen.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="out_time")
    ap.add_argument("--synthetic", default="out_synthetic")
    ap.add_argument("--ablation", default="../../TMLR_submission_package/Journal_manuscript/ablation_frozen_results.json")
    ap.add_argument("--dest", default="../Revised_manuscript")
    a = ap.parse_args()
    real(a.real, a.dest)
    if os.path.exists(f"{a.synthetic}/regimes.json"):
        synthetic(f"{a.synthetic}/regimes.json", a.dest)
    ablation(a.ablation, a.dest)


if __name__ == "__main__":
    main()
