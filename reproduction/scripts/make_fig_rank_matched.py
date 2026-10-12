"""Figures for the rank-matched controls (out_rank_matched/*.json).

    python make_fig_rank_matched.py
writes fig_rank_matched.png (paper, panels A-D), and for slides and poster
fig_rank_matched_AB.png (prediction, price of erasure) and fig_rank_matched_CD.png (simulator).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "out_rank_matched"
GROUP, CTRL = "#eb6834", "#2a78d6"        # uses the group / does not (palette slots 2 and 1)
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a63", "#e4e3dd", "#fcfcfb"
DATASETS = [("assistments", "ASSISTments"), ("eedi", "Eedi"), ("oulad_imd", "OULAD imd"),
            ("oulad_disability", "OULAD disab."), ("oulad_gender", "OULAD gender")]
ARMS = [("group_r4", "Group projection (rank 4)", True), ("leace_op_r1", "LEACE (rank 1)", True),
        ("permuted_r4", "Shuffled labels (rank 4)", False), ("random_r4", "Random (rank 4)", False),
        ("pca_top_r4", "Top-variance PCA (rank 4)", False), ("pca_bottom_r4", "Lowest-variance PCA (rank 4)", False),
        ("mindist_r4", "Least-used directions (rank 4)", False)]
Y = np.arange(len(ARMS))[::-1]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.facecolor": SURF, "figure.facecolor": "white"})


def style(ax, both=False):
    ax.grid(axis="both" if both else "x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def marker(uses_group):
    return dict(marker="o" if uses_group else "s", color=GROUP if uses_group else CTRL, ms=7,
                mec="white", mew=1.2, zorder=3)


def panel_a(ax, title="A. Same rank, very different cost"):
    res = {d: json.load(open(f"{OUT}/{d}.json"))["across_splits"] for d, _ in DATASETS}
    for yi, (arm, label, ug) in zip(Y, ARMS):
        ratios = [res[d][arm]["mse_k_h1"][0] / res[d]["raw"]["mse_k_h1"][0] for d, _ in DATASETS]
        ax.plot(ratios, [yi] * len(ratios), "|", color=MUTED, ms=9, zorder=2)
        ax.plot(np.exp(np.mean(np.log(ratios))), yi, **marker(ug))
    ax.set_xscale("log")
    ax.axvline(1, color=MUTED, lw=1, ls="--")
    ax.set_yticks(Y, [a[1] for a in ARMS])
    ax.set_xlabel("One-step error on knowledge / raw model\n(ticks: 5 datasets; marker: geometric mean)")
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    style(ax)


def panel_b(ax, title="B. The price of erasing the group"):
    ep = json.load(open(f"{OUT}/erasure_price.json"))
    allv = []
    for d, lab in DATASETS:
        f = [r["taus"]["0.0"]["formula"] for r in ep[d]]
        tr = [r["taus"]["0.0"]["excess_mse_train"] for r in ep[d]]
        te = [r["taus"]["0.0"]["excess_mse_test"] for r in ep[d]]
        ax.plot(f, tr, "o", color=GROUP, ms=6, mec="white", mew=1, zorder=3)
        ax.plot(f, te, "s", mfc="none", mec=CTRL, ms=6, mew=1.2, zorder=3)
        off = {"eedi": (-34, 4), "oulad_gender": (8, 6)}.get(d, (6, -10))
        ax.annotate(lab, (np.mean(f), np.mean(tr)), xytext=off, textcoords="offset points", fontsize=8, color=MUTED)
        allv += f + tr + te
    lim = [0, max(allv) * 1.08]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--", zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"Formula: Var$(\hat y)\,\rho(\hat y,\hat g)^2$")
    ax.set_ylabel("Measured extra error after erasure")
    ax.plot([], [], "o", color=GROUP, label="training data (exact)")
    ax.plot([], [], "s", mfc="none", mec=CTRL, label="held-out students")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    style(ax, both=True)


def panels_cd(ax_c, ax_d, labels_on_c=False, titles=("C. Simulator: true gap under MPC", "D. Simulator: mean knowledge"),
              legend=("c", "lower right")):
    sim = json.load(open(f"{OUT}/simulator.json"))
    cfg = next(c for c in sim["configs"] if c["gamma"] == 1.0)
    for ax, key, title, xl in ((ax_c, "abs_gap", titles[0], "Change in |gap| vs raw MPC"),
                               (ax_d, "knowledge", titles[1], "Change in knowledge vs raw MPC")):
        for yi, (arm, label, ug) in zip(Y, ARMS):
            e = cfg["effects_vs_raw"][arm][key]
            ax.plot(e["ci"], [yi, yi], color=GROUP if ug else CTRL, lw=2, zorder=2)
            ax.plot(e["mean"], yi, **marker(ug))
        ax.axvline(0, color=MUTED, lw=1, ls="--")
        ax.set_yticks(Y, [a[1] for a in ARMS] if (labels_on_c and ax is ax_c) else [""] * len(Y))
        ax.set_xlabel(xl + "\n(γ = 1, 10 replicates, 95% CI)")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        style(ax)
    ax_l = ax_c if legend[0] == "c" else ax_d
    ax_l.plot([], [], **{**marker(True), "ls": ""}, label="uses the group label")
    ax_l.plot([], [], **{**marker(False), "ls": ""}, label="does not use it")
    ax_l.legend(frameon=False, fontsize=8, loc=legend[1])


# paper: four panels
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
panel_a(axes[0]); panel_b(axes[1]); panels_cd(axes[2], axes[3])
fig.tight_layout(); fig.savefig("fig_rank_matched.png", dpi=200); plt.close(fig)
# slides and poster: prediction and price
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.3, 1]})
panel_a(axes[0]); panel_b(axes[1])
fig.tight_layout(); fig.savefig("fig_rank_matched_AB.png", dpi=200); plt.close(fig)
# slides: simulator
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.45, 1]})
panels_cd(axes[0], axes[1], labels_on_c=True, titles=("A. True gap under MPC", "B. Mean knowledge"),
          legend=("d", "lower left"))
fig.tight_layout(); fig.savefig("fig_rank_matched_CD.png", dpi=200); plt.close(fig)
print("written fig_rank_matched.png, fig_rank_matched_AB.png, fig_rank_matched_CD.png")
