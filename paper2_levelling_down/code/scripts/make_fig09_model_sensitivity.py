"""Figure: how much do the promise and the price depend on the learned model?

    python code/scripts/make_fig09_model_sensitivity.py

Reads outputs/13_model_sensitivity/results_knowledge.json, the same file as Table "sensitivity".  One point per refit
(ten refits for each group strength gamma and training size n).  Refits whose efficient reference has no model gap
(nothing to narrow) are left out, as in the table.
  (a) closable gap promised by the model, min(capacity, |model gap|), against the narrowing that the closing plan
      obtains on the simulator ("Closable gap" and "Obtained" columns);
  (b) price: extra budget, as a share of the reference budget, to narrow by half of the closable gap ("Price" column).
Writes figures/fig09_model_sensitivity.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "13_model_sensitivity" / "results_knowledge.json"
GAMMAS = ((0.5, "#0072B2", "o"), (1.0, "#D55E00", "s"), (2.0, "#009E73", "^"))   # Okabe-Ito, as in Figure 7
SIZES = (50, 150, 600)
INK, MUTED = "#222222", "#555555"


def gamma_label(g):
    return rf"$\gamma={g:g}$"


def main():
    rows = [r for r in json.load(open(SRC))["rows"] if r["close_target"] > 0]
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.05, 1]})

    # (a) promised against obtained, log-log, one point per refit
    lo, hi = 0.004, 0.4
    a.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
    diag = a.text(0.0095, 0.0068, "obtained = promised", fontsize=7.5, color=MUTED, ha="center", va="center")
    for g, colour, marker in GAMMAS:
        rs = [r for r in rows if r["gamma"] == g]
        a.scatter([r["close_target"] for r in rs], [r["true_full"] for r in rs], s=20, marker=marker,
                  color=colour, linewidth=0, alpha=0.85, zorder=3, label=gamma_label(g))
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    ticks = [0.01, 0.03, 0.1, 0.3]
    for axis in (a.xaxis, a.yaxis):
        axis.set_major_locator(FixedLocator(ticks)); axis.set_minor_locator(NullLocator())
    a.set_xticklabels([f"{t:g}" for t in ticks]); a.set_yticklabels([f"{t:g}" for t in ticks])
    a.set_xlabel("narrowing promised by the model")
    a.set_ylabel("narrowing obtained on the simulator")
    a.set_title("(a) promise and result of the closing plan", fontsize=9, color=INK, loc="left")

    # (b) price against training size, one point per refit, groups side by side
    for k, (g, colour, marker) in enumerate(GAMMAS):
        for i, n in enumerate(SIZES):
            ys = [r["price_half"] for r in rows if r["gamma"] == g and r["n_train"] == n]
            xs = [i + (k - 1) * 0.24 + (j - (len(ys) - 1) / 2) * 0.014 for j in range(len(ys))]
            b.scatter(xs, ys, s=18, marker=marker, color=colour, linewidth=0, alpha=0.85, zorder=3)
    b.set_xticks(range(len(SIZES)))
    b.set_xticklabels([f"{n} students" for n in SIZES])
    b.set_xlim(-0.5, len(SIZES) - 0.5)
    b.set_ylim(0, None)
    b.set_xlabel("training size of the learned model")
    b.set_ylabel("price (share of the reference budget)")
    b.set_title("(b) price of narrowing by half", fontsize=9, color=INK, loc="left")

    for ax in (a, b):
        ax.grid(True, which="major", color="#e5e5e5", lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], ls="", marker=m, ms=5.5, markerfacecolor=c, markeredgewidth=0, label=gamma_label(g))
               for g, c, m in GAMMAS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01),
               title="group strength", title_fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.canvas.draw()                                       # angle of the diagonal on the page, after the layout
    (x0, y0), (x1, y1) = a.transData.transform([(lo, lo), (hi, hi)])
    diag.set_rotation(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
    out = ROOT / "figures" / "fig09_model_sensitivity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print("written", out, "| refits shown:", len(rows))


if __name__ == "__main__":
    main()
