"""Figures 1 and 5 with named terms instead of letters (same data, same style).

    python code/scripts/make_fig01_fig05_named_terms.py

The figures of the decomposition use the letters I, D, R, P.  In a text where D already names the target outcome,
the terms are named instead: start, dynamics (drift), response to teaching, allocation of teaching.  The OULAD figure
also says "our earlier work" instead of "companion paper" for the data construction of the first study.
  * fig01: reads outputs/01_decomposition_simulator/results.json (same drawing as notebooks/01_gap_decomposition.ipynb);
  * fig05: reads outputs/06_oulad_start/results.json (same drawing as make_fig05_oulad_start.py).
Writes figures/named_terms/fig01_decomposition_simulator.png and figures/named_terms/fig05_oulad_start.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "named_terms"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64"


def fig01():
    res = json.load(open(ROOT / "outputs" / "01_decomposition_simulator" / "results.json"))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e5e0"})
    col = {"I": "#9e9e9e", "D": ORANGE, "R": "#eda100", "P": BLUE}
    names = {"I": "start", "D": "dynamics (drift)", "R": "response to teaching", "P": "allocation of teaching"}
    arms = [("no_teaching", "No teaching"), ("mpc_original", "Original MPC"), ("mpc_projected", "Projected MPC")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, g in zip(axes, ["0.5", "1.0", "2.0"]):
        for j, (arm, _) in enumerate(arms):
            r = res[g][arm]["disparity"]
            neg, pos = 0.0, 0.0
            for f in ["I", "D", "R", "P"]:
                val = r["shapley"][f]["mean"]
                base = pos if val >= 0 else neg
                ax.bar(j, val, bottom=base, color=col[f], width=0.6, edgecolor="white", linewidth=1.5,
                       label=names[f] if (g == "0.5" and j == 0) else None)
                if val >= 0:
                    pos += val
                else:
                    neg += val
            ax.plot(j, r["observed"]["mean"], "D", color="black", ms=6, label="observed gap" if (g == "0.5" and j == 0) else None)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(range(3), [a[1] for a in arms], fontsize=8)
        ax.set_title(f"γ = {float(g):g}", loc="left")
    axes[0].set_ylabel("final gap in disparity (group 1 − group 0)")
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index("observed gap")] + [i for i, x in enumerate(l) if x != "observed gap"]
    axes[0].legend([h[i] for i in order], [l[i] for i in order], fontsize=7, loc="lower left")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig01_decomposition_simulator.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    for g in ("0.5", "1.0", "2.0"):                         # numbers quoted in the text
        print(f"gamma={g}: " + " | ".join(
            f"{arm}: obs {res[g][arm]['disparity']['observed']['mean']:+.3f} drift {res[g][arm]['disparity']['shapley']['D']['mean']:+.3f} "
            f"alloc {res[g][arm]['disparity']['shapley']['P']['mean']:+.3f} err {res[g][arm]['disparity']['error']['mean']:+.3f}"
            for arm, _ in arms))


def fig05():
    plt.rcParams.update(plt.rcParamsDefault)
    R = json.load(open(ROOT / "outputs" / "06_oulad_start" / "results.json"))["part_b"]
    variants = [("paper", "as in our\nearlier work\n(0.5 imputed)"), ("observed", "students with\na score\n(0.5 imputed)"),
                ("backfill", "first observed\nscore carried\nback"), ("cohort4", "same, students\nscored by\nweek 4")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (attr, title) in zip(axes, (("deprivation", "Deprivation (IMD 0-30%)"), ("disability", "Declared disability"))):
        x = np.arange(len(variants))
        w = 0.34
        for j, (key, col, hatch, lab) in enumerate((("I", BLUE, None, "start term"), ("a", ORANGE, "///", "drift term"))):
            val = [R[attr][v]["ls"][key]["value"] for v, _ in variants]
            ci = np.array([R[attr][v]["ls"][key]["ci"] for v, _ in variants])
            pos = x + (j - 0.5) * w
            ax.bar(pos, val, w * 0.92, color=col, hatch=hatch, edgecolor="white", linewidth=0.0, label=lab, zorder=2)
            ax.errorbar(pos, val, yerr=[np.array(val) - ci[:, 0], ci[:, 1] - np.array(val)], fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
        start = [R[attr][v]["observed_initial"] for v, _ in variants]
        end = [R[attr][v]["observed"] for v, _ in variants]
        ax.scatter(x, start, marker="o", s=40, facecolor="white", edgecolor=INK, linewidth=1.2, zorder=4, label="observed gap at the start")
        ax.scatter(x, end, marker="D", s=36, color=INK, zorder=4, label="observed gap at the end")
        ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in variants], fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("knowledge gap (group 1 minus group 0)", fontsize=9, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig05_oulad_start.png", dpi=300)
    plt.close(fig)
    for attr in ("deprivation", "disability"):              # numbers quoted in the text
        for v, _ in variants:
            r = R[attr][v]["ls"]
            print(f"{attr:12s} {v:9s} start {r['I']['value']:+.3f} {np.round(r['I']['ci'], 3)} drift {r['a']['value']:+.3f} "
                  f"model gap {sum(r[k]['value'] for k in ('I', 'A', 'a')):+.3f} | observed start {R[attr][v]['observed_initial']:+.3f} end {R[attr][v]['observed']:+.3f}")


if __name__ == "__main__":
    fig01()
    fig05()
    print("written", OUT)
