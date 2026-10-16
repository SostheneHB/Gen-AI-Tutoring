"""Figure 6 (appendix): observed and imputed values of the OULAD knowledge proxy, week by week.

    python code/scripts/make_fig06_oulad_imputation.py     (reads outputs/06_oulad_start/results.json, part A)

Left: share of students whose proxy is still the imputed 0.5 (no assessment submitted yet), by group.
Right: gap of the proxy (group 1 minus group 0) over all students, and among the students with at least
one observed score at that week.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
A = json.load(open(ROOT / "outputs" / "06_oulad_start" / "results.json"))["part_a"]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3df"
LAB = {"deprivation": ("other students", "IMD 0-30%"), "disability": ("no declared disability", "declared disability")}

fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
for row, attr in enumerate(("deprivation", "disability")):
    r = A[attr]
    weeks = sorted(int(w) for w in r["imputed_share"])
    ax = axes[row, 0]
    for g, col, mk in (("0", BLUE, "o"), ("1", ORANGE, "s")):
        ax.plot(weeks, [r["imputed_share"][str(w)][g] for w in weeks], color=col, marker=mk, ms=5, lw=2, label=LAB[attr][int(g)])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(f"{attr}: share imputed (0.5)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[row, 1]
    ax.plot(weeks, [r["gap_all"][str(w)] for w in weeks], color=INK, lw=2, marker="o", ms=4, label="all students (0.5 imputed)")
    ax.plot(weeks, [r["gap_observed"][str(w)] for w in weeks], color=INK, lw=2, ls="--", marker="D", ms=4, label="students with an observed score")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel(f"{attr}: gap of the proxy", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8)
for ax in axes.ravel():
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
for ax in axes[1]:
    ax.set_xlabel("week", fontsize=9, color=INK)
fig.tight_layout()
out = ROOT / "figures" / "fig06_oulad_imputation.png"
fig.savefig(out, dpi=200)
print("written", out)
