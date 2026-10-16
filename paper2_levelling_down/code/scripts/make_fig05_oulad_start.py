"""Figure 5: how the split of the OULAD knowledge gap depends on the definition of the start.

    python code/scripts/make_fig05_oulad_start.py      (reads outputs/06_oulad_start/results.json)

For each attribute and each definition of the start: the start term I and the drift term a of the
least-squares decomposition (bars, 95% intervals from the student bootstrap with re-estimation), and
the observed gap of the proxy at the start and at the end (markers).  The persistence term A is below
0.006 in absolute value everywhere and is not drawn.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = json.load(open(ROOT / "outputs" / "06_oulad_start" / "results.json"))["part_b"]
VARIANTS = [("paper", "as in the\ncompanion paper\n(0.5 imputed)"), ("observed", "students with\na score\n(0.5 imputed)"),
            ("backfill", "first observed\nscore carried\nback"), ("cohort4", "same, students\nscored by\nweek 4")]
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (attr, title) in zip(axes, (("deprivation", "Deprivation (IMD 0-30%)"), ("disability", "Declared disability"))):
    x = np.arange(len(VARIANTS))
    w = 0.34
    for j, (key, col, hatch, lab) in enumerate((("I", BLUE, None, "start term $I$"), ("a", ORANGE, "///", "drift term $a$"))):
        val = [R[attr][v]["ls"][key]["value"] for v, _ in VARIANTS]
        ci = np.array([R[attr][v]["ls"][key]["ci"] for v, _ in VARIANTS])
        pos = x + (j - 0.5) * w
        ax.bar(pos, val, w * 0.92, color=col, hatch=hatch, edgecolor="white", linewidth=0.0, label=lab, zorder=2)
        ax.errorbar(pos, val, yerr=[np.array(val) - ci[:, 0], ci[:, 1] - np.array(val)], fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
    start = [R[attr][v]["observed_initial"] for v, _ in VARIANTS]
    end = [R[attr][v]["observed"] for v, _ in VARIANTS]
    ax.scatter(x, start, marker="o", s=40, facecolor="white", edgecolor=INK, linewidth=1.2, zorder=4, label="observed gap at the start")
    ax.scatter(x, end, marker="D", s=36, color=INK, zorder=4, label="observed gap at the end")
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in VARIANTS], fontsize=8, color=INK)
    ax.set_title(title, fontsize=10, color=INK)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
axes[0].set_ylabel("knowledge gap (group 1 minus group 0)", fontsize=9, color=INK)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
fig.tight_layout(rect=(0, 0.08, 1, 1))
out = ROOT / "figures" / "fig05_oulad_start.png"
fig.savefig(out, dpi=200)
print("written", out)
