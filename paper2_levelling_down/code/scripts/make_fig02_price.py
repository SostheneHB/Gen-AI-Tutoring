"""Figure 2: the price of fairness with knowledge as the fairness outcome (reads outputs/02_price_of_fairness/results<TAG>.json).

    P2_OUTCOME=knowledge python code/scripts/make_fig02_price.py

For each group strength: the narrowing of the knowledge gap promised by the model (line, from Theorem 2 = the linear
program) and the narrowing really obtained on the simulator (markers), against the extra budget, as a share of the
reference budget; mean over ten fitted models, with their range (band: promised; bars: obtained).  Dotted: the knowledge gap of the model; dashed: the true knowledge gap.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from p2.config import TAG  # noqa: E402

R = json.load(open(ROOT / "outputs" / "02_price_of_fairness" / f"results{TAG}.json"))
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3df"
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
for ax, g in zip(axes, ("0.5", "1.0", "2.0")):
    rows = [r for r in R[g]["frontier"] if r.get("feasible")]
    x = [r["extra_budget_model"] for r in rows]
    pm = R[g].get("per_model", [])
    if pm:                                      # spread over the fitted models: 5-95% ranges (min-max with ten models)
        fr = [r["fraction"] for r in rows]
        def band(key):
            v = [[next(x[key] for x in m["rows"] if x["fraction"] == f and x["feasible"]) for m in pm] for f in fr]
            return [min(a) for a in v], [max(a) for a in v]
        lo, hi = band("promised_reduction")
        ax.fill_between(x, lo, hi, color=BLUE, alpha=0.15, lw=0)
        lo, hi = band("true_reduction")
        ax.errorbar(x, [r["true_reduction"] for r in rows], yerr=[[m - l for m, l in zip([r["true_reduction"] for r in rows], lo)],
                    [h_ - m for m, h_ in zip([r["true_reduction"] for r in rows], hi)]], fmt="none", ecolor=ORANGE, elinewidth=1, capsize=2)
    ax.plot(x, [r["promised_reduction"] for r in rows], color=BLUE, lw=2, marker="o", ms=4, label="promised by the model")
    ax.plot(x, [r["true_reduction"] for r in rows], color=ORANGE, lw=0, marker="s", ms=6, label="obtained on the simulator")
    ax.axhline(R[g]["model_gap"], color=MUTED, ls=":", lw=1.2, label="knowledge gap, model")
    ax.axhline(R[g]["true_gap"], color=MUTED, ls="--", lw=1.2, label="knowledge gap, simulator")
    ax.set_title(f"γ = {float(g):g}", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("extra budget (share of the reference)", fontsize=9, color=INK)
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
axes[0].set_ylabel("narrowing of the knowledge gap", fontsize=9, color=INK)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
fig.tight_layout(rect=(0, 0.09, 1, 1))
out = ROOT / "figures" / "fig02_price_of_fairness_knowledge.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200)
print("written", out)
