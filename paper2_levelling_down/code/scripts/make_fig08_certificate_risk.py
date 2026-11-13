"""Figure: what the plan without a margin risks, and what the certificate risks.

    python code/scripts/make_fig08_certificate_risk.py

Reads outputs/12_certificate_v3/results_<regime>.json, the same files as Tables "certificate" and
"certificate-decisions" (make_tab_certificate.py).  For each regime, group strength gamma and fitted model:
  * plan without margin: 1 - success of the plan with delta = 0 applied without calibration ("no margin" column);
  * certificate and Learn then Test: risk = P(the plan is changed and fails), over calibrations and test cohorts.
Models whose efficient reference has no model gap are left out, as in the tables.
Writes figures/fig08_certificate_risk.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "12_certificate_v3"
ALPHA = 0.1
GAMMAS = ("0.5", "1.0", "2.0")
REGIMES = ("standard", "heterogeneous", "hard")
SERIES = (                                      # label, colour (Okabe-Ito), marker, horizontal shift
    ("plan without margin", "#7a7a7a", "o", -0.27),
    ("certificate", "#0072B2", "s", 0.0),
    ("Learn then Test", "#D55E00", "^", 0.27),
)
INK, MUTED = "#222222", "#555555"


def values(run):
    naive = 1 - run["naive"]["success"] if run["naive"] else None
    return naive, run["full"]["conformal"]["risk"], run["full"]["ltt"]["risk"]


def main():
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    for ax, regime in zip(axes, REGIMES):
        res = json.load(open(OUT / f"results_{regime}.json"))
        for i, g in enumerate(GAMMAS):
            runs = [r for r in res[g] if not r.get("skipped")]
            vals = [values(r) for r in runs]
            for k, (_, colour, marker, shift) in enumerate(SERIES):
                ys = [v[k] for v in vals if v[k] is not None]
                n = len(ys)
                xs = [i + shift + (j - (n - 1) / 2) * 0.045 for j in range(n)]      # small spread, one point per model
                ax.scatter(xs, ys, s=20, marker=marker, color=colour, linewidth=0, alpha=0.85, zorder=3)
        ax.axhline(ALPHA, color=INK, lw=1, ls="--", zorder=2)
        ax.set_xticks(range(3))
        ax.set_xticklabels([r"$\gamma=0.5$", r"$\gamma=1$", r"$\gamma=2$"])
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(f"{regime} regime", fontsize=9, color=INK)
        ax.grid(True, axis="y", color="#e5e5e5", lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("share of decisions that fail")
    axes[0].text(2.55, ALPHA + 0.025, r"$\alpha=0.1$", ha="right", va="bottom", fontsize=8, color=INK)
    axes[2].text(1.0, 0.45, "the certificate and Learn then Test\nkeep the reference plan\nin every calibration",
                 ha="center", va="center", fontsize=7.5, color=MUTED)
    handles = [plt.Line2D([], [], ls="", marker=m, ms=5.5, markerfacecolor=c, markeredgewidth=0, label=lab)
               for lab, c, m, _ in SERIES]
    handles.append(plt.Line2D([], [], color=INK, lw=1, ls="--", label=r"bound of the theorem ($\alpha$)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = ROOT / "figures" / "fig08_certificate_risk.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print("written", out)


if __name__ == "__main__":
    main()
