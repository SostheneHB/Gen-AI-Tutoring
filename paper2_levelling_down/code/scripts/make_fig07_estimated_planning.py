"""Figure: when does the certificate with estimated effects become usable?  (reviewer 1, point 5)

    python code/scripts/make_fig07_estimated_planning.py

For a target narrowing rho and a margin delta, the conservative score allows the margin delta only if, roughly,
2 w(n) <= delta * rho, where w(n) is the empirical Bernstein-Serfling half-width of one arm mean with n students per
group and arm (sigma = standard deviation of the final outcome, rho_n = 0.8, beta' = beta / (20 (L + 1)),
beta = 0.02, L = 4 plans; Bardenet and Maillard 2015, Thm 4.3).  The figure gives the smallest such n.
Horizontal lines: the cohort sizes of experiment 17 (10,000, 50,000 and 250,000 students, two groups, five arms).
Writes figures/fig07_estimated_planning.png.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KAPPA = 7 / 3 + 3 / math.sqrt(2)
BETA, L, RHO_N = 0.02, 4, 0.8
LOG = math.log(20 * (L + 1) / BETA)


def width(n, sigma):
    return sigma * math.sqrt(2 * RHO_N * LOG / n) + KAPPA * LOG / n


def needed(rho, delta, sigma):
    lo, hi = 1.0, 1e12
    for _ in range(200):                                   # bisection on log n
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if 2 * width(mid, sigma) > delta * rho else (lo, mid)
    return hi


def main():
    rhos = np.logspace(math.log10(0.003), math.log10(0.3), 120)
    colors = {0.1: "#0072B2", 0.3: "#D55E00", 0.5: "#009E73"}       # Okabe-Ito, colour-blind safe
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.fill_between(rhos, [needed(r, 0.3, 0.18) for r in rhos], [needed(r, 0.3, 0.32) for r in rhos], color=colors[0.3], alpha=0.15, lw=0)
    for d, c in colors.items():
        y = [needed(r, d, 0.25) for r in rhos]
        ax.plot(rhos, y, color=c, lw=2)
        ax.text(rhos[-1] * 1.04, y[-1], f"$\\delta={d}$", color="#333333", fontsize=9, va="center")
    for cohort, lab in ((10_000, "10,000"), (50_000, "50,000"), (250_000, "250,000")):
        ax.axhline(cohort / 10, color="#777777", lw=1, ls="--")
        ax.text(0.0033, cohort / 10 * 1.12, f"cohort of {lab}", color="#555555", fontsize=8)
    ax.plot([0.112], [5000], marker="o", ms=8, color="#333333")
    ax.annotate("first intervention\nin the simulator", (0.112, 5000), xytext=(0.03, 1.6e3), fontsize=8, color="#333333", ha="right",
                arrowprops=dict(arrowstyle="-", color="#777777", lw=1))
    ax.axvline(0.003, color="#999999", lw=1, ls=":")
    ax.text(0.0033, 1.2e5, "size of the effects in\nthe randomized experiments", fontsize=8, color="#555555")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(6e2, 1e8); ax.set_xlim(0.0028, 0.42)
    ax.set_xlabel("target narrowing of the gap $\\rho$")
    ax.set_ylabel("students needed per group and arm")
    ax.grid(True, which="major", color="#e5e5e5", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = ROOT / "figures" / "fig07_estimated_planning.png"
    fig.savefig(out, dpi=200)
    print("written", out, "| needed at rho=0.05, delta=0.3, sigma=0.25:", round(needed(0.05, 0.3, 0.25)),
          "| at rho=0.003:", round(needed(0.003, 0.3, 0.45)))


if __name__ == "__main__":
    main()
