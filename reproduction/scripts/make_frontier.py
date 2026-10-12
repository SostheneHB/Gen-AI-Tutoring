"""Figure and table of the budget frontier (fair_frontier.py outputs).

    python make_frontier.py --inputs out_frontier_g0.5.json out_frontier_g1.json out_frontier_g2.json \
        --dest ../Revised_manuscript
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f = lambda e: f"{e['mean']:+.3f} \\scriptsize[{e['ci'][0]:+.3f},\\,{e['ci'][1]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+")
    ap.add_argument("--dest", default="../Revised_manuscript")
    a = ap.parse_args()
    R = [json.load(open(p)) for p in a.inputs]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "legend.frameon": False,
                         "savefig.dpi": 200, "savefig.bbox": "tight"})
    fig, axs = plt.subplots(1, len(R), figsize=(3.4 * len(R), 3.3), sharey=False)
    for ax, r, lab in zip(axs, R, "abc"):
        for kind, col, name in (("plain", "#2a78d6", "more control (smaller $\\rho$)"),
                                ("fair", "#1baf7a", "Pareto-fair MPC (budget cap $c$)")):
            arms = [(k, v) for k, v in r["arms"].items() if k.startswith(kind)]
            arms.sort(key=lambda kv: kv[1]["u_ratio"]["mean"])
            x = [v["u_ratio"]["mean"] for _, v in arms]
            y = [-v["abs_gap"]["mean"] for _, v in arms]
            ax.plot(x, y, "-o", color=col, ms=4, lw=1.5, label=name)
        ax.set_xlabel("mean $|u|$ relative to the reference MPC")
        ax.set_title(f"({lab}) $\\gamma={r['gamma']:g}$", loc="left", fontsize=9)
    axs[0].set_ylabel("reduction of the final gap $|\\delta_T|$")
    axs[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{a.dest}/fig_frontier.png")

    out = [r"\begin{tabular}{@{}rlrllll@{}}", r"\toprule",
           r"\(\gamma\) & Arm & \(|u|\) ratio & \(\Delta|\delta_T|\) & \(\Delta\overline k_{T,0}\) & \(\Delta\overline k_{T,1}\) & worst group (min)\\",
           r"\midrule"]
    for i, r in enumerate(R):
        if i:
            out.append(r"\midrule")
        for name, e in sorted(r["arms"].items(), key=lambda kv: (kv[0].split("_")[0] != "plain", kv[1]["u_ratio"]["mean"])):
            kind, x = name.split("_")
            lab = f"more control, \\(\\rho={x}\\)" if kind == "plain" else f"Pareto-fair, \\(c={x}\\)"
            if kind == "plain" and x == "1":
                continue
            out.append(f"{r['gamma']:g} & {lab} & {e['u_ratio']['mean']:.2f} & {f(e['abs_gap'])} & {f(e['k0'])} & "
                       f"{f(e['k1'])} & {e['worst']['min']:+.4f}\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    open(f"{a.dest}/tab_frontier.tex", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
