"""Tables and calibration figure added in the third revision.

    python make_tables_R3.py --real out_time --synthetic out_synthetic --dest ../Revised_manuscript

Writes tab_paired_differences.tex, tab_calibration.tex, tab_regimes_full.tex,
tab_budget.tex and fig_real_calibration.png from the JSON outputs of
assistments_pipeline.py and synthetic_regimes.py.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

L = lambda p: json.load(open(p))


def paired_differences(R):
    f = lambda x: f"{1e3 * x:+.2f}"
    out = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
           r"Predictor minus raw EDMD & \(h\) & Split 1 & Split 2 & Split 3 & Split 4 & Split 5 & Mean [range]\\",
           r"\midrule"]
    for key, lab in [("persistence", "Persistence"), ("affine_ls", "Affine LS"), ("gbm", "Gradient boosting"),
                     ("edmd_groupcal", "Recalibrated EDMD"), ("edmd_proj", "Projected EDMD")]:
        for h in ["1", "25"]:
            cells, ds = [], []
            for r in R:
                d = r["replay"]["paired_vs_raw"][key][h]
                ds.append(d["diff"])
                cells.append(f"{f(d['diff'])} \\scriptsize[{f(d['school_ci'][0])},\\,{f(d['school_ci'][1])}]")
            out.append(f"{lab} & {h} & " + " & ".join(cells) + f" & {f(np.mean(ds))} [{f(min(ds))},\\,{f(max(ds))}]\\\\")
    out += [r"\midrule", r"\multicolumn{8}{@{}l}{\emph{Predicted minus observed final gap in \(k_{25}\) (one-step replay)}}\\"]
    for key, lab in [("edmd_raw", "Raw EDMD"), ("edmd_groupcal", "Recalibrated EDMD"), ("edmd_proj", "Projected EDMD")]:
        cells, ds = [], []
        for r in R:
            m = r["replay"]["models"][key]
            ds.append(m["gap_error"])
            cells.append(f"{f(m['gap_error'])} \\scriptsize[{f(m['gap_error_school_ci'][0])},\\,{f(m['gap_error_school_ci'][1])}]")
        out.append(f"{lab} & -- & " + " & ".join(cells) + f" & {f(np.mean(ds))} [{f(min(ds))},\\,{f(max(ds))}]\\\\")
    return out + [r"\bottomrule", r"\end{tabular}"]


def calibration_table(R):
    out = [r"\begin{tabular}{@{}lllrrrrr@{}}", r"\toprule",
           r"Model & Group & Quantity & Split 1 & Split 2 & Split 3 & Split 4 & Split 5\\", r"\midrule"]
    for m, lab in [("edmd_raw", "Raw EDMD"), ("edmd_groupcal", "Recalibrated EDMD")]:
        for g in "01":
            for q, ql, fmt in [("citl", "CITL \\(\\times10^3\\)", lambda x: f"{1e3 * x:+.1f}"),
                               ("slope", "Slope", lambda x: f"{x:.3f}")]:
                cells = []
                for r in R:
                    c = r["calibration"][m][g]
                    lo, hi = c[q + "_school_ci"]
                    cells.append(f"{fmt(c[q])} \\scriptsize[{fmt(lo)},\\,{fmt(hi)}]")
                first = lab if (g == "0" and q == "citl") else ""
                out.append(f"{first} & {g if q == 'citl' else ''} & {ql} & " + " & ".join(cells) + r"\\")
        if m == "edmd_raw":
            out.append(r"\midrule")
    return out + [r"\bottomrule", r"\end{tabular}"]


def calibration_figure(R, path):
    INK2, GRID = "#52514e", "#e6e5e0"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                         "savefig.dpi": 200, "savefig.bbox": "tight"})
    cols = {"edmd_raw": ("#2a78d6", "Raw EDMD"), "edmd_groupcal": ("#1baf7a", "Recalibrated EDMD"),
            "edmd_proj": ("#eb6834", "Projected EDMD")}
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4))
    for ax, g in zip(axs, "01"):
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
        for m, (c, lab) in cols.items():
            for j, r in enumerate(R):
                rel = np.array(r["calibration"][m][g]["reliability_deciles"])
                ax.plot(rel[:, 0], rel[:, 1], "-o", color=c, ms=3, lw=1.2, alpha=0.75, label=lab if j == 0 else None)
        ax.set_xlabel("Mean predicted $k_{t+1}$ (decile)")
        ax.set_ylabel("Mean observed $k_{t+1}$")
        ax.set_title(f"({'ab'[int(g)]}) Group {g}, five test splits", loc="left", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    axs[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)


def regimes_tables(S):
    rows, seen = [], set()
    for r in [r for v in S["one_factor"].values() for r in v] + S["gamma_x_beta"]:
        k = json.dumps(r["config"], sort_keys=True)
        if k not in seen:
            seen.add(k)
            rows.append(r)
    rows.sort(key=lambda r: tuple(r["config"][k] for k in ("gamma", "beta", "rank", "depth", "horizon")))
    f = lambda e: f"{e['mean']:+.3f} \\scriptsize[{e['ci'][0]:+.3f},\\,{e['ci'][1]:+.3f}]"
    full = [r"\begin{tabular}{@{}rrrrrllllrrr@{}}", r"\toprule",
            r"\(\gamma\) & \(\beta\) & rank & depth & \(T\) & Proj.: \(\Delta|\delta_T|\) & Proj.: \(\Delta\overline k_T\) & Constr.: \(\Delta|\delta_T|\) & Constr.: \(\Delta\overline k_T\) & \(|u|\) raw & \(|u|\) constr. & infeas.\\",
            r"\midrule"]
    for r in rows:
        c, P, C = r["config"], r["effect_mpc_proj_vs_raw"], r["effect_gap_constraint_vs_raw"]
        full.append(f"{c['gamma']:g} & {c['beta']:g} & {c['rank']} & {c['depth']} & {c['horizon']} & {f(P['abs_gap'])} & "
                    f"{f(P['knowledge'])} & {f(C['abs_gap'])} & {f(C['knowledge'])} & "
                    f"{r['mpc_raw']['mean_abs_control']['mean']:.3f} & {r['gap_constraint']['mean_abs_control']['mean']:.3f} & "
                    f"{100 * r['gap_constraint']['infeasible_fraction']['mean']:.1f}\\%\\\\")
    full += [r"\bottomrule", r"\end{tabular}"]
    budget = [r"\begin{tabular}{@{}rrrrll@{}}", r"\toprule",
              r"\(\gamma\) & \(\rho\) & \(|u|\) raw & \(|u|\) constr. & \(\Delta|\delta_T|\) & \(\Delta\overline k_T\)\\", r"\midrule"]
    for b in [b for b in S["budget_check"] if b["rho"] <= 5]:   # rho=8 saturates the bisection interval
        budget.append(f"{b['gamma']:g} & {b['rho']:g} & {b['raw_mean_abs_u']['mean']:.3f} & {b['con_mean_abs_u']['mean']:.3f} & "
                      f"{f(b['abs_gap_change'])} & {f(b['knowledge_change'])}\\\\")
    budget += [r"\bottomrule", r"\end{tabular}"]
    return full, budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="out_time")
    ap.add_argument("--synthetic", default="out_synthetic")
    ap.add_argument("--dest", default="../Revised_manuscript")
    a = ap.parse_args()
    R = [L(f"{a.real}/school_disjoint_{j}/results.json") for j in range(5)]
    w = lambda name, lines: open(f"{a.dest}/{name}", "w").write("\n".join(lines) + "\n")
    w("tab_paired_differences.tex", paired_differences(R))
    w("tab_calibration.tex", calibration_table(R))
    calibration_figure(R, f"{a.dest}/fig_real_calibration.png")
    full, budget = regimes_tables(L(f"{a.synthetic}/regimes.json"))
    w("tab_regimes_full.tex", full)
    w("tab_budget.tex", budget)


if __name__ == "__main__":
    main()
