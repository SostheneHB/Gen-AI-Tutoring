"""Tables and figure added in the fourth revision.

    python make_tables_R4.py --synthetic out_synthetic --extras out_synthetic_extras.json \
        --structural out_structural.json --flow sample_flow.csv --dest ../Revised_manuscript

Writes tab_groupwise.tex, tab_beta_gap.tex, tab_structural.tex, tab_flow.tex and
fig_horizon50.png, and prints the counts quoted in the text.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

L = lambda p: json.load(open(p))
f = lambda e: f"{e['mean']:+.3f} \\scriptsize[{e['ci'][0]:+.3f},\\,{e['ci'][1]:+.3f}]"
sig = lambda e: "lower" if e["ci"][1] < 0 else ("higher" if e["ci"][0] > 0 else "unclear")


def all_settings(S):
    rows, seen = [], set()
    for r in [r for v in S["one_factor"].values() for r in v] + S["gamma_x_beta"]:
        k = json.dumps(r["config"], sort_keys=True)
        if k not in seen:
            seen.add(k)
            rows.append(r)
    rows.sort(key=lambda r: tuple(r["config"][k] for k in ("gamma", "beta", "rank", "depth", "horizon")))
    return rows


def groupwise(S):
    rows = all_settings(S)
    arms = [("effect_mpc_proj_vs_raw", "Projected MPC"), ("effect_gap_penalty_vs_raw", "Gap-penalty MPC"),
            ("effect_gap_constraint_vs_raw", "Gap-constrained MPC")]
    out = [r"\begin{tabular}{@{}lllllll@{}}", r"\toprule",
           r"Setting & Arm & \(\Delta|\delta_T|\) & \(\Delta\overline k_{T,0}\) & \(\Delta\overline k_{T,1}\) & "
           r"\(\Delta\overline d_{T,0}\) & \(\Delta\overline d_{T,1}\)\\", r"\midrule"]
    keep = [dict(gamma=g, beta=b, rank=4, depth=3, horizon=T) for g, b, T in
            [(0.5, 8, 25), (1.0, 8, 25), (2.0, 8, 25), (-1.0, 8, 25), (1.0, 0, 25), (1.0, 32, 25), (1.0, 8, 50)]]
    for cfg in keep:
        r = next(r for r in rows if all(r["config"][k] == v for k, v in cfg.items()))
        lab = f"\\(\\gamma={cfg['gamma']:g}\\), \\(\\beta_{{\\mathrm{{ind}}}}={cfg['beta']:g}\\)" + \
              (f", \\(T={cfg['horizon']}\\)" if cfg["horizon"] != 25 else "")
        for j, (key, name) in enumerate(arms):
            e = r[key]
            out.append(f"{lab if j == 0 else ''} & {name} & {f(e['abs_gap'])} & {f(e['k_g0'])} & {f(e['k_g1'])} & "
                       f"{f(e['d_g0'])} & {f(e['d_g1'])}\\\\")
        if cfg != keep[-1]:
            out.append(r"\addlinespace")
    out += [r"\bottomrule", r"\end{tabular}"]
    # counts over all 28 settings (worst-group criterion on knowledge)
    counts = {}
    for key, name in arms:
        gapped = [r for r in rows if r["config"]["gamma"] != 0]
        worse = sum(1 for r in gapped if min(r[key]["k_g0"]["ci"][1], r[key]["k_g1"]["ci"][1]) < 0)
        both = sum(1 for r in gapped if r[key]["k_g0"]["ci"][0] > 0 and r[key]["k_g1"]["ci"][0] > 0)
        worst = [min(r[key]["k_g0"]["mean"], r[key]["k_g1"]["mean"]) for r in gapped]
        counts[name] = dict(n=len(gapped), one_group_lower=worse, both_higher=both,
                            worst_group_change=[float(min(worst)), float(max(worst))])
    return out, counts


def beta_gap_table(E):
    out = [r"\begin{tabular}{@{}rrrrllllr@{}}", r"\toprule",
           r"\(\gamma\) & \(\beta_{\mathrm{gap}}\) & \(|u|\) raw & \(|u|\) arm & \(\Delta|\delta_T|\) & \(\Delta\overline k_{T,0}\) & "
           r"\(\Delta\overline k_{T,1}\) & median \(|\lambda|\) & infeas.\\", r"\midrule"]
    for i, b in enumerate(E["beta_gap_grid"]):
        bg = "constraint" if b["beta_gap"] == "constraint" else f"{b['beta_gap']:g}"
        if i and b["gamma"] != E["beta_gap_grid"][i - 1]["gamma"]:
            out.append(r"\midrule")
        out.append(f"{b['gamma']:g} & {bg} & {b['u_raw']['mean']:.3f} & {b['u_alt']['mean']:.3f} & {f(b['abs_gap_change'])} & "
                   f"{f(b['k0_change'])} & {f(b['k1_change'])} & {b['median_abs_lambda']['mean']:.1f} & "
                   f"{100 * b['infeasible']['mean']:.1f}\\%\\\\")
    return out + [r"\bottomrule", r"\end{tabular}"]


def structural_table(D):
    def e(v):
        if v == 0:
            return "0"
        m, x = f"{v:.1e}".split("e")
        return f"\\({m}\\times10^{{{int(x)}}}\\)"
    rng = lambda v: f"{v[0]:.2f}--{v[1]:.2f}"
    ir = lambda v: f"{int(v[0])}" if v[0] == v[1] else f"{int(v[0])}--{int(v[1])}"
    fams = [("synthetic_all_fits", "Simulator (all grid fits)"),
            ("assistments:school_disjoint", "ASSISTments, school-disjoint"),
            ("assistments:school_disjoint_otherlabel", "ASSISTments, other-student labels"),
            ("assistments:student_split", "ASSISTments, student split"),
            ("assistments:subsample", "ASSISTments, 5{,}000-student subsamples"),
            ("assistments:student_split_full_strata", "ASSISTments, full-corpus label")]
    heads = ["Simulator", "School-disj.", "Other labels", "Student split", "Subsamples", "Full label"]
    out = [r"\begin{tabular}{@{}l" + "r" * len(fams) + "@{}}", r"\toprule",
           "Quantity & " + " & ".join(heads) + r"\\", r"\midrule"]
    V = [D[k] for k, _ in fams]
    lines = [("fits", lambda v, k: str(v["n_fits"])),
             (r"\(\max|K-A^\top|\)", lambda v, k: e(v["duality_err"][1])),
             (r"\(\max|P^2-P|\)", lambda v, k: e(v["idempotency_err"][1])),
             (r"\(\max|\Gamma^{-1}P^\top\Gamma-P|\)", lambda v, k: e(v["metric_duality_err"][1])),
             (r"\(\max\epsilon_q\)", lambda v, k: e(v["eps_q"][1])),
             (r"\(\max\|R^\top B_P\|\)", lambda v, k: "--" if k.startswith("assist") else e(v["leak_B"][1])),
             (r"\(\max\|R^\top a_P\|\)", lambda v, k: e(v["leak_a"][1])),
             (r"\(r_1\) (range)", lambda v, k: rng(v["r1"])),
             (r"\(r_2\) (range)", lambda v, k: rng(v["r2"])),
             ("requested rank", lambda v, k: ir(v["requested_rank"])),
             ("candidate rank", lambda v, k: ir(v["candidate_rank"])),
             ("applied rank", lambda v, k: ir(v["applied_rank"])),
             (r"numerical rank of \(G\)", lambda v, k: ir(v["numerical_rank_G"])),
             ("relative SVD cutoff", lambda v, k: e(v["svd_cutoff"][1]))]
    for lab, fn in lines:
        out.append(lab + " & " + " & ".join(fn(v, k) for v, (k, _) in zip(V, fams)) + r"\\")
    return out + [r"\bottomrule", r"\end{tabular}"]


def flow_table(path):
    F = pd.read_csv(path)
    lab = {"feature file": "Rows with a skill and a school", ">= 20 rows": "At least 20 attempts",
           ">= 26 rows": r"At least 26 attempts (full window \(x_0,\ldots,x_{25}\))",
           "labelled schools": r"School has a step after \(t=25\) (label window)"}
    out = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"Stage & Students & Schools & Rows\\", r"\midrule"]
    for _, r in F.iterrows():
        n = lambda v: f"{v:,}".replace(",", "{,}")
        out.append(f"{lab[r['stage']]} & {n(r['students'])} & {n(r['schools'])} & {n(r['rows'])}\\\\")
    return out + [r"\bottomrule", r"\end{tabular}"]


def horizon_figure(E, path):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "legend.frameon": False,
                         "savefig.dpi": 200, "savefig.bbox": "tight"})
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    t = np.arange(1, 51)
    for key, ls, lab in (("horizon50_train25", "-", "trained on 25 steps"), ("horizon50_train50", "--", "trained on 50 steps")):
        r = E[key]
        axs[0].plot(t, r["mpc_raw:err_raw"], ls, color="#2a78d6", lw=1.4, label=f"original model, {lab}")
        axs[0].plot(t, r["mpc_proj:err_proj"], ls, color="#eb6834", lw=1.4, label=f"projected model, {lab}")
        axs[1].plot(t, r["mpc_raw:share_md_above_train99"], ls, color="#2a78d6", lw=1.4, label=f"beyond training 99% radius, {lab}")
        axs[1].plot(t, r["mpc_raw:share_outside_range"], ls, color="#52514e", lw=1.4, label=f"outside training range, {lab}")
    for ax in axs:
        ax.axvline(25, color="#52514e", lw=0.8, ls=":")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("step $t$")
    axs[0].set_ylabel("one-step MSE of the arm's own model")
    axs[0].set_title("(a) One-step error along the MPC trajectories", loc="left", fontsize=9)
    axs[1].set_xlabel("step $t$")
    axs[1].set_ylabel("share of simulated students")
    axs[1].set_title("(b) Distance to the training states (original MPC)", loc="left", fontsize=9)
    axs[0].legend(fontsize=6.5)
    axs[1].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="out_synthetic")
    ap.add_argument("--extras", default="out_synthetic_extras.json")
    ap.add_argument("--structural", default="out_structural.json")
    ap.add_argument("--flow", default="sample_flow.csv")
    ap.add_argument("--dest", default="../Revised_manuscript")
    a = ap.parse_args()
    w = lambda name, lines: open(f"{a.dest}/{name}", "w").write("\n".join(lines) + "\n")
    gw, counts = groupwise(L(f"{a.synthetic}/regimes.json"))
    w("tab_groupwise.tex", gw)
    print(json.dumps(counts, indent=1))
    E = L(a.extras)
    w("tab_beta_gap.tex", beta_gap_table(E))
    w("tab_structural.tex", structural_table(L(a.structural)))
    w("tab_flow.tex", flow_table(a.flow))
    horizon_figure(E, f"{a.dest}/fig_horizon50.png")


if __name__ == "__main__":
    main()
