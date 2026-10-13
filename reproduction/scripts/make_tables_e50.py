"""
Tables of Section 7.4 and of the regime appendix from the 50-replicate runs (e50_runs.py):
  tab_regimes50_knowledge.tex  rows of Table tab:regimes (knowledge first: mean knowledge, |knowledge gap|, |disparity gap|)
  tab_regimes_full.tex, tab_budget.tex   (make_tables_R3 layout)
  tab_groupwise.tex, tab_beta_gap.tex    (make_tables_R4 layout)

    python make_tables_e50.py --src out_synthetic50 --dest ../Revised_manuscript
"""
import argparse
import json
import os

import make_tables_R3 as r3
import make_tables_R4 as r4

ARMS = ("effect_mpc_proj_vs_raw", "effect_gap_penalty_vs_raw", "effect_gap_constraint_vs_raw")
SEL = [((0.0, 8.0, 25), r"\(\gamma=0\), \(\beta_{\mathrm{ind}}=8\)"), ((0.5, 8.0, 25), r"\(\gamma=0.5\), \(\beta_{\mathrm{ind}}=8\)"),
       ((1.0, 8.0, 25), r"\(\gamma=1\), \(\beta_{\mathrm{ind}}=8\) (main setting)"), ((2.0, 8.0, 25), r"\(\gamma=2\), \(\beta_{\mathrm{ind}}=8\)"),
       ((-1.0, 8.0, 25), r"\(\gamma=-1\), \(\beta_{\mathrm{ind}}=8\)"), ((1.0, 0.0, 25), r"\(\gamma=1\), \(\beta_{\mathrm{ind}}=0\)"),
       ((1.0, 32.0, 25), r"\(\gamma=1\), \(\beta_{\mathrm{ind}}=32\)"), ((1.0, 8.0, 50), r"\(\gamma=1\), \(\beta_{\mathrm{ind}}=8\), \(T=50\)")]


def cell(e):
    m = e["mean"]
    s = f"{m:+.3f}" if abs(m) >= 1.5e-3 else (f"{m:+.4f}" if e["p_holm"] < .05 else "0.000")
    return f"\\({s}{'^*' if e['p_holm'] < .05 else ''}\\)"


def main_rows(rows):
    out = []
    for (g, b, T), lab in SEL:
        r = next(r for r in rows if (r["config"]["gamma"], r["config"]["beta"], r["config"]["horizon"]) == (g, b, T)
                 and r["config"]["rank"] == 4 and r["config"]["depth"] == 3)
        cells = [f"\\({r['mpc_raw']['kgap']['mean']:+.3f}\\)"]
        for a in ARMS:
            e = r[a]
            cells += [cell(e["knowledge"]), cell(e["abs_kgap"]), cell(e["abs_gap"])]
        out.append(lab + " & " + " & ".join(cells) + "\\\\")
    return out


def add_kgap(lines, S):
    """Add the change of the signed knowledge gap, with its interval from the paired replicates."""
    rows = r4.all_settings(S)
    keep = [dict(gamma=g, beta=b, rank=4, depth=3, horizon=T) for g, b, T in
            [(0.5, 8, 25), (1.0, 8, 25), (2.0, 8, 25), (-1.0, 8, 25), (1.0, 0, 25), (1.0, 32, 25), (1.0, 8, 50)]]
    arms = ["effect_mpc_proj_vs_raw", "effect_gap_penalty_vs_raw", "effect_gap_constraint_vs_raw"]
    vals = [r[a]["kgap"] for cfg in keep for r in rows if all(r["config"][k] == v for k, v in cfg.items()) for a in arms]
    out, i = [], 0
    for ln in lines:
        if ln.startswith("\\begin{tabular}"):
            ln = "\\begin{tabular}{@{}llllllll@{}}"
        elif "\\(\\Delta\\overline d_{T,1}\\)" in ln:
            ln = ln.replace("\\(\\Delta\\overline k_{T,1}\\) &", "\\(\\Delta\\overline k_{T,1}\\) & \\(\\Delta\\delta^k_T\\) &")
        elif "\\scriptsize" in ln and ln.rstrip().endswith("\\\\"):
            cells = ln.rstrip().rstrip("\\").split(" & ")
            cells.insert(5, r4.f(vals[i])); i += 1
            ln = " & ".join(cells) + "\\\\"
        out.append(ln)
    assert i == len(vals), (i, len(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out_synthetic50")
    ap.add_argument("--dest", default="../Revised_manuscript")
    a = ap.parse_args()
    S = json.load(open(os.path.join(a.src, "regimes.json")))
    E = json.load(open(os.path.join(a.src, "extras.json")))
    w = lambda name, lines: open(os.path.join(a.dest, name), "w").write("\n".join(lines) + "\n")
    w("tab_regimes50_knowledge.tex", main_rows(S["one_factor"]["all"]))
    full, budget = r3.regimes_tables(S)
    w("tab_regimes_full.tex", full)
    w("tab_budget.tex", budget)
    gw, counts = r4.groupwise(S)
    gw = add_kgap(gw, S)
    w("tab_groupwise.tex", gw)
    w("tab_beta_gap.tex", r4.beta_gap_table(E))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
