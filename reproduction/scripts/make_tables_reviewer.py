"""LaTeX tables from out_reviewer/*.json (reviewer_experiments.py).

    python make_tables_reviewer.py
writes tab_equal_budget.tex, tab_scenarios.tex and, when replicates.json exists, tab_replicates50.tex
"""
import json
import os

f = lambda e: f"\\({e['mean']:+.3f}\\) \\([{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]\\)"
r3 = lambda e: f"{e['mean']:.2f}"

b = json.load(open("out_reviewer/budget.json"))
rows = []
names = {"fair": "Floor-constrained MPC (\\(c=2\\))", "topup_u1": "Targeted top-up, equal \\(\\mathcal U_1\\)",
         "topup_u2": "Targeted top-up, equal \\(\\mathcal U_2\\)", "more_u1": "Uniform increase, equal \\(\\mathcal U_1\\)"}
for g in ("0.5", "1.0", "2.0"):
    for i, (arm, lab) in enumerate(names.items()):
        e = b[g][arm]
        rows.append(f"{('\\(\\gamma=' + g.rstrip('0').rstrip('.') + '\\)') if i == 0 else ''} & {lab} & {f(e['abs_gap'])} & {f(e['k0'])} & {f(e['k1'])} & {r3(e['u1_ratio'])} & {r3(e['u2_ratio'])}\\\\")
    rows.append("\\addlinespace")
open("tab_equal_budget.tex", "w").write(
    "\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n & Arm & \\(\\Delta|\\delta_T|\\) & \\(\\Delta\\overline k_{T,0}\\) & \\(\\Delta\\overline k_{T,1}\\) & \\(\\mathcal U_1\\) ratio & \\(\\mathcal U_2\\) ratio\\\\\n\\midrule\n"
    + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

s = json.load(open("out_reviewer/scenarios.json"))
labels = {"weaker_response_group1": "Group 1 responds half as much",
          "slower_learning_more_noise": "Slower learning, twice the noise",
          "diminishing_returns": "Diminishing returns to help"}
arms = [("proj_vs_raw", "Projected MPC vs.\\ original"), ("con_vs_raw", "Gap-constrained vs.\\ original"),
        ("fair_vs_ga_ref", "Floor-constrained vs.\\ group-aware reference"), ("topup_u1_vs_ga_ref", "Top-up (equal \\(\\mathcal U_1\\)) vs.\\ reference")]
rows = []
for sc, lab in labels.items():
    for i, (k, alab) in enumerate(arms):
        e = s[sc][k]
        rows.append(f"{lab if i == 0 else ''} & {alab} & {f(e['abs_gap'])} & {f(e['k0'])} & {f(e['k1'])}\\\\")
    rows.append("\\addlinespace")
open("tab_scenarios.tex", "w").write(
    "\\begin{tabular}{@{}llrrr@{}}\n\\toprule\nScenario & Comparison & \\(\\Delta|\\delta_T|\\) & \\(\\Delta\\overline k_{T,0}\\) & \\(\\Delta\\overline k_{T,1}\\)\\\\\n\\midrule\n"
    + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

if os.path.exists("out_reviewer/replicates.json"):
    r = json.load(open("out_reviewer/replicates.json"))
    rows = []
    for row in r["rows"]:
        c = row["config"]
        name = f"\\(\\gamma={c['gamma']:g}\\), rank {c['rank']}, depth {c['depth']}, \\(T={c['horizon']}\\), \\(\\beta_{{\\mathrm{{ind}}}}={c['beta']:g}\\)"
        p, q, l = row["proj_abs_gap"], row["con_abs_gap"], row["con_lower_group_k"]
        rows.append(f"{name} & {f(p)} & {p['p_holm']:.1e} & {f(q)} & {f(l)} & {l['p_holm']:.1e}\\\\")
    open("tab_replicates50.tex", "w").write(
        "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\nSetting & Projected: \\(\\Delta|\\delta_T|\\) & Holm \\(p\\) & Constrained: \\(\\Delta|\\delta_T|\\) & Constrained: lower group \\(\\Delta\\overline k_T\\) & Holm \\(p\\)\\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print("tables written")
