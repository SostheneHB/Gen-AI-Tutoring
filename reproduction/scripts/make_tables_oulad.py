"""Table of the OULAD study (declared attributes: deprivation, disability, gender).

    python make_tables_oulad.py --oulad <out_oulad> --dest ../Revised_manuscript

Writes tab_oulad.tex: the five splits for deprivation (the main attribute) and
mean rows for disability and gender.
"""
import argparse
import json
import os

import numpy as np

from make_tables_eedi import row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oulad", required=True)
    ap.add_argument("--dest", default="../Revised_manuscript")
    a = ap.parse_args()
    load = lambda att: [row(json.load(open(os.path.join(a.oulad, att, f"{att}_student_split_{j}", "results.json"))))
                        for j in range(5)]
    f = lambda x: f"\\({x:+.3f}\\)"
    out = [r"\begin{tabular}{@{}llrrrrrrl@{}}", r"\toprule",
           r"Attribute & Split & Observed [95\% CI] & Raw & Recal. & Proj. & MSE ratio & \(\widehat\tau_\Delta\) & Gap error \(\not\ni0\)\\",
           r"\midrule"]
    imd = load("imd")
    for j, x in enumerate(imd):
        flags = "/".join("Y" if x[k] else "n" for k in ("e_raw", "e_cal", "e_proj"))
        out.append(f"{'Deprivation (IMD 0--30\\%)' if j == 0 else ''} & {j + 1} & {f(x['obs'])} \\([{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]\\) & "
                   f"{f(x['raw'])} & {f(x['cal'])} & \\({x['proj']:+.4f}\\) & {x['ratio']:.1f} & \\({x['tau']:+.4f}\\) & {flags}\\\\")
    out.append(r"\midrule")
    for att, lab, rows in (("imd", "Deprivation", imd), ("disability", "Disability", load("disability")), ("gender", "Gender (female)", load("gender"))):
        m = lambda k: np.mean([x[k] for x in rows])
        nobs = sum((x["ci"][0] > 0 or x["ci"][1] < 0) for x in rows)
        out.append(f"{lab} & mean & {f(m('obs'))} ({nobs}/5 \\(\\not\\ni0\\)) & {f(m('raw'))} & {f(m('cal'))} & \\({m('proj'):+.4f}\\) & "
                   f"{min(x['ratio'] for x in rows):.1f}--{max(x['ratio'] for x in rows):.1f} & \\({m('tau'):+.4f}\\) & "
                   f"{sum(x['e_raw'] for x in rows)}/{sum(x['e_cal'] for x in rows)}/{sum(x['e_proj'] for x in rows)} of 5\\\\")
        print(att, "gap errors raw/recal:", round(m('raw') - m('obs'), 4), round(m('cal') - m('obs'), 4),
              "persistence 5/5:", sum(x["persist"] for x in rows))
    out += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.dest, "tab_oulad.tex"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
