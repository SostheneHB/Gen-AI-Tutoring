"""Table of the Eedi study (real sensitive attribute: PremiumPupil).

    python make_tables_eedi.py --main <out_eedi> --strict <out_eedi_strict> --dest ../Revised_manuscript

Writes tab_eedi.tex and prints the numbers quoted in the text.
"""
import argparse
import json
import os

import numpy as np


def load(folder, name):
    runs = []
    for j in range(5):
        p = os.path.join(folder, f"{name}_{j}", "results.json")
        if os.path.exists(p):
            runs.append(json.load(open(p)))
    return runs


def excl(ci):
    return ci[0] > 0 or ci[1] < 0


def row(r):
    p, m = r["replay"], r["replay"]["models"]
    mse = lambda k, h: m[k]["mse"][h]["k_t"]
    return dict(obs=p["terminal_gap_observed"], ci=p["terminal_gap_observed_school_ci"],
                raw=m["edmd_raw"]["terminal_gap_pred"], cal=m["edmd_groupcal"]["terminal_gap_pred"],
                proj=m["edmd_proj"]["terminal_gap_pred"], ratio=mse("edmd_proj", "1") / mse("edmd_raw", "1"),
                e_raw=excl(m["edmd_raw"]["gap_error_school_ci"]), e_cal=excl(m["edmd_groupcal"]["gap_error_school_ci"]),
                e_proj=excl(m["edmd_proj"]["gap_error_school_ci"]),
                persist=mse("edmd_proj", "1") > mse("persistence", "1") and mse("edmd_proj", "5") > mse("persistence", "5"),
                tau=r["free_run"]["tau_delta"], tau_ci=r["free_run"]["tau_delta_school_ci"], n_test=r["n_test"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--strict", default=None)
    ap.add_argument("--dest", default="../Revised_manuscript")
    a = ap.parse_args()
    rows = [row(r) for r in load(a.main, "class_disjoint")]
    f = lambda x: f"\\({x:+.3f}\\)"
    out = [r"\begin{tabular}{@{}lrrrrrrl@{}}", r"\toprule",
           r"Split & Observed [95\% CI] & Raw & Recal. & Proj. & MSE ratio & \(\widehat\tau_\Delta\) [95\% CI] & Gap error \(\not\ni0\)\\",
           r"\midrule"]
    for j, x in enumerate(rows):
        flags = "/".join("Y" if x[k] else "n" for k in ("e_raw", "e_cal", "e_proj"))
        out.append(f"{j + 1} & {f(x['obs'])} \\([{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]\\) & {f(x['raw'])} & {f(x['cal'])} & "
                   f"\\({x['proj']:+.4f}\\) & {x['ratio']:.1f} & \\({x['tau']:+.4f}\\) \\([{x['tau_ci'][0]:+.4f},{x['tau_ci'][1]:+.4f}]\\) & {flags}\\\\")
    mean = lambda k: np.mean([x[k] for x in rows])
    out += [r"\midrule",
            f"Mean & {f(mean('obs'))} & {f(mean('raw'))} & {f(mean('cal'))} & \\({mean('proj'):+.4f}\\) & "
            f"{min(x['ratio'] for x in rows):.1f}--{max(x['ratio'] for x in rows):.1f} & \\({mean('tau'):+.4f}\\) & "
            f"{sum(x['e_raw'] for x in rows)}/{sum(x['e_cal'] for x in rows)}/{sum(x['e_proj'] for x in rows)} of 5\\\\"]
    if a.strict:
        srows = [row(r) for r in load(a.strict, "class_disjoint_strict")]
        if srows:
            sm = lambda k: np.mean([x[k] for x in srows])
            out.append(f"Strict (no shared class) & {f(sm('obs'))} & {f(sm('raw'))} & {f(sm('cal'))} & \\({sm('proj'):+.4f}\\) & "
                       f"{min(x['ratio'] for x in srows):.1f}--{max(x['ratio'] for x in srows):.1f} & \\({sm('tau'):+.4f}\\) & "
                       f"{sum(x['e_raw'] for x in srows)}/{sum(x['e_cal'] for x in srows)}/{sum(x['e_proj'] for x in srows)} of {len(srows)}\\\\")
            print("strict: test students per split", [x["n_test"] for x in srows])
    out += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.dest, "tab_eedi.tex"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("projected worse than persistence at h=1 and h=5 in", sum(x["persist"] for x in rows), "of 5 splits")


if __name__ == "__main__":
    main()
