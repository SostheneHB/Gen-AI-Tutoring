"""
Assignment shares of the retained experiments (post-hoc documentation, 13 October 2026; no outcome is read).

For each of the 30 retained experiments: number of students, number treated, treated share, two-sided binomial
p-value against 1/2, number of classes, and the treated share within each prior-achievement group (low, high),
with a binomial p-value against 1/2.  The assignment probability itself is not recorded in the data; these
shares document how well the design value 1/2 describes each experiment.

    python randomization_table.py      -> outputs/randomization_table.json (aggregates only)
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

D = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main():
    c = pd.read_csv(os.path.join(D, "experiment_conditions.csv"), index_col=0)
    s = pd.read_csv(os.path.join(D, "experiment_prior_stats_and_targets.csv"))
    s = s.drop(columns=["assignment_completed", "problems_completed"])          # no outcome
    m = s.merge(c, on="assignment_log_id", how="inner")
    m = m[(m.in_control + m.in_treatment) == 1]
    m = m.sort_values("assignment_start_time").drop_duplicates(["target_sequence", "student_id"])
    per = m.groupby("target_sequence").agg(n=("student_id", "size"), n_t=("in_treatment", "sum"))
    per["n_c"] = per.n - per.n_t
    per["p_binom"] = [stats.binomtest(int(t), int(n), .5).pvalue for t, n in zip(per.n_t, per.n)]
    keep = per.index[(per.p_binom >= .10) & (per.n_t >= 56) & (per.n_c >= 56)]
    m = m[m.target_sequence.isin(keep)].copy()
    x = m.student_prior_average_problem_correctness
    q1, q2 = np.nanquantile(x, [1 / 3, 2 / 3])
    m["group"] = np.where(x.isna(), "missing", np.where(x <= q1, "low", np.where(x > q2, "high", "middle")))
    rows = []
    for i, (e, d) in enumerate(sorted(m.groupby("target_sequence"), key=lambda kv: -len(kv[1]))):
        r = dict(experiment=i + 1, n=int(len(d)), n_treated=int(d.in_treatment.sum()), share=float(d.in_treatment.mean()),
                 p_binom=float(stats.binomtest(int(d.in_treatment.sum()), len(d), .5).pvalue), classes=int(d.class_id.nunique()))
        for g in ("low", "high"):
            dg = d[d.group == g]
            r[f"share_{g}"] = float(dg.in_treatment.mean()) if len(dg) else None
            r[f"p_{g}"] = float(stats.binomtest(int(dg.in_treatment.sum()), len(dg), .5).pvalue) if len(dg) else None
        rows.append(r)
    sh = np.array([r["share"] for r in rows])
    out = dict(experiments=len(rows), share_min=float(sh.min()), share_max=float(sh.max()), share_median=float(np.median(sh)),
               min_p_binom=float(min(r["p_binom"] for r in rows)),
               groups_p_below_05=int(sum((r["p_low"] or 1) < .05 for r in rows) + sum((r["p_high"] or 1) < .05 for r in rows)),
               rows=rows)
    json.dump(out, open(os.path.join(OUT, "randomization_table.json"), "w"), indent=1)
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
