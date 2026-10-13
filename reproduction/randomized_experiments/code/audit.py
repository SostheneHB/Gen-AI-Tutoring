"""
Data audit of the ASSISTments randomized experiments (OSF k8ph9), before any effect is estimated.

Uses only assignment and baseline covariates (no outcome), so that the analysis plan can be fixed
before the outcomes are looked at.  Writes outputs/audit.json and prints the sample flow.

Rules (fixed here, before outcome analysis):
  R1  keep assignment records in exactly one of the two compared arms (in_control xor in_treatment);
  R2  keep experiments whose treated share is compatible with Bernoulli(1/2): two-sided binomial p >= 0.10;
  R3  keep experiments with at least 56 students in each arm (5(k+2)+1 with k = 9, as the data authors);
  R4  one record per student and experiment (the first assignment start);
  R5  group = prior achievement: tercile of student_prior_average_problem_correctness, cut points
      computed once on all retained records; low = bottom tercile, high = top tercile; records with a
      missing value form their own category and are reported, not imputed.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

D = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
COV = ["student_prior_assignments_started", "student_prior_assignments_percent_completed",
       "student_prior_average_problem_correctness", "student_prior_average_problem_attempt_count",
       "student_prior_average_problem_hint_count", "student_prior_median_ln_problem_time_on_task"]


def main():
    c = pd.read_csv(os.path.join(D, "experiment_conditions.csv"), index_col=0)
    s = pd.read_csv(os.path.join(D, "experiment_prior_stats_and_targets.csv"))
    s = s.drop(columns=["assignment_completed", "problems_completed"])          # no outcome in the audit
    m = s.merge(c, on="assignment_log_id", how="inner")
    flow = [("assignment records with a condition", len(m), m.target_sequence.nunique())]
    m = m[(m.in_control + m.in_treatment) == 1]
    flow.append(("R1 exactly one of the two arms", len(m), m.target_sequence.nunique()))
    m = m.sort_values("assignment_start_time").drop_duplicates(["target_sequence", "student_id"])
    flow.append(("R4 one record per student and experiment", len(m), m.target_sequence.nunique()))
    per = m.groupby("target_sequence").agg(n=("student_id", "size"), n_t=("in_treatment", "sum"))
    per["n_c"] = per.n - per.n_t
    per["p_binom"] = [stats.binomtest(int(t), int(n), .5).pvalue for t, n in zip(per.n_t, per.n)]
    keep2 = per.index[per.p_binom >= .10]
    m2 = m[m.target_sequence.isin(keep2)]
    flow.append(("R2 treated share compatible with 1/2", len(m2), m2.target_sequence.nunique()))
    keep3 = per.index[(per.p_binom >= .10) & (per.n_t >= 56) & (per.n_c >= 56)]
    m3 = m[m.target_sequence.isin(keep3)].copy()
    flow.append(("R3 at least 56 students per arm", len(m3), m3.target_sequence.nunique()))
    x = m3.student_prior_average_problem_correctness
    q1, q2 = np.nanquantile(x, [1 / 3, 2 / 3])
    m3["group"] = np.where(x.isna(), "missing", np.where(x <= q1, "low", np.where(x > q2, "high", "middle")))
    # balance of baseline covariates between arms (standardized mean differences), per experiment
    smd = []
    for e, d in m3.groupby("target_sequence"):
        for v in COV:
            a, b = d.loc[d.in_treatment == 1, v].dropna(), d.loc[d.in_treatment == 0, v].dropna()
            sd = np.sqrt((a.var() + b.var()) / 2)
            if sd > 0:
                smd.append(abs(a.mean() - b.mean()) / sd)
    smd = np.array(smd)
    grp = m3.groupby(["target_sequence", "group"]).size().unstack(fill_value=0)
    out = dict(
        flow=[dict(step=a, records=int(b), experiments=int(c_)) for a, b, c_ in flow],
        tercile_cut_points=[float(q1), float(q2)],
        group_shares=m3.group.value_counts(normalize=True).round(4).to_dict(),
        experiments_with_30_per_arm_in_low_and_high=int(sum(
            (m3[(m3.target_sequence == e) & (m3.group == g)].groupby("in_treatment").size().reindex([0, 1], fill_value=0).min() >= 30)
            for e in keep3 for g in ("low",)) ),
        students=int(m3.student_id.nunique()), classes=int(m3.class_id.nunique()),
        records_per_class_median=float(m3.groupby("class_id").size().median()),
        smd_abs=dict(median=float(np.median(smd)), p95=float(np.percentile(smd, 95)), max=float(smd.max()),
                     share_above_0_1=float((smd > .1).mean()), n=int(len(smd))),
        per_experiment_size=dict(min=int(per.loc[keep3, "n"].min()), median=float(per.loc[keep3, "n"].median()),
                                 max=int(per.loc[keep3, "n"].max())),
        group_size_per_experiment_median=grp.median().round(1).to_dict())
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "audit.json"), "w"), indent=1)
    for f in out["flow"]:
        print(f"{f['step']:45s} records {f['records']:7d}  experiments {f['experiments']:3d}")
    print({k: v for k, v in out.items() if k != "flow"})


if __name__ == "__main__":
    main()
