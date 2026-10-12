"""Weekly features for OULAD, in the format of assistments_features_time.csv,
so that assistments_pipeline.py can be reused.

    python build_features_oulad.py --data <OULAD folder> --out oulad_features_weekly.csv

One trajectory per student (first registration only), weeks t = 0..25 from
the start of the module presentation:
  k_t  mean assessment score (/100) of the assessments submitted up to week t,
       0.5 before the first submission
  e_t  log(1 + clicks in week t) / log(1 + 99th percentile of weekly clicks),
       clipped to [0, 1], minus 0.5
  p_t  1{k_t >= 0.4}  (the Open University pass mark)
  d_t  active_t - mean activity of the same module presentation and week
       (active_t = 1 if the student clicked at least once in week t); the
       analysis pipeline rebuilds it from training students only
Sensitive attributes (stored as columns, S is chosen by run_oulad.py):
  S_imd = 1 if imd_band is in the three most deprived deciles (0-30%)
  S_gender = 1 if female, S_disability = 1 if a disability is declared
Cluster school_id = module presentation (code_module + code_presentation).
Licence CC BY 4.0 (Kuzilek et al. 2017).
"""
import argparse
import os

import numpy as np
import pandas as pd

WEEKS = 26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="oulad_features_weekly.csv")
    a = ap.parse_args()
    P = lambda f: os.path.join(a.data, f)
    info = pd.read_csv(P("studentInfo.csv"))
    reg = pd.read_csv(P("studentRegistration.csv"))
    ass = pd.read_csv(P("assessments.csv"))
    sa = pd.read_csv(P("studentAssessment.csv"))
    print("studentInfo rows", len(info), "students", info.id_student.nunique(), flush=True)

    info["pres"] = info.code_module + "_" + info.code_presentation
    first = info.sort_values(["id_student", "code_presentation", "code_module"]).drop_duplicates("id_student")
    first = first[first.imd_band.notna() & (first.imd_band != "?")]
    keys = first[["id_student", "code_module", "code_presentation", "pres"]]

    # weekly clicks
    clicks = []
    for ch in pd.read_csv(P("studentVle.csv"), chunksize=2_000_000):
        ch = ch.merge(keys, on=["id_student", "code_module", "code_presentation"])
        ch = ch[(ch.date >= 0) & (ch.date < 7 * WEEKS)]
        ch["week"] = ch.date // 7
        clicks.append(ch.groupby(["id_student", "week"]).sum_click.sum())
    wk = pd.concat(clicks).groupby(level=[0, 1]).sum()

    grid = pd.MultiIndex.from_product([keys.id_student.values, range(WEEKS)], names=["id_student", "week"])
    d = pd.DataFrame(index=grid).join(wk.rename("clicks")).fillna({"clicks": 0}).reset_index()
    d = d.merge(first[["id_student", "pres", "gender", "imd_band", "disability", "code_module", "code_presentation"]],
                on="id_student")
    cap = np.log1p(np.percentile(d.clicks[d.clicks > 0], 99))
    d["e_t"] = np.clip(np.log1p(d.clicks) / cap, 0, 1) - 0.5
    d["correct"] = (d.clicks > 0).astype(float)          # weekly activity indicator
    d["skill_id"] = d.pres + "_w" + d.week.astype(str)

    # running mean assessment score
    sa["score"] = pd.to_numeric(sa.score, errors="coerce")
    sa["date_submitted"] = pd.to_numeric(sa.date_submitted, errors="coerce")
    sa = sa.merge(ass[["id_assessment", "code_module", "code_presentation"]], on="id_assessment")
    sa = sa.merge(keys, on=["id_student", "code_module", "code_presentation"])
    sa = sa[sa.score.notna() & (sa.date_submitted < 7 * WEEKS)]
    sa["week"] = np.maximum(sa.date_submitted, 0) // 7
    ws = sa.groupby(["id_student", "week"]).score.agg(["sum", "count"])
    d = d.merge(ws.reset_index(), on=["id_student", "week"], how="left").fillna({"sum": 0, "count": 0})
    d = d.sort_values(["id_student", "week"])
    cs, cn = d.groupby("id_student")["sum"].cumsum(), d.groupby("id_student")["count"].cumsum()
    d["k_t"] = np.where(cn > 0, cs / np.maximum(cn, 1) / 100.0, 0.5)
    d["p_t"] = (d.k_t >= 0.4).astype(int)
    d["d_t"] = d.correct - d.skill_id.map(d.groupby("skill_id").correct.mean())

    d["S_imd"] = d.imd_band.isin(["0-10%", "10-20", "10-20%", "20-30%"]).astype(int)
    d["S_gender"] = (d.gender == "F").astype(int)
    d["S_disability"] = (d.disability == "Y").astype(int)
    d["S"] = d.S_imd
    out = d.rename(columns={"id_student": "user_id", "pres": "school_id"})
    out["problem_id"] = out.week
    out["problem_log_id"] = np.arange(len(out))
    out["start_time"] = out.week
    out = out[["user_id", "problem_id", "k_t", "e_t", "p_t", "d_t", "S", "S_imd", "S_gender", "S_disability",
               "correct", "skill_id", "school_id", "problem_log_id", "start_time"]]
    out.to_csv(a.out, index=False)
    s = out.drop_duplicates("user_id")
    print(f"students {len(s)}, presentations {s.school_id.nunique()}, deprived {s.S_imd.mean():.3f}, "
          f"female {s.S_gender.mean():.3f}, disability {s.S_disability.mean():.3f}, "
          f"active weeks {out.correct.mean():.3f}, imd missing removed before; "
          f"students with at least one score by week 25 {(d.groupby('id_student')['count'].sum() > 0).mean():.3f}")


if __name__ == "__main__":
    main()
