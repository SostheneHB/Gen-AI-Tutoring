"""Experiment 14: OULAD first observed scores, in more detail (review of 16 October 2026, point 11).

    python code/scripts/run_oulad_first_scores.py

By group (deprivation, disability): the distribution of the week of the first submitted assessment (quantiles,
and the share submitting by weeks 2, 4, 8), the gap on the first observed score overall and within each course
presentation (module x presentation; mean of the within-presentation gaps weighted by presentation size, with a
student bootstrap), and who has no score at all (share by group, and by final result category).
Writes outputs/14_oulad_first_scores/results.json (aggregate numbers only).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from p2.paths import PATHS  # noqa: E402

raw = PATHS["oulad_raw"]
info = pd.read_csv(raw / "studentInfo.csv")
sa = pd.read_csv(raw / "studentAssessment.csv")
asm = pd.read_csv(raw / "assessments.csv")
sa["score"] = pd.to_numeric(sa.score, errors="coerce")
first = info.sort_values(["id_student", "code_presentation", "code_module"]).drop_duplicates("id_student")
first = first[first.imd_band.notna() & (first.imd_band != "?")].copy()
first["S_imd"] = first.imd_band.isin(["0-10%", "10-20", "10-20%", "20-30%"]).astype(int)
first["S_disability"] = (first.disability == "Y").astype(int)
first["pres"] = first.code_module + "_" + first.code_presentation
sa = sa.merge(asm[["id_assessment", "code_module", "code_presentation"]], on="id_assessment")
sa = sa.merge(first[["id_student", "code_module", "code_presentation"]], on=["id_student", "code_module", "code_presentation"])
sa = sa[sa.score.notna()].copy()
sa["week"] = np.maximum(sa.date_submitted, 0) // 7
f = sa.sort_values(["id_student", "date_submitted"]).groupby("id_student").first()
st = first.set_index("id_student").join(f[["week", "score"]].rename(columns={"week": "first_week", "score": "first_score"}))
st["first_score"] /= 100
rng = np.random.default_rng(20261016)
out = {}
for name, col in (("deprivation", "S_imd"), ("disability", "S_disability")):
    r = {}
    for g in (0, 1):
        s = st[st[col] == g]
        w = s.first_week.dropna()
        r[f"group{g}"] = dict(n=int(len(s)), first_week_quantiles=[float(x) for x in np.percentile(w, [10, 25, 50, 75, 90])],
                              share_by_week={k: float((s.first_week <= k).mean()) for k in (2, 4, 8)},
                              no_score=float(s.first_week.isna().mean()),
                              no_score_by_result={k: float(v) for k, v in s[s.first_week.isna()].final_result.value_counts(normalize=True).items()})
    obs = st[st.first_score.notna()]

    def within(d):
        gaps, wts = [], []
        for _, p in d.groupby("pres"):
            if p[col].nunique() == 2 and min((p[col] == 0).sum(), (p[col] == 1).sum()) >= 10:
                gaps.append(p[p[col] == 1].first_score.mean() - p[p[col] == 0].first_score.mean()); wts.append(len(p))
        return float(np.average(gaps, weights=wts)), len(gaps)

    overall = float(obs[obs[col] == 1].first_score.mean() - obs[obs[col] == 0].first_score.mean())
    w_gap, n_pres = within(obs)
    boots = [within(obs.iloc[rng.integers(0, len(obs), len(obs))])[0] for _ in range(300)]
    r.update(first_score_gap=overall, first_score_gap_within_presentation=w_gap,
             within_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], presentations=n_pres)
    out[name] = r
    print(f"{name}: first-week quantiles g0 {r['group0']['first_week_quantiles']} g1 {r['group1']['first_week_quantiles']}; "
          f"submitted by week 2/4/8: g0 {r['group0']['share_by_week']} g1 {r['group1']['share_by_week']}; no score g0 {r['group0']['no_score']:.3f} "
          f"g1 {r['group1']['no_score']:.3f} (no-score results g1 {r['group1']['no_score_by_result']}) | first-score gap {overall:+.4f}, within "
          f"presentation {w_gap:+.4f} {np.round(r['within_ci'], 4)} ({n_pres} presentations)", flush=True)
d = ROOT / "outputs" / "14_oulad_first_scores"
d.mkdir(parents=True, exist_ok=True)
json.dump(out, open(d / "results.json", "w"), indent=1)
