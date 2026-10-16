"""Experiment 06: is the OULAD knowledge gap absent at the start, or is this an effect of the imputation?

    python code/scripts/run_oulad_start_sensitivity.py [--boot 100]

The OULAD knowledge proxy k_t is the mean score of the assessments submitted up to week t, and 0.5
before the first submission (the companion paper [A] construction).  At week 0 almost every student is imputed, so a
gap near zero at the start can come from the imputation.  This script answers two questions.

A. Without a model (students resampled, 2,000 draws):
   * share of imputed k_t by group and week;
   * gap on the first observed score of each student, and the week of that score;
   * fixed cohorts (students with a score by week 2 or 4): gap at that week and at week 24, same students;
   * share of students with no score by week 25, and the final gap among students with a score.
B. The decomposition of the final knowledge gap (start I, persistence A, drift a; Shapley) under four
   definitions of the start:
     paper      k_t as built (0.5 before the first score), all students;
     observed   same k_t, only students with a score by week 24;
     backfill   first observed value carried back to the weeks before it, students with a score;
     cohort4    backfill, only students with a score by week 4 (the start is then mostly observed).
   Uncertainty: the whole procedure (lifting, decoder, group models, held-out initial means and gaps)
   is refitted on student bootstrap samples; in each draw, the unique students are split 60/40 into
   training and evaluation.  Estimators: least squares, and IV with z_{t-1} or z_{t-2} as instrument
   (with first-stage diagnostics).
Writes outputs/06_oulad_start/results.json (aggregate numbers only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from p2.decomposition import FREE_FACTORS, fit_free_group_models, fit_free_group_models_iv_lag, free_decomposition  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402

use_paper1_code()
import assistments_pipeline as ap  # noqa: E402

T = ap.N_TRANS
KIDX = ap.FEATURES.index("k_t")
ATTR = {"deprivation": "S_imd", "disability": "S_disability"}
SEED = 20261015


def first_scores():
    raw = PATHS["oulad_raw"]
    info = pd.read_csv(raw / "studentInfo.csv")
    sa = pd.read_csv(raw / "studentAssessment.csv")
    asm = pd.read_csv(raw / "assessments.csv")
    sa["score"] = pd.to_numeric(sa.score, errors="coerce")
    first = info.sort_values(["id_student", "code_presentation", "code_module"]).drop_duplicates("id_student")
    sa = sa.merge(asm[["id_assessment", "code_module", "code_presentation"]], on="id_assessment")
    sa = sa.merge(first[["id_student", "code_module", "code_presentation"]], on=["id_student", "code_module", "code_presentation"])
    sa = sa[sa.score.notna()].copy()
    sa["week"] = np.maximum(sa.date_submitted, 0) // 7
    sa = sa.sort_values(["id_student", "date_submitted"])
    f = sa.groupby("id_student").first()
    return pd.DataFrame({"first_week": f.week, "first_score": f.score / 100.0})


def boot_gap(x1, x0, rng, B=2000):
    d = [rng.choice(x1, len(x1)).mean() - rng.choice(x0, len(x0)).mean() for _ in range(B)]
    return dict(gap=float(x1.mean() - x0.mean()), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


def part_a(df, fs, rng):
    out = {}
    st = df.drop_duplicates("user_id").set_index("user_id")[list(ATTR.values())].join(fs)
    k = df.pivot(index="user_id", columns="step", values="k_t")
    for name, col in ATTR.items():
        g = st[col].astype(int)
        r = dict(imputed_share={}, gap_all={}, gap_observed={})
        for w in (0, 1, 2, 4, 8, 12, 24):
            obs = st.first_week <= w
            r["imputed_share"][w] = {str(gg): float(1 - obs[g == gg].mean()) for gg in (0, 1)}
            r["gap_all"][w] = float(k.loc[g.index[g == 1], w].mean() - k.loc[g.index[g == 0], w].mean())
            r["gap_observed"][w] = float(k.loc[g.index[(g == 1) & obs], w].mean() - k.loc[g.index[(g == 0) & obs], w].mean())
        s = st[st.first_week.notna()]
        gs = s[col].astype(int)
        r["first_score_gap"] = boot_gap(s.first_score[gs == 1].values, s.first_score[gs == 0].values, rng)
        r["first_week_mean"] = {str(gg): float(s.first_week[gs == gg].mean()) for gg in (0, 1)}
        r["no_score_by_25"] = {str(gg): float(st.first_week[g == gg].isna().mean() + (st.first_week[g == gg] > 24).mean()) for gg in (0, 1)}
        r["final_gap_all"] = boot_gap(k.loc[g.index[g == 1], 24].values, k.loc[g.index[g == 0], 24].values, rng)
        so = s.index[s.first_week <= 24]
        r["final_gap_observed"] = boot_gap(k.loc[so[gs.loc[so] == 1], 24].values, k.loc[so[gs.loc[so] == 0], 24].values, rng)
        r["cohorts"] = {}
        for w0 in (2, 4):
            c = s.index[s.first_week <= w0]
            gc = gs.loc[c]
            r["cohorts"][w0] = dict(n=int(len(c)),
                                    start=boot_gap(k.loc[c[gc == 1], w0].values, k.loc[c[gc == 0], w0].values, rng),
                                    end=boot_gap(k.loc[c[gc == 1], 24].values, k.loc[c[gc == 0], 24].values, rng))
        out[name] = r
        print(f"[A] {name}: imputed at week 0 {r['imputed_share'][0]}; first-score gap {r['first_score_gap']['gap']:+.4f} "
              f"{np.round(r['first_score_gap']['ci'], 4)}; final gap all {r['final_gap_all']['gap']:+.4f}, observed only "
              f"{r['final_gap_observed']['gap']:+.4f}; no score by week 25 {r['no_score_by_25']}", flush=True)
        for w0, c in r["cohorts"].items():
            print(f"    cohort scored by week {w0} (n={c['n']}): {c['start']['gap']:+.4f} {np.round(c['start']['ci'], 4)} -> "
                  f"week 24 {c['end']['gap']:+.4f} {np.round(c['end']['ci'], 4)}", flush=True)
    return out


def variant_arrays(df, fs, variant):
    """Trajectories (N, T+1, F) and per-student first observed week for one definition of the start."""
    ids, X, _, _ = ap.trajectories(df, df.user_id.unique())
    fw = fs.reindex(ids).first_week.values
    keep = np.ones(len(ids), bool) if variant == "paper" else (fw <= 24)
    if variant == "cohort4":
        keep &= fw <= 4
    X = X.copy()
    if variant in ("backfill", "cohort4"):
        for i in np.where(keep)[0]:
            w = int(fw[i])
            X[i, :w, KIDX] = X[i, w, KIDX]
    return ids[keep], X[keep]


def decompose(Xtr, Gtr, Xte, Gte):
    fit = ap.fit_edmd(Xtr)
    Zs = ap.encode(fit, Xtr)
    Z = Zs[:, :-1].reshape(-1, fit.A.shape[0])
    Zn = Zs[:, 1:].reshape(-1, fit.A.shape[0])
    models = {"ls": (fit_free_group_models(Z, Zn, np.repeat(Gtr, T)), None)}
    for lag in (1, 2):
        models[f"iv{lag}"] = fit_free_group_models_iv_lag(Zs, Gtr, lag)
    Z0 = ap.encode(fit, Xte[:, 0])
    m0 = {g: Z0[Gte == g].mean(0) for g in (0, 1)}
    obs = float(Xte[Gte == 1, T, KIDX].mean() - Xte[Gte == 0, T, KIDX].mean())
    init = float(Xte[Gte == 1, 0, KIDX].mean() - Xte[Gte == 0, 0, KIDX].mean())
    res = {}
    for est, (mod, diag) in models.items():
        dec = free_decomposition(mod, m0, T, fit.C[KIDX], fit.c[KIDX])
        res[est] = dict(model=dec["gap"], error=obs - dec["gap"], **{f: dec["shapley"][f] for f in FREE_FACTORS},
                        diag=diag)
    return dict(observed=obs, observed_initial=init, est=res)


def part_b(df, fs, attr, variant, B, rng):
    ids, X = variant_arrays(df, fs, variant)
    G = df.drop_duplicates("user_id").set_index("user_id")[attr].reindex(ids).astype(int).values
    point = None
    draws = []
    for b in range(B + 1):
        if b == 0:                                            # the sample itself, one 60/40 split
            idx = np.arange(len(ids))
        else:
            idx = rng.integers(0, len(ids), len(ids))
        u = np.unique(idx)
        tr_ids = set(rng.choice(u, int(0.6 * len(u)), replace=False))
        tr = np.array([i in tr_ids for i in idx])
        r = decompose(X[idx[tr]], G[idx[tr]], X[idx[~tr]], G[idx[~tr]])
        if b == 0:
            point = r
        else:
            draws.append(r)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = dict(n_students=int(len(ids)), observed=point["observed"], observed_ci=ci([d["observed"] for d in draws]),
               observed_initial=point["observed_initial"], observed_initial_ci=ci([d["observed_initial"] for d in draws]))
    for est in point["est"]:
        e = {}
        for key in ("model", "error") + FREE_FACTORS:
            e[key] = dict(value=point["est"][est][key], ci=ci([d["est"][est][key] for d in draws]))
        if point["est"][est]["diag"]:
            e["diag"] = {g: {k: dict(value=point["est"][est]["diag"][g][k],
                                     range=[float(np.min([d["est"][est]["diag"][g][k] for d in draws])),
                                            float(np.max([d["est"][est]["diag"][g][k] for d in draws]))])
                             for k in point["est"][est]["diag"][g]} for g in (0, 1)}
        out[est] = e
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--boot", type=int, default=100)
    a_.add_argument("--out", default="results.json", help="output file name in outputs/06_oulad_start")
    a = a_.parse_args()
    rng = np.random.default_rng(SEED)
    df = ap.keep_full_trajectories(ap.load_features(str(PATHS["oulad_features"])))
    fs = first_scores()
    out = dict(part_a=part_a(df, fs, rng), part_b={}, boot=a.boot)
    for name, col in ATTR.items():
        out["part_b"][name] = {}
        for variant in ("paper", "observed", "backfill", "cohort4"):
            r = part_b(df, fs, col, variant, a.boot, rng)
            out["part_b"][name][variant] = r
            ls, iv = r["ls"], r["iv1"]
            print(f"[B] {name:11s} {variant:8s} n={r['n_students']:5d} observed start {r['observed_initial']:+.4f} "
                  f"{np.round(r['observed_initial_ci'], 4)} end {r['observed']:+.4f} | LS I {ls['I']['value']:+.4f} "
                  f"{np.round(ls['I']['ci'], 4)} A {ls['A']['value']:+.4f} a {ls['a']['value']:+.4f} {np.round(ls['a']['ci'], 4)} "
                  f"err {ls['error']['value']:+.4f} | IV1 I {iv['I']['value']:+.4f} a {iv['a']['value']:+.4f} "
                  f"model {iv['model']['value']:+.4f} | IV2 model {r['iv2']['model']['value']:+.4f}", flush=True)
            d = ROOT / "outputs" / "06_oulad_start"
            d.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(d / a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
