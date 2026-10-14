"""
Eedi: sensitivity to the recording of PremiumPupil (review of 13 October 2026, item 5).

The attribute is recorded for 29% of pupils.  We model the probability of being recorded from observables known
for every pupil (number of answers, accuracy, gender and year of birth, with indicators of missing values), weight
each recorded pupil by the inverse of this probability (trimmed at the 99th percentile), and recompute on the test
classes of the five class-disjoint splits: the observed gap at step 25 and the one-step predicted gaps of the
original, group-aware and projected EDMD models, and the one-step error ratio of the projected model, all weighted.
The models are fitted without weights, as in the main study, and then with weights.  Under the assumption that the
status is missing at random given these observables, the weighted values describe all Eedi pupils.

    DATA_ROOT=/path/to/root python eedi_reweight.py     -> out_reviewer/eedi_reweight.json (aggregates only)
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import assistments_pipeline as ap
import rank_matched_controls as rm
from dynamics_only import GroupAware

ROOT = os.environ.get("DATA_ROOT", ".")
DATA = os.path.join(ROOT, "Datasets", "Eedi_NeurIPS2020")
SEED = 20260914


def propensity():
    meta = pd.read_csv(glob.glob(os.path.join(DATA, "data", "**", "student_metadata_task_1_2.csv"), recursive=True)[0])
    ans = pd.read_csv(glob.glob(os.path.join(DATA, "data", "**", "train_task_1_2.csv"), recursive=True)[0], usecols=["UserId", "IsCorrect"])
    per = ans.groupby("UserId").IsCorrect.agg(n="size", acc="mean").reset_index()
    m = meta.merge(per, on="UserId", how="inner")
    m["recorded"] = m.PremiumPupil.notna().astype(int)
    year = pd.to_datetime(m.DateOfBirth, errors="coerce").dt.year
    ln = np.log(m.n)
    X = np.column_stack([ln, ln ** 2, m.acc, m.acc ** 2, ln * m.acc, (m.Gender == 1).astype(float), (~m.Gender.isin([1, 2])).astype(float),
                         year.fillna(year.median()) - year.median(), year.isna().astype(float)])
    X = (X - X.mean(0)) / X.std(0)
    lr = LogisticRegression(max_iter=2000).fit(X, m.recorded)
    p = lr.predict_proba(X)[:, 1]
    rec = m.recorded.values == 1
    w = 1 / p[rec]
    w = np.minimum(w, np.percentile(w, 99))
    # balance check: standardized differences after weighting (recorded, weighted) vs all pupils
    def smd(v):
        a = np.average(v[rec], weights=w)
        return float((a - v.mean()) / v.std())
    bal = dict(log_answers=smd(ln.values), accuracy=smd(m.acc.values))
    raw = dict(log_answers=float((ln[rec].mean() - ln.mean()) / ln.std()), accuracy=float((m.acc[rec].mean() - m.acc.mean()) / m.acc.std()))
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return pd.Series(w, index=m.UserId.values[rec]), dict(balance_before=raw, balance_after=bal, effective_sample_size=ess,
                                                           n_recorded=int(rec.sum()), weight_max_over_mean=float(w.max() / w.mean()))


def wgap(v, g, w):
    return float(np.average(v[g == 1], weights=w[g == 1]) - np.average(v[g == 0], weights=w[g == 0]))


def one_split(df, seed, W):
    rng = np.random.default_rng(seed)
    tr, va, te = ap.split_schools(df, rng)
    d = ap.rebuild_dt(df, tr)
    idtr, Xtr, Gtr, _ = ap.trajectories(d, tr)
    idte, Xte, Gte, _ = ap.trajectories(d, te)
    wte = W.reindex(idte).values
    out = {}
    for mode in ("unweighted_fit", "weighted_fit"):
        if mode == "unweighted_fit":
            fit = ap.fit_edmd(Xtr)
            ga = GroupAware(fit, Xtr, Gtr)
        else:   # weighted least squares by repeating rows in proportion to the weights (rounded, mean 4 copies)
            wtr = W.reindex(idtr).values
            reps = np.maximum(np.rint(4 * wtr / wtr.mean()), 1).astype(int)
            Xw, Gw = np.repeat(Xtr, reps, 0), np.repeat(Gtr, reps)
            fit = ap.fit_edmd(Xw)
            ga = GroupAware(fit, Xw, Gw)
            Xtr_, Gtr_ = Xw, Gw
        Gamma, _ = ap.metric(fit.G, ap.RIDGE_RATIO)
        c, _ = ap.contrast(fit, Xtr if mode == "unweighted_fit" else Xtr_, Gtr if mode == "unweighted_fit" else Gtr_)
        P = ap.build_projector(fit.K, Gamma, c, rank=4)[0]
        models = {"original": rm.LinPredictor(fit, fit.A, fit.a), "projected": rm.LinPredictor(fit, P.T @ fit.A @ P.T, P.T @ fit.a),
                  "group_aware": ga}
        T = ap.N_TRANS
        cur = Xte[:, :-1].reshape(-1, 4)
        g_rows, w_rows = np.repeat(Gte, T), np.repeat(wte, T)
        nxt = Xte[:, 1:, 0].reshape(-1)
        r = dict(observed_gap_weighted=wgap(Xte[:, -1, 0], Gte, wte), observed_gap_unweighted=wgap(Xte[:, -1, 0], Gte, np.ones(len(Gte))))
        mse = {}
        for name, m in models.items():
            r[f"gap_{name}_weighted"] = wgap(m.predict(Xte[:, -2], 1, Gte)[:, 0], Gte, wte)
            mse[name] = float(np.average((m.predict(cur, 1, g_rows)[:, 0] - nxt) ** 2, weights=w_rows))
        r["ratio_weighted"] = mse["projected"] / mse["original"]
        out[mode] = r
    return out


def main():
    W, info = propensity()
    df = ap.keep_full_trajectories(ap.load_features(os.path.join(DATA, "eedi_features_time.csv")))
    runs = [one_split(df, SEED + 200 + j, W) for j in range(5)]
    summ = {mode: {k: dict(mean=float(np.mean([r[mode][k] for r in runs])), range=[float(min(r[mode][k] for r in runs)), float(max(r[mode][k] for r in runs))])
                   for k in runs[0][mode]} for mode in runs[0]}
    out = dict(propensity=info, splits=runs, summary=summ)
    os.makedirs("out_reviewer", exist_ok=True)
    json.dump(out, open("out_reviewer/eedi_reweight.json", "w"), indent=1)
    print(json.dumps(info, indent=1))
    for mode, s in summ.items():
        print(mode, {k: (round(v["mean"], 4), [round(x, 4) for x in v["range"]]) for k, v in s.items()})


if __name__ == "__main__":
    main()
