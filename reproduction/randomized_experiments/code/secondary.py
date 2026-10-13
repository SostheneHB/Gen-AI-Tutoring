"""
Pre-specified items of notes/analysis_plan.md not covered by analysis.py (real outcomes, after approval):

  1. Per-experiment tests (plan, Section 7): for P3, P4, P5 vs P2, differences in low, high and gap inside
     each experiment, cluster bootstrap of classes (2,000 draws), two-sided percentile p-value, Holm
     correction across experiments.
  2. Secondary outcome (plan, Section 4): problems_completed among students who completed.  The policies
     are the ones learnt on completion (same folds); their value on the secondary outcome is estimated by
     the same doubly robust formula among completers.  Completion follows randomization, so this is a
     descriptive comparison, not a causal one (recorded in the plan, Deviations).

    python secondary.py          # writes outputs/per_experiment_holm.json and outputs/secondary_problems.json
"""
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from analysis import (HERE, OUT, POLICIES, boot_weights, dr_scores, features, fit_outcome, load,
                      policies_on, values)

B = 2000
COMPARED = ("P3", "P4", "P5")
QUANT = ("low", "high", "gap")


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def experiment_both(d, y, z):
    """DR scores for completion (y) on everybody and for problems_completed (z) on completers."""
    X, a = features(d), d.A.to_numpy()
    sy = {k: np.full(len(d), np.nan) for k in POLICIES}
    sz = {k: np.full(len(d), np.nan) for k in POLICIES}
    comp = y == 1
    for tr, te in GroupKFold(n_splits=min(5, d.class_id.nunique())).split(X, y, groups=d.class_id):
        m0, m1 = fit_outcome(X[tr], a[tr], y[tr], 0), fit_outcome(X[tr], a[tr], y[tr], 1)
        mu0, mu1 = m0.predict_proba(X[te])[:, 1], m1.predict_proba(X[te])[:, 1]
        tmu0, tmu1 = m0.predict_proba(X[tr])[:, 1], m1.predict_proba(X[tr])[:, 1]
        pols = policies_on(d.iloc[te], mu0, mu1, tmu0, tmu1, d.iloc[tr])
        trc = tr[comp[tr]]
        r = {arm: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3, random_state=0)
             .fit(X[trc][a[trc] == arm], z[trc][a[trc] == arm]) for arm in (0, 1)}
        nu0, nu1 = r[0].predict(X[te]), r[1].predict(X[te])
        for k, pol in pols.items():
            sy[k][te] = dr_scores(pol, a[te], y[te], mu0, mu1)
            sz[k][te] = dr_scores(pol, a[te], z[te], nu0, nu1)
    return sy, sz


def boot_diffs(S, grp, cls, rng, keep=None, ex=None):
    """Differences with P2; classes resampled within each experiment (ex), as in analysis.py."""
    keep = np.ones(len(grp), bool) if keep is None else keep
    S = {k: v[keep] for k, v in S.items()}
    grp, cls = grp[keep], cls[keep]
    ex = np.zeros(len(grp)) if ex is None else ex[keep]
    point = values(S, grp)
    draws = {k: {q: [] for q in QUANT} for k in COMPARED}
    W = np.zeros((B, len(grp)))
    for e in np.unique(ex):
        idx = np.flatnonzero(ex == e)
        W[:, idx] = boot_weights(cls[idx], rng, B)
    for w in W:
        v = values(S, grp, w)
        for k in COMPARED:
            for q in QUANT:
                draws[k][q].append(v[k][q] - v["P2"][q])
    out = {}
    for k in COMPARED:
        out[k] = {}
        for q in QUANT:
            dq = np.asarray(draws[k][q])
            dropped = int(np.isnan(dq).sum())       # draws with no student of a group (small experiments)
            dq = dq[~np.isnan(dq)]
            p = min(1.0, 2 * min((dq <= 0).mean(), (dq >= 0).mean())) if np.any(dq != 0) else 1.0
            out[k][q] = dict(mean=point[k][q] - point["P2"][q],
                             ci=[float(np.percentile(dq, 2.5)), float(np.percentile(dq, 97.5))], p=float(p),
                             draws_dropped=dropped)
    return point, out


def main():
    rng = np.random.default_rng(20261012)
    m = load(real=True)
    import pandas as pd
    z_all = pd.read_csv(os.path.join(HERE, "..", "data", "experiment_prior_stats_and_targets.csv"),
                        usecols=["assignment_log_id", "problems_completed"])
    m = m.merge(z_all, on="assignment_log_id", how="left")
    per, Sz, grp_all, cls_all, comp_all, ex_all = {}, {k: [] for k in POLICIES}, [], [], [], []
    for e, d in m.groupby("target_sequence"):
        d = d.reset_index(drop=True)
        y, z = d.assignment_completed.to_numpy(int), d.problems_completed.fillna(0).to_numpy(float)
        sy, sz = experiment_both(d, y, z)
        _, diffs = boot_diffs(sy, d.group.to_numpy(), d.class_id.to_numpy(), rng)
        per[e] = dict(n=len(d), classes=int(d.class_id.nunique()), vs_P2=diffs)
        for k in POLICIES:
            Sz[k].append(sz[k])
        grp_all.append(d.group.to_numpy()); comp_all.append(y == 1); ex_all.append(np.full(len(d), e))
        cls_all.append((e + "_" + d.class_id.astype(str)).to_numpy())
    exps = list(per)
    summary = {}
    for k in COMPARED:
        summary[k] = {}
        for q in QUANT:
            adj = holm([per[e]["vs_P2"][k][q]["p"] for e in exps])
            for e, a in zip(exps, adj):
                per[e]["vs_P2"][k][q]["p_holm"] = float(a)
            summary[k][q] = dict(raw_p_below_05=int(sum(per[e]["vs_P2"][k][q]["p"] < .05 for e in exps)),
                                 holm_p_below_05=[e for e, a in zip(exps, adj) if a < .05])
    json.dump(dict(experiments=len(exps), draws=B, summary=summary, per_experiment=per),
              open(os.path.join(OUT, "per_experiment_holm.json"), "w"), indent=1)
    print(f"[per-experiment] {len(exps)} experiments; experiments with Holm-adjusted p < 0.05 (raw p < 0.05):")
    for k in COMPARED:
        print(f"  {k}: " + " | ".join(f"{q} {summary[k][q]['holm_p_below_05']} ({summary[k][q]['raw_p_below_05']})"
                                       for q in QUANT))
    # secondary outcome, pooled exactly as the primary one (student-weighted, joint cluster bootstrap)
    Sz = {k: np.concatenate(v) for k, v in Sz.items()}
    grp, cls, comp = np.concatenate(grp_all), np.concatenate(cls_all), np.concatenate(comp_all)
    point, diffs = boot_diffs(Sz, grp, cls, rng, keep=comp, ex=np.concatenate(ex_all))
    json.dump(dict(outcome="problems_completed among completers (descriptive)", students=int(comp.sum()),
                   values=point, vs_P2=diffs), open(os.path.join(OUT, "secondary_problems.json"), "w"), indent=1)
    print(f"[secondary] problems completed among {int(comp.sum())} completers; P2: low {point['P2']['low']:.2f}, "
          f"high {point['P2']['high']:.2f}")
    for k in COMPARED:
        print(f"  {k}: " + " | ".join(f"{q} {diffs[k][q]['mean']:+.3f} [{diffs[k][q]['ci'][0]:+.3f},"
                                       f"{diffs[k][q]['ci'][1]:+.3f}]" for q in QUANT))


if __name__ == "__main__":
    main()
