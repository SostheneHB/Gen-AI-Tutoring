"""Experiment 07: is the instrumental-variable check valid on Eedi?

    python code/scripts/run_eedi_iv_diagnostics.py [--boot 50]

The Eedi knowledge proxy k_t is a rolling mean of correctness over the last 10 answers.  Two states
less than 10 steps apart share answers, so their measurement errors are correlated, and z_{t-1} is
not a valid instrument for z_t.  A lag of at least 10 removes the shared answers.  This script fits
the per-group models by least squares and by IV with z_{t-1}, z_{t-2} and z_{t-10} as instruments,
reports first-stage diagnostics (condition number of W'X, first-stage R^2, spectral radius of A),
and decomposes the final knowledge gap (start I, persistence A, drift a; Shapley).
Uncertainty: classes are resampled (cluster bootstrap) and the whole procedure is refitted; in each
draw the unique classes are split 60/40 into training and evaluation.
Writes outputs/07_eedi_iv/results.json (aggregate numbers only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from p2.decomposition import FREE_FACTORS, fit_free_group_models, fit_free_group_models_iv_lag, free_decomposition  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402

use_paper1_code()
import assistments_pipeline as ap  # noqa: E402

T = ap.N_TRANS
KIDX = ap.FEATURES.index("k_t")
LAGS = (1, 2, 10)
SEED = 20261015 + 7


def decompose(Xtr, Gtr, Xte, Gte):
    fit = ap.fit_edmd(Xtr)
    Zs = ap.encode(fit, Xtr)
    n = fit.A.shape[0]
    models = {"ls": (fit_free_group_models(Zs[:, :-1].reshape(-1, n), Zs[:, 1:].reshape(-1, n), np.repeat(Gtr, T)), None)}
    for lag in LAGS:
        models[f"iv{lag}"] = fit_free_group_models_iv_lag(Zs, Gtr, lag)
    Z0 = ap.encode(fit, Xte[:, 0])
    m0 = {g: Z0[Gte == g].mean(0) for g in (0, 1)}
    obs = float(Xte[Gte == 1, T, KIDX].mean() - Xte[Gte == 0, T, KIDX].mean())
    init = float(Xte[Gte == 1, 0, KIDX].mean() - Xte[Gte == 0, 0, KIDX].mean())
    res = {}
    for est, (mod, diag) in models.items():
        dec = free_decomposition(mod, m0, T, fit.C[KIDX], fit.c[KIDX])
        rho = max(float(np.abs(np.linalg.eigvals(mod[g]["A"])).max()) for g in (0, 1))
        res[est] = dict(model=dec["gap"], error=obs - dec["gap"], rho=rho, **{f: dec["shapley"][f] for f in FREE_FACTORS}, diag=diag)
    return dict(observed=obs, observed_initial=init, est=res)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--boot", type=int, default=50)
    a = a_.parse_args()
    rng = np.random.default_rng(SEED)
    df = ap.keep_full_trajectories(ap.load_features(str(PATHS["eedi_features"])))
    ids, X, G, school = ap.trajectories(df, df.user_id.unique())
    classes = np.unique(school)
    rows = {c: np.where(school == c)[0] for c in classes}
    point, draws = None, []
    for b in range(a.boot + 1):
        cl = classes if b == 0 else rng.choice(classes, len(classes))
        u = np.unique(cl)
        tr_cl = set(rng.choice(u, int(0.6 * len(u)), replace=False))
        tr = np.concatenate([rows[c] for c in cl if c in tr_cl])
        te = np.concatenate([rows[c] for c in cl if c not in tr_cl])
        r = decompose(X[tr], G[tr], X[te], G[te])
        if b == 0:
            point = r
        else:
            draws.append(r)
        if b % 10 == 0:
            print(f"draw {b}: " + " | ".join(f"{e} model {r['est'][e]['model']:+.3f} rho {r['est'][e]['rho']:.3f}" for e in r["est"]), flush=True)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = dict(n_students=int(len(ids)), n_classes=int(len(classes)), boot=a.boot, lags=LAGS,
               observed=point["observed"], observed_ci=ci([d["observed"] for d in draws]),
               observed_initial=point["observed_initial"], observed_initial_ci=ci([d["observed_initial"] for d in draws]))
    for est in point["est"]:
        e = {k: dict(value=point["est"][est][k], ci=ci([d["est"][est][k] for d in draws])) for k in ("model", "error", "rho") + FREE_FACTORS}
        if point["est"][est]["diag"]:
            e["diag"] = {str(g): {k: dict(value=point["est"][est]["diag"][g][k],
                                          range=[float(np.min([d["est"][est]["diag"][g][k] for d in draws])),
                                                 float(np.max([d["est"][est]["diag"][g][k] for d in draws]))])
                                  for k in point["est"][est]["diag"][g]} for g in (0, 1)}
        out[est] = e
        print(f"{est:5s} model {e['model']['value']:+.4f} {np.round(e['model']['ci'], 4)} I {e['I']['value']:+.4f} {np.round(e['I']['ci'], 4)} "
              f"A {e['A']['value']:+.4f} a {e['a']['value']:+.4f} {np.round(e['a']['ci'], 4)} rho {e['rho']['value']:.3f} {np.round(e['rho']['ci'], 3)}"
              + (f" | cond {e['diag']['0']['cond_WX']['value']:.1e} firstR2min {e['diag']['0']['min_first_stage_r2']['value']:.3f}" if "diag" in e else ""), flush=True)
    print(f"observed: start {out['observed_initial']:+.4f} {np.round(out['observed_initial_ci'], 4)} end {out['observed']:+.4f} {np.round(out['observed_ci'], 4)}")
    d = ROOT / "outputs" / "07_eedi_iv"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / "results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
