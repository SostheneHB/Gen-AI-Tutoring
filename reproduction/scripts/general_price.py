"""
The price of erasure in general form, and its horizon profile.

Claim checked here.  For features z, target y and binary group g with shares p0, p1, let f* = E[y|z] and
q(z) = P(g = 1 | z).  Among all predictors of z whose predicted group gap is a fraction tau of the gap
Delta of f*, the smallest extra MSE is
        (1 - tau)^2 * p0 p1 * Delta^2 / R2,        R2 = Var(q) / (p0 p1),
reached by f_tau = f* - (1 - tau) * Cov(f*, q) / Var(q) * (q - p1).  The linear version (Proposition
erasure-price) is the same formula with f* and q replaced by their best linear versions.

Part 1 (linear, horizons h = 1..25): target k_{t+h}, features = EDMD features of x_t; best linear predictor
  fitted on the training students.  Formula vs measured extra MSE of the gap-free linear predictor
  (training: exact identity; held-out students: check).
Part 2 (nonlinear, h = 1 and 10): f* and q estimated by gradient boosting on x_t with out-of-fold
  predictions; formula vs measured extra MSE of f_0 on held-out students, and its remaining gap.

    python general_price.py   -> out_rank_matched/general_price.json  (aggregates only)
"""
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, GroupKFold

import assistments_pipeline as ap
import rank_matched_controls as rm

ROOT = os.environ.get("DATA_ROOT", ".")
HORIZONS = [1, 2, 5, 10, 15, 20, 25]
NL_HORIZONS = [1, 10]


def design(fit, X, G, h):
    """Rows (student, t) with t + h <= 25: lifted features of x_t, physical x_t, target k_{t+h}, group, student id."""
    n = X.shape[0]
    T = ap.N_TRANS + 1 - h
    Z = ap.encode(fit, X[:, :T]).reshape(-1, len(fit.mu))
    Xp = X[:, :T].reshape(-1, X.shape[-1])
    y = X[:, h:h + T, 0].reshape(-1)
    g = np.repeat(G, T)
    sid = np.repeat(np.arange(n), T)
    return Z, Xp, y, g, sid


def gap(v, g):
    return float(v[g == 1].mean() - v[g == 0].mean())


def linear_part(fit, Xtr, Gtr, Xte, Gte, h):
    Z, _, y, g, _ = design(fit, Xtr, Gtr, h)
    Zt, _, yt, gt, _ = design(fit, Xte, Gte, h)
    mz, my, p1 = Z.mean(0), y.mean(), g.mean()
    Zc, Ztc = Z - mz, Zt - mz
    S = Zc.T @ Zc / len(Zc)
    beta = np.linalg.lstsq(S, Zc.T @ (y - my) / len(Zc), rcond=None)[0]
    gam = np.linalg.lstsq(S, Zc.T @ (g - p1) / len(Zc), rcond=None)[0]
    yhat, ghat = Zc @ beta, Zc @ gam
    r2 = float(ghat.var() / (p1 * (1 - p1)))
    D = gap(yhat, g)
    formula = p1 * (1 - p1) * D ** 2 / r2
    w0 = beta - gam * (beta @ S @ gam) / (gam @ S @ gam)
    mse = lambda pred, t: float(((t - pred) ** 2).mean())
    return dict(h=h, p1=float(p1), predicted_gap=D, r2_group=r2, formula=float(formula),
                formula_rho_form=float(yhat.var() * np.corrcoef(yhat, ghat)[0, 1] ** 2),
                measured_train=mse(Zc @ w0 + my, y) - mse(yhat + my, y),
                measured_test=mse(Ztc @ w0 + my, yt) - mse(Ztc @ beta + my, yt),
                remaining_gap_test=gap(Ztc @ w0 + my, gt))


def nonlinear_part(fit, Xtr, Gtr, Xte, Gte, h, seed):
    _, Xp, y, g, sid = design(fit, Xtr, Gtr, h)
    _, Xpt, yt, gt, _ = design(fit, Xte, Gte, h)
    reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=seed)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    cv = GroupKFold(n_splits=3)
    f_oof = cross_val_predict(reg, Xp, y, cv=cv, groups=sid)
    q_oof = cross_val_predict(clf, Xp, g, cv=cv, groups=sid, method="predict_proba")[:, 1]
    p1 = g.mean()
    D = gap(f_oof, g)
    r2 = float(q_oof.var() / (p1 * (1 - p1)))
    formula = p1 * (1 - p1) * D ** 2 / r2
    c = np.cov(f_oof, q_oof)[0, 1] / q_oof.var()
    reg.fit(Xp, y); clf.fit(Xp, g)
    ft, qt = reg.predict(Xpt), clf.predict_proba(Xpt)[:, 1]
    f0 = ft - c * (qt - p1)
    mse = lambda pred: float(((yt - pred) ** 2).mean())
    return dict(h=h, predicted_gap=D, r2_group=r2, formula=float(formula),
                measured_test=mse(f0) - mse(ft), remaining_gap_test=gap(f0, gt), original_gap_test=gap(ft, gt))


def run(Xtr, Gtr, Xte, Gte, seed):
    fit = ap.fit_edmd(Xtr)
    return dict(linear=[linear_part(fit, Xtr, Gtr, Xte, Gte, h) for h in HORIZONS],
                nonlinear=[nonlinear_part(fit, Xtr, Gtr, Xte, Gte, h, seed) for h in NL_HORIZONS])


def main():
    pick = lambda t: [v for i, v in enumerate(t) if i in (1, 2, 4, 5)]
    res = {}
    df = ap.keep_full_trajectories(ap.load_features("assistments_features_time.csv"))
    res["assistments"] = [run(*pick(rm.prepare(df, 20260914 + 200 + j, "school", "train")), j) for j in range(5)]
    df = ap.keep_full_trajectories(ap.load_features(f"{ROOT}/Datasets/Eedi_NeurIPS2020/eedi_features_time.csv"))
    res["eedi"] = [run(*pick(rm.prepare(df, 20260914 + 200 + j, "school", "given")), j) for j in range(5)]
    base = ap.keep_full_trajectories(ap.load_features(f"{ROOT}/Datasets/OULAD/oulad_features_weekly.csv"))
    for att in ("imd", "gender", "disability"):
        d = base.copy(); d["S"] = d[f"S_{att}"]; d["G"] = d["S"].astype(int)
        res[f"oulad_{att}"] = [run(*pick(rm.prepare(d, 20260914 + 300 + j, "student", "given")), j) for j in range(5)]
    json.dump(res, open(os.path.join(rm.OUT, "general_price.json"), "w"), indent=1)
    for k, runs in res.items():
        print(f"== {k}")
        for i, h in enumerate(HORIZONS):
            L = [r["linear"][i] for r in runs]
            m = lambda q: np.mean([x[q] for x in L])
            print(f"   linear h={h:2d}: gap {m('predicted_gap'):+.4f} R2 {m('r2_group'):.3f} formula {m('formula'):.5f} "
                  f"(rho form {m('formula_rho_form'):.5f}) train {m('measured_train'):.5f} test {m('measured_test'):.5f} "
                  f"gap left {m('remaining_gap_test'):+.4f}")
        for i, h in enumerate(NL_HORIZONS):
            N = [r["nonlinear"][i] for r in runs]
            m = lambda q: np.mean([x[q] for x in N])
            print(f"   nonlin h={h:2d}: gap {m('predicted_gap'):+.4f} R2 {m('r2_group'):.3f} formula {m('formula'):.5f} "
                  f"test {m('measured_test'):.5f} gap {m('original_gap_test'):+.4f} -> {m('remaining_gap_test'):+.4f}")


if __name__ == "__main__":
    main()
