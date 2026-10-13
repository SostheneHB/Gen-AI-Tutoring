"""
Price of mean-contrast removal over the forecast horizon, with a common set of starting points
(referee report 5, item 2).

For every horizon h in HORIZONS the information is x_t with t in a COMMON set of starts (t = 0..T0), so the law
of the information, and hence R^2 of the group, is the same for every h.

  direct   : best linear predictor of k_{t+h} from the EDMD features of x_t, fitted on the training students;
             Proposition 2 applies to it (formula = measured on the training data).
  iterated : the fitted EDMD one-step model applied h times, then decoded. It is NOT the best linear predictor of
             k_{t+h}, so its cost under the same removal d = -c (g_hat - p1) is
                 E[d^2] + 2 E[(f - Y) d]
             and the second (cross) term need not vanish; both terms are reported.

    python horizon_check.py      -> out_rank_matched/horizon_check.json (aggregates only)
"""
import json
import os

import numpy as np

import assistments_pipeline as ap
import rank_matched_controls as rm

ROOT = os.environ.get("DATA_ROOT", ".")
HORIZONS = [1, 2, 5, 10, 15, 20]
T0 = ap.N_TRANS - max(HORIZONS)          # common starts t = 0..T0


def rows(fit, X, G, h):
    t = np.arange(T0 + 1)
    Z = ap.encode(fit, X[:, t]).reshape(-1, len(fit.mu))
    y = X[:, t + h, 0].reshape(-1)
    g = np.repeat(G, len(t))
    return Z, y, g, t


def iterate(fit, Z, h, A=None, a=None):
    A = fit.A if A is None else A
    a = fit.a if a is None else a
    z = Z.copy()
    for _ in range(h):
        z = z @ A.T + a
    return z @ fit.C[0] + fit.c[0]


def one(fit, Xtr, Gtr, Xte, Gte, h, P=None):
    Z, y, g, _ = rows(fit, Xtr, Gtr, h)
    Zt, yt, gt, _ = rows(fit, Xte, Gte, h)
    mz, my, p1 = Z.mean(0), y.mean(), g.mean()
    Zc, Ztc = Z - mz, Zt - mz
    S = Zc.T @ Zc / len(Zc)
    Sp = np.linalg.pinv(S)                                    # covariance is singular (p_t is binary)
    beta = Sp @ (Zc.T @ (y - my) / len(Zc))
    gam = Sp @ (Zc.T @ (g - p1) / len(Zc))
    ghat, ghat_t = Zc @ gam, Ztc @ gam
    r2 = float(ghat.var() / (p1 * (1 - p1)))
    mse = lambda f, t: float(((t - f) ** 2).mean())
    gap = lambda v, gg: float(v[gg == 1].mean() - v[gg == 0].mean())
    out = dict(h=h, r2_group=r2, p1=float(p1))
    # direct best linear predictor
    yhat, yhat_t = Zc @ beta + my, Ztc @ beta + my
    c = np.cov(yhat, ghat)[0, 1] / ghat.var()
    d, dt = -c * ghat, -c * ghat_t
    out["direct"] = dict(gap=gap(yhat, g), formula=float(p1 * (1 - p1) * gap(yhat, g) ** 2 / r2),
                         measured_train=mse(yhat + d, y) - mse(yhat, y), measured_test=mse(yhat_t + dt, yt) - mse(yhat_t, yt),
                         gap_left_test=gap(yhat_t + dt, gt))
    # iterated EDMD forecast
    fi, fi_t = iterate(fit, Z, h), iterate(fit, Zt, h)
    c = np.cov(fi, ghat)[0, 1] / ghat.var()
    d, dt = -c * ghat, -c * ghat_t
    out["iterated"] = dict(gap=gap(fi, g), formula=float(p1 * (1 - p1) * gap(fi, g) ** 2 / r2),
                           d2_train=float((d ** 2).mean()), cross_train=float(2 * ((fi - y) * d).mean()),
                           measured_train=mse(fi + d, y) - mse(fi, y),
                           d2_test=float((dt ** 2).mean()), cross_test=float(2 * ((fi_t - yt) * dt).mean()),
                           measured_test=mse(fi_t + dt, yt) - mse(fi_t, yt), mse_vs_direct_test=mse(fi_t, yt) - mse(yhat_t, yt))
    if P is not None:                                         # forecast error of the raw and projected models, common starts
        fp_t = iterate(fit, Zt, h, P.T @ fit.A @ P.T, P.T @ fit.a)
        out["mse_raw_test"], out["mse_proj_test"] = mse(fi_t, yt), mse(fp_t, yt)
        out["mse_ratio_proj_raw"] = out["mse_proj_test"] / out["mse_raw_test"]
    return out


def run(df, seed, split, strata):
    _, Xtr, Gtr, _, Xte, Gte, _ = rm.prepare(df, seed, split, strata)
    fit = ap.fit_edmd(Xtr)
    Gamma, _ = ap.metric(fit.G, ap.RIDGE_RATIO)
    c, _ = ap.contrast(fit, Xtr, Gtr)
    P = ap.build_projector(fit.K, Gamma, c, rank=4)[0]
    return [one(fit, Xtr, Gtr, Xte, Gte, h, P) for h in HORIZONS]


def main():
    seed, res = 20260914, {}
    df = ap.keep_full_trajectories(ap.load_features("assistments_features_time.csv"))
    res["assistments"] = [run(df, seed + 200 + j, "school", "train") for j in range(5)]
    df = ap.keep_full_trajectories(ap.load_features(f"{ROOT}/Datasets/Eedi_NeurIPS2020/eedi_features_time.csv"))
    res["eedi"] = [run(df, seed + 200 + j, "school", "given") for j in range(5)]
    base = ap.keep_full_trajectories(ap.load_features(f"{ROOT}/Datasets/OULAD/oulad_features_weekly.csv"))
    for att in ("imd", "disability", "gender"):
        d = base.copy(); d["S"] = d[f"S_{att}"]; d["G"] = d["S"].astype(int)
        res[f"oulad_{att}"] = [run(d, seed + 300 + j, "student", "given") for j in range(5)]
    summ = {}
    for k, runs in res.items():
        summ[k] = []
        for i, h in enumerate(HORIZONS):
            m = lambda a, q: float(np.mean([r[i][a][q] for r in runs]))
            row = dict(h=h, r2_group=float(np.mean([r[i]["r2_group"] for r in runs])),
                       mse_raw_test=float(np.mean([r[i]["mse_raw_test"] for r in runs])),
                       mse_proj_test=float(np.mean([r[i]["mse_proj_test"] for r in runs])),
                       mse_ratio_proj_raw=float(np.mean([r[i]["mse_ratio_proj_raw"] for r in runs])),
                       direct={q: m("direct", q) for q in runs[0][i]["direct"]},
                       iterated={q: m("iterated", q) for q in runs[0][i]["iterated"]})
            summ[k].append(row)
            D, I = row["direct"], row["iterated"]
            print(f"{k:17s} h={h:2d} R2 {row['r2_group']:.3f} | direct formula {D['formula']:.5f} train {D['measured_train']:.5f} "
                  f"test {D['measured_test']:.5f} | iterated formula {I['formula']:.5f} d2 {I['d2_test']:.5f} cross {I['cross_test']:+.5f} "
                  f"measured {I['measured_test']:.5f} (iter-direct mse {I['mse_vs_direct_test']:+.4f}) | MSE raw {row['mse_raw_test']:.4f} proj {row['mse_proj_test']:.4f} x{row['mse_ratio_proj_raw']:.1f}", flush=True)
    os.makedirs(rm.OUT, exist_ok=True)
    json.dump({"common_starts": T0 + 1, "horizons": HORIZONS, "splits": res, "summary": summ},
              open(os.path.join(rm.OUT, "horizon_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
