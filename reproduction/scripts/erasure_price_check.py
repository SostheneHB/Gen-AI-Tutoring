"""
Numerical check of the price of linear group erasure (proposed new proposition).

For features z (centred, covariance S), a target y and a binary group g, let
beta = S^-1 Cov(z, y) (best linear predictor yhat) and gamma = S^-1 Cov(z, g)
(best linear group score ghat).  Among linear predictors w'z whose predicted group
gap is a fraction tau of the gap of yhat, the smallest MSE is
    MSE(yhat) + Var(yhat) * rho(yhat, ghat)^2 * (1 - tau)^2,
reached by w = beta - (1 - tau) gamma (beta'S gamma)/(gamma'S gamma).  tau = 0 is
LEACE followed by refitting: the predicted gap is then exactly zero.

Target: one-step knowledge k_{t+1} from the EDMD features of x_t, as in the paper.
Checked on the training data (exact identity) and on held-out students (no guarantee).

    python erasure_price_check.py
Writes out_rank_matched/erasure_price.json (aggregates only).
"""
import json
import os

import numpy as np

import assistments_pipeline as ap
import rank_matched_controls as rm

ROOT = os.path.expanduser("~/Documents/Directive_research_copy_code")
TAUS = [0.0, 0.25, 0.5, 0.75, 1.0]


def check(Xtr, Gtr, Xte, Gte):
    fit = ap.fit_edmd(Xtr)
    def design(X, G):
        Z = ap.encode(fit, X[:, :-1]).reshape(-1, len(fit.mu))
        return Z, X[:, 1:, 0].reshape(-1), np.repeat(G, ap.N_TRANS)
    Z, y, g = design(Xtr, Gtr)
    mz, my, mg = Z.mean(0), y.mean(), g.mean()
    Zc = Z - mz
    S = Zc.T @ Zc / len(Zc)
    beta = np.linalg.lstsq(S, Zc.T @ (y - my) / len(Zc), rcond=None)[0]
    gam = np.linalg.lstsq(S, Zc.T @ (g - mg) / len(Zc), rcond=None)[0]
    yhat, ghat = Zc @ beta, Zc @ gam
    rho = float(np.corrcoef(yhat, ghat)[0, 1])
    var = float(yhat.var())
    Zt, yt, gt = design(Xte, Gte)
    out = dict(rho_yhat_ghat=rho, var_yhat=var, r2_group=float(np.corrcoef(ghat, g)[0, 1] ** 2),
               r2_k=float(1 - ((y - my - yhat) ** 2).mean() / y.var()), taus={})
    gap = lambda v, gg: float(v[gg == 1].mean() - v[gg == 0].mean())
    for tau in TAUS:
        w = beta - (1 - tau) * gam * (beta @ S @ gam) / (gam @ S @ gam)
        f_tr, f_te = Zc @ w + my, (Zt - mz) @ w + my
        out["taus"][str(tau)] = dict(
            excess_mse_train=float(((y - f_tr) ** 2).mean() - ((y - my - yhat) ** 2).mean()),
            formula=var * rho ** 2 * (1 - tau) ** 2,
            excess_mse_test=float(((yt - f_te) ** 2).mean() - ((yt - (Zt - mz) @ beta - my) ** 2).mean()),
            gap_ratio_train=gap(f_tr, g) / gap(Zc @ beta + my, g),
            gap_pred_test=gap(f_te, gt), gap_obs_test=gap(yt, gt))
    return out


def main():
    res = {}
    df = ap.keep_full_trajectories(ap.load_features("assistments_features_time.csv"))
    res["assistments"] = [check(*[v for i, v in enumerate(rm.prepare(df, 20260914 + 200 + j, "school", "train")) if i in (1, 2, 4, 5)])
                          for j in range(5)]
    df = ap.keep_full_trajectories(ap.load_features(f"{ROOT}/Datasets/Eedi_NeurIPS2020/eedi_features_time.csv"))
    res["eedi"] = [check(*[v for i, v in enumerate(rm.prepare(df, 20260914 + 200 + j, "school", "given")) if i in (1, 2, 4, 5)])
                   for j in range(5)]
    base = ap.keep_full_trajectories(ap.load_features(f"{ROOT}/Datasets/OULAD/oulad_features_weekly.csv"))
    for att in ("imd", "gender", "disability"):
        df = base.copy()
        df["S"] = df[f"S_{att}"]
        df["G"] = df["S"].astype(int)
        res[f"oulad_{att}"] = [check(*[v for i, v in enumerate(rm.prepare(df, 20260914 + 300 + j, "student", "given")) if i in (1, 2, 4, 5)])
                               for j in range(5)]
    os.makedirs(rm.OUT, exist_ok=True)
    json.dump(res, open(os.path.join(rm.OUT, "erasure_price.json"), "w"), indent=1)
    for k, runs in res.items():
        r = runs[0]
        t0 = r["taus"]["0.0"]
        print(f"{k:18s} rho {np.mean([x['rho_yhat_ghat'] for x in runs]):+.3f}  R2(group) {np.mean([x['r2_group'] for x in runs]):.3f}  "
              f"R2(k) {np.mean([x['r2_k'] for x in runs]):.3f} | split0 tau=0: excess train {t0['excess_mse_train']:.5f} "
              f"formula {t0['formula']:.5f} test {t0['excess_mse_test']:.5f}; gap ratio {t0['gap_ratio_train']:.1e}; "
              f"pred gap test {t0['gap_pred_test']:+.4f} vs obs {t0['gap_obs_test']:+.4f}")


if __name__ == "__main__":
    main()
