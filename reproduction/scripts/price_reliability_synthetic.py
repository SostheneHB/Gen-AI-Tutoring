"""
Controlled study of the price of removing the mean group contrast and of its estimation
(review of 13 October 2026, items 2 and 3).  Everything is known in closed form or by large Monte Carlo.

Data.  z in R^6, G ~ Bernoulli(p1 = 0.4), z | G ~ N(G a u, I), y = beta'z + N(0, 1), with |beta| = |u| = 1.
       The true group posterior q(z) = P(G = 1 | z) is logistic in z, and f*(z) = beta'z.
Factors.
  * alignment of the group signal with the predictive signal, cos(beta, u) in {0, 0.5, 1}
    ("useless", "partial", "aligned");
  * group separation a in {0.2, 0.7, 1.5}, which sets R^2_G = Var(q) / (p0 p1) (small, medium, large);
  * two shift cells, where the usefulness of the group signal changes between training and test
    (a = 0.7): "utility lost" (cos 1 in training, 0 in test) and "utility gained" (cos 0, then 1);
  * training size n in {2000, 20000} (test set of the same size);
  * learner for f: least squares ("ols", well specified) or gradient boosting ("gbm", flexible).
Procedure (the audit of Section 4, as on the real data).
  1. Fit f and q (logistic regression) on the training set; announce price = p0 p1 Delta^2 / R^2, and the
     correction f0 = f - c (q - p1).
  2. Refit bootstrap (B draws of the training set): f and q refitted for ols, q and c refitted for gbm.
  3. Apply the correction on the test set: achieved tau, measured cost MSE(f0) - MSE(f), frontier (1 - tau)^2 price.
  4. Rule: interpretable only if (R1) the test contrast of f has a 95% interval excluding 0, (R2) q is calibrated
     on the test set (slope in [0.8, 1.2], ECE <= 0.05), (R3) |tau_test| <= 0.2, (R4) the coefficient of
     variation of the announced price over refits is <= 0.5.
Outputs per cell: true price (training and test laws), announced price, coverage of the refit interval,
held-out measured cost and frontier, share declared interpretable, and the error of the announced price as a
description of the test cost, separately for cells declared interpretable or not.

    python price_reliability_synthetic.py --workers 2   -> out_reviewer/price_reliability_synthetic.json
"""
import argparse
import json
import os
import zlib
from multiprocessing import Pool

import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

D, P1 = 6, 0.4
RULE = dict(slope=(0.8, 1.2), ece=0.05, tau=0.2, price_cv=0.5)
B_REFIT, B_ROWS = 100, 200
REPS = dict(ols=100, gbm=60)
CELLS = ([dict(name=f"{al}_a{a}", cos_train=c, cos_test=c, a=a) for al, c in (("useless", 0.0), ("partial", 0.5), ("aligned", 1.0))
          for a in (0.2, 0.7, 1.5)]
         + [dict(name="shift_utility_lost", cos_train=1.0, cos_test=0.0, a=0.7),
            dict(name="shift_utility_gained", cos_train=0.0, cos_test=1.0, a=0.7)])
BETA = np.eye(D)[0]


def direction(cos):
    return cos * np.eye(D)[0] + np.sqrt(max(1 - cos ** 2, 0)) * np.eye(D)[1]


def draw(rng, n, a, cos):
    g = (rng.random(n) < P1).astype(float)
    z = rng.standard_normal((n, D)) + np.outer(g, a * direction(cos))
    return z, g, z @ BETA + rng.standard_normal(n)


def true_price(a, cos, n=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    z, g, _ = draw(rng, n, a, cos)
    u = direction(cos)
    q = expit(a * z @ u - a ** 2 / 2 + np.log(P1 / (1 - P1)))
    f = z @ BETA
    r2 = q.var() / (P1 * (1 - P1))
    delta = a * cos                                    # E[f* | G=1] - E[f* | G=0]
    return dict(price=P1 * (1 - P1) * delta ** 2 / r2, r2=r2, delta=delta, check_delta=f[g == 1].mean() - f[g == 0].mean())


def fit_f(learner, z, y, w=None):
    if learner == "ols":
        X = np.column_stack([np.ones(len(z)), z])
        sw = np.ones(len(z)) if w is None else w
        coef = np.linalg.lstsq(X * np.sqrt(sw)[:, None], y * np.sqrt(sw), rcond=None)[0]
        return lambda zz: np.column_stack([np.ones(len(zz)), zz]) @ coef
    m = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=0).fit(z, y, sample_weight=w)
    return m.predict


def fit_q(z, g, w=None):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, g, sample_weight=w)
    return lambda zz: m.predict_proba(zz)[:, 1]


def announce(f, q, g, w):
    wm = lambda v: (v * w).sum() / w.sum()
    p1 = wm(g)
    vq = wm((q - wm(q)) ** 2)
    c = wm((f - wm(f)) * (q - wm(q))) / vq
    delta = wm(f * g) / wm(g) - wm(f * (1 - g)) / wm(1 - g)
    r2 = vq / (p1 * (1 - p1))
    return dict(price=p1 * (1 - p1) * delta ** 2 / r2, c=c, p1=p1, r2=r2)


def calib(q, g):
    lq = np.log(np.clip(q, 1e-9, 1 - 1e-9) / np.clip(1 - q, 1e-9, 1))
    slope = float(LogisticRegression(C=1e6).fit(lq[:, None], g).coef_[0, 0])
    b = np.minimum((q * 10).astype(int), 9)
    ece = float(sum(abs(g[b == k].mean() - q[b == k].mean()) * (b == k).mean() for k in range(10) if (b == k).any()))
    return slope, ece


def gap(v, g):
    return v[g == 1].mean() - v[g == 0].mean()


def replicate(args):
    cell, n, learner, rep = args
    rng = np.random.default_rng([rep, n, zlib.crc32(cell["name"].encode()), int(learner == "gbm")])
    ztr, gtr, ytr = draw(rng, n, cell["a"], cell["cos_train"])
    zte, gte, yte = draw(rng, n, cell["a"], cell["cos_test"])
    f, q = fit_f(learner, ztr, ytr), fit_q(ztr, gtr)
    ftr, fte, qtr, qte = f(ztr), f(zte), q(ztr), q(zte)
    A = announce(ftr, qtr, gtr, np.ones(n))
    f0 = fte - A["c"] * (qte - A["p1"])
    gf = gap(fte, gte)
    tau = gap(f0, gte) / gf
    measured = ((f0 - yte) ** 2).mean() - ((fte - yte) ** 2).mean()
    prices = []
    for _ in range(B_REFIT):
        w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
        fb = fit_f(learner, ztr, ytr, w) if learner == "ols" else f
        qb = fit_q(ztr, gtr, w)
        prices.append(announce(fb(ztr) if learner == "ols" else ftr, qb(ztr), gtr, w)["price"])
    prices = np.array(prices)
    boot = []
    for _ in range(B_ROWS):
        i = rng.integers(0, n, n)
        boot.append(gap(fte[i], gte[i]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    slope, ece = calib(qte, gte)
    cv = float(prices.std() / prices.mean()) if prices.mean() > 0 else np.inf
    checks = dict(R1=bool(lo > 0 or hi < 0), R2=bool(RULE["slope"][0] <= slope <= RULE["slope"][1] and ece <= RULE["ece"]),
                  R3=bool(abs(tau) <= RULE["tau"]), R4=bool(cv <= RULE["price_cv"]))
    return dict(cell=cell["name"], n=n, learner=learner, rep=rep, price=A["price"], r2=A["r2"],
                ci=[float(np.percentile(prices, 2.5)), float(np.percentile(prices, 97.5))], cv=cv,
                tau=float(tau), measured=float(measured), frontier=float((1 - tau) ** 2 * A["price"]),
                full_removal_frontier=float(A["price"]), slope=slope, ece=ece, contrast=float(gf),
                checks=checks, interpretable=bool(all(checks.values())))


def summarise(rows, truth_tr, truth_te):
    pr = np.array([r["price"] for r in rows])
    t_tr, t_te = truth_tr["price"], truth_te["price"]
    cover = np.mean([r["ci"][0] <= t_tr <= r["ci"][1] for r in rows])
    ok = np.array([r["interpretable"] for r in rows])
    # error of the announced price as a description of the cost of full removal under the TEST law
    err = np.abs(pr - t_te)
    rel = lambda e: float(np.median(e / max(t_te, 1e-3)))
    meas = np.array([r["measured"] for r in rows])
    front = np.array([r["frontier"] for r in rows])
    return dict(true_price_train=t_tr, true_price_test=t_te, true_r2_train=truth_tr["r2"], true_r2_test=truth_te["r2"],
                announced_mean=float(pr.mean()), announced_median=float(np.median(pr)),
                announced_q10_q90=[float(np.percentile(pr, 10)), float(np.percentile(pr, 90))],
                coverage_train_truth=float(cover), mean_cv=float(np.mean([min(r["cv"], 10) for r in rows])),
                tau_mean=float(np.mean([r["tau"] for r in rows])), tau_median=float(np.median([r["tau"] for r in rows])),
                measured_mean=float(meas.mean()), frontier_mean=float(front.mean()),
                measured_minus_frontier_median=float(np.median(meas - front)),
                share_interpretable=float(ok.mean()),
                checks_pass={k: float(np.mean([r["checks"][k] for r in rows])) for k in ("R1", "R2", "R3", "R4")},
                abs_error_vs_test_truth_interpretable=float(np.median(err[ok])) if ok.any() else None,
                abs_error_vs_test_truth_flagged=float(np.median(err[~ok])) if (~ok).any() else None,
                rel_error_vs_test_truth_all=rel(err), n_reps=len(rows))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--workers", type=int, default=2)
    a = a_.parse_args()
    truths = {}
    for c in CELLS:
        truths[(c["name"], "train")] = true_price(c["a"], c["cos_train"])
        truths[(c["name"], "test")] = true_price(c["a"], c["cos_test"], seed=1)
    tasks = [(c, n, l, r) for c in CELLS for n in (2000, 20000) for l in ("ols", "gbm") for r in range(REPS[l])]
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(replicate, tasks, chunksize=4)):
            rows.append(r)
            if (i + 1) % 400 == 0:
                print(f"{i + 1}/{len(tasks)}", flush=True)
    out = {"rule": RULE, "b_refit": B_REFIT, "cells": {}}
    for c in CELLS:
        for n in (2000, 20000):
            for l in ("ols", "gbm"):
                sel = [r for r in rows if r["cell"] == c["name"] and r["n"] == n and r["learner"] == l]
                s = summarise(sel, truths[(c["name"], "train")], truths[(c["name"], "test")])
                out["cells"][f"{c['name']}|{n}|{l}"] = dict(cell=c, n=n, learner=l, **s)
                print(f"{c['name']:22s} n={n:5d} {l}: true {s['true_price_train']:.4f}/{s['true_price_test']:.4f} "
                      f"announced {s['announced_median']:.4f} cover {s['coverage_train_truth']:.2f} tau {s['tau_median']:+.2f} "
                      f"measured {s['measured_mean']:.4f} frontier {s['frontier_mean']:.4f} interpretable {s['share_interpretable']:.2f} "
                      f"checks {s['checks_pass']}", flush=True)
    os.makedirs("out_reviewer", exist_ok=True)
    json.dump(out, open("out_reviewer/price_reliability_synthetic.json", "w"), indent=1)


if __name__ == "__main__":
    main()
