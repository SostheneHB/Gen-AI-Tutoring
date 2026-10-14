"""
Controlled study of the price and of its estimate, second version (reviews of 14 October 2026).

Changes from price_reliability_synthetic.py:
  * a validation set (same law as training) separates the stages: fit on training units, announce and diagnose
    on validation units, evaluate on test units;
  * two estimators: "v1", the published one (training units, R^2 = Var(q)/(p0 p1), c = Cov(f,q)/Var(q),
    rule R1-R4 on test units), and "v2" (Proposition 3: on validation units, c = Cov(f,G)/Cov(q,G) and
    price p0 p1 Delta^2 / corr(q,G)^2, the cost of the correction built from q; diagnostics D1-D3 on validation
    units);
  * two targets: the minimum price (Proposition 1) and the cost of the correction built from the population
    limit of the fitted group model (they differ only when the group model is misspecified);
  * a cell where the group model is calibrated but misspecified (z = (X1, X2) in {-1,1}^2,
    q = 0.5 + 0.2 X1 + 0.2 X1 X2, y = X1 + noise): a main-effects logistic model is calibrated and misses the
    interaction (minimum 0.5, cost of the built correction 1.0); the same cell with the correct model;
  * error and interval coverage reported overall and among the replicates that the diagnostics accept;
    coverage is reported also when the true price is zero (a positive interval then never covers it).

    python price_reliability_synthetic_v2.py --workers 5   -> out_reviewer/price_reliability_synthetic_v2.json
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

import price_reliability_synthetic as v1

D, P1 = v1.D, v1.P1
B = 60
REPS = dict(ols=100, gbm=50)
RULE = v1.RULE
CELLS = v1.CELLS + [dict(name="calibrated_misspecified", kind="discrete", qmodel="main"),
                    dict(name="calibrated_misspecified_correct_model", kind="discrete", qmodel="interaction")]


def draw(rng, n, cell, which):
    if cell.get("kind") == "discrete":
        X = rng.choice([-1.0, 1.0], size=(n, 2))
        q = 0.5 + 0.2 * X[:, 0] + 0.2 * X[:, 0] * X[:, 1]
        g = (rng.random(n) < q).astype(float)
        return X, g, X[:, 0] + rng.standard_normal(n)
    cos = cell["cos_train"] if which != "test" else cell["cos_test"]
    return v1.draw(rng, n, cell["a"], cos)


def features_q(cell, z):
    if cell.get("kind") == "discrete" and cell["qmodel"] == "interaction":
        return np.column_stack([z, z[:, 0] * z[:, 1]])
    return z


def truths(cell):
    if cell.get("kind") == "discrete":
        built = 1.0 if cell["qmodel"] == "main" else 0.5
        return dict(min=0.5, built=built, r2=0.32, test_min=0.5)
    t = v1.true_price(cell["a"], cell["cos_train"])
    tt = v1.true_price(cell["a"], cell["cos_test"], seed=1)
    return dict(min=t["price"], built=t["price"], r2=t["r2"], test_min=tt["price"])


def fit_f(learner, z, y, w=None):
    return v1.fit_f(learner, z, y, w)


def fit_q(X, g, w=None):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(X, g, sample_weight=w)
    return lambda XX: m.predict_proba(XX)[:, 1]


def wm(v, w):
    return (v * w).sum() / w.sum()


def v2q(f, s, g, w):
    p1 = wm(g, w)
    cfg_ = wm((f - wm(f, w)) * (g - p1), w)
    csg = wm((s - wm(s, w)) * (g - p1), w)
    vs = wm((s - wm(s, w)) ** 2, w)
    corr2 = csg ** 2 / (vs * p1 * (1 - p1))
    delta = wm(f * g, w) / wm(g, w) - wm(f * (1 - g), w) / wm(1 - g, w)
    return dict(c=cfg_ / csg, center=wm(s, w), price=p1 * (1 - p1) * delta ** 2 / corr2, delta=delta, cov_sg=csg)


def gap(v, g):
    return v[g == 1].mean() - v[g == 0].mean()


def replicate(args):
    cell, n, learner, rep = args
    rng = np.random.default_rng([rep, n, zlib.crc32(cell["name"].encode()), int(learner == "gbm"), 2])
    (ztr, gtr, ytr), (zva, gva, yva), (zte, gte, yte) = (draw(rng, n, cell, w) for w in ("train", "val", "test"))
    Q = lambda z: features_q(cell, z)
    f, q = fit_f(learner, ztr, ytr), fit_q(Q(ztr), gtr)
    ftr, fva, fte = f(ztr), f(zva), f(zte)
    qtr, qva, qte = q(Q(ztr)), q(Q(zva)), q(Q(zte))
    one = lambda m: np.ones(m)
    # v1
    A = v1.announce(ftr, qtr, gtr, one(n))
    d1 = -A["c"] * (qte - A["p1"])
    tau1 = gap(fte + d1, gte) / gap(fte, gte)
    # v2
    V = v2q(fva, qva, gva, one(n))
    d2 = -V["c"] * (qte - V["center"])
    tau2 = gap(fte + d2, gte) / gap(fte, gte)
    meas2 = ((fte + d2 - yte) ** 2).mean() - ((fte - yte) ** 2).mean()
    cross2 = 2 * ((fte - yte) * d2).mean()
    p1draws, p2draws, deltas, covs = [], [], [], []
    for _ in range(B):
        w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
        fb = fit_f(learner, ztr, ytr, w) if learner == "ols" else f
        qb = fit_q(Q(ztr), gtr, w)
        p1draws.append(v1.announce(fb(ztr) if learner == "ols" else ftr, qb(Q(ztr)), gtr, w)["price"])
        iv = rng.integers(0, n, n)
        Vb = v2q((fb(zva) if learner == "ols" else fva)[iv], qb(Q(zva))[iv], gva[iv], one(n))
        p2draws.append(Vb["price"]); deltas.append(Vb["delta"]); covs.append(Vb["cov_sg"])
    p1d, p2d = np.array(p1draws), np.array(p2draws)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    # v1 rule (as published, on test units)
    bt = [gap(fte[i], gte[i]) for i in (rng.integers(0, n, n) for _ in range(200))]
    lo, hi = np.percentile(bt, [2.5, 97.5])
    slope, ece = v1.calib(qte, gte)
    cv1 = float(p1d.std() / p1d.mean()) if p1d.mean() > 0 else np.inf
    ok1 = bool((lo > 0 or hi < 0) and RULE["slope"][0] <= slope <= RULE["slope"][1] and ece <= RULE["ece"]
               and abs(tau1) <= RULE["tau"] and cv1 <= RULE["price_cv"])
    # v2 diagnostics (on validation units)
    cv2 = float(p2d.std() / p2d.mean()) if p2d.mean() > 0 else np.inf
    c2 = dict(D1=bool(ci(deltas)[0] > 0 or ci(deltas)[1] < 0), D2=bool(ci(covs)[0] > 0 or ci(covs)[1] < 0), D3=bool(cv2 <= 0.5))
    t2 = dict(T1=bool(lo > 0 or hi < 0), T2=bool(abs(tau2) <= 0.2))
    return dict(cell=cell["name"], n=n, learner=learner, rep=rep,
                v1=dict(price=float(A["price"]), ci=ci(p1d), accepted=ok1, tau=float(tau1)),
                v2=dict(price=float(V["price"]), ci=ci(p2d), accepted=bool(all(c2.values())), checks=c2, tau=float(tau2), test_checks=t2,
                        confirmed=bool(all(c2.values()) and all(t2.values())),
                        measured=float(meas2), cross=float(cross2)))


def summarise(rows, T):
    out = dict(truth_min=T["min"], truth_built=T["built"], r2=T["r2"], n_reps=len(rows))
    for est in ("v1", "v2"):
        pr = np.array([r[est]["price"] for r in rows])
        acc = np.array([r[est]["accepted"] for r in rows])
        cov = lambda t, m: float(np.mean([r[est]["ci"][0] <= t <= r[est]["ci"][1] for r, k in zip(rows, m) if k])) if m.any() else None
        rel = lambda t, m: float(np.median(np.abs(pr[m] - t) / max(t, 1e-3))) if m.any() else None
        allm = np.ones(len(rows), bool)
        out[est] = dict(median=float(np.median(pr)), q10_q90=[float(np.percentile(pr, 10)), float(np.percentile(pr, 90))],
                        share_accepted=float(acc.mean()),
                        coverage_min=cov(T["min"], allm), coverage_min_accepted=cov(T["min"], acc),
                        coverage_built=cov(T["built"], allm), coverage_built_accepted=cov(T["built"], acc),
                        rel_error_min=rel(T["min"], allm), rel_error_min_accepted=rel(T["min"], acc),
                        rel_error_min_rejected=rel(T["min"], ~acc), tau_median=float(np.median([r[est]["tau"] for r in rows])))
    out["v2"]["measured_mean"] = float(np.mean([r["v2"]["measured"] for r in rows]))
    out["v2"]["cross_mean"] = float(np.mean([r["v2"]["cross"] for r in rows]))
    out["v2"]["checks_pass"] = {k: float(np.mean([r["v2"]["checks"][k] for r in rows])) for k in ("D1", "D2", "D3")}
    out["v2"]["test_checks_pass"] = {k: float(np.mean([r["v2"]["test_checks"][k] for r in rows])) for k in ("T1", "T2")}
    conf = np.array([r["v2"]["confirmed"] for r in rows])
    out["v2"]["share_confirmed"] = float(conf.mean())
    tt = T.get("test_min", T["min"])
    out["truth_test_min"] = tt
    out["v2"]["measured_minus_test_truth_median"] = float(np.median([r["v2"]["measured"] for r in rows]) - tt)
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--workers", type=int, default=5)
    a = a_.parse_args()
    T = {c["name"]: truths(c) for c in CELLS}
    tasks = [(c, n, l, r) for c in CELLS for n in (2000, 20000) for l in ("ols", "gbm") for r in range(REPS[l])]
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(replicate, tasks, chunksize=2)):
            rows.append(r)
            if (i + 1) % 400 == 0:
                print(f"{i + 1}/{len(tasks)}", flush=True)
    out = {"b": B, "rule_v1": RULE, "cells": {}}
    for c in CELLS:
        for n in (2000, 20000):
            for l in ("ols", "gbm"):
                sel = [r for r in rows if r["cell"] == c["name"] and r["n"] == n and r["learner"] == l]
                s = summarise(sel, T[c["name"]])
                out["cells"][f"{c['name']}|{n}|{l}"] = dict(cell=c, n=n, learner=l, **s)
                print(f"{c['name']:38s} n={n:5d} {l}: min {s['truth_min']:.3f} built {s['truth_built']:.3f} | "
                      f"v1 {s['v1']['median']:.3f} acc {s['v1']['share_accepted']:.2f} cov {s['v1']['coverage_min']} cov|acc {s['v1']['coverage_min_accepted']} | "
                      f"v2 {s['v2']['median']:.3f} acc {s['v2']['share_accepted']:.2f} cov {s['v2']['coverage_min']:.2f} cov|acc {s['v2']['coverage_min_accepted']} "
                      f"covbuilt|acc {s['v2']['coverage_built_accepted']} tau {s['v2']['tau_median']:+.2f} confirmed {s['v2']['share_confirmed']:.2f} T {s['v2']['test_checks_pass']}", flush=True)
    os.makedirs("out_reviewer", exist_ok=True)
    json.dump(out, open("out_reviewer/price_reliability_synthetic_v2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
