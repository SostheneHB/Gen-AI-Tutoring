"""
Controlled study of the price and of its estimate, fourth version: more bootstrap draws, coverage after abstention
for other thresholds, and a second type of interval (review of 31 October 2026, point 3).

Same cells, learners, sample sizes, replicates and random streams as price_reliability_synthetic_v3.py (the data
of every replicate, and its first 60 bootstrap draws, are the same). Changes:
  * B = 1000 refit draws for the audit interval (v3: 60);
  * two intervals from the same draws: percentile (as in v3) and basic, [2 x - q97.5, 2 x - q2.5];
  * the diagnostic is stored in parts (D1, D2, the coefficient of variation of the price, T1, tau), so that the
    coverage among accepted replicates and the share whose gap reduction transfers can be computed for other
    thresholds: D3 in {0.25, 0.5, 1, none} and T2 in {0.1, 0.2, 0.3}.

    python price_reliability_synthetic_v4.py --workers 3 [--b 1000] [--learners ols,gbm] [--reps-ols 100 --reps-gbm 50]
        -> out_reviewer/price_reliability_synthetic_v4.json   (aggregates only)
    python price_reliability_synthetic_v4.py --time-one       (time one replicate of the heaviest cell, then stop)
"""
import argparse
import json
import os
import time
import zlib
from multiprocessing import Pool

import numpy as np

import price_reliability_synthetic_v2 as v2
import price_reliability_synthetic_v3 as v3

D3_GRID = (0.25, 0.5, 1.0, None)
T2_GRID = (0.1, 0.2, 0.3)
ci = v3.ci


def replicate(args):
    cell, n, learner, rep, b = args
    rng = np.random.default_rng([rep, n, zlib.crc32(cell["name"].encode()), int(learner == "gbm"), 3])   # as in v3
    (ztr, gtr, ytr), (zva, gva, yva), (zte, gte, yte) = (v2.draw(rng, n, cell, w) for w in ("train", "val", "test"))
    Q = lambda z: v2.features_q(cell, z)
    f, q = v2.fit_f(learner, ztr, ytr), v2.fit_q(Q(ztr), gtr)
    fva, fte, qva, qte = f(zva), f(zte), q(Q(zva)), q(Q(zte))
    one = np.ones(n)
    A = v3.ann(fva, qva, gva, yva, one)
    d = -A["c"] * (qte - A["center"])
    tau = v2.gap(fte + d, gte) / v2.gap(fte, gte)
    real = {w: v3.realised(rng, cell, f, q, Q, A, w) for w in ("val", "test")}
    full = []
    for _ in range(b):
        w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
        fb = v2.fit_f(learner, ztr, ytr, w) if learner == "ols" else f
        qb = v2.fit_q(Q(ztr), gtr, w)
        iv = rng.integers(0, n, n)
        Ab = v3.ann((fb(zva) if learner == "ols" else fva)[iv], qb(Q(zva))[iv], gva[iv], yva[iv], one)
        full.append((Ab["price"], Ab["total"], Ab["delta"], Ab["cov_sg"]))
    full = np.array(full)
    pr = full[:, 0]
    bt = [v2.gap(fte[i], gte[i]) for i in (rng.integers(0, n, n) for _ in range(200))]
    lo, hi = np.percentile(bt, [2.5, 97.5])
    basic = lambda x, draws: [float(2 * x - np.percentile(draws, 97.5)), float(2 * x - np.percentile(draws, 2.5))]
    return dict(cell=cell["name"], n=n, learner=learner, rep=rep,
                ann=dict(quad=float(A["price"]), total=float(A["total"])), realised=real, tau=float(tau),
                ci_pct=dict(quad=ci(pr), total=ci(full[:, 1])),
                ci_basic=dict(quad=basic(A["price"], pr), total=basic(A["total"], full[:, 1])),
                ci_pct_60=dict(quad=ci(pr[:60]), total=ci(full[:60, 1])),
                D1=bool(ci(full[:, 2])[0] > 0 or ci(full[:, 2])[1] < 0),
                D2=bool(ci(full[:, 3])[0] > 0 or ci(full[:, 3])[1] < 0),
                cv=float(pr.std() / pr.mean()) if pr.mean() > 0 else float("inf"),
                T1=bool(lo > 0 or hi < 0))


def summarise(rows, T):
    inside = lambda iv, t: iv[0] <= t <= iv[1]
    o = dict(truth_min=T["min"], truth_built=T["built"], n_reps=len(rows))
    for d3 in D3_GRID:
        acc = np.array([r["D1"] and r["D2"] and (d3 is None or r["cv"] <= d3) for r in rows])
        key = "none" if d3 is None else str(d3)
        o[f"share_accepted_d3_{key}"] = float(acc.mean())
        for t2 in T2_GRID:
            conf = acc & np.array([r["T1"] and abs(r["tau"]) <= t2 for r in rows])
            o[f"share_transfers_d3_{key}_t2_{t2}"] = float(conf.mean())
        for kind in ("pct", "basic", "pct_60"):
            for part in ("quad", "total"):
                hit = np.array([inside(r[f"ci_{kind}"][part], r["realised"]["val"][part]) for r in rows])
                o[f"cov_{kind}_{part}_d3_{key}"] = float(hit[acc].mean()) if acc.any() else None
                o[f"cov_{kind}_{part}_d3_{key}_n"] = int(acc.sum())
    for part in ("quad", "total"):
        o[f"cov_pct_{part}_all"] = float(np.mean([inside(r["ci_pct"][part], r["realised"]["val"][part]) for r in rows]))
        o[f"width_pct_{part}_median"] = float(np.median([r["ci_pct"][part][1] - r["ci_pct"][part][0] for r in rows]))
        o[f"ann_{part}_median"] = float(np.median([r["ann"][part] for r in rows]))
        o[f"realised_val_{part}_median"] = float(np.median([r["realised"]["val"][part] for r in rows]))
    return o


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--workers", type=int, default=3)
    a_.add_argument("--b", type=int, default=1000)
    a_.add_argument("--learners", default="ols,gbm")
    a_.add_argument("--reps-ols", type=int, default=v3.REPS["ols"])
    a_.add_argument("--reps-gbm", type=int, default=v3.REPS["gbm"])
    a_.add_argument("--time-one", action="store_true")
    a = a_.parse_args()
    cells = v3.CELLS
    if a.time_one:
        c = next(c for c in cells if c["name"] == "aligned_a1.5")
        for l in a.learners.split(","):
            t0 = time.time()
            replicate((c, 20000, l, 0, a.b))
            print(f"one replicate, n=20000, {l}, B={a.b}: {time.time() - t0:.1f}s")
        return
    reps = dict(ols=a.reps_ols, gbm=a.reps_gbm)
    learners = a.learners.split(",")
    T = {c["name"]: v2.truths(c) for c in cells}
    tasks = [(c, n, l, r, a.b) for c in cells for n in (2000, 20000) for l in learners for r in range(reps[l])]
    rows = []
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(replicate, tasks, chunksize=2)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    out = {"b": a.b, "d3_grid": [x if x is not None else "none" for x in D3_GRID], "t2_grid": list(T2_GRID), "cells": {}}
    for c in cells:
        for n in (2000, 20000):
            for l in learners:
                sel = [r for r in rows if r["cell"] == c["name"] and r["n"] == n and r["learner"] == l]
                s = summarise(sel, T[c["name"]])
                out["cells"][f"{c['name']}|{n}|{l}"] = dict(cell=c, n=n, learner=l, **s)
                print(f"{c['name']:38s} n={n:5d} {l}: acc(D3=0.5) {s['share_accepted_d3_0.5']:.2f} | cov total pct {s['cov_pct_total_d3_0.5']} "
                      f"basic {s['cov_basic_total_d3_0.5']} pct60 {s['cov_pct_60_total_d3_0.5']}", flush=True)
    os.makedirs("out_reviewer", exist_ok=True)
    json.dump(out, open(os.path.join("out_reviewer", "price_reliability_synthetic_v4.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
