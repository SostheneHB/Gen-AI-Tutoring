"""
Controlled study of the price and of its estimate, third version: explicit targets.

Same cells, splits, learners and estimators as price_reliability_synthetic_v2.py. Added:
  * the realised cost of the fitted correction (f, q, c and centre fixed as fitted) on a large independent sample
    (N_BIG draws of the validation law and of the test law), both the quadratic part E[d^2] and the total change
    of the squared error E[(f + d - y)^2] - E[(f - y)^2];
  * two announcements on validation units: quadratic part pi_s = c^2 Var(s), and total = pi_s + 2 E_val[(f - y) d];
  * two intervals: "audit" (training units resampled with the group model refitted, f refitted for least squares,
    and validation units resampled; the interval used on the real data) and "conditional" (validation units only,
    the fitted f and q fixed);
  * coverage of each target (realised cost of the fitted correction, cost of the population built correction,
    minimum price) by each interval;
  * a stability check of the bootstrap: in the stability cells, B_STAB refit draws, and the intervals computed
    from the first 40, 60, 200 and all draws.

    python price_reliability_synthetic_v3.py --workers 3        -> out_reviewer/price_reliability_synthetic_v3.json
    python price_reliability_synthetic_v3.py --stability ...    -> out_reviewer/price_reliability_synthetic_v3_stability.json
"""
import argparse
import json
import os
import zlib
from multiprocessing import Pool

import numpy as np

import price_reliability_synthetic_v2 as v2

B, B_COND, B_STAB, N_BIG = 60, 400, 400, 200_000
REPS = dict(ols=100, gbm=50)
CELLS = v2.CELLS
STAB_CELLS = ["aligned_a0.2", "partial_a0.7", "aligned_a1.5", "calibrated_misspecified"]
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def ann(f, s, g, y, w):
    V = v2.v2q(f, s, g, w)
    d = -V["c"] * (s - V["center"])
    return dict(V, total=float(V["price"] + 2 * v2.wm((f - y) * d, w)))


def realised(rng, cell, f, q, Q, V, which):
    z, g, y = v2.draw(rng, N_BIG, cell, which)
    fz = f(z)
    d = -V["c"] * (q(Q(z)) - V["center"])
    return dict(quad=float((d ** 2).mean()), total=float(((fz + d - y) ** 2).mean() - ((fz - y) ** 2).mean()))


def replicate(args):
    cell, n, learner, rep, b = args
    rng = np.random.default_rng([rep, n, zlib.crc32(cell["name"].encode()), int(learner == "gbm"), 3])
    (ztr, gtr, ytr), (zva, gva, yva), (zte, gte, yte) = (v2.draw(rng, n, cell, w) for w in ("train", "val", "test"))
    Q = lambda z: v2.features_q(cell, z)
    f, q = v2.fit_f(learner, ztr, ytr), v2.fit_q(Q(ztr), gtr)
    fva, fte, qva, qte = f(zva), f(zte), q(Q(zva)), q(Q(zte))
    one = np.ones(n)
    A = ann(fva, qva, gva, yva, one)
    d = -A["c"] * (qte - A["center"])
    tau = v2.gap(fte + d, gte) / v2.gap(fte, gte)
    meas = dict(quad=float((d ** 2).mean()), total=float(((fte + d - yte) ** 2).mean() - ((fte - yte) ** 2).mean()))
    real = {w: realised(rng, cell, f, q, Q, A, w) for w in ("val", "test")}
    # audit interval: refit on training units, resample validation units
    full = []
    for _ in range(b):
        w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
        fb = v2.fit_f(learner, ztr, ytr, w) if learner == "ols" else f
        qb = v2.fit_q(Q(ztr), gtr, w)
        iv = rng.integers(0, n, n)
        Ab = ann((fb(zva) if learner == "ols" else fva)[iv], qb(Q(zva))[iv], gva[iv], yva[iv], one)
        full.append((Ab["price"], Ab["total"], Ab["delta"], Ab["cov_sg"]))
    full = np.array(full)
    cond = []
    for _ in range(B_COND):
        iv = rng.integers(0, n, n)
        Ab = ann(fva[iv], qva[iv], gva[iv], yva[iv], one)
        cond.append((Ab["price"], Ab["total"]))
    cond = np.array(cond)
    pr = full[:, 0]
    cv = float(pr.std() / pr.mean()) if pr.mean() > 0 else np.inf
    checks = dict(D1=bool(ci(full[:, 2])[0] > 0 or ci(full[:, 2])[1] < 0),
                  D2=bool(ci(full[:, 3])[0] > 0 or ci(full[:, 3])[1] < 0), D3=bool(cv <= 0.5))
    bt = [v2.gap(fte[i], gte[i]) for i in (rng.integers(0, n, n) for _ in range(200))]
    lo, hi = np.percentile(bt, [2.5, 97.5])
    tchecks = dict(T1=bool(lo > 0 or hi < 0), T2=bool(abs(tau) <= 0.2))
    out = dict(cell=cell["name"], n=n, learner=learner, rep=rep,
               ann=dict(quad=float(A["price"]), total=float(A["total"])), measured=meas, realised=real, tau=float(tau),
               ci_full=dict(quad=ci(pr), total=ci(full[:, 1])), ci_cond=dict(quad=ci(cond[:, 0]), total=ci(cond[:, 1])),
               accepted=bool(all(checks.values())), confirmed=bool(all(checks.values()) and all(tchecks.values())))
    if b > B:
        out["stab"] = {str(k): dict(quad=ci(pr[:k]), total=ci(full[:k, 1])) for k in (40, 60, 200, b)}
    return out


def summarise(rows, T):
    acc = np.array([r["accepted"] for r in rows])
    inside = lambda iv, t: iv[0] <= t <= iv[1]
    o = dict(truth_min=T["min"], truth_built=T["built"], n_reps=len(rows), share_accepted=float(acc.mean()),
             share_confirmed=float(np.mean([r["confirmed"] for r in rows])))
    for part in ("quad", "total"):
        for kind in ("full", "cond"):
            for law in ("val", "test"):
                hit = np.array([inside(r[f"ci_{kind}"][part], r["realised"][law][part]) for r in rows])
                o[f"cov_{kind}_{part}_realised_{law}"] = float(hit.mean())
                o[f"cov_{kind}_{part}_realised_{law}_accepted"] = float(hit[acc].mean()) if acc.any() else None
        o[f"ann_{part}_median"] = float(np.median([r["ann"][part] for r in rows]))
        o[f"realised_val_{part}_median"] = float(np.median([r["realised"]["val"][part] for r in rows]))
        o[f"realised_test_{part}_median"] = float(np.median([r["realised"]["test"][part] for r in rows]))
        o[f"rel_err_{part}_vs_realised_val_median"] = float(np.median([abs(r["ann"][part] - r["realised"]["val"][part]) / max(abs(r["realised"]["val"][part]), 1e-3) for r in rows]))
        o[f"width_full_{part}_median"] = float(np.median([r["ci_full"][part][1] - r["ci_full"][part][0] for r in rows]))
        o[f"width_cond_{part}_median"] = float(np.median([r["ci_cond"][part][1] - r["ci_cond"][part][0] for r in rows]))
    o["cov_full_quad_built"] = float(np.mean([inside(r["ci_full"]["quad"], T["built"]) for r in rows]))
    o["cov_full_quad_built_accepted"] = float(np.mean([inside(r["ci_full"]["quad"], T["built"]) for r, a in zip(rows, acc) if a])) if acc.any() else None
    o["cross_realised_val_median"] = float(np.median([r["realised"]["val"]["total"] - r["realised"]["val"]["quad"] for r in rows]))
    if "stab" in rows[0]:
        st = {}
        for k in rows[0]["stab"]:
            for part in ("quad", "total"):
                st[f"{k}_{part}_cov_realised_val"] = float(np.mean([inside(r["stab"][k][part], r["realised"]["val"][part]) for r in rows]))
                st[f"{k}_{part}_width_median"] = float(np.median([r["stab"][k][part][1] - r["stab"][k][part][0] for r in rows]))
                kmax = max(rows[0]["stab"], key=int)
                st[f"{k}_{part}_endpoint_rel_dev_median"] = float(np.median([
                    max(abs(r["stab"][k][part][j] - r["stab"][kmax][part][j]) for j in (0, 1)) / max(abs(r["stab"][kmax][part][1] - r["stab"][kmax][part][0]), 1e-9)
                    for r in rows]))
        o["stability"] = st
    return o


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--workers", type=int, default=3)
    a_.add_argument("--stability", action="store_true")
    a = a_.parse_args()
    T = {c["name"]: v2.truths(c) for c in CELLS}
    if a.stability:
        cells = [c for c in CELLS if c["name"] in STAB_CELLS]
        tasks = [(c, 2000, l, r, B_STAB) for c in cells for l in ("ols", "gbm") for r in range(30)]
        name = "price_reliability_synthetic_v3_stability.json"
    else:
        cells = CELLS
        tasks = [(c, n, l, r, B) for c in cells for n in (2000, 20000) for l in ("ols", "gbm") for r in range(REPS[l])]
        name = "price_reliability_synthetic_v3.json"
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(replicate, tasks, chunksize=2)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(tasks)}", flush=True)
    out = {"b": B_STAB if a.stability else B, "b_cond": B_COND, "n_big": N_BIG, "cells": {}}
    for c in cells:
        for n in sorted({t[1] for t in tasks}):
            for l in ("ols", "gbm"):
                sel = [r for r in rows if r["cell"] == c["name"] and r["n"] == n and r["learner"] == l]
                s = summarise(sel, T[c["name"]])
                out["cells"][f"{c['name']}|{n}|{l}"] = dict(cell=c, n=n, learner=l, **s)
                print(f"{c['name']:38s} n={n:5d} {l}: acc {s['share_accepted']:.2f} | quad ann {s['ann_quad_median']:.3f} real {s['realised_val_quad_median']:.3f} "
                      f"cov full {s['cov_full_quad_realised_val']:.2f} (acc {s['cov_full_quad_realised_val_accepted']}) cond {s['cov_cond_quad_realised_val']:.2f} | "
                      f"total ann {s['ann_total_median']:.3f} real {s['realised_val_total_median']:.3f} cov full {s['cov_full_total_realised_val']:.2f} "
                      f"cond {s['cov_cond_total_realised_val']:.2f}" + (f" | stab {s['stability']}" if a.stability else ""), flush=True)
    os.makedirs("out_reviewer", exist_ok=True)
    json.dump(out, open(os.path.join("out_reviewer", name), "w"), indent=1)


if __name__ == "__main__":
    main()
