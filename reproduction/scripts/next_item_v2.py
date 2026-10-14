"""
Next-item prediction, second revision of the estimation procedure (reviews of 14 October 2026).

For each split, the DKT configuration selected on validation in next_item_tuned.py is retrained with the same seed
(no new grid search). Then, on the same training / validation / test units:

  A. Published procedure ("v1"): group model q fitted on training units, price p0 p1 Delta^2 / R^2 with
     R^2 = Var(q)/(p0 p1) and coefficient c = Cov(f, q)/Var(q), both on training units; rule R1-R4 on test units.
  B. Reviewer 1: the same, after Platt recalibration of q on the validation units.
  C. Three-stage procedure ("v2", Proposition 3): q fitted on training units; on the validation units, the
     coefficient c = Cov(f, G)/Cov(q, G), which removes the validation gap exactly whether or not q is calibrated,
     and the announced price p0 p1 Delta^2 / corr(q, G)^2, the cost of the correction built from q (an upper
     bound of the minimum when f is the best predictor); diagnostics on the validation units (D1 a gap is
     detectable, D2 q carries group information, D3 the price is stable under refitting); final evaluation on
     the test units: achieved tau, measured cost, its two terms E[d^2] and 2E[(f - y) d] (Eq. cross-term).
     Intervals: joint bootstrap over training schools/classes (q refitted, f fixed: conditional on the fitted
     network), validation and test schools/classes.
  D. The four removals of next_item_tuned.py (two on the frozen network: LEACE on the hidden state and within
     each next skill, and the output correction; one that retrains the network: mean-parity fine-tuning), each
     with its achieved tau, its cost and the two terms of its cost; and, for each, the cost of the output
     correction scaled to the same achieved tau (comparison at equal remaining gap).

    DATA_ROOT=/path/to/data python next_item_v2.py --dataset assistments|eedi [--split-ids 0,1] [--suffix _a]
    DATA_ROOT=/path/to/data python next_item_v2.py --dataset eedi --long        (longer histories, splits 0 and 1)
        -> out_rank_matched/next_item_v2_<dataset>[_long]<suffix>.json  (aggregates only)
"""
import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

import assistments_pipeline as ap
import next_item as ni
import next_item_tuned as nt
import rank_matched_controls as rm
from dynamics_only import ROOT

SEED = nt.SEED
B = int(os.environ.get("B_V2", 40))
DIAG = dict(cv=0.5)


def selected(dataset, long=False):
    out = {}
    for f in sorted(glob.glob(f"out_rank_matched/next_item_tuned_{dataset}{'_long' if long else ''}*.json")):
        if not long and "_long" in f:
            continue
        for k, r in enumerate(json.load(open(f))["splits"]):
            j = r.get("split", k)
            out[j] = ({L: r["results"][L]["selected"] for L in r["results"]} if long else
                      dict(cfg=r["selected"], lam=r["parity"]["lam"]))
    return out


def wmean(v, w):
    return float((v * w).sum() / w.sum())


def wcov(a, b, w):
    return wmean((a - wmean(a, w)) * (b - wmean(b, w)), w)


def v2_quantities(f, s, g, w):
    """On (weighted) validation rows: coefficient that removes the gap, price of the correction built from s."""
    p1 = wmean(g, w)
    cfg_ = wcov(f, g, w)
    csg = wcov(s, g, w)
    vs = wcov(s, s, w)
    corr2 = csg ** 2 / (vs * p1 * (1 - p1))
    delta = wmean(f, w * g) - wmean(f, w * (1 - g))
    return dict(c=cfg_ / csg, center=wmean(s, w), price=p1 * (1 - p1) * delta ** 2 / corr2, corr2=corr2, delta=delta, p1=p1,
                cov_sg=csg)


def cost_terms(f, d, y):
    """MSE(f + d) - MSE(f) = E[d^2] + 2 E[(f - y) d]."""
    return dict(total=float(((f + d - y) ** 2).mean() - ((f - y) ** 2).mean()), d2=float((d ** 2).mean()),
                cross=float(2 * ((f - y) * d).mean()))


def platt(s_val, g_val):
    ls = lambda s: np.log(np.clip(s, 1e-6, 1 - 1e-6) / np.clip(1 - s, 1e-6, 1))
    m = LogisticRegression(C=1e6).fit(ls(s_val)[:, None], g_val)
    return lambda s: m.predict_proba(ls(s)[:, None])[:, 1]


def analyse(model, parts, sl, nv, seed, extra_preds=None):
    T_ = sl.stop - sl.start
    rng = np.random.default_rng(seed)
    Hs, ks, gs, fs, ys = [], [], [], [], []
    for p in parts:
        f, H = nt.predict(model, p, sl, states=True)
        Hs.append(H); ks.append(p["k"][:, 1:][:, sl]); gs.append(np.repeat(p["G"], T_).astype(float))
        fs.append(f.ravel()); ys.append(nt.targets(p, sl).ravel())
    (Htr, Hva, Hte), (ktr, kva, kte), (gtr, gva, gte), (ftr, fva, fte), (ytr, yva, yte) = Hs, ks, gs, fs, ys
    theta = nt.fit_q(Htr, ktr, gtr, nv, iters=300)
    qtr, qva, qte = (nt.q_of(theta, H, k) for H, k in ((Htr, ktr), (Hva, kva), (Hte, kte)))
    _, cl_tr, rows_tr = nt.cluster_rows(parts[0]["S"], T_)
    _, cl_va, rows_va = nt.cluster_rows(parts[1]["S"], T_)
    _, cl_te, rows_te = nt.cluster_rows(parts[2]["S"], T_)
    gap = lambda v, g: float(v[g == 1].mean() - v[g == 0].mean())
    one = lambda n: np.ones(n)
    out = {}
    # ---- A and B: published procedure, without and with Platt recalibration on validation units
    for name, (a_tr, a_te) in (("v1", (qtr, qte)), ("v1_platt", tuple(platt(qva, gva)(q) for q in (qtr, qte)))):
        A = nt.announce(ftr, a_tr, gtr, one(len(gtr)))
        d = -A["c"] * (a_te - A["p1"])
        tau = gap(fte + d, gte) / gap(fte, gte)
        slope, ece = nt.calibration(a_te, gte)
        out[name] = dict(price=float(A["price"]), tau=float(tau), measured=cost_terms(fte, d, yte), q_slope=slope, q_ece=ece,
                         frontier=float((1 - tau) ** 2 * A["price"]))
    # ---- C: three-stage procedure
    V = v2_quantities(fva, qva, gva, one(len(gva)))
    d_te = -V["c"] * (qte - V["center"])
    tau = gap(fte + d_te, gte) / gap(fte, gte)
    draws = []
    for _ in range(B):
        cnt = rng.multinomial(len(cl_tr), np.full(len(cl_tr), 1 / len(cl_tr)))
        w = np.zeros(len(gtr))
        for c, m in zip(cl_tr, cnt):
            if m:
                w[rows_tr[c]] = m
        th = nt.fit_q(Htr, ktr, gtr, nv, w=w, init=theta, iters=60)
        sva, ste = nt.q_of(th, Hva, kva), nt.q_of(th, Hte, kte)
        iv = np.concatenate([rows_va[c] for c in rng.choice(cl_va, len(cl_va))])
        Vb = v2_quantities(fva[iv], sva[iv], gva[iv], one(len(iv)))
        it = np.concatenate([rows_te[c] for c in rng.choice(cl_te, len(cl_te))])
        db = -Vb["c"] * (ste[it] - Vb["center"])
        taub = gap(fte[it] + db, gte[it]) / gap(fte[it], gte[it])
        mb = cost_terms(fte[it], db, yte[it])["total"]
        draws.append(dict(price=Vb["price"], delta=Vb["delta"], cov_sg=Vb["cov_sg"], tau=taub, measured=mb,
                          frontier=(1 - taub) ** 2 * Vb["price"], diff=mb - (1 - taub) ** 2 * Vb["price"]))
    ci = lambda k: [float(np.percentile([x[k] for x in draws], 2.5)), float(np.percentile([x[k] for x in draws], 97.5))]
    pr = np.array([x["price"] for x in draws])
    checks = dict(D1_gap_detectable=bool(ci("delta")[0] > 0 or ci("delta")[1] < 0),
                  D2_score_informative=bool(ci("cov_sg")[0] > 0 or ci("cov_sg")[1] < 0),
                  D3_price_stable=bool(pr.std() / pr.mean() <= DIAG["cv"]))
    meas = cost_terms(fte, d_te, yte)
    pci = ci("price")
    out["v2"] = dict(price=float(V["price"]), price_ci=pci, corr2=float(V["corr2"]), r2_v1=float(np.var(qtr) / (gtr.mean() * (1 - gtr.mean()))),
                     tau=float(tau), tau_ci=ci("tau"), measured=meas, measured_ci=ci("measured"),
                     frontier=float((1 - tau) ** 2 * V["price"]), diff_ci=ci("diff"), price_cv=float(pr.std() / pr.mean()),
                     checks=checks, accepted=bool(all(checks.values())),
                     measured_in_price_ci=bool(pci[0] <= meas["total"] <= pci[1]))
    # ---- D: the other removals, their two cost terms, and the output correction at the same achieved tau
    if extra_preds:
        comp = {}
        for name, p in extra_preds.items():
            dm = p - fte
            tm = gap(p, gte) / gap(fte, gte)
            dq = -(1 - tm) * V["c"] * (qte - V["center"])
            comp[name] = dict(tau=float(tm), cost=cost_terms(fte, dm, yte), output_same_tau=cost_terms(fte, dq, yte),
                              frontier_same_tau=float((1 - tm) ** 2 * V["price"]))
        out["comparison"] = comp
    return out


def one_split(df, j, strata, sel, dataset):
    t0 = time.time()
    seed = SEED + 200 + j
    parts = nt.prepare(df, seed, strata)
    nv = nt.encode(parts)
    sl = slice(0, ni.T)
    cfg, lam = sel["cfg"], sel["lam"]
    model, _, ep = nt.train(parts, cfg, SEED + j, nv, sl)
    T_ = ni.T
    f_va = nt.predict(model, parts[1], sl)
    _, Htr = nt.predict(model, parts[0], sl, states=True)
    _, Hva = nt.predict(model, parts[1], sl, states=True)
    _, Hte = nt.predict(model, parts[2], sl, states=True)
    ks = [p["k"][:, 1:][:, sl] for p in parts]
    gs = [np.repeat(p["G"], T_).reshape(-1, T_) for p in parts]
    extra = {}
    for name, by in (("leace_hidden", False), ("leace_by_skill", True)):
        Es, _ = nt.erase([Htr, Hva, Hte], ks, gs, by)
        extra[name] = nt.refit_head(model, Es, parts, sl).ravel()
    m2, _, _ = nt.train(parts, cfg, SEED + j, nv, sl, lam=lam, init=copy.deepcopy(model.state_dict()))
    extra["parity_training"] = nt.predict(m2, parts[2], sl).ravel()
    res = analyse(model, parts, sl, nv, SEED + j, extra)
    yte, gte = nt.targets(parts[2], sl).ravel(), np.repeat(parts[2]["G"], T_)
    fte = nt.predict(model, parts[2], sl).ravel()
    res["dkt_auc"] = float(ni.scores(fte, yte, gte)["auc"])
    res["selected"], res["epochs"], res["split"] = cfg, ep, j
    v = res["v2"]
    print(f"[{dataset} v2 split {j}] auc {res['dkt_auc']:.3f} | v1 price {res['v1']['price']:.4f} tau {res['v1']['tau']:+.2f} slope {res['v1']['q_slope']:.2f} | "
          f"platt tau {res['v1_platt']['tau']:+.2f} slope {res['v1_platt']['q_slope']:.2f} | v2 price {v['price']:.4f} {[round(x, 4) for x in v['price_ci']]} "
          f"tau {v['tau']:+.2f} measured {v['measured']['total']:.4f} (d2 {v['measured']['d2']:.4f} cross {v['measured']['cross']:+.4f}) "
          f"accepted {v['accepted']} {v['checks']} ({time.time() - t0:.0f}s)", flush=True)
    for name, c in res.get("comparison", {}).items():
        print(f"     {name:16s} tau {c['tau']:+.2f} cost {c['cost']['total']:.4f} | output correction at same tau {c['output_same_tau']['total']:.4f}", flush=True)
    return res


def one_split_long(df, j, sel):
    seed = SEED + 200 + j
    parts = nt.prepare(df, seed, "given", last=100, min_len=101)
    nv = nt.encode(parts)
    res = {}
    for L in ("25", "50", "100"):
        Li = int(L)
        sub = [dict(p, k=p["k"][:, 100 - Li:], y=p["y"][:, 100 - Li:], Y=p["Y"][:, 100 - Li:]) for p in parts]
        sl = slice(Li - 25, Li)
        model, _, _ = nt.train(sub, sel[L], SEED + j, nv, sl)
        r = analyse(model, sub, sl, nv, SEED + j)
        res[L] = r
        v = r["v2"]
        print(f"[eedi long v2 split {j} L={L}] v2 price {v['price']:.4f} {[round(x, 4) for x in v['price_ci']]} tau {v['tau']:+.2f} "
              f"measured {v['measured']['total']:.4f} accepted {v['accepted']} | v1 tau {r['v1']['tau']:+.2f}", flush=True)
    return dict(split=j, results=res)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dataset", choices=["assistments", "eedi"], required=True)
    a_.add_argument("--long", action="store_true")
    a_.add_argument("--split-ids", default=None)
    a_.add_argument("--suffix", default="")
    a = a_.parse_args()
    if a.dataset == "assistments":
        df = ap.keep_full_trajectories(ap.load_features("assistments_features_time.csv")); strata = "train"
    else:
        df = ap.keep_full_trajectories(ap.load_features(f"{ROOT}/Datasets/Eedi_NeurIPS2020/eedi_features_time.csv")); strata = "given"
    sel = selected(a.dataset, a.long)
    ids = [int(x) for x in a.split_ids.split(",")] if a.split_ids else sorted(sel)
    name = f"next_item_v2_{a.dataset}{'_long' if a.long else ''}{a.suffix}.json"
    runs = []
    for j in ids:
        runs.append(one_split_long(df, j, sel[j]) if a.long else one_split(df, j, strata, sel[j], a.dataset))
        json.dump({"splits": runs, "b": B, "diag": DIAG}, open(os.path.join(rm.OUT, name), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
