"""
Next-item prediction, three-stage audit with explicit targets.

Same splits, networks (selected configuration and seed retrained) and removals as next_item_v2.py. Changes:
  * two announced quantities, both computed on validation units only: the quadratic part
    pi_s = c_s^2 Var(s) = E[d^2], and the total change of the Brier score pi_s + 2 E_val[(f - y) d];
  * intervals: the announced interval resamples training units (group model refitted) and validation units
    only; the measured cost on test units has its own interval (test units resampled, correction fixed); the
    difference "measured minus announced" resamples all three;
  * a second, simpler group score (hidden state only, no skill term), to show that a high built cost can come
    from the score rather than from the constraint;
  * the output correction of Proposition 3 is evaluated like the other arms on test units (AUC with paired
    school/class-cluster intervals; predictions clipped to [0, 1] only for AUC and log-loss, costs unclipped);
  * comparison with the other removals at the same remaining gap, with paired cluster-bootstrap intervals of the
    cost difference (descriptive: the scaling uses the gap measured on test units), and an operational version in
    which the intensity of the output correction is chosen on validation units to match each method's validation
    gap, then both are evaluated on test units.

    DATA_ROOT=/path/to/data python next_item_v3.py --dataset assistments|eedi [--long] [--split-ids 0] [--suffix _x]
    B_V3=200 ... (bootstrap draws; default 40)
        -> out_rank_matched/next_item_v3_<dataset>[_long]<suffix>.json  (aggregates only)
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score

import assistments_pipeline as ap
import next_item as ni
import next_item_tuned as nt
import next_item_v2 as v2
import rank_matched_controls as rm
from dynamics_only import ROOT

SEED = nt.SEED
B = int(os.environ.get("B_V3", 40))
gap = lambda v, g: float(v[g == 1].mean() - v[g == 0].mean())


def refit_head_vt(model, Es, parts, sl, lr=3e-3):
    """nt.refit_head, returning validation and test predictions."""
    head = torch.nn.Embedding(*model.out.weight.shape)
    head.weight.data.copy_(model.out.weight.data)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    kn = [p["k"][:, 1:][:, sl] for p in parts]
    ys = [torch.tensor(nt.targets(p, sl)) for p in parts]

    def pred(j):
        with torch.no_grad():
            w = head(kn[j])
            return torch.sigmoid((Es[j] * w[..., :-1]).sum(-1) + w[..., -1]).numpy().astype(np.float64).ravel()
    best, state, bad = np.inf, None, 0
    n = len(Es[0])
    for _ in range(nt.MAX_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, nt.BATCH):
            b = perm[i:i + nt.BATCH]
            w = head(kn[0][b])
            loss = lossf((Es[0][b] * w[..., :-1]).sum(-1) + w[..., -1], ys[0][b].float())
            opt.zero_grad(); loss.backward(); opt.step()
        v = log_loss(ys[1].numpy().ravel(), np.clip(pred(1), 1e-6, 1 - 1e-6))
        if v < best - 1e-5:
            best, state, bad = v, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= nt.PATIENCE:
                break
    head.load_state_dict(state)
    return pred(1), pred(2)


def announce(f, s, g, y):
    V = v2.v2_quantities(f, s, g, np.ones(len(g)))
    d = -V["c"] * (s - V["center"])
    cross = float(2 * ((f - y) * d).mean())
    return dict(V, quad=float(V["price"]), cross=cross, total=float(V["price"] + cross))


def analyse(model, parts, sl, nv, seed, extra=None):
    T_ = sl.stop - sl.start
    rng = np.random.default_rng(seed)
    Hs, ks, gs, fs, ys = [], [], [], [], []
    for p in parts:
        f, H = nt.predict(model, p, sl, states=True)
        Hs.append(H); ks.append(p["k"][:, 1:][:, sl]); gs.append(np.repeat(p["G"], T_).astype(float))
        fs.append(f.ravel()); ys.append(nt.targets(p, sl).ravel())
    (Htr, Hva, Hte), (ktr, kva, kte), (gtr, gva, gte), (ftr, fva, fte), (ytr, yva, yte) = Hs, ks, gs, fs, ys
    one = lambda n: np.ones(n)
    _, cl_tr, rows_tr = nt.cluster_rows(parts[0]["S"], T_)
    _, cl_va, rows_va = nt.cluster_rows(parts[1]["S"], T_)
    _, cl_te, rows_te = nt.cluster_rows(parts[2]["S"], T_)
    out = {}
    for score, kz in (("skill", False), ("hidden_only", True)):
        K = [torch.zeros_like(k) if kz else k for k in (ktr, kva, kte)]
        theta = nt.fit_q(Htr, K[0], gtr, nv, iters=300)
        sva, ste = nt.q_of(theta, Hva, K[1]), nt.q_of(theta, Hte, K[2])
        A = announce(fva, sva, gva, yva)
        d = -A["c"] * (ste - A["center"])
        meas = v2.cost_terms(fte, d, yte)
        tau = gap(fte + d, gte) / gap(fte, gte)
        res = dict(announced=dict(quad=A["quad"], cross=A["cross"], total=A["total"], corr2=float(A["corr2"])),
                   tau=float(tau), measured=meas)
        if score == "skill":
            draws = []
            for _ in range(B):
                cnt = rng.multinomial(len(cl_tr), np.full(len(cl_tr), 1 / len(cl_tr)))
                w = np.zeros(len(gtr))
                for c, m in zip(cl_tr, cnt):
                    if m:
                        w[rows_tr[c]] = m
                th = nt.fit_q(Htr, ktr, gtr, nv, w=w, init=theta, iters=60)
                sv, st = nt.q_of(th, Hva, kva), nt.q_of(th, Hte, kte)
                iv = np.concatenate([rows_va[c] for c in rng.choice(cl_va, len(cl_va))])
                Ab = announce(fva[iv], sv[iv], gva[iv], yva[iv])
                it = np.concatenate([rows_te[c] for c in rng.choice(cl_te, len(cl_te))])
                db = -Ab["c"] * (st[it] - Ab["center"])
                mb = v2.cost_terms(fte[it], db, yte[it])
                draws.append(dict(quad=Ab["quad"], total=Ab["total"], delta=Ab["delta"], cov_sg=Ab["cov_sg"],
                                  m_total=mb["total"], m_quad=mb["d2"], diff_total=mb["total"] - Ab["total"],
                                  diff_quad=mb["d2"] - Ab["quad"]))
            # measured cost: test units only, correction fixed
            mdraw, tgap = [], []
            for _ in range(max(B, 200)):
                it = np.concatenate([rows_te[c] for c in rng.choice(cl_te, len(cl_te))])
                mdraw.append(v2.cost_terms(fte[it], d[it], yte[it]))
                tgap.append(gap(fte[it], gte[it]))
            ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
            col = lambda k: np.array([x[k] for x in draws])
            pr = col("quad")
            checks = dict(D1=bool(ci(col("delta"))[0] > 0 or ci(col("delta"))[1] < 0),
                          D2=bool(ci(col("cov_sg"))[0] > 0 or ci(col("cov_sg"))[1] < 0),
                          D3=bool(pr.std() / pr.mean() <= 0.5))
            res.update(quad_ci=ci(pr), total_ci=ci(col("total")), measured_total_ci=ci([x["total"] for x in mdraw]),
                       measured_quad_ci=ci([x["d2"] for x in mdraw]), diff_total_ci=ci(col("diff_total")),
                       diff_quad_ci=ci(col("diff_quad")), checks=checks, accepted=bool(all(checks.values())),
                       test_checks=dict(T1=bool(ci(tgap)[0] > 0 or ci(tgap)[1] < 0), T2=bool(abs(tau) <= 0.2)),
                       confirmed=bool(all(checks.values()) and (ci(tgap)[0] > 0 or ci(tgap)[1] < 0) and abs(tau) <= 0.2),
                       total_in_ci=bool(ci(col("total"))[0] <= meas["total"] <= ci(col("total"))[1]),
                       quad_in_ci=bool(ci(pr)[0] <= meas["d2"] <= ci(pr)[1]),
                       quantiles={q: dict(quad=float(np.percentile(pr, q)), total=float(np.percentile(col("total"), q))) for q in (2.5, 97.5)})
            # the corrected predictions as an arm (AUC on clipped predictions; costs unclipped)
            p0 = fte + d
            out["clip_share"] = float(((p0 < 0) | (p0 > 1)).mean())
            out["pred_output"] = np.clip(p0, 1e-6, 1 - 1e-6)
            out["d_output"], out["s_te"], out["s_va"], out["A"] = d, ste, sva, A
        out[score] = res
    # ---- comparison with the other removals
    if extra:
        A, ste, sva = out["A"], out["s_te"], out["s_va"]
        dq_unit_te = -A["c"] * (ste - A["center"])
        comp = {}
        for name, (pv, pt) in extra.items():
            tm_te = gap(pt, gte) / gap(fte, gte)
            tm_va = gap(pv, gva) / gap(fva, gva)
            same = (1 - tm_te) * dq_unit_te
            oper = (1 - tm_va) * dq_unit_te
            diffs = []
            for _ in range(200):
                it = np.concatenate([rows_te[c] for c in rng.choice(cl_te, len(cl_te))])
                tb = gap(pt[it], gte[it]) / gap(fte[it], gte[it])
                cm = v2.cost_terms(fte[it], pt[it] - fte[it], yte[it])["total"]
                co = v2.cost_terms(fte[it], (1 - tb) * dq_unit_te[it], yte[it])["total"]
                diffs.append(cm - co)
            comp[name] = dict(tau_test=float(tm_te), tau_val=float(tm_va), cost=v2.cost_terms(fte, pt - fte, yte),
                              output_same_tau=v2.cost_terms(fte, same, yte),
                              output_val_tau=dict(v2.cost_terms(fte, oper, yte), tau_test=float(gap(fte + oper, gte) / gap(fte, gte))),
                              diff_ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                              diff=float(v2.cost_terms(fte, pt - fte, yte)["total"] - v2.cost_terms(fte, same, yte)["total"]))
        out["comparison"] = comp
    for k in ("A", "s_te", "s_va", "d_output"):
        out.pop(k, None)
    return out, fte, yte, gte


def one_split(df, j, strata, sel, dataset):
    t0 = time.time()
    parts = nt.prepare(df, SEED + 200 + j, strata)
    nv = nt.encode(parts)
    sl = slice(0, ni.T)
    cfg, lam = sel["cfg"], sel["lam"]
    model, _, ep = nt.train(parts, cfg, SEED + j, nv, sl)
    T_ = ni.T
    _, Htr = nt.predict(model, parts[0], sl, states=True)
    _, Hva = nt.predict(model, parts[1], sl, states=True)
    _, Hte = nt.predict(model, parts[2], sl, states=True)
    ks = [p["k"][:, 1:][:, sl] for p in parts]
    gs = [np.repeat(p["G"], T_).reshape(-1, T_) for p in parts]
    extra = {}
    for name, by in (("leace_hidden", False), ("leace_by_skill", True)):
        Es, _ = nt.erase([Htr, Hva, Hte], ks, gs, by)
        extra[name] = refit_head_vt(model, Es, parts, sl)
    m2, _, _ = nt.train(parts, cfg, SEED + j, nv, sl, lam=lam, init=copy.deepcopy(model.state_dict()))
    extra["parity_training"] = (nt.predict(m2, parts[1], sl).ravel(), nt.predict(m2, parts[2], sl).ravel())
    res, fte, yte, gte = analyse(model, parts, sl, nv, SEED + j, extra)
    po = res.pop("pred_output")
    preds = {"dkt": fte, "output_v3": po}
    for name, (_, pt) in extra.items():
        preds[name] = pt
    res["paired_vs_dkt"] = nt.paired(preds, yte, gte, parts[2]["S"], T_, "dkt", SEED + j)
    res["scores"] = {a: ni.scores(np.clip(p, 1e-6, 1 - 1e-6), yte, gte) for a, p in preds.items()}
    res["selected"], res["epochs"], res["split"] = cfg, ep, j
    s = res["skill"]
    print(f"[{dataset} v3 split {j}] quad {s['announced']['quad']:.4f} {[round(x, 4) for x in s['quad_ci']]} total {s['announced']['total']:.4f} "
          f"{[round(x, 4) for x in s['total_ci']]} | measured {s['measured']['total']:.4f} (d2 {s['measured']['d2']:.4f}) tau {s['tau']:+.2f} "
          f"in-ci {s['total_in_ci']} acc {s['accepted']} conf {s['confirmed']} | hidden-only total {res['hidden_only']['announced']['total']:.4f} "
          f"measured {res['hidden_only']['measured']['total']:.4f} | output auc {res['scores']['output_v3']['auc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    for name, c in res["comparison"].items():
        print(f"     {name:16s} tau {c['tau_test']:+.2f} cost {c['cost']['total']:.4f} | same tau {c['output_same_tau']['total']:.4f} diff {c['diff']:+.4f} "
              f"{[round(x, 4) for x in c['diff_ci']]} | val-chosen: tau {c['output_val_tau']['tau_test']:+.2f} cost {c['output_val_tau']['total']:.4f}", flush=True)
    return res


def one_split_long(df, j, sel):
    parts = nt.prepare(df, SEED + 200 + j, "given", last=100, min_len=101)
    nv = nt.encode(parts)
    res = {}
    for L in ("25", "50", "100"):
        Li = int(L)
        sub = [dict(p, k=p["k"][:, 100 - Li:], y=p["y"][:, 100 - Li:], Y=p["Y"][:, 100 - Li:]) for p in parts]
        sl = slice(Li - 25, Li)
        model, _, _ = nt.train(sub, sel[L], SEED + j, nv, sl)
        r, _, _, _ = analyse(model, sub, sl, nv, SEED + j)
        r.pop("pred_output", None)
        res[L] = r
        s = r["skill"]
        print(f"[eedi long v3 split {j} L={L}] total {s['announced']['total']:.4f} {[round(x, 4) for x in s['total_ci']]} measured "
              f"{s['measured']['total']:.4f} tau {s['tau']:+.2f} in-ci {s['total_in_ci']} acc {s['accepted']} conf {s['confirmed']}", flush=True)
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
    sel = v2.selected(a.dataset, a.long)
    ids = [int(x) for x in a.split_ids.split(",")] if a.split_ids else sorted(sel)
    name = f"next_item_v3_{a.dataset}{'_long' if a.long else ''}{a.suffix}.json"
    runs = []
    for j in ids:
        runs.append(one_split_long(df, j, sel[j]) if a.long else one_split(df, j, strata, sel[j], a.dataset))
        json.dump({"splits": runs, "b": B}, open(os.path.join(rm.OUT, name), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
