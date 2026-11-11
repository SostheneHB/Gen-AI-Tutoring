"""
Next-item prediction, fourth version: a separate split for model selection, and the clipped and unclipped output
correction side by side (review of 31 October 2026, points 3 and 4).

Same data, five splits, training and test units, DKT grid, removals and three-stage audit as next_item_v3.py.
Changes:
  * a fourth split. The validation schools (ASSISTments) or classes (Eedi) of v3 are divided in two halves, by
    students. The first half (selection) selects the DKT configuration, stops its training, stops the refit of the
    LEACE readouts and sets the lambda of mean-parity fine-tuning. The second half (announcement) is used only to
    announce the cost and to run the diagnostic D1-D3. The announcement is then independent of the fit;
  * the output correction with and without clipping to [0, 1], on announcement and test units: extra Brier score,
    remaining share tau of the gap, share of values outside [0, 1], with cluster-bootstrap intervals (correction
    fixed, 200 draws).

    DATA_ROOT=/path/to/data B_V3=400 python next_item_v4.py --dataset assistments|eedi [--split-ids 0] [--suffix _x]
        -> out_rank_matched/next_item_v4_<dataset><suffix>.json  (aggregates only)
    B_V3=5 python next_item_v4.py --dataset assistments --split-ids 0 --smoke   (one configuration, one epoch)
"""
import argparse
import copy
import hashlib
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import log_loss

import assistments_pipeline as ap
import next_item as ni
import next_item_tuned as nt
import next_item_v3 as v3
import rank_matched_controls as rm
from dynamics_only import ROOT

SEED = nt.SEED
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

# nt.fit_q (LBFGS, multithreaded float32) is not bit-reproducible: two fits on the same rows can differ by a few
# hundredths in the coefficients. The unweighted fits are cached by the content of their inputs, so that
# clip_side_by_side uses exactly the group score of v3.analyse. Bootstrap refits (weighted) are not cached.
_fit_q, _cache = nt.fit_q, {}


def _fit_q_cached(H, k, g, nv, w=None, init=None, iters=100):
    if w is not None or init is not None:
        return _fit_q(H, k, g, nv, w=w, init=init, iters=iters)
    key = tuple(hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest() for x in (H.numpy(), k.numpy(), g)) + (nv, iters)
    if key not in _cache:
        _cache[key] = _fit_q(H, k, g, nv, iters=iters)
    return _cache[key]


nt.fit_q = _fit_q_cached


def prepare4(df_all, seed, strata):
    """Training and test units of v3; its validation units split in two halves (selection, announcement)."""
    rng = np.random.default_rng(seed)
    tr, va, te = ap.split_schools(df_all, rng)
    school = df_all.drop_duplicates("user_id").set_index("user_id")["school_id"].loc[va]
    counts = school.value_counts()
    order = np.random.default_rng(seed + 7).permutation(counts.index.values)
    cum = np.cumsum(counts.loc[order].values) / counts.sum()
    first = set(order[:max(1, min(int((cum <= 0.5).sum()), len(order) - 1))])
    in_first = school.isin(first).values
    sel, ann = va[in_first], va[~in_first]
    df = df_all
    if strata == "train":
        df, _ = ap.rebuild_school_strata(df, tr, True)
    df = ap.rebuild_dt(df, tr)
    parts = []
    for ids in (tr, sel, ann, te):
        users, Y, K, g, s = nt.sequences(df, ids, ni.T)
        parts.append(dict(users=users, Y=Y, K=K, G=g, S=s))
    return parts


def refit_head4(model, Es, parts, sl, lr=3e-3):
    """v3.refit_head_vt with early stopping on the selection units; predictions on announcement and test units."""
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
    return pred(2), pred(3)


def clip_side_by_side(model, P3, sl, nv, seed):
    """The output correction of v3.analyse (same group score: fit_q is deterministic), with and without clipping."""
    T_ = sl.stop - sl.start
    fH = [nt.predict(model, p, sl, states=True) for p in P3]
    fs, Hs = [f.ravel() for f, _ in fH], [H for _, H in fH]
    ks = [p["k"][:, 1:][:, sl] for p in P3]
    gs = [np.repeat(p["G"], T_).astype(float) for p in P3]
    ys = [nt.targets(p, sl).ravel() for p in P3]
    theta = nt.fit_q(Hs[0], ks[0], gs[0], nv, iters=300)
    s = [None] + [nt.q_of(theta, Hs[i], ks[i]) for i in (1, 2)]
    A = v3.announce(fs[1], s[1], gs[1], ys[1])
    rng = np.random.default_rng(seed + 99)
    out = {}
    for name, i in (("announce", 1), ("test", 2)):
        f, g, y = fs[i], gs[i], ys[i]
        d = -A["c"] * (s[i] - A["center"])

        def stats(it):
            p = f[it] + d[it]
            pc = np.clip(p, 0, 1)
            base = ((f[it] - y[it]) ** 2).mean()
            g0 = v3.gap(f[it], g[it])
            return dict(cost=((p - y[it]) ** 2).mean() - base, cost_clip=((pc - y[it]) ** 2).mean() - base,
                        tau=v3.gap(p, g[it]) / g0, tau_clip=v3.gap(pc, g[it]) / g0, share_out=((p < 0) | (p > 1)).mean())
        est = stats(np.arange(len(f)))
        _, cl, rows = nt.cluster_rows(P3[i]["S"], T_)
        draws = [stats(np.concatenate([rows[c] for c in rng.choice(cl, len(cl))])) for _ in range(200)]
        out[name] = {k: dict(est=float(v), ci=ci([x[k] for x in draws])) for k, v in est.items()}
    return out


def one_split(df, j, strata, dataset):
    t0 = time.time()
    P = prepare4(df, SEED + 200 + j, strata)
    nv = nt.encode(P)
    sl = slice(0, ni.T)
    T_ = ni.T
    model, cfg, table = nt.select([P[0], P[1]], nt.GRID, SEED + j, nv, sl)
    print(f"[{dataset} v4 split {j}] selected {cfg} on the selection units in {time.time() - t0:.0f}s", flush=True)
    Hs = [nt.predict(model, p, sl, states=True)[1] for p in P]
    ks = [p["k"][:, 1:][:, sl] for p in P]
    gs = [np.repeat(p["G"], T_).reshape(-1, T_) for p in P]
    extra = {}
    for name, by in (("leace_hidden", False), ("leace_by_skill", True)):
        Es, _ = nt.erase(Hs, ks, gs, by)
        extra[name] = refit_head4(model, Es, P, sl)
    f_sel = nt.predict(model, P[1], sl)
    for lam in nt.LAMBDAS:
        m2, _, _ = nt.train([P[0], P[1]], cfg, SEED + j, nv, sl, lam=lam, init=copy.deepcopy(model.state_dict()))
        tau_sel = nt.gap(nt.predict(m2, P[1], sl).mean(1), P[1]["G"]) / nt.gap(f_sel.mean(1), P[1]["G"])
        if abs(tau_sel) <= 0.1:
            break
    extra["parity_training"] = (nt.predict(m2, P[2], sl).ravel(), nt.predict(m2, P[3], sl).ravel())
    P3 = [P[0], P[2], P[3]]                      # training, announcement, test: the three stages of the audit
    res, fte, yte, gte = v3.analyse(model, P3, sl, nv, SEED + j, extra)
    po = res.pop("pred_output")
    preds = {"dkt": fte, "output_v4": po}
    for name, (_, pt) in extra.items():
        preds[name] = pt
    res["paired_vs_dkt"] = nt.paired(preds, yte, gte, P[3]["S"], T_, "dkt", SEED + j)
    res["scores"] = {a: ni.scores(np.clip(p, 1e-6, 1 - 1e-6), yte, gte) for a, p in preds.items()}
    res["clip"] = clip_side_by_side(model, P3, sl, nv, SEED + j)
    tc = res["clip"]["test"]
    res["clip"]["test_clip_cost_in_announced_ci"] = bool(res["skill"]["total_ci"][0] <= tc["cost_clip"]["est"] <= res["skill"]["total_ci"][1])
    res["selected"], res["grid"], res["lam"], res["split"] = cfg, table, lam, j
    res["n_units"] = dict(train=len(P[0]["G"]), selection=len(P[1]["G"]), announcement=len(P[2]["G"]), test=len(P[3]["G"]))
    res["n_clusters"] = {k: int(len(np.unique(p["S"]))) for k, p in zip(("train", "selection", "announcement", "test"), P)}
    s = res["skill"]
    print(f"[{dataset} v4 split {j}] units {res['n_units']} | total {s['announced']['total']:.4f} {[round(x, 4) for x in s['total_ci']]} "
          f"measured {s['measured']['total']:.4f} in-ci {s['total_in_ci']} acc {s['accepted']} conf {s['confirmed']} | clipped test cost "
          f"{tc['cost_clip']['est']:.4f} (unclipped {tc['cost']['est']:.4f}), tau {tc['tau_clip']['est']:+.2f} ({tc['tau']['est']:+.2f}), "
          f"outside [0,1] {tc['share_out']['est']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return res


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dataset", choices=["assistments", "eedi"], required=True)
    a_.add_argument("--split-ids", default="0,1,2,3,4")
    a_.add_argument("--suffix", default="")
    a_.add_argument("--smoke", action="store_true", help="one configuration, one epoch, one lambda: checks the code only")
    a = a_.parse_args()
    if a.smoke:
        nt.GRID, nt.MAX_EPOCHS, nt.LAMBDAS, a.suffix = nt.GRID[:1], 1, (10.0,), a.suffix + "_smoke"
    if a.dataset == "assistments":
        df = ap.keep_full_trajectories(ap.load_features("assistments_features_time.csv")); strata = "train"
    else:
        df = ap.keep_full_trajectories(ap.load_features(f"{ROOT}/Datasets/Eedi_NeurIPS2020/eedi_features_time.csv")); strata = "given"
    name = f"next_item_v4_{a.dataset}{a.suffix}.json"
    runs = []
    for j in (int(x) for x in a.split_ids.split(",")):
        runs.append(one_split(df, j, strata, a.dataset))
        json.dump({"splits": runs, "b": v3.B}, open(os.path.join(rm.OUT, name), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
