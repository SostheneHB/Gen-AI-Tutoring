"""
Next-item prediction with a DKT selected on validation schools, a full comparison of ways to remove the mean
group contrast, and the reliability of the announced cost (review of 13 October 2026, items 2 and 4).

For each of the five school-disjoint splits of next_item.py:
  1. DKT selected on the validation schools: hidden size {32, 64, 128} x learning rate {1e-3, 3e-3} x dropout
     {0, 0.2} on the hidden state, early stopping on the validation log-loss (patience 3, at most 30 epochs).
  2. Ways to remove the mean group contrast, all built from the selected DKT:
       leace_hidden     LEACE on the hidden state; per-skill readout refitted (early stopping on validation);
       leace_by_skill   LEACE on the hidden state within each next-item skill, so that the whole readout input
                        (hidden state, next skill) carries no linear group signal; readout refitted;
       output_removal   the correction of Proposition 1 on the prediction, f - c (q - p1), with
                        q = P(g = 1 | hidden state, next skill): the same information as the prediction;
       parity_training  DKT fine-tuned with a penalty lambda * (mean prediction gap)^2 on every input;
                        lambda is the smallest of 10, 100, 1000, 10000 with |tau| <= 0.1 on the validation schools;
     and, for reference, the raw and projected EDMD readouts of next_item.py.
  3. For every arm on the held-out schools: AUC, Brier score, log-loss, the remaining fraction tau of the mean
     contrast, and school-cluster bootstrap intervals of the paired differences with the selected DKT.
  4. Reliability of the announced cost: the price p0 p1 Delta^2 / R^2 is announced on the training schools;
     q and the correction are refitted on B_REFIT bootstrap resamples of the training schools (the DKT is fixed
     within a split; the five splits retrain it), and each refit is applied to a bootstrap resample of the
     held-out schools.  The rule of Section 4 is evaluated on the held-out schools (contrast interval,
     calibration slope and ECE of q, spread of the announced price over refits).
  5. --long (Eedi only): longer histories.  Students with at least 101 items; the same 25 targets (items
     77..101) are predicted from windows of 25, 50 and 100 previous items.  On ASSISTments the group is defined
     from the items after the 26th, so a longer history would contain the items that define the group.

    python next_item_tuned.py --dataset assistments|eedi [--long]
        -> out_rank_matched/next_item_tuned_<dataset>[_long].json
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

import assistments_pipeline as ap
import next_item as ni
import rank_matched_controls as rm
from dynamics_only import ROOT

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 3)))
GRID = [dict(hidden=h, lr=lr, dropout=dr) for h in (32, 64, 128) for lr in (1e-3, 3e-3) for dr in (0.0, 0.2)]
GRID_LONG = [dict(hidden=h, lr=3e-3, dropout=0.0) for h in (64, 128)]
MAX_EPOCHS, PATIENCE, BATCH = 30, 3, 256
LAMBDAS = (10.0, 100.0, 1000.0, 10000.0)
B_REFIT = int(os.environ.get("B_REFIT", 40))
B_BOOT = 300
SEED = 20260914
# rule of Section 4: the estimated cost is declared not interpretable unless all four conditions hold
RULE = dict(slope=(0.8, 1.2), ece=0.05, price_cv=0.5)


class DKT2(torch.nn.Module):
    def __init__(self, n_skills, hidden=64, emb=32, dropout=0.0):
        super().__init__()
        self.inp = torch.nn.Embedding(2 * (n_skills + 1), emb)
        self.gru = torch.nn.GRU(emb, hidden, batch_first=True)
        self.drop = torch.nn.Dropout(dropout)
        self.out = torch.nn.Embedding(n_skills + 1, hidden + 1)

    def hidden(self, K, Y):
        return self.gru(self.inp(K * 2 + Y))[0]

    def logits(self, H, Knext):
        w = self.out(Knext)
        return (self.drop(H) * w[..., :-1]).sum(-1) + w[..., -1]


# --------------------------------------------------------------------------- data
def sequences(df, ids, last):
    sub = df[df["user_id"].isin(ids) & (df["step"] <= last)].sort_values(["user_id", "step"])
    users = sub["user_id"].drop_duplicates().values
    Y = (sub["correct"].values >= 1).reshape(-1, last + 1).astype(np.int64)
    K = sub["skill_id"].values.reshape(-1, last + 1)
    g = sub.groupby("user_id", sort=True)["G"].first().loc[users].values.astype(int)
    s = sub.groupby("user_id", sort=True)["school_id"].first().loc[users].values
    return users, Y, K, g, s


def prepare(df_all, seed, strata, last=ni.T, min_len=None):
    rng = np.random.default_rng(seed)
    tr, va, te = ap.split_schools(df_all, rng)            # same split as next_item.py for the same seed
    df = df_all
    if strata == "train":
        df, _ = ap.rebuild_school_strata(df, tr, True)
    if min_len is not None:
        n = df.groupby("user_id")["step"].transform("max") + 1
        df = df[n >= min_len]
    df = ap.rebuild_dt(df, tr)
    parts = []
    for ids in (tr, va, te):
        users, Y, K, g, s = sequences(df, ids, last)
        part = dict(users=users, Y=Y, K=K, G=g, S=s)
        if last == ni.T:
            _, X, G2, S2 = ap.trajectories(df, ids)
            assert (G2 == g).all()
            part["X"] = X
        parts.append(part)
    return parts


def encode(parts):
    vocab = {s: i + 1 for i, s in enumerate(np.unique(parts[0]["K"]))}
    for p in parts:
        p["k"] = torch.tensor(pd.Series(p["K"].ravel()).map(vocab).fillna(0).astype(np.int64).values.reshape(p["K"].shape))
        p["y"] = torch.tensor(p["Y"])
        p["g"] = torch.tensor(p["G"])
    return len(vocab)


# --------------------------------------------------------------------------- DKT
def predict(model, p, sl, bs=4096, states=False):
    model.eval()
    out, hs = [], []
    with torch.no_grad():
        for i in range(0, len(p["k"]), bs):
            k, y = p["k"][i:i + bs], p["y"][i:i + bs]
            H = model.hidden(k[:, :-1], y[:, :-1])
            out.append(torch.sigmoid(model.logits(H, k[:, 1:]))[:, sl])
            if states:
                hs.append(H[:, sl])
    P = torch.cat(out).numpy().astype(np.float64)
    return (P, torch.cat(hs)) if states else P


def targets(p, sl):
    return p["Y"][:, 1:][:, sl].astype(float)


def gap(v, g):
    return v[g == 1].mean() - v[g == 0].mean()


def train(parts, cfg, seed, nv, sl, lam=0.0, init=None):
    tr, va = parts[0], parts[1]
    torch.manual_seed(seed)
    model = DKT2(nv, cfg["hidden"], 32, cfg["dropout"])
    if init is not None:
        model.load_state_dict(init)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"] if init is None else cfg["lr"] / 3)
    lossf = torch.nn.BCEWithLogitsLoss()
    yva, gva = targets(va, sl), va["G"]

    def objective():
        P = predict(model, va, sl)
        v = log_loss(yva.ravel(), np.clip(P.ravel(), 1e-6, 1 - 1e-6))
        return v + lam * gap(P.mean(1), gva) ** 2
    best, state, best_ep, bad = (objective() if init is not None else np.inf), copy.deepcopy(model.state_dict()), 0, 0
    n = len(tr["k"])
    for ep in range(MAX_EPOCHS if init is None else 10):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            k, y, g = tr["k"][b], tr["y"][b], tr["g"][b]
            lg = model.logits(model.hidden(k[:, :-1], y[:, :-1]), k[:, 1:])
            loss = lossf(lg, y[:, 1:].float())
            if lam and 0 < int(g.sum()) < len(g):
                pm = torch.sigmoid(lg[:, sl]).mean(1)
                loss = loss + lam * (pm[g == 1].mean() - pm[g == 0].mean()) ** 2
            opt.zero_grad(); loss.backward(); opt.step()
        v = objective()
        if v < best - 1e-5:
            best, state, best_ep, bad = v, copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()
    return model, float(best), best_ep


def select(parts, grid, seed, nv, sl):
    table, best = [], None
    for cfg in grid:
        t0 = time.time()
        model, val, ep = train(parts, cfg, seed, nv, sl)
        table.append(dict(cfg, val_logloss=val, epochs=ep, seconds=round(time.time() - t0, 1)))
        if best is None or val < best[1]:
            best = (model, val, cfg)
    return best[0], best[2], table


# --------------------------------------------------------------------------- erasers
def leace_fit(H, g):
    """LEACE for a binary concept (closed form, whitened): x -> x - (x - mu) Pm^T."""
    n, d = H.shape
    mu = H.mean(0, dtype=np.float64)
    S, cov, gm = np.zeros((d, d)), np.zeros(d), g.mean()
    for i in range(0, n, 100000):
        Hc = H[i:i + 100000].astype(np.float64) - mu
        S += Hc.T @ Hc
        cov += Hc.T @ (g[i:i + 100000] - gm)
    S, cov = S / n + 1e-6 * np.eye(d), cov / n
    w, V = np.linalg.eigh(S)
    Wh, Wih = (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T
    v = Wih @ cov
    v /= np.linalg.norm(v)
    return mu, Wh @ np.outer(v, v) @ Wih


def leace_apply(H, mu, Pm):
    return (H - (H.astype(np.float64) - mu) @ Pm.T).astype(np.float32)


def erase(Hs, ks, gs, by_skill, min_rows=50):
    """Hs, ks, gs: lists (train, val, test) of states (n, T, d), next skills (n, T), groups per row (n, T).
    The eraser is fitted on the training rows and applied to all three."""
    flat = lambda H: H.reshape(-1, H.shape[-1]).numpy()
    Htr, ktr, gtr = flat(Hs[0]), ks[0].reshape(-1).numpy(), gs[0].reshape(-1).astype(float)
    glob = leace_fit(Htr, gtr)
    fits = {}
    if by_skill:
        for s in np.unique(ktr):
            m = ktr == s
            if min(int((gtr[m] == 1).sum()), int((gtr[m] == 0).sum())) >= min_rows:
                fits[s] = leace_fit(Htr[m], gtr[m])
    out = []
    for H, k in zip(Hs, ks):
        Hn, kn = flat(H), k.reshape(-1).numpy()
        E = leace_apply(Hn, *glob)
        for s, f in fits.items():
            m = kn == s
            if m.any():
                E[m] = leace_apply(Hn[m], *f)
        out.append(torch.tensor(E.reshape(H.shape)))
    return out, len(fits)


def refit_head(model, Es, parts, sl, lr=3e-3):
    """Per-skill readout refitted on erased states, early stopping on the validation schools."""
    head = torch.nn.Embedding(*model.out.weight.shape)
    head.weight.data.copy_(model.out.weight.data)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    kn = [p["k"][:, 1:][:, sl] for p in parts]
    ys = [torch.tensor(targets(p, sl)) for p in parts]

    def pred(j):
        with torch.no_grad():
            w = head(kn[j])
            return torch.sigmoid((Es[j] * w[..., :-1]).sum(-1) + w[..., -1]).numpy().astype(np.float64)
    best, state, bad = np.inf, None, 0
    n = len(Es[0])
    for ep in range(MAX_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            w = head(kn[0][b])
            loss = lossf((Es[0][b] * w[..., :-1]).sum(-1) + w[..., -1], ys[0][b].float())
            opt.zero_grad(); loss.backward(); opt.step()
        v = log_loss(ys[1].numpy().ravel(), np.clip(pred(1).ravel(), 1e-6, 1 - 1e-6))
        if v < best - 1e-5:
            best, state, bad = v, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    head.load_state_dict(state)
    return pred(2)


# --------------------------------------------------------------------------- group model q and the price
def fit_q(H, k, g, nv, w=None, init=None, iters=100):
    """Logistic model of the group on (hidden state, next skill): logit = H a + b[k] (the readout's information)."""
    Hf = H.reshape(-1, H.shape[-1]).float()
    kf = k.reshape(-1)
    gf = torch.tensor(g.reshape(-1), dtype=torch.float32)
    wf = torch.ones_like(gf) if w is None else torch.tensor(w, dtype=torch.float32)
    a = torch.zeros(Hf.shape[1], requires_grad=True)
    b = torch.zeros(nv + 1, requires_grad=True)
    c = torch.zeros(1, requires_grad=True)
    if init is not None:
        with torch.no_grad():
            a.copy_(init[0]); b.copy_(init[1]); c.copy_(init[2])
    opt = torch.optim.LBFGS([a, b, c], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    lossf = torch.nn.BCEWithLogitsLoss(weight=wf, reduction="sum")
    tot = float(wf.sum())

    def closure():
        opt.zero_grad()
        loss = lossf(Hf @ a + b[kf] + c, gf) / tot + 1e-6 * (a @ a + b @ b)
        loss.backward()
        return loss
    opt.step(closure)
    return (a.detach(), b.detach(), c.detach())


def q_of(theta, H, k):
    a, b, c = theta
    with torch.no_grad():
        return torch.sigmoid(H.reshape(-1, H.shape[-1]).float() @ a + b[k.reshape(-1)] + c).numpy().astype(np.float64)


def wmean(v, w):
    return (v * w).sum() / w.sum()


def announce(f, q, g, w):
    """Price p0 p1 Delta^2 / R^2, correction coefficient c and p1 on (weighted) training rows."""
    p1 = wmean(g, w)
    vq = wmean((q - wmean(q, w)) ** 2, w)
    c = wmean((f - wmean(f, w)) * (q - wmean(q, w)), w) / vq
    delta = wmean(f, w * g) - wmean(f, w * (1 - g))
    r2 = vq / (p1 * (1 - p1))
    return dict(price=p1 * (1 - p1) * delta ** 2 / r2, c=c, p1=p1, r2=r2, delta=delta)


def calibration(q, g):
    lq = np.log(np.clip(q, 1e-6, 1 - 1e-6) / np.clip(1 - q, 1e-6, 1))
    slope = float(LogisticRegression(C=1e6).fit(lq[:, None], g).coef_[0, 0])
    bins = np.minimum((q * 10).astype(int), 9)
    ece = float(sum(abs(g[bins == b].mean() - q[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()))
    return slope, ece


def cluster_rows(S, T):
    rows = np.arange(len(S) * T).reshape(len(S), T)
    cl = np.unique(S)
    return rows, cl, {c: rows[S == c].ravel() for c in cl}


def reliability(model, parts, sl, nv, f_tr, f_te, seed, ytr, yte):
    """Announced cost on training schools, refit bootstrap of q and c, held-out comparison, rule diagnostics."""
    T_ = f_te.shape[1]
    _, Htr = predict(model, parts[0], sl, states=True)
    _, Hte = predict(model, parts[2], sl, states=True)
    ktr, kte = parts[0]["k"][:, 1:][:, sl], parts[2]["k"][:, 1:][:, sl]
    gtr, gte = np.repeat(parts[0]["G"], T_).astype(float), np.repeat(parts[2]["G"], T_).astype(float)
    ftr, fte = f_tr.ravel(), f_te.ravel()
    theta = fit_q(Htr, ktr, gtr, nv, iters=300)
    qtr, qte = q_of(theta, Htr, ktr), q_of(theta, Hte, kte)
    A0 = announce(ftr, qtr, gtr, np.ones_like(gtr))

    def held_out(A, q, idx):
        f0 = fte[idx] - A["c"] * (q[idx] - A["p1"])
        tau = gap(f0, gte[idx]) / gap(fte[idx], gte[idx])
        meas = ((f0 - yte[idx]) ** 2).mean() - ((fte[idx] - yte[idx]) ** 2).mean()
        return tau, meas, (1 - tau) ** 2 * A["price"]
    all_te = np.arange(len(gte))
    tau0, meas0, front0 = held_out(A0, qte, all_te)
    f0_full = fte - A0["c"] * (qte - A0["p1"])
    slope, ece = calibration(qte, gte)
    # held-out contrast of the prediction, school-cluster bootstrap
    rng = np.random.default_rng(seed)
    _, cl_te, rows_te = cluster_rows(parts[2]["S"], T_)
    _, cl_tr, rows_tr = cluster_rows(parts[0]["S"], T_)
    dte = []
    for _ in range(B_BOOT):
        idx = np.concatenate([rows_te[c] for c in rng.choice(cl_te, len(cl_te))])
        dte.append(gap(fte[idx], gte[idx]))
    # refit bootstrap: training schools resampled (weights), q and c refitted; held-out schools resampled
    refits = []
    for r in range(B_REFIT):
        cnt = rng.multinomial(len(cl_tr), np.full(len(cl_tr), 1 / len(cl_tr)))
        w = np.zeros(len(gtr))
        for c, m in zip(cl_tr, cnt):
            if m:
                w[rows_tr[c]] = m
        th = fit_q(Htr, ktr, gtr, nv, w=w, init=theta, iters=60)
        qa, qb = q_of(th, Htr, ktr), q_of(th, Hte, kte)
        A = announce(ftr, qa, gtr, w)
        idx = np.concatenate([rows_te[c] for c in rng.choice(cl_te, len(cl_te))])
        tau, meas, front = held_out(A, qb, idx)
        refits.append(dict(price=A["price"], r2=A["r2"], tau=tau, measured=meas, frontier=front, diff=meas - front))
    pr = np.array([d["price"] for d in refits])
    ci = lambda k: [float(np.percentile([d[k] for d in refits], 2.5)), float(np.percentile([d[k] for d in refits], 97.5))]
    dci = [float(np.percentile(dte, 2.5)), float(np.percentile(dte, 97.5))]
    checks = dict(contrast_excludes_zero=bool(dci[0] > 0 or dci[1] < 0),
                  q_slope_ok=bool(RULE["slope"][0] <= slope <= RULE["slope"][1]),
                  q_ece_ok=bool(ece <= RULE["ece"]),
                  price_stable=bool(pr.std() / pr.mean() <= RULE["price_cv"]))
    return dict(announced=A0, tau=float(tau0), measured=float(meas0), frontier=float(front0),
                heldout_contrast=float(gap(fte, gte)), heldout_contrast_ci=dci, q_slope=slope, q_ece=ece,
                price_cv=float(pr.std() / pr.mean()), price_ci=ci("price"), r2_ci=ci("r2"), tau_ci=ci("tau"),
                measured_ci=ci("measured"), frontier_ci=ci("frontier"), measured_minus_frontier_ci=ci("diff"),
                checks=checks, interpretable=bool(all(checks.values())), n_refits=B_REFIT,
                clip_share=float(((f0_full < 0) | (f0_full > 1)).mean())), f0_full, (Htr, Hte, ktr, kte)


# --------------------------------------------------------------------------- comparison
def paired(preds, y, g, S, T_, ref, seed):
    rows, cl, by = cluster_rows(S, T_)
    rng = np.random.default_rng(seed)
    arms = [a for a in preds if a != ref]
    stat = lambda a, idx: dict(auc=roc_auc_score(y[idx], preds[a][idx]) - roc_auc_score(y[idx], preds[ref][idx]),
                               brier=((preds[a][idx] - y[idx]) ** 2).mean() - ((preds[ref][idx] - y[idx]) ** 2).mean(),
                               tau=gap(preds[a][idx], g[idx]) / gap(preds[ref][idx], g[idx]))
    allr = np.arange(len(y))
    point = {a: stat(a, allr) for a in arms}
    draws = {a: [] for a in arms}
    for _ in range(B_BOOT):
        idx = np.concatenate([by[c] for c in rng.choice(cl, len(cl))])
        for a in arms:
            draws[a].append(stat(a, idx))
    return {a: {k: dict(diff=float(point[a][k]), ci=[float(np.percentile([d[k] for d in draws[a]], 2.5)),
                                                    float(np.percentile([d[k] for d in draws[a]], 97.5))])
                for k in ("auc", "brier", "tau")} for a in arms}


def one_split(df, j, strata, dataset):
    seed = SEED + 200 + j
    parts = prepare(df, seed, strata)
    nv = encode(parts)
    sl = slice(0, ni.T)
    t0 = time.time()
    model, cfg, table = select(parts, GRID, SEED + j, nv, sl)
    print(f"[{dataset} split {j}] selected {cfg} in {time.time() - t0:.0f}s", flush=True)
    T_ = ni.T
    ytr, yte = targets(parts[0], sl).ravel(), targets(parts[2], sl).ravel()
    gte = np.repeat(parts[2]["G"], T_)
    f_tr, f_va, f_te = (predict(model, p, sl) for p in parts)
    preds = {"dkt": f_te.ravel()}
    rel, f0, (Htr, Hte, ktr, kte) = reliability(model, parts, sl, nv, f_tr, f_te, SEED + j, ytr, yte)
    preds["output_removal"] = np.clip(f0, 1e-6, 1 - 1e-6)
    _, Hva = predict(model, parts[1], sl, states=True)
    kva = parts[1]["k"][:, 1:][:, sl]
    gs = [np.repeat(p["G"], T_).reshape(-1, T_) for p in parts]
    leace_info = {}
    for name, by in (("leace_hidden", False), ("leace_by_skill", True)):
        Es, nfit = erase([Htr, Hva, Hte], [ktr, kva, kte], gs, by)
        preds[name] = refit_head(model, Es, parts, sl).ravel()
        leace_info[name] = dict(skill_specific_erasers=nfit)
    # mean-parity fine-tuning on every input
    par = None
    for lam in LAMBDAS:
        m2, _, ep = train(parts, cfg, SEED + j, nv, sl, lam=lam, init=copy.deepcopy(model.state_dict()))
        tau_va = gap(predict(m2, parts[1], sl).mean(1), parts[1]["G"]) / gap(f_va.mean(1), parts[1]["G"])
        par = (lam, m2, tau_va, ep)
        if abs(tau_va) <= 0.1:
            break
    preds["parity_training"] = predict(par[1], parts[2], sl).ravel()
    # EDMD readouts of next_item.py on the same students
    ni.PRED.clear()
    ni.edmd_arms(parts[0]["X"], parts[0]["G"], parts[0]["Y"].astype(float), parts[2]["X"], parts[2]["G"], parts[2]["Y"].astype(float))
    for k in ("edmd_raw", "edmd_projected"):
        preds[k] = ni.PRED[k]
    scores = {a: ni.scores(p, yte, gte) for a, p in preds.items()}
    pc = paired(preds, yte, gte, parts[2]["S"], T_, "dkt", SEED + j)
    for a in ("leace_hidden", "leace_by_skill", "parity_training", "output_removal"):
        tau = pc[a]["tau"]["diff"]
        pc[a]["frontier_at_tau"] = float((1 - tau) ** 2 * rel["announced"]["price"])
    out = dict(selected=cfg, grid=table, scores=scores, paired_vs_dkt=pc, reliability=rel, leace=leace_info,
               parity=dict(lam=par[0], val_tau=float(par[2]), epochs=par[3]), n_train=len(parts[0]["G"]),
               n_val=len(parts[1]["G"]), n_test=len(parts[2]["G"]), n_test_schools=int(len(np.unique(parts[2]["S"]))))
    s = scores
    print(f"[{dataset} split {j}] AUC dkt {s['dkt']['auc']:.3f} edmd {s['edmd_raw']['auc']:.3f} proj {s['edmd_projected']['auc']:.3f} | "
          + " | ".join(f"{a} dAUC {pc[a]['auc']['diff']:+.3f} tau {pc[a]['tau']['diff']:+.2f} dBrier {pc[a]['brier']['diff']:+.4f}"
                       for a in ("leace_hidden", "leace_by_skill", "parity_training", "output_removal"))
          + f" | price {rel['announced']['price']:.4f} {[round(x, 4) for x in rel['price_ci']]} measured {rel['measured']:.4f} "
          f"frontier {rel['frontier']:.4f} slope {rel['q_slope']:.2f} ece {rel['q_ece']:.3f} interpretable {rel['interpretable']} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out


def one_split_long(df, j):
    seed = SEED + 200 + j
    parts = prepare(df, seed, "given", last=100, min_len=101)
    nv = encode(parts)
    res = {}
    for L in (25, 50, 100):
        sub = [dict(p, k=p["k"][:, 100 - L:], y=p["y"][:, 100 - L:], Y=p["Y"][:, 100 - L:]) for p in parts]
        sl = slice(L - 25, L)
        t0 = time.time()
        model, cfg, table = select(sub, GRID_LONG, SEED + j, nv, sl)
        f_tr, f_te = predict(model, sub[0], sl), predict(model, sub[2], sl)
        ytr, yte = targets(sub[0], sl).ravel(), targets(sub[2], sl).ravel()
        gte = np.repeat(sub[2]["G"], 25)
        rel, f0, _ = reliability(model, sub, sl, nv, f_tr, f_te, SEED + j, ytr, yte)
        preds = {"dkt": f_te.ravel(), "output_removal": np.clip(f0, 1e-6, 1 - 1e-6)}
        pc = paired(preds, yte, gte, sub[2]["S"], 25, "dkt", SEED + j)
        res[L] = dict(selected=cfg, grid=table, scores={a: ni.scores(p, yte, gte) for a, p in preds.items()},
                      paired_vs_dkt=pc, reliability=rel)
        print(f"[eedi long split {j} L={L}] {cfg} AUC {res[L]['scores']['dkt']['auc']:.3f} removal dAUC {pc['output_removal']['auc']['diff']:+.3f} "
              f"dBrier {pc['output_removal']['brier']['diff']:+.4f} price {rel['announced']['price']:.4f} measured {rel['measured']:.4f} "
              f"interpretable {rel['interpretable']} ({time.time() - t0:.0f}s)", flush=True)
    return dict(results=res, n_train=len(parts[0]["G"]), n_test=len(parts[2]["G"]))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dataset", choices=["assistments", "eedi"], required=True)
    a_.add_argument("--long", action="store_true")
    a_.add_argument("--splits", type=int, default=5)
    a_.add_argument("--split-ids", default=None, help="comma-separated split indices (default: 0..splits-1)")
    a_.add_argument("--suffix", default="", help="added to the output name, to run splits in parallel processes")
    a = a_.parse_args()
    if a.dataset == "assistments":
        df = ap.keep_full_trajectories(ap.load_features("assistments_features_time.csv")); strata = "train"
    else:
        df = ap.keep_full_trajectories(ap.load_features(f"{ROOT}/Datasets/Eedi_NeurIPS2020/eedi_features_time.csv")); strata = "given"
    os.makedirs(rm.OUT, exist_ok=True)
    name = f"next_item_tuned_{a.dataset}{'_long' if a.long else ''}{a.suffix}.json"
    ids = [int(x) for x in a.split_ids.split(",")] if a.split_ids else list(range(a.splits))
    runs = []
    for j in ids:
        runs.append(one_split_long(df, j) if a.long else one_split(df, j, strata, a.dataset))
        runs[-1]["split"] = j
        json.dump({"splits": runs, "rule": RULE, "grid": GRID_LONG if a.long else GRID, "b_refit": B_REFIT},
                  open(os.path.join(rm.OUT, name), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
