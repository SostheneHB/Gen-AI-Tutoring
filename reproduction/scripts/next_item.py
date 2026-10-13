"""
Next-item target and a knowledge-tracing baseline (referee report 2, item 8), on the splits of the paper.

Target: correctness of the next item, y_{t+1}, for t = 0..24, on held-out students (same students and the same
26-step windows as the EDMD studies).

Arms
  * EDMD readouts: raw, projected (rank 4), group-aware and dynamics-only models predict the next state
    x_{t+1} from x_t; a logistic readout fitted on the training students maps it to P(y_{t+1} = 1).
  * DKT: a GRU over (skill, correctness) of items 0..t, with the skill of item t+1 given at the output
    (Piech et al., 2015), trained on the training students.
  * DKT, LEACE on the hidden state: LEACE (closed form, binary concept) applied to the hidden state, and the
    skill-specific output layer refitted on the erased states.
  * DKT, Proposition 5 erasure: f_0 = f - c (q - p1) on the DKT predictions, with q = P(g = 1 | hidden state);
    its Brier-score cost is compared with the frontier (1 - tau)^2 p0 p1 Delta^2 / R^2 at the achieved tau.

Metrics on held-out students: AUC and log-loss (overall and worst group), the predicted next-item gap
(mean prediction, group 1 minus group 0) against the observed one, and calibration in the large per group.

    python next_item.py --dataset assistments|eedi      -> out_rank_matched/next_item_<dataset>.json (aggregates)
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

import assistments_pipeline as ap
import rank_matched_controls as rm
from dynamics_only import GroupAware, ROOT

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
T = ap.N_TRANS


def prepare_items(df_all, seed, split, strata):
    """rank_matched_controls.prepare, plus the aligned correctness and skill of the same rows."""
    rng = np.random.default_rng(seed)
    df = df_all
    tr, va, te = ap.split_schools(df, rng) if split == "school" else ap.split_students(df["user_id"].unique(), rng)
    if strata == "train":
        df, _ = ap.rebuild_school_strata(df, tr, split == "school")
    df = ap.rebuild_dt(df, tr)
    out = []
    for ids in (tr, te):
        _, X, G, _ = ap.trajectories(df, ids)
        sub = df[df["user_id"].isin(ids) & (df["step"] <= T)].sort_values(["user_id", "step"])
        Y = (sub["correct"].values >= 1).reshape(-1, T + 1).astype(float)   # full credit (0.01% partial credit)
        K = sub["skill_id"].values.reshape(-1, T + 1)
        out.append((X, G, Y, K))
    return out


def scores(p, y, g):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    r = dict(auc=float(roc_auc_score(y, p)), logloss=float(log_loss(y, p)), brier=float(((p - y) ** 2).mean()),
             pred_gap=float(p[g == 1].mean() - p[g == 0].mean()), obs_gap=float(y[g == 1].mean() - y[g == 0].mean()))
    for gv in (0, 1):
        m = g == gv
        r[f"auc_g{gv}"] = float(roc_auc_score(y[m], p[m])) if len(np.unique(y[m])) > 1 else float("nan")
        r[f"logloss_g{gv}"] = float(log_loss(y[m], p[m], labels=[0, 1]))
        r[f"citl_g{gv}"] = float(y[m].mean() - p[m].mean())
    r["worst_logloss"] = max(r["logloss_g0"], r["logloss_g1"])
    r["worst_auc"] = min(r["auc_g0"], r["auc_g1"])
    return r


def edmd_arms(Xtr, Gtr, Ytr, Xte, Gte, Yte):
    fit = ap.fit_edmd(Xtr)
    Gamma, _ = ap.metric(fit.G, ap.RIDGE_RATIO)
    c, _ = ap.contrast(fit, Xtr, Gtr)
    P = ap.build_projector(fit.K, Gamma, c, rank=4)[0]
    ga = GroupAware(fit, Xtr, Gtr)
    do = GroupAware(fit, Xtr, Gtr); do.use_group = False
    models = {"edmd_raw": rm.LinPredictor(fit, fit.A, fit.a), "edmd_projected": rm.LinPredictor(fit, P.T @ fit.A @ P.T, P.T @ fit.a),
              "edmd_group_aware": ga, "edmd_dynamics_only": do}
    out = {}
    for name, m in models.items():
        feats = lambda X, G: m.predict(X[:, :-1].reshape(-1, 4), 1, np.repeat(G, T))
        lr = LogisticRegression(max_iter=1000).fit(feats(Xtr, Gtr), Ytr[:, 1:].reshape(-1))
        p = lr.predict_proba(feats(Xte, Gte))[:, 1]
        out[name] = scores(p, Yte[:, 1:].reshape(-1), np.repeat(Gte, T))
    return out


class DKT(torch.nn.Module):
    def __init__(self, n_skills, hidden=64, emb=32):
        super().__init__()
        self.inp = torch.nn.Embedding(2 * (n_skills + 1), emb)
        self.gru = torch.nn.GRU(emb, hidden, batch_first=True)
        self.out = torch.nn.Embedding(n_skills + 1, hidden + 1)      # per-skill readout weights and bias

    def hidden(self, K, Y):
        return self.gru(self.inp(K * 2 + Y))[0]                       # (n, T+1, hidden), state after item t

    def logits(self, H, Knext):
        w = self.out(Knext)
        return (H * w[..., :-1]).sum(-1) + w[..., -1]


def leace_binary(H, g):
    """LEACE eraser for a binary concept: x -> x - S u (u'S u)^-1 u'(x - mean), with u = S^-1 Cov(x, g) (whitened form)."""
    mu = H.mean(0)
    Hc = H - mu
    S = Hc.T @ Hc / len(Hc) + 1e-6 * np.eye(H.shape[1])
    cov = Hc.T @ (g - g.mean()) / len(Hc)
    w, V = np.linalg.eigh(S)
    Wh, Wih = (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T
    v = Wih @ cov
    v = v / np.linalg.norm(v)
    Pm = Wh @ np.outer(v, v) @ Wih                                    # oblique projection onto the concept
    return lambda X: X - (X - mu) @ Pm.T


def dkt_arms(Ktr, Ytr, Gtr, Kte, Yte, Gte, seed, epochs=int(os.environ.get("DKT_EPOCHS", 20))):
    torch.manual_seed(seed); np.random.seed(seed)
    vocab = {s: i + 1 for i, s in enumerate(np.unique(Ktr))}
    enc = lambda K: np.vectorize(lambda s: vocab.get(s, 0))(K)
    ktr, kte = torch.tensor(enc(Ktr)), torch.tensor(enc(Kte))
    ytr, yte = torch.tensor(Ytr).long(), torch.tensor(Yte).long()
    model = DKT(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = len(ktr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            H = model.hidden(ktr[b, :-1], ytr[b, :-1])
            loss = lossf(model.logits(H, ktr[b, 1:]), ytr[b, 1:].float())
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Htr = model.hidden(ktr[:, :-1], ytr[:, :-1])
        Hte = model.hidden(kte[:, :-1], yte[:, :-1])
        p = torch.sigmoid(model.logits(Hte, kte[:, 1:])).numpy().reshape(-1)
    y, g = Yte[:, 1:].reshape(-1), np.repeat(Gte, T)
    out = {"dkt": scores(p, y, g)}
    # group erased from the hidden state, readout refitted (same per-skill form) on the erased training states
    Htr_np, Hte_np = Htr.numpy().reshape(-1, Htr.shape[-1]), Hte.numpy().reshape(-1, Hte.shape[-1])
    E = leace_binary(Htr_np, np.repeat(Gtr, T).astype(float))
    Htr_e = torch.tensor(E(Htr_np).reshape(Htr.shape), dtype=torch.float32)
    Hte_e = torch.tensor(E(Hte_np).reshape(Hte.shape), dtype=torch.float32)
    head = torch.nn.Embedding(len(vocab) + 1, Htr.shape[-1] + 1)
    head.weight.data.copy_(model.out.weight.data)
    opt2 = torch.optim.Adam(head.parameters(), lr=3e-3)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            w = head(ktr[b, 1:])
            loss = lossf((Htr_e[b] * w[..., :-1]).sum(-1) + w[..., -1], ytr[b, 1:].float())
            opt2.zero_grad(); loss.backward(); opt2.step()
    with torch.no_grad():
        w = head(kte[:, 1:])
        pe = torch.sigmoid((Hte_e * w[..., :-1]).sum(-1) + w[..., -1]).numpy().reshape(-1)
    out["dkt_leace_hidden"] = scores(pe, y, g)
    # Proposition 5 applied to DKT: f_0 = f - c (q - p1), q = P(g = 1 | hidden state) by logistic regression;
    # cost measured with the Brier score (the MSE of Proposition 5) and compared with the frontier at the achieved tau
    with torch.no_grad():
        ptr = torch.sigmoid(model.logits(Htr, ktr[:, 1:])).numpy().reshape(-1)
    gtr_ = np.repeat(Gtr, T).astype(float)
    qm = LogisticRegression(max_iter=1000).fit(Htr_np, gtr_)
    qtr, qte = qm.predict_proba(Htr_np)[:, 1], qm.predict_proba(Hte_np)[:, 1]
    p1 = gtr_.mean()
    cc = np.cov(ptr, qtr)[0, 1] / qtr.var()
    f0_raw = p - cc * (qte - p1)                                   # the predictor of Proposition 5 (may leave [0, 1])
    f0 = np.clip(f0_raw, 1e-6, 1 - 1e-6)                           # clipped, for AUC and log-loss
    out["dkt_prop5_erased"] = scores(f0, y, g)
    clip_share = float(((f0_raw < 0) | (f0_raw > 1)).mean())
    brier_raw = float(((f0_raw - y) ** 2).mean())
    gap_raw = float(f0_raw[g == 1].mean() - f0_raw[g == 0].mean())
    d_tr = ptr[gtr_ == 1].mean() - ptr[gtr_ == 0].mean()
    r2 = qtr.var() / (p1 * (1 - p1))
    price = p1 * (1 - p1) * d_tr ** 2 / r2
    tau = out["dkt_prop5_erased"]["pred_gap"] / out["dkt"]["pred_gap"]
    tau_raw = gap_raw / out["dkt"]["pred_gap"]
    out["dkt_prop5"] = dict(price=float(price), r2_group=float(r2), tau=float(tau), frontier=float((1 - tau) ** 2 * price),
                            measured_brier=float(out["dkt_prop5_erased"]["brier"] - out["dkt"]["brier"]),
                            clip_share=clip_share, tau_unclipped=float(tau_raw),
                            frontier_unclipped=float((1 - tau_raw) ** 2 * price),
                            measured_brier_unclipped=float(brier_raw - out["dkt"]["brier"]))
    # how well the hidden state predicts the group (R^2 of the linear group score), for the price formula
    gtr = np.repeat(Gtr, T).astype(float)
    Hc = Htr_np - Htr_np.mean(0)
    coef, *_ = np.linalg.lstsq(Hc, gtr - gtr.mean(), rcond=None)
    out["dkt_hidden_r2_group"] = float((Hc @ coef).var() / gtr.var())
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--dataset", choices=["assistments", "eedi"], required=True)
    a = ap_.parse_args()
    seed = 20260914
    if a.dataset == "assistments":
        df = ap.keep_full_trajectories(ap.load_features("assistments_features_time.csv")); strata = "train"
    else:
        df = ap.keep_full_trajectories(ap.load_features(f"{ROOT}/Datasets/Eedi_NeurIPS2020/eedi_features_time.csv")); strata = "given"
    runs = []
    for j in range(5):
        (Xtr, Gtr, Ytr, Ktr), (Xte, Gte, Yte, Kte) = prepare_items(df, seed + 200 + j, "school", strata)
        r = edmd_arms(Xtr, Gtr, Ytr, Xte, Gte, Yte)
        r.update(dkt_arms(Ktr, Ytr, Gtr, Kte, Yte, Gte, seed + j))
        runs.append(r)
        print(f"split {j}: " + " | ".join(f"{k} auc {v['auc']:.3f} gap {v['pred_gap']:+.3f}" for k, v in r.items() if isinstance(v, dict) and "auc" in v)
              + f" | obs gap {r['dkt']['obs_gap']:+.3f}", flush=True)
    keys = [k for k in runs[0] if isinstance(runs[0][k], dict) and k != "dkt_prop5"]
    summ = {k: {q: float(np.mean([r[k][q] for r in runs])) for q in runs[0][k]} for k in keys}
    summ["dkt_hidden_r2_group"] = float(np.mean([r["dkt_hidden_r2_group"] for r in runs]))
    summ["dkt_prop5"] = {q: float(np.mean([r["dkt_prop5"][q] for r in runs])) for q in runs[0]["dkt_prop5"]}
    os.makedirs(rm.OUT, exist_ok=True)
    json.dump({"splits": runs, "summary": summ}, open(os.path.join(rm.OUT, f"next_item_{a.dataset}.json"), "w"), indent=1)
    for k in keys:
        s = summ[k]
        print(f"{k:20s} AUC {s['auc']:.3f} (worst {s['worst_auc']:.3f}) logloss {s['logloss']:.4f} (worst {s['worst_logloss']:.4f}) "
              f"pred gap {s['pred_gap']:+.4f} obs {s['obs_gap']:+.4f}")
    print("DKT hidden R2 of group:", round(summ["dkt_hidden_r2_group"], 4))
    print("Proposition 5 on DKT:", {k: round(v, 5) for k, v in summ["dkt_prop5"].items()})


if __name__ == "__main__":
    main()
