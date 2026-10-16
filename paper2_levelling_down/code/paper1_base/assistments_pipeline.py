"""
ASSISTments 2012-2013 pipeline for the revised manuscript
"Operator Compression for Sequential Fairness: Structural, Predictive, and
Policy-Level Diagnostics" (version 2, second revision).

Every ASSISTments number in the manuscript is produced here and written to
<out>/<run_name>/results.json and <out>/summary.json.

    python build_features.py --raw <raw csv> --order time --out assistments_features_time.csv
    python assistments_pipeline.py --features assistments_features_time.csv --out out_time

Design (each point answers a reviewer comment):

* Temporal order.  The features file is built by build_features.py with every
  student's rows sorted by (start_time, problem_log_id).  Steps are taken in
  file order.  `--features assistments_features_pid.csv` reproduces the
  historical problem-identifier order as a sensitivity analysis.
* Group variable.  G in {0,1} is an endogenous school-performance stratum,
  rebuilt from training students only on a label window (steps > 25) that
  never overlaps the evaluated steps 0..25.  The full-corpus upstream label S
  is used only in the sensitivity run `student_split_full_strata`.
* Primary analysis.  Five school-disjoint 60/20/20 splits.  Results are
  summarised at the level of the five splits (t interval, 4 df).
* Uncertainty.  Every held-out quantity carries a school-cluster bootstrap
  interval (schools resampled with replacement, B = 2000).  Cohorts are kept
  only for the TRS, which is defined per cohort.
* Order of estimation.  Split first; skill means in d_t, dictionary scaling,
  EDMD, metric Gamma, contrast, projector, decoder and all baselines are
  fitted on training students only and frozen.
* Baselines.  Persistence, affine least squares on the physical state,
  gradient-boosted trees on the physical state, raw EDMD; errors at horizons
  1, 5, 10, 25, per feature and per stratum.
* Fairness alternatives.  Projected EDMD (hard removal), soft projections
  P_alpha = I - alpha R, and per-stratum recalibration of raw EDMD (keeps the
  gap, equalises calibration).
* TRS (formerly TIB).  Definition 1 with divisor n, exact zero handling, plus
  norms, normalised Euclidean distance, student rank correlations and
  sensitivity to centring/standardisation.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["k_t", "e_t", "p_t", "d_t"]
N_TRANS = 25                 # transitions per student -> states x_0..x_25
RANK = 4
DEPTH = 3
RIDGE_RATIO = 1e-3
RIDGE_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
COHORT_SIZE = 150
N_COHORTS = 10
ZERO_TOL = 1e-12
HORIZONS = [1, 5, 10, 25]
ALPHAS = [0.25, 0.5, 0.75]
B_BOOT = 2000


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_features(path: str, school_file: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    cnt = df["user_id"].value_counts()
    df = df[df["user_id"].isin(cnt[cnt >= 20].index)].copy()
    if "correct" not in df:          # historical file: recover from d_t
        df["correct"] = (df["d_t"] > 0).astype(float)
        df["skill_id"] = (df["correct"] - df["d_t"]).round(10)
    df["G"] = df["S"].astype(int)
    df["step"] = df.groupby("user_id").cumcount()      # file order
    if "school_id" not in df and school_file and os.path.exists(school_file):
        sch = pd.read_csv(school_file).drop_duplicates("user_id")
        df = df.merge(sch[["user_id", "school_id"]], on="user_id", how="left")
    return df


def keep_full_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    n = df.groupby("user_id")["step"].transform("max") + 1
    return df[n >= N_TRANS + 1].copy()


def split_students(ids, rng):
    ids = rng.permutation(ids)
    n = len(ids)
    return ids[: int(0.6 * n)], ids[int(0.6 * n): int(0.8 * n)], ids[int(0.8 * n):]


def split_schools(df, rng):
    """School-disjoint 60/20/20 split (by student counts)."""
    sch = df.groupby("school_id")["user_id"].nunique()
    order = rng.permutation(sch.index.values)
    cum = np.cumsum(sch.loc[order].values) / sch.sum()
    parts = order[cum <= 0.6], order[(cum > 0.6) & (cum <= 0.8)], order[cum > 0.8]
    users = df.drop_duplicates("user_id").set_index("user_id")["school_id"]
    return [users.index[users.isin(s)].values for s in parts]


def rebuild_dt(df, train_ids):
    """d_t = correct - training-only skill mean (evaluated steps), clipped."""
    tr = df[df["user_id"].isin(train_ids) & (df["step"] <= N_TRANS)]
    m = tr.groupby("skill_id")["correct"].mean()
    df = df.copy()
    df["d_t"] = np.clip(df["correct"] - df["skill_id"].map(m).fillna(tr["correct"].mean()), -1, 1)
    return df


def rebuild_school_strata(df, train_ids, school_disjoint):
    """Train-only strata from the label window (steps > 25)."""
    lab = df[df["step"] > N_TRANS]
    tr_means = lab[lab["user_id"].isin(train_ids)].groupby("school_id")["correct"].mean()
    thr = float(np.median(tr_means.values))
    if school_disjoint:          # held-out schools: their own label window
        means = lab.groupby("school_id")["correct"].mean()
        means.loc[tr_means.index] = tr_means
    else:
        means = tr_means
    df = df.copy()
    df["G"] = df["school_id"].map((means > thr).astype(int))
    df = df[df["G"].notna()].copy()
    df["G"] = df["G"].astype(int)
    return df, thr


def rebuild_school_strata_other(df, train_ids, val_ids, test_ids, rng, frac=0.5):
    """Prospective variant for school-disjoint splits.  Training schools are
    labelled as in rebuild_school_strata (training students, label window).
    In every validation/test school a random half of the students is used only
    to label the school (all their steps); these students are then removed
    from evaluation, so the label of a new school is known before any
    evaluated student of that school is predicted."""
    lab = df[df["step"] > N_TRANS]
    tr_means = lab[lab["user_id"].isin(train_ids)].groupby("school_id")["correct"].mean()
    thr = float(np.median(tr_means.values))
    users = df.drop_duplicates("user_id").set_index("user_id")["school_id"]
    held = users[users.index.isin(np.concatenate([val_ids, test_ids]))]
    label_students = []
    for sch, grp in held.groupby(held):
        ids = rng.permutation(grp.index.values)
        label_students.extend(ids[: int(np.ceil(frac * len(ids)))])
    label_students = np.array(label_students)
    held_means = df[df["user_id"].isin(label_students)].groupby("school_id")["correct"].mean()
    means = pd.concat([tr_means, held_means])
    df = df[~df["user_id"].isin(label_students)].copy()
    df["G"] = df["school_id"].map((means > thr).astype(int))
    df = df[df["G"].notna()].copy()
    df["G"] = df["G"].astype(int)
    keep = set(df["user_id"].unique())
    return (df, thr, np.array([u for u in val_ids if u in keep]),
            np.array([u for u in test_ids if u in keep]), int(len(label_students)))


def calibration(model, X, G, boot):
    """Per-stratum calibration of one-step k predictions on held-out students:
    calibration-in-the-large (mean observed minus mean predicted), calibration
    slope (regression of observed on predicted) and a decile reliability table,
    with school-cluster bootstrap intervals for the first two."""
    n = X.shape[0]
    g_rep = np.repeat(G, N_TRANS)
    p = model.predict(X[:, :-1].reshape(-1, 4), 1, g_rep)[:, 0].reshape(n, N_TRANS)
    y = X[:, 1:, 0]
    stats_ = [np.full(n, N_TRANS, float), p.sum(1), y.sum(1), (p * p).sum(1), (p * y).sum(1)]
    out = {}
    for gv in (0, 1):
        m = (G == gv).astype(float)
        def est(w):
            N, Sp, Sy, Spp, Spy = [np.asarray(w) @ (v * m) for v in stats_]
            citl = (Sy - Sp) / N
            slope = (Spy - Sp * Sy / N) / (Spp - Sp * Sp / N)
            return citl, slope
        c0, s0 = est(np.ones(n))
        cb, sb = est(boot.w)
        pp, yy = p[G == gv].ravel(), y[G == gv].ravel()
        edges = np.quantile(pp, np.linspace(0, 1, 11))
        idx = np.clip(np.searchsorted(edges, pp, side="right") - 1, 0, 9)
        rel = [[float(pp[idx == b].mean()), float(yy[idx == b].mean())] for b in range(10) if (idx == b).any()]
        out[str(gv)] = dict(citl=float(c0), citl_school_ci=SchoolBoot.ci(cb),
                            slope=float(s0), slope_school_ci=SchoolBoot.ci(sb),
                            reliability_deciles=rel)
    return out


def trajectories(df, ids):
    sub = df[df["user_id"].isin(ids) & (df["step"] <= N_TRANS)].sort_values(["user_id", "step"])
    X = sub[FEATURES].values.reshape(-1, N_TRANS + 1, len(FEATURES)).astype(float)
    first = sub.groupby("user_id", sort=True).first()
    return first.index.values, X, first["G"].values.astype(int), first["school_id"].values


# ---------------------------------------------------------------------------
# Dictionary, EDMD, metric, contrast, projector (unchanged from version 1)
# ---------------------------------------------------------------------------
def lift(X):
    k, e, p, d = (X[..., i] for i in range(4))
    return np.stack([k, e, p, d, k * k, e * e, p * p, d * d, k * e, k * p, k * d,
                     e * p, e * d, p * d], axis=-1)


def sym_sqrt(M):
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


@dataclass
class Fit:
    mu: np.ndarray
    sd: np.ndarray
    A: np.ndarray
    a: np.ndarray
    K: np.ndarray
    G: np.ndarray
    C: np.ndarray
    c: np.ndarray
    num_rank: int


def fit_edmd(Xtr):
    Psi = lift(Xtr)
    cur = Psi[:, :-1].reshape(-1, Psi.shape[-1])
    nxt = Psi[:, 1:].reshape(-1, Psi.shape[-1])
    mu, sd = cur.mean(0), cur.std(0) + 1e-12
    Z, Zp = (cur - mu) / sd, (nxt - mu) / sd
    Zaug = np.hstack([Z, np.ones((Z.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(Zaug, Zp, rcond=None)
    A, a = coef[:-1].T, coef[-1]
    G = Z.T @ Z / Z.shape[0]
    dec, *_ = np.linalg.lstsq(Zaug, Xtr[:, :-1].reshape(-1, 4), rcond=None)
    num_rank = int(np.linalg.matrix_rank(G, tol=1e-10 * np.linalg.eigvalsh(G).max()))
    return Fit(mu, sd, A, a, A.T, G, dec[:-1].T, dec[-1], num_rank)


def encode(fit, X):
    return (lift(X) - fit.mu) / fit.sd


def decode(fit, Z):
    return Z @ fit.C.T + fit.c


def metric(G, ratio):
    eta = ratio * np.linalg.eigvalsh(G).max()
    return G + eta * np.eye(G.shape[0]), eta


def contrast(fit, Xtr, Gtr):
    Z = encode(fit, Xtr[:, :-1]).reshape(-1, fit.A.shape[0])
    g = np.repeat(Gtr, N_TRANS)
    p1 = g.mean()
    c, *_ = np.linalg.lstsq(Z, (g - p1) / (p1 * (1 - p1)), rcond=None)
    return c, float(p1)


def build_projector(K, Gamma, c, rank=RANK, depth=DEPTH, cutoff=1e-8):
    Gh, Gih = sym_sqrt(Gamma)
    W = Gh @ K @ Gih
    b = Gh @ c
    seed = b / np.linalg.norm(b)
    words, frontier = [seed], [seed]
    for _ in range(depth):
        new = []
        for v in frontier:
            new += [W @ v, W.T @ v]
        words += new
        frontier = new
    Cand = np.stack(words[1:], axis=1)
    Cand = Cand - np.outer(seed, seed @ Cand)
    U, s, _ = np.linalg.svd(Cand, full_matrices=False)
    cand_rank = int((s > cutoff * s.max()).sum())
    Qw, _ = np.linalg.qr(np.column_stack([seed, U[:, : min(rank - 1, cand_rank)]]))
    Q = Gih @ Qw
    R = Q @ Q.T @ Gamma
    return np.eye(K.shape[0]) - R, R, cand_rank + 1


def wnorm(L, Gamma, ord="fro"):
    Gh, Gih = sym_sqrt(Gamma)
    return float(np.linalg.norm(Gh @ L @ Gih, ord))


def propagator(fit, P):
    return P.T @ fit.A @ P.T, P.T @ fit.a


# ---------------------------------------------------------------------------
# Predictors: every model maps a batch of physical states to the h-step
# prediction, iterating its own one-step map.
# ---------------------------------------------------------------------------
class EDMDModel:
    def __init__(self, fit, prop, calib=None):
        self.fit, (self.A, self.a), self.calib = fit, prop, calib

    def predict(self, X0, h, g=None):
        z = encode(self.fit, X0)
        for _ in range(h):
            z = z @ self.A.T + self.a
        X = decode(self.fit, z)
        if self.calib is not None:              # per-stratum affine map on k
            X = X.copy()
            for gv, (b0, b1) in self.calib.items():
                X[g == gv, 0] = b0 + b1 * X[g == gv, 0]
        return X


class PhysModel:
    def __init__(self, step):
        self.step = step

    def predict(self, X0, h, g=None):
        X = X0
        for _ in range(h):
            X = self.step(X)
        return X


def fit_affine(Xtr):
    cur = Xtr[:, :-1].reshape(-1, 4)
    nxt = Xtr[:, 1:].reshape(-1, 4)
    W, *_ = np.linalg.lstsq(np.hstack([cur, np.ones((len(cur), 1))]), nxt, rcond=None)
    return PhysModel(lambda X: np.hstack([X, np.ones((len(X), 1))]) @ W)


def fit_gbm(Xtr, seed):
    cur = Xtr[:, :-1].reshape(-1, 4)
    nxt = Xtr[:, 1:].reshape(-1, 4)
    ms = [HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                        random_state=seed).fit(cur, nxt[:, j])
          for j in range(4)]
    return PhysModel(lambda X: np.column_stack([m.predict(X) for m in ms]))


def fit_group_calibration(model, Xtr, Gtr):
    """Per-stratum affine recalibration of one-step k predictions (training)."""
    Y = model.predict(Xtr[:, :-1].reshape(-1, 4), 1, np.repeat(Gtr, N_TRANS))[:, 0]
    y = Xtr[:, 1:, 0].reshape(-1)
    g = np.repeat(Gtr, N_TRANS)
    cal = {}
    for gv in (0, 1):
        m = g == gv
        b1, b0 = np.polyfit(Y[m], y[m], 1)
        cal[gv] = (float(b0), float(b1))
    return cal


def horizon_errors(model, X, G, h):
    """Per-student squared errors of h-step predictions from every observed
    start x_t, t = 0..T-h.  Returns (n, 4) mean squared error per feature."""
    n = X.shape[0]
    starts = X[:, : N_TRANS + 1 - h].reshape(-1, 4)
    target = X[:, h:].reshape(n, -1, 4)
    g = np.repeat(G, N_TRANS + 1 - h)
    pred = model.predict(starts, h, g).reshape(n, -1, 4)
    return ((pred - target) ** 2).mean(1)


# ---------------------------------------------------------------------------
# School-cluster bootstrap
# ---------------------------------------------------------------------------
class SchoolBoot:
    def __init__(self, schools, rng, B=B_BOOT):
        u, inv = np.unique(schools, return_inverse=True)
        W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=B)
        self.w = W[:, inv].astype(float)               # (B, n_students)

    def mean(self, v, mask=None):
        w = self.w if mask is None else self.w * mask
        s = w.sum(1)
        return np.where(s > 0, (w * v).sum(1) / np.where(s > 0, s, 1), np.nan)

    def gap(self, v, g):
        return self.mean(v, g == 1) - self.mean(v, g == 0)

    @staticmethod
    def ci(draws):
        return [float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))]


def gap(v, g):
    return float(v[g == 1].mean() - v[g == 0].mean())


def tcI(x, level=0.95):
    x = np.asarray(x, float)
    m = float(x.mean())
    if len(x) < 2:
        return [m, float("nan"), float("nan")]
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [m, float(m - h), float(m + h)]


# ---------------------------------------------------------------------------
# Rollouts, cohorts, TRS
# ---------------------------------------------------------------------------
def rollout(fit, prop, X0, T=N_TRANS):
    A, a = prop
    z = encode(fit, X0)
    Zs = []
    for _ in range(T):
        z = z @ A.T + a
        Zs.append(z)
    return decode(fit, np.stack(Zs, axis=1))


def disjoint_cohorts(G, rng, n_cohorts=N_COHORTS, size=COHORT_SIZE):
    i0 = rng.permutation(np.where(G == 0)[0])
    i1 = rng.permutation(np.where(G == 1)[0])
    h = size // 2
    n = min(n_cohorts, min(len(i0), len(i1)) // h)
    return [np.concatenate([i0[j * h:(j + 1) * h], i1[j * h:(j + 1) * h]]) for j in range(n)]


def dcos0(a, b, tol_a, tol_b):
    za, zb = np.linalg.norm(a) <= tol_a, np.linalg.norm(b) <= tol_b
    if za and zb:
        return 0.0, "zero-zero"
    if za or zb:
        return 1.0, "zero-nonzero"
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), "nonzero"


def trs_vectors(tr0, tr1, centring="separate", scale=None):
    n = tr0.shape[0]
    g0, g1 = tr0.reshape(n, -1).copy(), tr1.reshape(n, -1).copy()
    if scale is not None:
        g0, g1 = g0 / scale, g1 / scale
    if centring == "separate":
        g0, g1 = g0 - g0.mean(0), g1 - g1.mean(0)
    elif centring == "common":
        m = g0.mean(0)
        g0, g1 = g0 - m, g1 - m
    return g0, g1


def trs_cohort(tr0, tr1, centring="separate", scale=None):
    g0, g1 = trs_vectors(tr0, tr1, centring, scale)
    t0 = ZERO_TOL * max(1.0, np.abs(g0).max()) * np.sqrt(g0.shape[1])
    t1 = ZERO_TOL * max(1.0, np.abs(g1).max()) * np.sqrt(g1.shape[1])
    out = [dcos0(g0[i], g1[i], t0, t1) for i in range(len(g0))]
    return np.array([o[0] for o in out]), [o[1] for o in out], g0, g1


def encoded_residual(fit, Xtraj, R, Gamma):
    Z = encode(fit, Xtraj)
    Zc = Z - Z.mean(0, keepdims=True)
    Gi = np.linalg.inv(Gamma)
    RZ = Zc @ R
    num = np.sqrt(np.einsum("ntj,jk,ntk->n", RZ, Gi, RZ))
    den = np.sqrt(np.einsum("ntj,jk,ntk->n", Zc, Gi, Zc))
    return num / np.maximum(den, 1e-300)


# ---------------------------------------------------------------------------
# One complete run
# ---------------------------------------------------------------------------
def run(df_all, name, seed, split="school", strata="train", student_ids=None,
        ridge_sweep=True, outdir="out"):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    df = df_all if student_ids is None else df_all[df_all["user_id"].isin(student_ids)]
    info = {"name": name, "seed": seed, "split": split, "strata": strata}
    if split == "school":
        tr, va, te = split_schools(df, rng)
    else:
        tr, va, te = split_students(df["user_id"].unique(), rng)
    if strata == "train":
        df, thr = rebuild_school_strata(df, tr, split == "school")
        info["stratum_threshold_train_only"] = thr
    elif strata == "other":
        df, thr, va, te, n_lab = rebuild_school_strata_other(df, tr, va, te, rng)
        info["stratum_threshold_train_only"] = thr
        info["n_label_only_students"] = n_lab
    df = rebuild_dt(df, tr)
    _, Xtr, Gtr, Str = trajectories(df, tr)
    _, Xva, Gva, Sva = trajectories(df, va)
    _, Xte, Gte, Ste = trajectories(df, te)

    def sc(S, G):
        return {"total": int(len(set(S))), "G0": int(len(set(S[G == 0]))),
                "G1": int(len(set(S[G == 1])))}
    info.update(n_students=len(Xtr) + len(Xva) + len(Xte), n_train=len(Xtr),
                n_val=len(Xva), n_test=len(Xte),
                n_train_transitions=int(len(Xtr) * N_TRANS),
                n_test_transitions=int(len(Xte) * N_TRANS),
                p1_train=float(Gtr.mean()), p1_test=float(Gte.mean()),
                schools={"train": sc(Str, Gtr), "val": sc(Sva, Gva), "test": sc(Ste, Gte)},
                schools_shared_train_test=int(len(set(Str) & set(Ste))))

    fit = fit_edmd(Xtr)
    Gamma, eta = metric(fit.G, RIDGE_RATIO)
    c, _ = contrast(fit, Xtr, Gtr)
    P, R, cand_rank = build_projector(fit.K, Gamma, c)
    raw, proj = (fit.A, fit.a), propagator(fit, P)
    F = P @ fit.K @ P
    info["diagnostics"] = dict(
        dictionary_dim=int(fit.K.shape[0]), numerical_rank_G=fit.num_rank, eta=float(eta),
        K_norm_Gamma2=wnorm(fit.K, Gamma, 2), F_norm_Gamma2=wnorm(F, Gamma, 2),
        spectral_radius_A=float(np.abs(np.linalg.eigvals(fit.A)).max()),
        spectral_radius_AP=float(np.abs(np.linalg.eigvals(proj[0])).max()),
        r1=wnorm(P @ fit.K @ R, Gamma), r2=wnorm(R @ fit.K @ P, Gamma),
        eps_q=float(np.sqrt(max((P @ c) @ Gamma @ (P @ c), 0))),
        idempotency_err=float(np.abs(P @ P - P).max()),
        applied_rank=RANK, candidate_rank=cand_rank)

    # ---- (1) predictors: baselines and fairness alternatives
    models = {
        "persistence": PhysModel(lambda X: X),
        "affine_ls": fit_affine(Xtr),
        "gbm": fit_gbm(Xtr, seed),
        "edmd_raw": EDMDModel(fit, raw),
        "edmd_proj": EDMDModel(fit, proj),
    }
    for al in ALPHAS:
        models[f"edmd_soft_{al}"] = EDMDModel(fit, propagator(fit, np.eye(len(P)) - al * R))
    models["edmd_groupcal"] = EDMDModel(fit, raw, fit_group_calibration(models["edmd_raw"], Xtr, Gtr))

    boot = SchoolBoot(Ste, rng)
    err = {m: {h: horizon_errors(mod, Xte, Gte, h) for h in HORIZONS} for m, mod in models.items()}
    kT = Xte[:, -1, 0]
    obs_gap = gap(kT, Gte)
    pred = {}
    info["replay"] = {"terminal_gap_observed": obs_gap,
                      "terminal_gap_observed_school_ci": SchoolBoot.ci(boot.gap(kT, Gte)),
                      "models": {}}
    for m, mod in models.items():
        pk = mod.predict(Xte[:, -2], 1, Gte)[:, 0]      # one-step prediction of k_25
        pred[m] = pk
        rec = {"terminal_gap_pred": gap(pk, Gte),
               "terminal_gap_pred_school_ci": SchoolBoot.ci(boot.gap(pk, Gte)),
               "gap_error": gap(pk, Gte) - obs_gap,
               "gap_error_school_ci": SchoolBoot.ci(boot.gap(pk - kT, Gte)),
               "mse": {}, "mse_k_by_stratum": {}}
        for h in HORIZONS:
            e = err[m][h]
            rec["mse"][str(h)] = {f: float(e[:, j].mean()) for j, f in enumerate(FEATURES)}
            rec["mse"][str(h)]["k_t_school_ci"] = SchoolBoot.ci(boot.mean(e[:, 0]))
            rec["mse_k_by_stratum"][str(h)] = {str(gv): float(e[Gte == gv, 0].mean()) for gv in (0, 1)}
        info["replay"]["models"][m] = rec
    # paired differences against raw EDMD (k_t), school-cluster intervals
    info["replay"]["paired_vs_raw"] = {
        m: {str(h): {"diff": float((err[m][h][:, 0] - err["edmd_raw"][h][:, 0]).mean()),
                     "school_ci": SchoolBoot.ci(boot.mean(err[m][h][:, 0] - err["edmd_raw"][h][:, 0]))}
            for h in HORIZONS}
        for m in models if m != "edmd_raw"}
    per_step = {"observed": [gap(Xte[:, t + 1, 0], Gte) for t in range(N_TRANS)]}
    for m in ("edmd_raw", "edmd_proj"):
        per_step[m] = [gap(models[m].predict(Xte[:, t], 1, Gte)[:, 0], Gte) for t in range(N_TRANS)]
    info["replay"]["per_step_gap"] = per_step
    info["calibration"] = {m: calibration(models[m], Xte, Gte, boot)
                           for m in ("edmd_raw", "edmd_groupcal", "edmd_proj")}

    # ---- (2) free-running surrogate rollouts from x_0 on all test students
    Xr_all = rollout(fit, raw, Xte[:, 0])
    Xp_all = rollout(fit, proj, Xte[:, 0])
    tau_all = gap(Xp_all[:, -1, 0], Gte) - gap(Xr_all[:, -1, 0], Gte)
    info["free_run"] = dict(
        tau_delta=tau_all,
        tau_delta_school_ci=SchoolBoot.ci(boot.gap(Xp_all[:, -1, 0], Gte) - boot.gap(Xr_all[:, -1, 0], Gte)),
        tau_k=float(Xp_all[:, -1, 0].mean() - Xr_all[:, -1, 0].mean()),
        tau_k_school_ci=SchoolBoot.ci(boot.mean(Xp_all[:, -1, 0] - Xr_all[:, -1, 0])),
        raw_gap=gap(Xr_all[:, -1, 0], Gte), proj_gap=gap(Xp_all[:, -1, 0], Gte),
        raw_g0=float(Xr_all[Gte == 0, -1, 0].mean()), raw_g1=float(Xr_all[Gte == 1, -1, 0].mean()),
        proj_g0=float(Xp_all[Gte == 0, -1, 0].mean()), proj_g1=float(Xp_all[Gte == 1, -1, 0].mean()),
        obs_g0=float(kT[Gte == 0].mean()), obs_g1=float(kT[Gte == 1].mean()),
        spearman_terminal_k=float(stats.spearmanr(Xr_all[:, -1, 0], Xp_all[:, -1, 0]).correlation),
        spearman_mean_k=float(stats.spearmanr(Xr_all[:, :, 0].mean(1), Xp_all[:, :, 0].mean(1)).correlation))

    # ---- (3) balanced student-disjoint cohorts (TRS is defined per cohort)
    cohorts = disjoint_cohorts(Gte, rng)
    rows, trs_all, kinds_all, rr_all, rp_all = [], [], [], [], []
    sens = {k: [] for k in ("common_centring", "no_centring", "standardised")}
    norms0, norms1, ned = [], [], []
    scale = Xr_all.reshape(len(Xr_all), -1).std(0) + 1e-12
    for j, idx in enumerate(cohorts):
        g = Gte[idx]
        Xr, Xp = Xr_all[idx], Xp_all[idx]
        rows.append(dict(cohort=j, schools=int(len(set(Ste[idx]))),
                         obs_gap=gap(Xte[idx, -1, 0], g),
                         raw_gap=gap(Xr[:, -1, 0], g), proj_gap=gap(Xp[:, -1, 0], g),
                         tau_delta=gap(Xp[:, -1, 0], g) - gap(Xr[:, -1, 0], g)))
        tv, kinds, g0, g1 = trs_cohort(Xr, Xp)
        trs_all.append(tv)
        kinds_all += kinds
        n0, n1 = np.linalg.norm(g0, axis=1), np.linalg.norm(g1, axis=1)
        norms0.append(n0)
        norms1.append(n1)
        ned.append(np.linalg.norm(g1 - g0, axis=1) / np.maximum(n0 + n1, 1e-300))
        sens["common_centring"].append(trs_cohort(Xr, Xp, "common")[0])
        sens["no_centring"].append(trs_cohort(Xr, Xp, "none")[0])
        sens["standardised"].append(trs_cohort(Xr, Xp, "separate", scale)[0])
        rr_all.append(encoded_residual(fit, Xr, R, Gamma))
        rp_all.append(encoded_residual(fit, Xp, R, Gamma))
    tau = [r["tau_delta"] for r in rows]
    info["cohorts"] = dict(n_cohorts=len(cohorts), cohort_size=COHORT_SIZE, rows=rows,
                           tau_delta=tcI(tau), tau_delta_n_negative=int(np.sum(np.array(tau) < 0)))
    trs = np.concatenate(trs_all)
    n0, n1, ned = np.concatenate(norms0), np.concatenate(norms1), np.concatenate(ned)
    rr, rp = np.concatenate(rr_all), np.concatenate(rp_all)
    kinds = pd.Series(kinds_all).value_counts().to_dict()
    q = lambda v: [float(np.percentile(v, p)) for p in (5, 25, 50, 75, 95)]
    info["trs"] = dict(
        n=int(len(trs)), mean=float(trs.mean()), median=float(np.median(trs)),
        sd_pop_n=float(trs.std(ddof=0)),
        pairs=dict(zero_zero=int(kinds.get("zero-zero", 0)),
                   zero_nonzero=int(kinds.get("zero-nonzero", 0)),
                   nonzero=int(kinds.get("nonzero", 0))),
        n_below_0p05=int((trs < 0.05).sum()), n_above_1=int((trs > 1).sum()),
        norm_raw_quantiles=q(n0), norm_proj_quantiles=q(n1),
        norm_ratio_proj_over_raw_median=float(np.median(n1 / n0)),
        small_norm_fraction=float(np.mean(np.minimum(n0, n1) < 0.01 * np.median(np.r_[n0, n1]))),
        normalised_euclidean_mean=float(ned.mean()), normalised_euclidean_median=float(np.median(ned)),
        spearman_trs_vs_min_norm=float(stats.spearmanr(trs, np.minimum(n0, n1)).correlation),
        sensitivity_mean={k: float(np.concatenate(v).mean()) for k, v in sens.items()},
        encoded_residual_raw_mean=float(rr.mean()), encoded_residual_proj_mean=float(rp.mean()))

    # ---- (4) ridge sweep: same fit and test set; metric/projector change
    if ridge_sweep:
        sweep = []
        for ratio in RIDGE_GRID:
            Gm, _ = metric(fit.G, ratio)
            Pm, Rm, cr = build_projector(fit.K, Gm, c)
            Xpm = rollout(fit, propagator(fit, Pm), Xte[:, 0])
            sweep.append(dict(ratio=ratio, r1=wnorm(Pm @ fit.K @ Rm, Gm), r2=wnorm(Rm @ fit.K @ Pm, Gm),
                              candidate_rank=cr,
                              tau_delta=gap(Xpm[:, -1, 0], Gte) - gap(Xr_all[:, -1, 0], Gte),
                              school_ci=SchoolBoot.ci(boot.gap(Xpm[:, -1, 0], Gte)
                                                      - boot.gap(Xr_all[:, -1, 0], Gte))))
        info["ridge_sweep"] = sweep
    info["seconds"] = time.time() - t_start

    os.makedirs(os.path.join(outdir, name), exist_ok=True)
    with open(os.path.join(outdir, name, "results.json"), "w") as f:
        json.dump(info, f, indent=2, default=float)
    np.savez(os.path.join(outdir, name, "arrays.npz"), trs=trs, res_raw=rr, res_proj=rp,
             norm_raw=n0, norm_proj=n1, ned=ned, Gte=Gte, kT=kT,
             **{f"pred_{m}": v for m, v in pred.items()},
             per_step_obs=np.array(per_step["observed"]),
             per_step_raw=np.array(per_step["edmd_raw"]),
             per_step_proj=np.array(per_step["edmd_proj"]))
    return info


# ---------------------------------------------------------------------------
# Split-level summary of the primary (school-disjoint) analysis
# ---------------------------------------------------------------------------
def summarise(infos):
    get = {
        "tau_delta": lambda r: r["free_run"]["tau_delta"],
        "tau_k": lambda r: r["free_run"]["tau_k"],
        "observed_gap": lambda r: r["replay"]["terminal_gap_observed"],
        "replay_gap_raw": lambda r: r["replay"]["models"]["edmd_raw"]["terminal_gap_pred"],
        "replay_gap_proj": lambda r: r["replay"]["models"]["edmd_proj"]["terminal_gap_pred"],
        "replay_gap_groupcal": lambda r: r["replay"]["models"]["edmd_groupcal"]["terminal_gap_pred"],
        "trs_mean": lambda r: r["trs"]["mean"],
        "encoded_residual_proj": lambda r: r["trs"]["encoded_residual_proj_mean"],
        "p1_train": lambda r: r["p1_train"], "p1_test": lambda r: r["p1_test"],
    }
    for m in infos[0]["replay"]["models"]:
        for h in HORIZONS:
            get[f"mse_k_h{h}_{m}"] = (lambda mm, hh: lambda r: r["replay"]["models"][mm]["mse"][str(hh)]["k_t"])(m, h)
    out = {}
    for k, f in get.items():
        v = np.array([f(r) for r in infos])
        m, lo, hi = tcI(v)
        out[k] = dict(values=v.tolist(), mean=m, t_ci=[lo, hi], min=float(v.min()), max=float(v.max()))
    out["n_splits_tau_ci_excluding_zero"] = int(sum(
        r["free_run"]["tau_delta_school_ci"][1] < 0 or r["free_run"]["tau_delta_school_ci"][0] > 0 for r in infos))
    out["n_splits_obs_gap_ci_excluding_zero"] = int(sum(
        r["replay"]["terminal_gap_observed_school_ci"][0] > 0 for r in infos))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="assistments_features_time.csv")
    ap.add_argument("--school-file", default=None)
    ap.add_argument("--out", default="out_time")
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    df = keep_full_trajectories(load_features(args.features, args.school_file))
    summary = {"features": os.path.basename(args.features),
               "n_students_full_traj": int(df.user_id.nunique()),
               "n_schools": int(df.school_id.nunique())}
    print(summary, flush=True)

    # Primary: five school-disjoint splits, train-only strata
    prim = [run(df, f"school_disjoint_{j}", args.seed + 200 + j, "school", "train", outdir=args.out)
            for j in range(5)]
    summary["school_disjoint"] = summarise(prim)
    # Secondary: random student split, train-only strata; five independently
    # seeded re-fits on the same corpus and three disjoint 5,000-student subsamples
    sec = [run(df, f"student_split_{j}", args.seed + 100 * (j > 0) + j, "student", "train",
               ridge_sweep=(j == 0), outdir=args.out) for j in range(5)]
    summary["student_split"] = summarise(sec)
    rng = np.random.default_rng(args.seed + 1)
    ids = rng.permutation(df["user_id"].unique())
    sub = [run(df, f"subsample_{j}", args.seed + 10 + j, "student", "train",
               student_ids=ids[j * 5000:(j + 1) * 5000], ridge_sweep=False, outdir=args.out)
           for j in range(3)]
    summary["subsample"] = summarise(sub)
    # Sensitivity: historical full-corpus label (uses validation/test outcomes)
    run(df, "student_split_full_strata", args.seed, "student", "full", outdir=args.out)
    # Prospective labels: new schools labelled from other students of the school
    oth = [run(df, f"school_disjoint_otherlabel_{j}", args.seed + 200 + j, "school", "other",
               ridge_sweep=False, outdir=args.out) for j in range(5)]
    summary["school_disjoint_otherlabel"] = summarise(oth)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: (v["mean"], v["t_ci"]) for k, v in summary["school_disjoint"].items()
                      if isinstance(v, dict) and not k.startswith("mse")}, indent=1))


if __name__ == "__main__":
    main()
