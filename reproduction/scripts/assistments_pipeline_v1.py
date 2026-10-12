"""
Clean, single-file ASSISTments 2012-2013 pipeline for the revised manuscript
"Mathematical Foundations of Operator Compression for Fairness in GenAI
Tutoring Systems".

It replaces the exploratory notebook cells.  Every quantity reported in the
ASSISTments sections of the manuscript is produced here and written to
out/<run_name>/results.json.

Design choices (each one answers a reviewer point):

* Group variable.  The generic group label is G in {0,1}.  In the features
  file it is the upstream column `S`, which is an *endogenous school-
  performance stratum* (school mean correctness above / below the median).
  It is never called a sensitive attribute here.
* Order of operations.  The student split is made first.  Everything that is
  estimated (dictionary scaling, skill means used in d_t, EDMD fit, metric
  Gamma, contrast, projector, decoder) is estimated on the training students
  only and then frozen before any validation/test evaluation.
  The upstream stratum label in the features file was computed on the full
  corpus before our split; with `--school-file` the stratum is recomputed
  from training students only (and, for the school-disjoint split, from a
  label window that never overlaps the evaluated steps).
* Cohorts.  Evaluation cohorts are *disjoint* in students.  Pipeline-level
  replication uses disjoint student subsamples, each with its own fit.
* Metric.  Gamma = G + eta I is estimated once per fit from the training
  Gram matrix and then held fixed for every evaluation of that fit.
* TIB.  Uses Definition 1 with the population (divisor n) standard deviation,
  exact zero-vector handling, and reports the counts of zero-zero and
  zero-nonzero pairs.  The dimensionally consistent encoded residual
  ||R_T (z_i - zbar)||_{I_T (x) Gamma^{-1}} is computed with
  R_T = I_T (x) R^T acting on stacked encoded state trajectories in R^{TN}.
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

FEATURES = ["k_t", "e_t", "p_t", "d_t"]
N_TRANS = 25                 # transitions per student -> states x_0..x_25
RANK = 4
DEPTH = 3
RIDGE_RATIO = 1e-3
RIDGE_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
COHORT_SIZE = 150
N_COHORTS = 10
ZERO_TOL = 1e-12             # relative tolerance for "zero" centred vectors


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_features(path: str, school_file: str | None = None,
                  time_file: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    # >= 20 interactions, as in the upstream preprocessing (18,366 students)
    cnt = df["user_id"].value_counts()
    df = df[df["user_id"].isin(cnt[cnt >= 20].index)].copy()
    if time_file is not None and os.path.exists(time_file):
        tm = pd.read_csv(time_file)
        tm["start_time"] = pd.to_datetime(tm["start_time"], errors="coerce")
        tm = tm.groupby(["user_id", "problem_id"], as_index=False)["start_time"].min()
        df = df.merge(tm, on=["user_id", "problem_id"], how="left")
        df = df.sort_values(["user_id", "start_time", "problem_id"], kind="stable")
    # The upstream d_t is (correct - full-corpus skill mean).  Recover the
    # binary response and the skill key so that d_t can be rebuilt from
    # training students only.
    df["correct"] = (df["d_t"] > 0).astype(float)
    df["skill_key"] = (df["correct"] - df["d_t"]).round(10)
    df["G"] = df["S"].astype(int)
    df["step"] = df.groupby("user_id").cumcount()
    if school_file is not None and os.path.exists(school_file):
        sch = pd.read_csv(school_file).drop_duplicates("user_id")
        df = df.merge(sch[["user_id", "school_id"]], on="user_id", how="left")
    return df


def keep_full_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Students with at least N_TRANS+1 recorded steps (all get 25 transitions)."""
    n = df.groupby("user_id")["step"].transform("max") + 1
    return df[n >= N_TRANS + 1].copy()


def split_students(ids: np.ndarray, rng: np.random.Generator):
    ids = rng.permutation(ids)
    n = len(ids)
    return ids[: int(0.6 * n)], ids[int(0.6 * n): int(0.8 * n)], ids[int(0.8 * n):]


def split_schools(df: pd.DataFrame, rng: np.random.Generator):
    """School-disjoint 60/20/20 split (by student counts)."""
    sch = df.groupby("school_id")["user_id"].nunique()
    order = rng.permutation(sch.index.values)
    cum = np.cumsum(sch.loc[order].values) / sch.sum()
    tr_s = order[cum <= 0.6]
    va_s = order[(cum > 0.6) & (cum <= 0.8)]
    te_s = order[cum > 0.8]
    users = df.drop_duplicates("user_id").set_index("user_id")["school_id"]
    f = lambda s: users.index[users.isin(s)].values
    return f(tr_s), f(va_s), f(te_s), (tr_s, va_s, te_s)


def rebuild_dt(df: pd.DataFrame, train_ids) -> pd.DataFrame:
    """d_t = correct - training-only skill mean, clipped to [-1, 1]."""
    ev = df["step"] <= N_TRANS
    tr = df[df["user_id"].isin(train_ids) & ev]
    m = tr.groupby("skill_key")["correct"].mean()
    glob = tr["correct"].mean()
    df = df.copy()
    df["d_t"] = np.clip(df["correct"] - df["skill_key"].map(m).fillna(glob), -1, 1)
    return df


def rebuild_school_strata(df, train_ids, school_disjoint: bool):
    """Train-only school-performance strata from a label window (steps > 25)
    that never overlaps the evaluated steps 0..25."""
    lab = df[df["step"] > N_TRANS]
    tr_lab = lab[lab["user_id"].isin(train_ids)]
    tr_means = tr_lab.groupby("school_id")["correct"].mean()
    thr = float(np.median(tr_means.values))
    if school_disjoint:
        # held-out schools: mean from their own label window (never evaluated)
        all_means = lab.groupby("school_id")["correct"].mean()
        means = all_means.copy()
        means.loc[tr_means.index] = tr_means
    else:
        means = tr_means
    g = (means > thr).astype(int)
    df = df.copy()
    df["G"] = df["school_id"].map(g)
    df = df[df["G"].notna()].copy()
    df["G"] = df["G"].astype(int)
    return df, thr


def trajectories(df: pd.DataFrame, ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = df[df["user_id"].isin(ids) & (df["step"] <= N_TRANS)]
    sub = sub.sort_values(["user_id", "step"])
    uid = sub["user_id"].values.reshape(-1, N_TRANS + 1)[:, 0]
    X = sub[FEATURES].values.reshape(-1, N_TRANS + 1, len(FEATURES))
    G = sub["G"].values.reshape(-1, N_TRANS + 1)[:, 0]
    return uid, X.astype(float), G.astype(int)


# ---------------------------------------------------------------------------
# Dictionary, EDMD, metric, contrast, projector
# ---------------------------------------------------------------------------
def lift(X: np.ndarray) -> np.ndarray:
    """Psi_A(x) = (k,e,p,d,k^2,e^2,p^2,d^2,ke,kp,kd,ep,ed,pd)."""
    k, e, p, d = (X[..., i] for i in range(4))
    cols = [k, e, p, d, k * k, e * e, p * p, d * d, k * e, k * p, k * d, e * p, e * d, p * d]
    return np.stack(cols, axis=-1)


def sym_sqrt(M):
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


@dataclass
class Fit:
    mu: np.ndarray
    sd: np.ndarray
    A: np.ndarray          # value propagator  z+ = A z + a
    a: np.ndarray
    K: np.ndarray          # coefficient operator K = A^T
    G: np.ndarray
    C: np.ndarray          # decoder x = C z + c
    c: np.ndarray
    num_rank: int


def fit_edmd(Xtr: np.ndarray) -> Fit:
    Psi = lift(Xtr)                                    # (n, T+1, N)
    cur = Psi[:, :-1].reshape(-1, Psi.shape[-1])
    nxt = Psi[:, 1:].reshape(-1, Psi.shape[-1])
    mu = cur.mean(0)
    sd = cur.std(0) + 1e-12
    Z = (cur - mu) / sd
    Zp = (nxt - mu) / sd
    M = Z.shape[0]
    Zaug = np.hstack([Z, np.ones((M, 1))])
    coef, *_ = np.linalg.lstsq(Zaug, Zp, rcond=None)   # min-norm LS
    A = coef[:-1].T
    a = coef[-1]
    G = Z.T @ Z / M
    xs = Xtr[:, :-1].reshape(-1, 4)
    dec, *_ = np.linalg.lstsq(Zaug, xs, rcond=None)
    num_rank = int(np.linalg.matrix_rank(G, tol=1e-10 * np.linalg.eigvalsh(G).max()))
    return Fit(mu, sd, A, a, A.T, G, dec[:-1].T, dec[-1], num_rank)


def encode(fit: Fit, X):
    return (lift(X) - fit.mu) / fit.sd


def decode(fit: Fit, Z):
    return Z @ fit.C.T + fit.c


def metric(G, ratio):
    eta = ratio * np.linalg.eigvalsh(G).max()
    return G + eta * np.eye(G.shape[0]), eta


def contrast(fit: Fit, Xtr, Gtr):
    Z = encode(fit, Xtr[:, :-1]).reshape(-1, fit.A.shape[0])
    g = np.repeat(Gtr, N_TRANS)
    p1 = g.mean()
    p0 = 1 - p1
    y = (g - p1) / (p0 * p1)
    c, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return c, float(p1)


def build_projector(K, Gamma, c, rank=RANK, depth=DEPTH, cutoff=1e-8):
    Gh, Gih = sym_sqrt(Gamma)
    W = Gh @ K @ Gih
    b = Gh @ c
    seed = b / np.linalg.norm(b)
    words = [seed]
    frontier = [seed]
    for _ in range(depth):
        new = []
        for v in frontier:
            new += [W @ v, W.T @ v]
        words += new
        frontier = new
    Cand = np.stack(words[1:], axis=1)
    Cand = Cand - np.outer(seed, seed @ Cand)
    U, s, _ = np.linalg.svd(Cand, full_matrices=False)
    keep = s > cutoff * s.max()
    cand_rank = int(keep.sum())
    Qw = np.column_stack([seed, U[:, : min(rank - 1, cand_rank)]])
    Qw, _ = np.linalg.qr(Qw)
    Q = Gih @ Qw                      # Q^T Gamma Q = I
    R = Q @ Q.T @ Gamma
    P = np.eye(K.shape[0]) - R
    return P, R, cand_rank + 1


def wnorm(L, Gamma, ord="fro"):
    Gh, Gih = sym_sqrt(Gamma)
    return float(np.linalg.norm(Gh @ L @ Gih, ord))


# ---------------------------------------------------------------------------
# Surrogates and evaluation
# ---------------------------------------------------------------------------
def propagators(fit: Fit, P):
    A_P = P.T @ fit.A @ P.T
    a_P = P.T @ fit.a
    return (fit.A, fit.a), (A_P, a_P)


def rollout(fit: Fit, prop, X0, T=N_TRANS):
    A, a = prop
    z = encode(fit, X0)
    Zs = []
    for _ in range(T):
        z = z @ A.T + a
        Zs.append(z)
    Zs = np.stack(Zs, axis=1)                # (n, T, N) latent path
    return decode(fit, Zs), Zs


def one_step(fit: Fit, prop, X):
    A, a = prop
    z = encode(fit, X[:, :-1])
    return decode(fit, z @ A.T + a)          # predictions for steps 1..T


def gap(v, g):
    return float(v[g == 1].mean() - v[g == 0].mean())


def disjoint_cohorts(G, rng, n_cohorts=N_COHORTS, size=COHORT_SIZE):
    i0 = rng.permutation(np.where(G == 0)[0])
    i1 = rng.permutation(np.where(G == 1)[0])
    h = size // 2
    n_max = min(len(i0), len(i1)) // h
    n = min(n_cohorts, n_max)
    return [np.concatenate([i0[j * h:(j + 1) * h], i1[j * h:(j + 1) * h]]) for j in range(n)]


def tcI(x, level=0.95):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(m), float(m - h), float(m + h)


def dcos0(a, b, tol_a, tol_b):
    za = np.linalg.norm(a) <= tol_a
    zb = np.linalg.norm(b) <= tol_b
    if za and zb:
        return 0.0, "zero-zero"
    if za or zb:
        return 1.0, "zero-nonzero"
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), "nonzero"


def tib_cohort(traj0, traj1):
    """traj*: (n, T, d) stacked physical-state predictions for one cohort."""
    n = traj0.shape[0]
    g0 = traj0.reshape(n, -1)
    g1 = traj1.reshape(n, -1)
    g0 = g0 - g0.mean(0)
    g1 = g1 - g1.mean(0)
    t0 = ZERO_TOL * max(1.0, np.abs(traj0).max()) * np.sqrt(g0.shape[1])
    t1 = ZERO_TOL * max(1.0, np.abs(traj1).max()) * np.sqrt(g1.shape[1])
    vals, kinds = [], []
    for i in range(n):
        v, k = dcos0(g0[i], g1[i], t0, t1)
        vals.append(v)
        kinds.append(k)
    return np.array(vals), kinds


def encoded_residual(fit: Fit, Xtraj, R, Gamma):
    """||R_T (z_i - zbar)||_{I_T (x) Gamma^{-1}} / ||z_i - zbar||_{...},
    z_i = vec(Psi(x_{i,1}),...,Psi(x_{i,T})) in R^{TN} (standardised Psi),
    R_T = I_T (x) R^T (value-space projector, Gamma^{-1}-orthogonal)."""
    Z = encode(fit, Xtraj)                         # (n, T, N)
    Zc = Z - Z.mean(0, keepdims=True)
    Gi = np.linalg.inv(Gamma)
    RZ = Zc @ R                                    # row-vector form of R^T z
    num = np.sqrt(np.einsum("ntj,jk,ntk->n", RZ, Gi, RZ))
    den = np.sqrt(np.einsum("ntj,jk,ntk->n", Zc, Gi, Zc))
    return num, num / np.maximum(den, 1e-300)


# ---------------------------------------------------------------------------
# One complete run
# ---------------------------------------------------------------------------
def run(df_all: pd.DataFrame, name: str, seed: int, student_ids=None,
        school_disjoint=False, recompute_strata=False, ridge_sweep=True,
        outdir="out", save_arrays=False):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    df = df_all if student_ids is None else df_all[df_all["user_id"].isin(student_ids)]
    info = {"name": name, "seed": seed}
    if school_disjoint:
        tr, va, te, schools = split_schools(df, rng)
        info["n_schools"] = [int(len(s)) for s in schools]
    else:
        tr, va, te = split_students(df["user_id"].unique(), rng)
    if recompute_strata:
        df, thr = rebuild_school_strata(df, tr, school_disjoint)
        info["stratum_threshold_train_only"] = thr
    df = rebuild_dt(df, tr)
    _, Xtr, Gtr = trajectories(df, tr)
    uid_te, Xte, Gte = trajectories(df, te)
    info.update(n_students=int(df["user_id"].nunique()), n_train=len(Xtr),
                n_val=int(len(va)), n_test=len(Xte),
                n_train_transitions=int(len(Xtr) * N_TRANS),
                n_test_transitions=int(len(Xte) * N_TRANS),
                p1_train=float(Gtr.mean()), p1_test=float(Gte.mean()))
    if "school_id" in df:
        s_tr = set(df[df.user_id.isin(tr)].school_id.dropna())
        s_te = set(df[df.user_id.isin(te)].school_id.dropna())
        info["schools_shared_train_test"] = len(s_tr & s_te)
        info["schools_test"] = len(s_te)

    fit = fit_edmd(Xtr)
    Gamma, eta = metric(fit.G, RIDGE_RATIO)
    c, p1 = contrast(fit, Xtr, Gtr)
    P, R, cand_rank = build_projector(fit.K, Gamma, c)
    raw, proj = propagators(fit, P)
    F = P @ fit.K @ P
    info["diagnostics"] = dict(
        dictionary_dim=int(fit.K.shape[0]), numerical_rank_G=fit.num_rank,
        eta=float(eta), K_norm_2=float(np.linalg.norm(fit.K, 2)),
        K_norm_Gamma2=wnorm(fit.K, Gamma, 2), F_norm_Gamma2=wnorm(F, Gamma, 2),
        spectral_radius_A=float(np.abs(np.linalg.eigvals(fit.A)).max()),
        spectral_radius_AP=float(np.abs(np.linalg.eigvals(proj[0])).max()),
        r1=wnorm(P @ fit.K @ R, Gamma), r2=wnorm(R @ fit.K @ P, Gamma),
        RKR=wnorm(R @ fit.K @ R, Gamma),
        eps_q=float(np.sqrt(max((P @ c) @ Gamma @ (P @ c), 0))),
        c_norm_Gamma=float(np.sqrt(c @ Gamma @ c)),
        idempotency_err=float(np.abs(P @ P - P).max()),
        applied_rank=RANK, candidate_rank=cand_rank)

    # ---- (1) observed-state-feedback one-step replay on all test students
    Yr = one_step(fit, raw, Xte)
    Yp = one_step(fit, proj, Xte)
    obs = Xte[:, 1:]
    mse_r = ((Yr - obs) ** 2).reshape(-1, 4).mean(0)
    mse_p = ((Yp - obs) ** 2).reshape(-1, 4).mean(0)
    per_step = {
        "observed": [gap(obs[:, t, 0], Gte) for t in range(N_TRANS)],
        "raw": [gap(Yr[:, t, 0], Gte) for t in range(N_TRANS)],
        "projected": [gap(Yp[:, t, 0], Gte) for t in range(N_TRANS)],
    }
    info["replay"] = dict(
        mse_raw=dict(zip(FEATURES, map(float, mse_r))),
        mse_proj=dict(zip(FEATURES, map(float, mse_p))),
        terminal_gap_observed=per_step["observed"][-1],
        terminal_gap_raw=per_step["raw"][-1],
        terminal_gap_proj=per_step["projected"][-1],
        per_step_gap=per_step)
    # student-bootstrap interval for the observed terminal gap
    B = 2000
    bs = []
    i0, i1 = np.where(Gte == 0)[0], np.where(Gte == 1)[0]
    kT = obs[:, -1, 0]
    for _ in range(B):
        bs.append(kT[rng.choice(i1, len(i1))].mean() - kT[rng.choice(i0, len(i0))].mean())
    info["replay"]["terminal_gap_observed_ci"] = [float(np.percentile(bs, 2.5)),
                                                  float(np.percentile(bs, 97.5))]

    # ---- (2) disjoint cohorts, free-running surrogate rollouts from x_0
    cohorts = disjoint_cohorts(Gte, rng)
    rows = []
    tib_all, kinds_all, res_raw_all, res_proj_all, tib_cohort_means = [], [], [], [], []
    tib_feat = {}
    for j, idx in enumerate(cohorts):
        X0 = Xte[idx, 0]
        g = Gte[idx]
        Xr, _ = rollout(fit, raw, X0)
        Xp, _ = rollout(fit, proj, X0)
        row = dict(cohort=j,
                   obs_gap=gap(Xte[idx, -1, 0], g),
                   raw_gap=gap(Xr[:, -1, 0], g), proj_gap=gap(Xp[:, -1, 0], g),
                   raw_k=float(Xr[:, -1, 0].mean()), proj_k=float(Xp[:, -1, 0].mean()),
                   raw_g0=float(Xr[g == 0, -1, 0].mean()), raw_g1=float(Xr[g == 1, -1, 0].mean()),
                   proj_g0=float(Xp[g == 0, -1, 0].mean()), proj_g1=float(Xp[g == 1, -1, 0].mean()))
        row["tau_delta"] = row["proj_gap"] - row["raw_gap"]
        row["tau_k"] = row["proj_k"] - row["raw_k"]
        rows.append(row)
        tv, kinds = tib_cohort(Xr, Xp)
        for fi in range(4):
            tf, _ = tib_cohort(Xr[:, :, [fi]], Xp[:, :, [fi]])
            tib_feat.setdefault(FEATURES[fi], []).append(tf)
        tib_all.append(tv)
        kinds_all += kinds
        tib_cohort_means.append(tv.mean())
        _, rr = encoded_residual(fit, Xr, R, Gamma)
        _, rp = encoded_residual(fit, Xp, R, Gamma)
        res_raw_all.append(rr)
        res_proj_all.append(rp)
    tau = [r["tau_delta"] for r in rows]
    tk = [r["tau_k"] for r in rows]
    tib = np.concatenate(tib_all)
    rr = np.concatenate(res_raw_all)
    rp = np.concatenate(res_proj_all)
    kinds = pd.Series(kinds_all).value_counts().to_dict()
    info["cohorts"] = dict(
        n_cohorts=len(cohorts), cohort_size=COHORT_SIZE, disjoint=True, rows=rows,
        tau_delta=tcI(tau), tau_delta_sd=float(np.std(tau, ddof=1)),
        tau_delta_n_negative=int(np.sum(np.array(tau) < 0)),
        tau_k=tcI(tk),
        obs_gap=tcI([r["obs_gap"] for r in rows]),
        raw_gap=tcI([r["raw_gap"] for r in rows]),
        proj_gap=tcI([r["proj_gap"] for r in rows]),
        d_g0=float(np.mean([r["proj_g0"] - r["raw_g0"] for r in rows])),
        d_g1=float(np.mean([r["proj_g1"] - r["raw_g1"] for r in rows])),
        raw_g0=float(np.mean([r["raw_g0"] for r in rows])),
        raw_g1=float(np.mean([r["raw_g1"] for r in rows])),
        proj_g0=float(np.mean([r["proj_g0"] for r in rows])),
        proj_g1=float(np.mean([r["proj_g1"] for r in rows])))
    rho = stats.spearmanr(tib, rr - rp).correlation if len(tib) > 2 else np.nan
    info["tib"] = dict(
        n=int(len(tib)), mean=float(tib.mean()), median=float(np.median(tib)),
        sd_pop_n=float(tib.std(ddof=0)), sd_sample_n1=float(tib.std(ddof=1)),
        cohort_mean_ci=tcI(tib_cohort_means),
        pairs=dict(zero_zero=int(kinds.get("zero-zero", 0)),
                   zero_nonzero=int(kinds.get("zero-nonzero", 0)),
                   nonzero=int(kinds.get("nonzero", 0))),
        n_below_0p05=int((tib < 0.05).sum()), n_above_0p5=int((tib >= 0.5).sum()),
        n_above_1=int((tib > 1).sum()),
        per_feature_mean={k: float(np.concatenate(v).mean()) for k, v in tib_feat.items()},
        encoded_residual_raw_mean=float(rr.mean()),
        encoded_residual_proj_mean=float(rp.mean()),
        encoded_residual_proj_max=float(rp.max()),
        spearman_tib_vs_residual_drop=float(rho))

    # ---- (3) ridge sweep: same fit, same cohorts; metric/projector change
    if ridge_sweep:
        sweep = []
        for ratio in RIDGE_GRID:
            Gm, _ = metric(fit.G, ratio)
            Pm, Rm, cr = build_projector(fit.K, Gm, c)
            _, pm = propagators(fit, Pm)
            taus = []
            for idx in cohorts:
                X0 = Xte[idx, 0]
                g = Gte[idx]
                Xr, _ = rollout(fit, raw, X0)
                Xp, _ = rollout(fit, pm, X0)
                taus.append(gap(Xp[:, -1, 0], g) - gap(Xr[:, -1, 0], g))
            m, lo, hi = tcI(taus)
            sweep.append(dict(ratio=ratio, r1=wnorm(Pm @ fit.K @ Rm, Gm),
                              r2=wnorm(Rm @ fit.K @ Pm, Gm), candidate_rank=cr,
                              tau_mean=m, ci=[lo, hi],
                              n_negative=int(np.sum(np.array(taus) < 0))))
        info["ridge_sweep"] = sweep
    info["seconds"] = time.time() - t0

    os.makedirs(os.path.join(outdir, name), exist_ok=True)
    with open(os.path.join(outdir, name, "results.json"), "w") as f:
        json.dump(info, f, indent=2, default=float)
    if save_arrays:
        np.savez(os.path.join(outdir, name, "arrays.npz"), tib=tib, res_raw=rr,
                 res_proj=rp, obs=obs, Yr=Yr, Yp=Yp, Gte=Gte)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="assistments_features_with_pid.csv")
    ap.add_argument("--school-file", default="user_school.csv")
    ap.add_argument("--time-file", default="user_problem_time.csv")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    df = keep_full_trajectories(load_features(args.features, args.school_file, args.time_file))
    have_school = "school_id" in df.columns
    summary = {"have_school_id": have_school,
               "chronological_order": os.path.exists(args.time_file),
               "n_students_full_traj": int(df.user_id.nunique())}
    print(summary)

    # Main analysis: full corpus of complete trajectories, random student split
    main_info = run(df, "full_corpus", args.seed, outdir=args.out, save_arrays=True)
    summary["full_corpus"] = main_info["name"]

    # Pipeline-level replication: three disjoint 5,000-student subsamples
    rng = np.random.default_rng(args.seed + 1)
    ids = rng.permutation(df["user_id"].unique())
    for j in range(3):
        run(df, f"subsample_{j}", args.seed + 10 + j, student_ids=ids[j * 5000:(j + 1) * 5000],
            ridge_sweep=False, outdir=args.out)
    # Seed replication on the full corpus (different split/cohorts)
    for j in range(5):
        run(df, f"full_seed_{j}", args.seed + 100 + j, ridge_sweep=False, outdir=args.out)

    if have_school:
        run(df, "train_only_strata", args.seed, recompute_strata=True, outdir=args.out,
            save_arrays=True)
        for j in range(5):
            run(df, f"school_disjoint_{j}", args.seed + 200 + j, school_disjoint=True,
                recompute_strata=True, ridge_sweep=False, outdir=args.out)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
