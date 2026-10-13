"""
Rank-matched controls for the group projection (supervisor's request, October 2026).

Question: is the loss of the projected model caused by removing GROUP information,
or by compressing the model by the same rank?  Every arm below uses the same data
split, the same fitted EDMD model, the same metric Gamma and the same rank; only
the removed subspace changes.

Arms (removed subspace span(Q) in coefficient space; P = I - Q (Q'GQ)^-1 Q'G with
G = Gamma, applied as in the paper: A_P = P'AP', B_P = P'B, a_P = P'a):
  group        published projection (contrast seed + Krylov words in K, K*)
  random       Q uniformly random in whitened coordinates (n_draws draws)
  permuted     same construction as `group`, contrast fitted on student-permuted
               group labels (n_draws draws): the same algorithm without group information
  pca_top      the r directions of largest variance of the standardised features
  pca_bottom   the r directions of smallest variance (the cheapest unsupervised compression)
  mindist      the r whitened directions least used by the operator
               (smallest eigenvalues of W'W + WW', W = Gamma^1/2 K Gamma^-1/2)
  leace_op     LEACE eraser for the binary group (Belrose et al. 2023), applied to
               the operator: E = I - S c c'/(c'Sc), S = feature covariance (rank 1)
  leace_refit  LEACE as usually deployed: erase the features, then refit the
               dynamics (and decoder) on the erased features (rank 1)
Rank-1 versions of group/random/permuted/pca/mindist are also run, so that LEACE
is compared at its own rank.

Part A (prediction): ASSISTments (5 school-disjoint splits), Eedi (5 class-disjoint
splits) and OULAD (imd, 5 student splits), reusing assistments_pipeline unchanged.
Part B (policy): the tutoring simulator, MPC on each compressed model, 10 replicates,
group strength gamma in {0, 0.5, 1, 2}.

    python rank_matched_controls.py --part A --dataset assistments
    python rank_matched_controls.py --part B --sim ../../../Final_Folder_on_DR
Outputs (aggregates only, no per-student arrays): out_rank_matched/<dataset>.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
from scipy import stats

import assistments_pipeline as ap

N_DRAWS = 50
RANKS = (1, 4)
HOR = ap.HORIZONS
OUT = "out_rank_matched"


# ---------------------------------------------------------------------------
# Projectors
# ---------------------------------------------------------------------------
def removal_projector(Q, Gamma):
    """Gamma-orthogonal projector that removes span(Q) (coefficient space)."""
    M = Q.T @ Gamma @ Q
    return np.eye(len(Gamma)) - Q @ np.linalg.solve(M, Q.T @ Gamma)


def sym_sqrt(M):
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


def leace_projector(c, S):
    """LEACE for a binary concept: value-space eraser E = I - S c c'/(c'Sc).
    Returned as the coefficient projector P = E' so that it plugs into P'AP'."""
    Sc = S @ c
    E = np.eye(len(c)) - np.outer(Sc, c) / (c @ Sc)
    return E.T


def control_projectors(K, Gamma, S, c, rank, rng, n_draws, permuted_contrasts, build):
    """All arms at one rank.  `build(K, Gamma, c, rank)` is the published construction."""
    N = len(Gamma)
    Gh, Gih = sym_sqrt(Gamma)
    arms = {"group": build(K, Gamma, c, rank)}
    w, V = np.linalg.eigh(S)                       # ascending variance
    arms["pca_top"] = removal_projector(V[:, -rank:], Gamma)
    arms["pca_bottom"] = removal_projector(V[:, :rank], Gamma)
    W = Gh @ K @ Gih
    _, U = np.linalg.eigh(W.T @ W + W @ W.T)
    arms["mindist"] = removal_projector(Gih @ U[:, :rank], Gamma)
    arms["random"] = [removal_projector(Gih @ np.linalg.qr(rng.standard_normal((N, rank)))[0], Gamma)
                      for _ in range(n_draws)]
    arms["permuted"] = [build(K, Gamma, cp, rank) for cp in permuted_contrasts]
    return arms


def distortion(K, P, Gamma):
    Gh, Gih = sym_sqrt(Gamma)
    n = lambda L: np.linalg.norm(Gh @ L @ Gih)
    return float(n(K - P @ K @ P) / n(K))


def contrast_left(P, c, Gamma):
    v = P @ c
    return float(np.sqrt(max(v @ Gamma @ v, 0) / (c @ Gamma @ c)))


# ---------------------------------------------------------------------------
# Part A: prediction on observational data
# ---------------------------------------------------------------------------
class LinPredictor:
    """z -> A z + a in the encoded space of `fit`, decoded by C z + c."""

    def __init__(self, fit, A, a, C=None, c=None):
        self.fit, self.A, self.a = fit, A, a
        self.C = fit.C if C is None else C
        self.c = fit.c if c is None else c

    def predict(self, X0, h, g=None):
        z = ap.encode(self.fit, X0)
        for _ in range(h):
            z = z @ self.A.T + self.a
        return z @ self.C.T + self.c


def prepare(df_all, seed, split, strata, student_ids=None):
    """The first lines of assistments_pipeline.run, unchanged (same splits)."""
    rng = np.random.default_rng(seed)
    df = df_all if student_ids is None else df_all[df_all["user_id"].isin(student_ids)]
    tr, va, te = ap.split_schools(df, rng) if split == "school" else ap.split_students(df["user_id"].unique(), rng)
    if strata == "train":
        df, _ = ap.rebuild_school_strata(df, tr, split == "school")
    elif strata == "other":
        df, _, va, te, _ = ap.rebuild_school_strata_other(df, tr, va, te, rng)
    df = ap.rebuild_dt(df, tr)
    _, Xtr, Gtr, Str = ap.trajectories(df, tr)
    _, Xte, Gte, Ste = ap.trajectories(df, te)
    return rng, Xtr, Gtr, Str, Xte, Gte, Ste


def calib(pred1, y, g):
    out = {}
    for gv in (0, 1):
        p, t = pred1[g == gv].ravel(), y[g == gv].ravel()
        out[gv] = (float(t.mean() - p.mean()), float(np.polyfit(p, t, 1)[0]))
    return out


def pred_metrics(model, Xte, Gte, raw_terminal):
    errs = {h: ap.horizon_errors(model, Xte, Gte, h) for h in HOR}
    n = len(Xte)
    p1 = model.predict(Xte[:, :-1].reshape(-1, 4), 1)[:, 0].reshape(n, -1)
    cal = calib(p1, Xte[:, 1:, 0], Gte[:, None].repeat(ap.N_TRANS, 1))
    kT = Xte[:, -1, 0]
    pk = model.predict(Xte[:, -2], 1)[:, 0]
    term = model.predict(Xte[:, 0], ap.N_TRANS)[:, 0]
    m = {f"mse_k_h{h}": float(errs[h][:, 0].mean()) for h in HOR}
    for h in (1, 10):
        for gv in (0, 1):
            m[f"mse_k_h{h}_g{gv}"] = float(errs[h][Gte == gv, 0].mean())
        m[f"mse_k_h{h}_worst_group"] = max(m[f"mse_k_h{h}_g0"], m[f"mse_k_h{h}_g1"])
    m["mse_all_h1"] = float(errs[1].mean())
    m["replay_gap_pred"] = ap.gap(pk, Gte)
    m["replay_gap_abs_error"] = abs(ap.gap(pk, Gte) - ap.gap(kT, Gte))
    m["free_run_gap"] = ap.gap(term, Gte)
    m["free_run_tau"] = ap.gap(term, Gte) - ap.gap(raw_terminal, Gte)
    for gv in (0, 1):
        m[f"citl_g{gv}"], m[f"slope_g{gv}"] = cal[gv]
    m["citl_gap_abs"] = abs(cal[1][0] - cal[0][0])
    return m, errs[1][:, 0]


def leace_refit(fit, Xtr, c):
    """Erase the group concept from the encoded features (LEACE), refit dynamics and decoder."""
    Z = ap.encode(fit, Xtr)
    cur, nxt = Z[:, :-1].reshape(-1, Z.shape[-1]), Z[:, 1:].reshape(-1, Z.shape[-1])
    S = cur.T @ cur / len(cur)
    E = leace_projector(c, S).T                    # value-space eraser
    ce, ne = cur @ E.T, nxt @ E.T
    aug = np.hstack([ce, np.ones((len(ce), 1))])
    coef, *_ = np.linalg.lstsq(aug, ne, rcond=None)
    dec, *_ = np.linalg.lstsq(aug, Xtr[:, :-1].reshape(-1, 4), rcond=None)
    A, a = coef[:-1].T, coef[-1]
    return LinPredictor(fit, A @ E, a, dec[:-1].T @ E, dec[-1]), E


def part_a_split(df, name, seed, split, strata, n_draws, student_ids=None):
    t0 = time.time()
    rng, Xtr, Gtr, Str, Xte, Gte, Ste = prepare(df, seed, split, strata, student_ids)
    fit = ap.fit_edmd(Xtr)
    Gamma, _ = ap.metric(fit.G, ap.RIDGE_RATIO)
    c, _ = ap.contrast(fit, Xtr, Gtr)
    S = fit.G
    crng = np.random.default_rng(seed + 7)
    perm_c = [ap.contrast(fit, Xtr, crng.permutation(Gtr))[0] for _ in range(n_draws)]
    build = lambda K, Gm, cc, r: ap.build_projector(K, Gm, cc, rank=r)[0]
    raw = LinPredictor(fit, fit.A, fit.a)
    raw_term = raw.predict(Xte[:, 0], ap.N_TRANS)[:, 0]
    res = {"name": name, "n_train": len(Xtr), "n_test": len(Xte), "dictionary_dim": int(len(Gamma)),
           "observed_terminal_gap": ap.gap(Xte[:, -1, 0], Gte), "arms": {}}
    m_raw, e_raw = pred_metrics(raw, Xte, Gte, raw_term)
    res["arms"]["raw"] = {"metrics": m_raw}
    boot = ap.SchoolBoot(Ste, rng, B=1000)
    per_student = {}

    def record(key, P_list, extra=None):
        Ms, E1 = [], []
        for P in P_list:
            mod = LinPredictor(fit, P.T @ fit.A @ P.T, P.T @ fit.a)
            m, e1 = pred_metrics(mod, Xte, Gte, raw_term)
            m["distortion"] = distortion(fit.K, P, Gamma)
            m["contrast_left"] = contrast_left(P, c, Gamma)
            Ms.append(m)
            E1.append(e1)
        agg = {k: float(np.mean([m[k] for m in Ms])) for k in Ms[0]}
        rec = {"metrics": agg, "n": len(Ms)}
        if len(Ms) > 1:
            rec["q025"] = {k: float(np.percentile([m[k] for m in Ms], 2.5)) for k in Ms[0]}
            rec["q975"] = {k: float(np.percentile([m[k] for m in Ms], 97.5)) for k in Ms[0]}
            rec["draws"] = {k: [m[k] for m in Ms] for k in ("mse_k_h1", "mse_k_h10", "replay_gap_pred",
                                                            "free_run_tau", "distortion", "mse_k_h1_worst_group")}
        if extra:
            rec["metrics"].update(extra)
        res["arms"][key] = rec
        per_student[key] = np.mean(E1, axis=0)

    for r in RANKS:
        arms = control_projectors(fit.K, Gamma, S, c, r, rng, n_draws, perm_c, build)
        for a, P in arms.items():
            record(f"{a}_r{r}", P if isinstance(P, list) else [P])
    record("leace_op_r1", [leace_projector(c, S)])
    mod, E = leace_refit(fit, Xtr, c)
    m, e1 = pred_metrics(mod, Xte, Gte, raw_term)
    res["arms"]["leace_refit_r1"] = {"metrics": m}
    per_student["leace_refit_r1"] = e1
    # paired h1 error differences against the group projection, school-cluster bootstrap
    res["paired_h1_vs_group_r4"] = {}
    for k, e in per_student.items():
        d = per_student["group_r4"] - e
        res["paired_h1_vs_group_r4"][k] = {"diff": float(d.mean()), "school_ci": ap.SchoolBoot.ci(boot.mean(d))}
    d = per_student["group_r4"] - e_raw
    res["paired_h1_vs_group_r4"]["raw"] = {"diff": float(d.mean()), "school_ci": ap.SchoolBoot.ci(boot.mean(d))}
    # where does the group projection fall in the null distributions?
    res["percentile_of_group_in_draws"] = {}
    for r in RANKS:
        g = res["arms"][f"group_r{r}"]["metrics"]
        for null in ("random", "permuted"):
            dr = res["arms"][f"{null}_r{r}"]["draws"]
            res["percentile_of_group_in_draws"][f"{null}_r{r}"] = {
                k: float(np.mean(np.array(dr[k]) <= g[k])) for k in dr}
    res["seconds"] = time.time() - t0
    print(f"{name}: {res['seconds']:.0f}s  h1 raw {m_raw['mse_k_h1']:.4f} group {res['arms']['group_r4']['metrics']['mse_k_h1']:.4f} "
          f"random {res['arms']['random_r4']['metrics']['mse_k_h1']:.4f} permuted {res['arms']['permuted_r4']['metrics']['mse_k_h1']:.4f} "
          f"pca_bottom {res['arms']['pca_bottom_r4']['metrics']['mse_k_h1']:.4f} mindist {res['arms']['mindist_r4']['metrics']['mse_k_h1']:.4f} "
          f"leace {res['arms']['leace_op_r1']['metrics']['mse_k_h1']:.4f}/{m['mse_k_h1']:.4f}", flush=True)
    return res


def across_splits(runs):
    """t intervals over splits of every arm metric and of (group_r4 - arm)."""
    keys = runs[0]["arms"]["raw"]["metrics"].keys()
    out = {}
    for arm in runs[0]["arms"]:
        out[arm] = {}
        for k in runs[0]["arms"][arm]["metrics"]:
            v = [r["arms"][arm]["metrics"][k] for r in runs]
            out[arm][k] = ap.tcI(v)
            if k in keys and arm != "group_r4":
                out[arm][f"group_r4_minus_this__{k}"] = ap.tcI([r["arms"]["group_r4"]["metrics"][k] - x
                                                                for r, x in zip(runs, v)])
    return out


def part_a(dataset, n_draws, seed=20260914):
    os.makedirs(OUT, exist_ok=True)
    root = os.environ.get("DATA_ROOT", ".")
    if dataset == "assistments":
        df = ap.keep_full_trajectories(ap.load_features("assistments_features_time.csv"))
        runs = [part_a_split(df, f"school_disjoint_{j}", seed + 200 + j, "school", "train", n_draws) for j in range(5)]
    elif dataset == "eedi":
        df = ap.keep_full_trajectories(ap.load_features(f"{root}/Datasets/Eedi_NeurIPS2020/eedi_features_time.csv"))
        runs = [part_a_split(df, f"class_disjoint_{j}", seed + 200 + j, "school", "given", n_draws) for j in range(5)]
    elif dataset.startswith("oulad"):
        att = dataset.split("_", 1)[1] if "_" in dataset else "imd"
        df = ap.keep_full_trajectories(ap.load_features(f"{root}/Datasets/OULAD/oulad_features_weekly.csv"))
        df["S"] = df[f"S_{att}"]
        df["G"] = df["S"].astype(int)
        runs = [part_a_split(df, f"{att}_student_split_{j}", seed + 300 + j, "student", "given", n_draws) for j in range(5)]
    summary = {"dataset": dataset, "n_draws": n_draws, "ranks": RANKS, "splits": runs, "across_splits": across_splits(runs)}
    json.dump(summary, open(os.path.join(OUT, f"{dataset}.json"), "w"), indent=1, default=float)
    return summary


# ---------------------------------------------------------------------------
# Part B: policy outcomes in the simulator
# ---------------------------------------------------------------------------
def part_b(sim_path, n_draws, gammas=(0.0, 0.5, 1.0, 2.0), seeds=range(10), rank=4, depth=3, beta=8.0, T=25):
    sys.path.insert(0, sim_path)
    import fa_edmd_mpc_v3_corrected as fa
    import synthetic_regimes as sr
    os.makedirs(OUT, exist_ok=True)

    def sim_build(K, Gm, cc, r):
        basis, _ = fa.reducing_basis(K, Gm, cc, r, depth)
        return np.eye(len(Gm)) - basis @ basis.T @ Gm

    def run_arm(model, init, noise, groups, projected=True):
        pol = fa.MPCPolicy(*model.dynamics(projected), model.decoder, model.decoder_offset, beta=beta)
        x, states, zs, us = init.copy(), [init.copy()], [], []
        for t in range(T):
            z = model.encode(x)
            u = pol(z)
            zs.append(z); us.append(u)
            x = fa.step(x, u, groups, noise[:, t])
            states.append(x.copy())
        S_, Z, U = np.stack(states, 1), np.stack(zs, 1), np.stack(us, 1)
        A, B, a = model.dynamics(projected)
        xhat = (Z @ A.T + U[..., None] * B + a) @ model.decoder.T + model.decoder_offset
        fc = fa.forecast_terminal(model, init, T, pol, projected)
        return dict(gap=float(fa.contrast(S_[:, -1, fa.D_IDX], groups)),
                    knowledge=float(S_[:, -1, 0].mean()),
                    k_g0=float(S_[groups == 0, -1, 0].mean()), k_g1=float(S_[groups == 1, -1, 0].mean()),
                    onestep_mse=float(((xhat - S_[:, 1:]) ** 2).mean()),
                    forecast_gap=float(fa.contrast(fc[:, fa.D_IDX], groups)),
                    forecast_gap_error=float(fa.contrast(fc[:, fa.D_IDX], groups) - fa.contrast(S_[:, -1, fa.D_IDX], groups)),
                    mean_abs_control=float(np.abs(U).mean()))

    results = {"gammas": list(gammas), "rank": rank, "depth": depth, "beta": beta, "n_draws": n_draws, "configs": []}
    for gamma in gammas:
        fa.step = sr.make_step(gamma)
        per_seed = []
        for seed in seeds:
            model = fa.fit_model(seed, rank, depth)
            init, noise, groups = fa.make_cohort(seed, horizon=T)
            states, controls, tg = fa.training_data(seed)
            lifted = (fa.psi_raw(states[:, :-1].reshape(-1, 4)) - model.mean) / model.scale
            lifted_next = (fa.psi_raw(states[:, 1:].reshape(-1, 4)) - model.mean) / model.scale
            labels = np.repeat(tg, T)
            rng = np.random.default_rng(1000 + seed)

            def contrast_of(lab):
                p = lab.mean()
                return np.linalg.lstsq(lifted, (lab - p) / (p * (1 - p)), rcond=None)[0]
            perm_c = [contrast_of(np.repeat(rng.permutation(tg), T)) for _ in range(n_draws)]
            K, Gm, c, S = model.propagation.T, model.metric, model.seed, model.gram
            out = {"raw": run_arm(model, init, noise, groups, projected=False)}
            for r in sorted({1, rank}):
                arms = control_projectors(K, Gm, S, c, r, rng, n_draws, perm_c, sim_build)
                for a, P in arms.items():
                    Ps = P if isinstance(P, list) else [P]
                    vals = []
                    for Pk in Ps:
                        m = run_arm(dataclasses.replace(model, projector=Pk), init, noise, groups)
                        m["distortion"] = distortion(K, Pk, Gm)
                        m["contrast_left"] = contrast_left(Pk, c, Gm)
                        vals.append(m)
                    out[f"{a}_r{r}"] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
                    if len(vals) > 1:
                        out[f"{a}_r{r}"]["_draws_abs_gap_change"] = [abs(v["gap"]) - abs(out["raw"]["gap"]) for v in vals]
                        out[f"{a}_r{r}"]["_draws_knowledge_change"] = [v["knowledge"] - out["raw"]["knowledge"] for v in vals]
            # LEACE on the operator, and LEACE erase-then-refit (with the control column)
            m = run_arm(dataclasses.replace(model, projector=leace_projector(c, S)), init, noise, groups)
            m["distortion"] = distortion(K, leace_projector(c, S), Gm)
            out["leace_op_r1"] = m
            E = leace_projector(c, S).T
            ce, ne = lifted @ E.T, lifted_next @ E.T
            reg = np.column_stack([ce, controls.reshape(-1), np.ones(len(ce))])
            coef = np.linalg.lstsq(reg, ne, rcond=None)[0]
            n_ = len(c)
            dec = np.linalg.lstsq(np.column_stack([ce, np.ones(len(ce))]), states[:, :-1].reshape(-1, 4), rcond=None)[0]
            refit = dataclasses.replace(model, propagation=coef[:n_].T @ E, control=coef[n_], offset=coef[-1],
                                        decoder=dec[:-1].T @ E, decoder_offset=dec[-1])
            out["leace_refit_r1"] = run_arm(refit, init, noise, groups, projected=False)
            per_seed.append(out)
            print(f"gamma={gamma} seed={seed}: |gap| raw {abs(out['raw']['gap']):.3f} group {abs(out[f'group_r{rank}']['gap']):.3f} "
                  f"random {abs(out[f'random_r{rank}']['gap']):.3f} permuted {abs(out[f'permuted_r{rank}']['gap']):.3f} "
                  f"pca_bottom {abs(out[f'pca_bottom_r{rank}']['gap']):.3f} | k raw {out['raw']['knowledge']:.3f} "
                  f"group {out[f'group_r{rank}']['knowledge']:.3f} random {out[f'random_r{rank}']['knowledge']:.3f}", flush=True)
        cfg = {"gamma": gamma, "per_seed": per_seed, "effects_vs_raw": {}, "group_minus_arm": {}}
        for arm in per_seed[0]:
            if arm == "raw":
                continue
            f = lambda k: [p[arm][k] - p["raw"][k] for p in per_seed]
            cfg["effects_vs_raw"][arm] = {
                "abs_gap": sr.summarise([abs(p[arm]["gap"]) - abs(p["raw"]["gap"]) for p in per_seed]),
                "knowledge": sr.summarise(f("knowledge")), "k_g0": sr.summarise(f("k_g0")), "k_g1": sr.summarise(f("k_g1")),
                "worst_group_k": sr.summarise([min(p[arm]["k_g0"] - p["raw"]["k_g0"], p[arm]["k_g1"] - p["raw"]["k_g1"]) for p in per_seed]),
                "onestep_mse": sr.summarise(f("onestep_mse")),
                "abs_forecast_gap_error": sr.summarise([abs(p[arm]["forecast_gap_error"]) - abs(p["raw"]["forecast_gap_error"]) for p in per_seed]),
                "mean_abs_control": sr.summarise(f("mean_abs_control")),
                "distortion": sr.summarise([p[arm].get("distortion", np.nan) for p in per_seed])}
            g = f"group_r{rank}" if arm.endswith(f"r{rank}") else "group_r1"
            if arm != g:
                cfg["group_minus_arm"][arm] = {
                    "abs_gap": sr.summarise([abs(p[g]["gap"]) - abs(p[arm]["gap"]) for p in per_seed]),
                    "knowledge": sr.summarise([p[g]["knowledge"] - p[arm]["knowledge"] for p in per_seed]),
                    "onestep_mse": sr.summarise([p[g]["onestep_mse"] - p[arm]["onestep_mse"] for p in per_seed])}
        results["configs"].append(cfg)
    fa.step = sr.make_step(1.0)
    json.dump(results, open(os.path.join(OUT, "simulator.json"), "w"), indent=1, default=float)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["A", "B"], required=True)
    p.add_argument("--dataset", default="assistments")
    p.add_argument("--sim", default=os.environ.get("SIM_DIR", "."))
    p.add_argument("--draws", type=int, default=N_DRAWS)
    a = p.parse_args()
    part_a(a.dataset, a.draws) if a.part == "A" else part_b(a.sim, a.draws)
