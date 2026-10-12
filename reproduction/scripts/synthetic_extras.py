"""Extra synthetic checks requested in the fourth review.

    python synthetic_extras.py --sim ../../Final_Folder_on_DR --out out_synthetic_extras.json

1. beta_gap grid: the gap-penalty MPC with a log-scale grid of the gap weight,
   reported with its mean |u| (budget) and group-wise outcomes, next to the
   gap-constrained MPC and its multipliers.
2. Horizon T = 50: the model is fitted on 25-step exploratory trajectories as
   in the main study (fit_model never sees the evaluation horizon).  We report
   the one-step error of the raw and projected surrogates at every step
   1..50 of the raw- and projected-MPC trajectories, and the distance of the
   visited states to the training distribution (Mahalanobis distance of the
   lifted state and share of coordinates outside the training range).
3. Sensitivity: the same T = 50 comparison when the exploratory training
   trajectories also have 50 transitions.
"""
import argparse
import json

import numpy as np

import synthetic_regimes as sr

SEEDS = list(range(10))


def rollout(fa, model, pol, init, noise, groups, T, projected_for_error):
    x = init.copy()
    S, Z, U = [x.copy()], [], []
    for t in range(T):
        z = model.encode(x)
        u = pol(z, groups) if isinstance(pol, sr.CohortMPC) else pol(z)
        Z.append(z); U.append(u)
        x = fa.step(x, u, groups, noise[:, t])
        S.append(x.copy())
    S, Z, U = np.stack(S, 1), np.stack(Z, 1), np.stack(U, 1)
    errs = {}
    for name, proj in (("raw", False), ("proj", True)):
        A, B, c = model.dynamics(proj)
        xhat = (Z @ A.T + U[..., None] * B + c) @ model.decoder.T + model.decoder_offset
        errs[name] = ((xhat - S[:, 1:]) ** 2).mean(axis=(0, 2))       # per step
    return S, Z, U, errs


def beta_gap_grid(fa):
    out = []
    for g in (0.5, 1.0, 2.0):
        fa.step = sr.make_step(g)
        for bg in (1.0, 8.0, 64.0, 512.0, 4096.0, 32768.0, "constraint"):
            per = []
            for seed in SEEDS:
                model = fa.fit_model(seed, 4, 3)
                init, noise, groups = fa.make_cohort(seed, horizon=25)
                raw = fa.MPCPolicy(*model.dynamics(False), model.decoder, model.decoder_offset, beta=8.0)
                alt = (sr.CohortMPC(fa, model, 0.0, "constraint") if bg == "constraint"
                       else sr.CohortMPC(fa, model, 0.0, "penalty", beta_gap=bg))
                r = {}
                for name, pol in (("raw", raw), ("alt", alt)):
                    S, _, U, _ = rollout(fa, model, pol, init, noise, groups, 25, False)
                    r[name] = dict(gap=float(fa.contrast(S[:, -1, fa.D_IDX], groups)), k=float(S[:, -1, 0].mean()),
                                   k0=float(S[groups == 0, -1, 0].mean()), k1=float(S[groups == 1, -1, 0].mean()),
                                   u=float(np.abs(U).mean()))
                r["lam"] = float(np.median(alt.lambdas)) if alt.lambdas else 0.0
                r["infeasible"] = alt.infeasible / max(alt.calls, 1)
                per.append(r)
            out.append(dict(gamma=g, beta_gap=bg,
                            u_raw=sr.summarise([p["raw"]["u"] for p in per]),
                            u_alt=sr.summarise([p["alt"]["u"] for p in per]),
                            abs_gap_change=sr.summarise([abs(p["alt"]["gap"]) - abs(p["raw"]["gap"]) for p in per]),
                            k_change=sr.summarise([p["alt"]["k"] - p["raw"]["k"] for p in per]),
                            k0_change=sr.summarise([p["alt"]["k0"] - p["raw"]["k0"] for p in per]),
                            k1_change=sr.summarise([p["alt"]["k1"] - p["raw"]["k1"] for p in per]),
                            median_abs_lambda=sr.summarise([p["lam"] for p in per]),
                            infeasible=sr.summarise([p["infeasible"] for p in per])))
            o = out[-1]
            print(f"gamma={g} beta_gap={bg}: |u| {o['u_raw']['mean']:.3f}->{o['u_alt']['mean']:.3f} "
                  f"d|gap| {o['abs_gap_change']['mean']:+.3f} dk0 {o['k0_change']['mean']:+.3f} dk1 {o['k1_change']['mean']:+.3f} "
                  f"lam {o['median_abs_lambda']['mean']:.1f}", flush=True)
    fa.step = sr.make_step(1.0)
    return out


def horizon50(fa, train_T):
    orig = fa.training_data
    fa.training_data = lambda seed, students=150, horizon=25: orig(seed, students, train_T)
    fa.step = sr.make_step(1.0)
    per = []
    for seed in SEEDS:
        model = fa.fit_model(seed, 4, 3)
        tr, _, _ = fa.training_data(seed)
        Ztr = model.encode(tr[:, :-1].reshape(-1, 4))
        mu, C = Ztr.mean(0), np.cov(Ztr.T) + 1e-9 * np.eye(Ztr.shape[1])
        Ci = np.linalg.inv(C)
        lo, hi = tr.reshape(-1, 4).min(0), tr.reshape(-1, 4).max(0)
        md_ref = np.percentile(np.sqrt(np.einsum("ij,jk,ik->i", Ztr - mu, Ci, Ztr - mu)), 99)
        init, noise, groups = fa.make_cohort(seed, horizon=50)
        rec = {}
        for arm, proj in (("mpc_raw", False), ("mpc_proj", True)):
            pol = fa.MPCPolicy(*model.dynamics(proj), model.decoder, model.decoder_offset, beta=8.0)
            S, Z, U, errs = rollout(fa, model, pol, init, noise, groups, 50, proj)
            md = np.sqrt(np.einsum("ntj,jk,ntk->nt", Z - mu, Ci, Z - mu))
            outside = ((S[:, :-1] < lo) | (S[:, :-1] > hi)).any(-1)
            rec[arm] = dict(err_raw=errs["raw"].tolist(), err_proj=errs["proj"].tolist(),
                            md_mean=md.mean(0).tolist(), share_md_above_train99=(md > md_ref).mean(0).tolist(),
                            share_outside_range=outside.mean(0).tolist(),
                            gap=float(fa.contrast(S[:, -1, fa.D_IDX], groups)), k=float(S[:, -1, 0].mean()))
        per.append(rec)
    fa.training_data = orig

    def curve(arm, key):
        v = np.array([p[arm][key] for p in per])
        return v.mean(0).tolist()
    res = {"train_T": train_T,
           "effect_abs_gap": sr.summarise([abs(p["mpc_proj"]["gap"]) - abs(p["mpc_raw"]["gap"]) for p in per]),
           "effect_k": sr.summarise([p["mpc_proj"]["k"] - p["mpc_raw"]["k"] for p in per])}
    for arm in ("mpc_raw", "mpc_proj"):
        for key in ("err_raw", "err_proj", "md_mean", "share_md_above_train99", "share_outside_range"):
            res[f"{arm}:{key}"] = curve(arm, key)
    print(f"train_T={train_T}: T=50 effect |gap| {res['effect_abs_gap']['mean']:+.3f} {res['effect_abs_gap']['ci']} "
          f"k {res['effect_k']['mean']:+.3f}; raw-MPC one-step err raw model steps1-25 {np.mean(res['mpc_raw:err_raw'][:25]):.4f} "
          f"steps26-50 {np.mean(res['mpc_raw:err_raw'][25:]):.4f}; share beyond train-99% MD steps 26-50 "
          f"{np.mean(res['mpc_raw:share_md_above_train99'][25:]):.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="../../Final_Folder_on_DR")
    ap.add_argument("--out", default="out_synthetic_extras.json")
    a = ap.parse_args()
    fa = sr.load_sim(a.sim)
    out = {"horizon50_train25": horizon50(fa, 25), "horizon50_train50": horizon50(fa, 50),
           "beta_gap_grid": beta_gap_grid(fa)}
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
