"""
The three simulator checks that used ten replicates, repeated with 50 replicates and the certified
gap-constrained solver (CertifiedCohortMPC):

  regimes  : the 28 settings with every arm, control use, certified infeasible share and group-wise
             knowledge and disparity  -> <out>/regimes.json (same layout as synthetic_regimes.py)
  budget   : the gap-constrained MPC with control cost rho in {1, 1.5, 2, 3, 5}  -> regimes.json["budget_check"]
  betagap  : the gap-penalty grid beta_gap in {1, ..., 32768} and the constraint  -> <out>/extras.json

Tasks (setting x replicate) run in parallel; every task rebuilds its model from its seed, so the result does
not depend on the number of workers.

    SIM_DIR=/path/to/sim python e50_runs.py --workers 7 --out out_synthetic50
"""
import argparse
import itertools
import json
import os
from multiprocessing import Pool

import numpy as np

import synthetic_regimes as sr
import synthetic_extras as se
import reviewer_experiments as rx

SEEDS = list(range(int(os.environ.get("E50_SEEDS", 50))))
FA = None


def init_worker(sim):
    global FA
    FA = sr.load_sim(sim)


class Certified(rx.CertifiedCohortMPC):
    """CertifiedCohortMPC that also records |lambda| of every step (the last multiplier it solves with)."""

    def _solve(self, latent, lam_vec):
        self._last = lam_vec
        return super()._solve(latent, lam_vec)

    def __call__(self, latent, groups):
        u = super().__call__(latent, groups)
        n1 = (groups == 1).sum()
        self.lambdas.append(float(abs(self._last[groups == 1][0] * n1)))
        return u


def stat(pol):
    if isinstance(pol, rx.CertifiedCohortMPC):
        return pol.stats["infeasible"] / max(pol.stats["steps"], 1), float(np.median(pol.lambdas)) if pol.lambdas else 0.0
    if isinstance(pol, sr.CohortMPC):
        return pol.infeasible / max(pol.calls, 1), float(np.median(pol.lambdas)) if pol.lambdas else 0.0
    return 0.0, 0.0


def one_config(cfg, seed):
    """synthetic_regimes.one_config with the certified solver for the constraint arm."""
    fa = FA
    fa.step = sr.make_step(cfg["gamma"])
    model = fa.fit_model(seed, cfg["rank"], cfg["depth"])
    init, noise, groups = fa.make_cohort(seed, horizon=cfg["horizon"])
    pols = {"mpc_raw": fa.MPCPolicy(*model.dynamics(False), model.decoder, model.decoder_offset, beta=cfg["beta"]),
            "mpc_proj": fa.MPCPolicy(*model.dynamics(True), model.decoder, model.decoder_offset, beta=cfg["beta"]),
            "gap_penalty": sr.CohortMPC(fa, model, 0.0, "penalty", beta_gap=cfg["beta"]),
            "gap_constraint": Certified(fa, model)}
    out = {}
    for name, pol in pols.items():
        x = init.copy(); states, us = [x.copy()], []
        for t in range(cfg["horizon"]):
            z = model.encode(x)
            u = pol(z, groups) if isinstance(pol, sr.CohortMPC) else pol(z)
            us.append(u); x = fa.step(x, u, groups, noise[:, t]); states.append(x.copy())
        S, U = np.stack(states, 1), np.stack(us, 1)
        infeas, lam = stat(pol)
        out[name] = dict(gap=float(fa.contrast(S[:, -1, fa.D_IDX], groups)), knowledge=float(S[:, -1, 0].mean()),
                         kgap=float(fa.contrast(S[:, -1, 0], groups)), mean_abs_control=float(np.abs(U).mean()),
                         infeasible_fraction=float(infeas), median_abs_lambda=lam,
                         k_g0=float(S[groups == 0, -1, 0].mean()), k_g1=float(S[groups == 1, -1, 0].mean()),
                         d_g0=float(S[groups == 0, -1, fa.D_IDX].mean()), d_g1=float(S[groups == 1, -1, fa.D_IDX].mean()))
    fa.step = sr.make_step(1.0)
    return out


def budget_task(g, rho, seed):
    fa = FA
    cfg = dict(sr.BASE, gamma=g)
    fa.step = sr.make_step(g)
    model = fa.fit_model(seed, cfg["rank"], cfg["depth"])
    init, noise, groups = fa.make_cohort(seed, horizon=cfg["horizon"])
    arms = {"raw": fa.MPCPolicy(*model.dynamics(False), model.decoder, model.decoder_offset, beta=cfg["beta"]),
            "con": Certified(fa, model, rho=rho)}
    r = {}
    for name, pol in arms.items():
        x = init.copy(); us = []
        for t in range(cfg["horizon"]):
            z = model.encode(x)
            u = pol(z, groups) if name == "con" else pol(z)
            us.append(u); x = fa.step(x, u, groups, noise[:, t])
        r[name] = (float(fa.contrast(x[:, fa.D_IDX], groups)), float(x[:, 0].mean()), float(np.abs(np.stack(us)).mean()))
    fa.step = sr.make_step(1.0)
    return r


def betagap_task(g, bg, seed):
    fa = FA
    fa.step = sr.make_step(g)
    model = fa.fit_model(seed, 4, 3)
    init, noise, groups = fa.make_cohort(seed, horizon=25)
    raw = fa.MPCPolicy(*model.dynamics(False), model.decoder, model.decoder_offset, beta=8.0)
    alt = Certified(fa, model) if bg == "constraint" else sr.CohortMPC(fa, model, 0.0, "penalty", beta_gap=bg)
    r = {}
    for name, pol in (("raw", raw), ("alt", alt)):
        S, _, U, _ = se.rollout(fa, model, pol, init, noise, groups, 25, False)
        r[name] = dict(gap=float(fa.contrast(S[:, -1, fa.D_IDX], groups)), k=float(S[:, -1, 0].mean()),
                       k0=float(S[groups == 0, -1, 0].mean()), k1=float(S[groups == 1, -1, 0].mean()),
                       u=float(np.abs(U).mean()))
    r["infeasible"], r["lam"] = stat(alt)
    fa.step = sr.make_step(1.0)
    return r


def run_task(t):
    kind, args = t
    return {"regimes": one_config, "budget": budget_task, "betagap": betagap_task}[kind](*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=os.environ.get("SIM_DIR", "."))
    ap.add_argument("--out", default="out_synthetic50")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfgs = rx.configs28()
    budget_grid = [(g, rho) for g in (0.5, 1.0, 2.0) for rho in (1.0, 1.5, 2.0, 3.0, 5.0)]
    bg_grid = [(g, bg) for g in (0.5, 1.0, 2.0) for bg in (1.0, 8.0, 64.0, 512.0, 4096.0, 32768.0, "constraint")]
    tasks = ([("regimes", (c, s)) for c in cfgs for s in SEEDS] + [("budget", (g, r, s)) for g, r in budget_grid for s in SEEDS]
             + [("betagap", (g, b, s)) for g, b in bg_grid for s in SEEDS])
    print(f"{len(tasks)} tasks on {a.workers} workers", flush=True)
    res = []
    with Pool(a.workers, initializer=init_worker, initargs=(a.sim,)) as pool:
        for i, r in enumerate(pool.imap(run_task, tasks, chunksize=2)):
            res.append(r)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(tasks)} done", flush=True)
    it = iter(res)
    S = sr.summarise
    regimes = {"one_factor": {"all": []}, "gamma_x_beta": []}
    for c in cfgs:
        per = [next(it) for _ in SEEDS]
        r = {"config": c}
        for arm in per[0]:
            r[arm] = {k: S([p[arm][k] for p in per]) for k in per[0][arm]}
        for arm in ("mpc_proj", "gap_penalty", "gap_constraint"):
            r[f"effect_{arm}_vs_raw"] = {
                "gap": S([p[arm]["gap"] - p["mpc_raw"]["gap"] for p in per]),
                "abs_gap": S([abs(p[arm]["gap"]) - abs(p["mpc_raw"]["gap"]) for p in per]),
                "knowledge": S([p[arm]["knowledge"] - p["mpc_raw"]["knowledge"] for p in per]),
                "kgap": S([p[arm]["kgap"] - p["mpc_raw"]["kgap"] for p in per]),
                "abs_kgap": S([abs(p[arm]["kgap"]) - abs(p["mpc_raw"]["kgap"]) for p in per]),
                "worst_group_k": S([min(p[arm]["k_g0"] - p["mpc_raw"]["k_g0"], p[arm]["k_g1"] - p["mpc_raw"]["k_g1"]) for p in per]),
                **{q: S([p[arm][q] - p["mpc_raw"][q] for p in per]) for q in ("k_g0", "k_g1", "d_g0", "d_g1")}}
        regimes["one_factor"]["all"].append(r)
    regimes["budget_check"] = []
    for g, rho in budget_grid:
        per = [next(it) for _ in SEEDS]
        regimes["budget_check"].append(dict(gamma=g, rho=rho, raw_mean_abs_u=S([p["raw"][2] for p in per]),
                                            con_mean_abs_u=S([p["con"][2] for p in per]),
                                            abs_gap_change=S([abs(p["con"][0]) - abs(p["raw"][0]) for p in per]),
                                            knowledge_change=S([p["con"][1] - p["raw"][1] for p in per])))
    extras = {"beta_gap_grid": []}
    for g, bg in bg_grid:
        per = [next(it) for _ in SEEDS]
        extras["beta_gap_grid"].append(dict(gamma=g, beta_gap=bg, u_raw=S([p["raw"]["u"] for p in per]),
                                            u_alt=S([p["alt"]["u"] for p in per]),
                                            abs_gap_change=S([abs(p["alt"]["gap"]) - abs(p["raw"]["gap"]) for p in per]),
                                            k_change=S([p["alt"]["k"] - p["raw"]["k"] for p in per]),
                                            k0_change=S([p["alt"]["k0"] - p["raw"]["k0"] for p in per]),
                                            k1_change=S([p["alt"]["k1"] - p["raw"]["k1"] for p in per]),
                                            median_abs_lambda=S([p["lam"] for p in per]),
                                            infeasible=S([p["infeasible"] for p in per])))
    from scipy import stats as st
    for arm in ("mpc_proj", "gap_penalty", "gap_constraint"):
        for q in ("abs_gap", "abs_kgap", "knowledge", "k_g0", "k_g1"):
            ps = []
            for r in regimes["one_factor"]["all"]:
                e = r[f"effect_{arm}_vs_raw"][q]
                se = (e["ci"][1] - e["mean"]) / st.t.ppf(.975, len(SEEDS) - 1)
                ps.append(1.0 if se == 0 else float(2 * st.t.sf(abs(e["mean"]) / se, len(SEEDS) - 1)))
            for r, padj in zip(regimes["one_factor"]["all"], rx.holm(ps)):
                r[f"effect_{arm}_vs_raw"][q]["p_holm"] = float(padj)
    json.dump(regimes, open(os.path.join(a.out, "regimes.json"), "w"), indent=1)
    json.dump(extras, open(os.path.join(a.out, "extras.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
