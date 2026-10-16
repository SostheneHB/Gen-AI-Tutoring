"""
Regime study for the synthetic EDMD--MPC experiment (reviewer comment 4).

Uses the simulator, EDMD fit, projector and exact box-constrained MPC of
fa_edmd_mpc_v3_corrected.py unchanged, and varies one factor at a time around
the published configuration (group strength 1, rank 4, depth 3, horizon 25,
beta 8), plus a group-strength x beta map.  Every configuration is run on the
same ten replicate identifiers 0..9 with paired arms.

Controllers compared on the same cohort, noise and fitted model:
  mpc_raw        per-student MPC on the raw surrogate (published)
  mpc_proj       per-student MPC on the projected surrogate (published)
  gap_penalty    raw surrogate, cohort-level objective: knowledge reward minus
                 beta_gap * (predicted group-1 minus group-0 disparity at the
                 end of the MPC horizon)^2; no per-student disparity penalty
  gap_constraint raw surrogate, knowledge reward subject to a zero predicted
                 cohort disparity gap at the end of the MPC horizon (hard,
                 whenever the box constraints allow it)
  postproc       raw MPC actions; the terminal forecast is post-processed by
                 removing its group-mean difference (a predictive
                 post-processing baseline: it changes the forecast only)
Reported per configuration: paired effects on the terminal disparity gap and
on knowledge, the absolute gap of every arm, one-step model error of the raw
and projected surrogates on held-out closed-loop trajectories, and the
terminal-gap forecast error of each arm.

    python synthetic_regimes.py --sim ../../Final_Folder_on_DR --out out_synthetic
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy.special import expit
from scipy.stats import t as student_t

SEEDS = list(range(10))
BASE = dict(gamma=1.0, rank=4, depth=3, horizon=25, beta=8.0)


def load_sim(path):
    sys.path.insert(0, path)
    import fa_edmd_mpc_v3_corrected as fa
    return fa


def make_step(gamma):
    """Simulator of fa.step with the two group coefficients scaled by gamma."""
    def step(states, controls, groups, noise):
        k, e, p, d = np.moveaxis(np.asarray(states), -1, 0)
        nk, ne, nd = np.moveaxis(np.asarray(noise), -1, 0)
        k1 = np.clip(.85 * k + .15 * expit(3 * e - 1.5) + .20 * controls
                     - .06 * gamma * groups + nk, 0, 1)
        e1 = np.clip(.80 * e + .10 * np.tanh(p) + .15 * controls + ne, 0, 1)
        return np.stack([k1, e1, expit(6 * (k1 - .5)), .90 * d - .05 * gamma * groups
                         + .05 * controls + nd], axis=-1)
    return step


class CohortMPC:
    """Raw-surrogate MPC with a cohort-level term on the predicted disparity
    gap at the end of the MPC horizon.  The per-student problem stays the
    exact box-constrained QP of fa.MPCPolicy; the coupling enters through one
    Lagrange multiplier found by bisection."""

    def __init__(self, fa, model, beta_student, mode, beta_gap=8.0, H=3, rho=1.0):
        prop, ctrl, off = model.dynamics(False)
        self.pol = fa.MPCPolicy(prop, ctrl, off, model.decoder, model.decoder_offset,
                                horizon=H, beta=beta_student, rho=rho)
        dim = len(off)
        infl = np.zeros((dim, H))
        smap, drift = np.eye(dim), np.zeros(dim)
        for t in range(H):
            smap, drift = prop @ smap, prop @ drift + off
            infl = prop @ infl
            infl[:, t] += ctrl
        self.v = model.decoder[fa.D_IDX] @ infl            # d-influence at step H
        self.base_map = model.decoder[fa.D_IDX] @ smap
        self.base_off = model.decoder[fa.D_IDX] @ drift + model.decoder_offset[fa.D_IDX]
        self.mode, self.beta_gap = mode, beta_gap
        self.calls = self.infeasible = 0          # diagnostics of the coupling step
        self.max_residual = 0.0                   # |predicted gap| (constraint) at the solution
        self.lambdas = []                         # multipliers of feasible steps

    def _solve(self, latent, lam_vec):
        p = self.pol
        grad = latent @ p.gradient_map.T + p.gradient_offset + lam_vec[:, None] * self.v
        cand = np.einsum('aij,...j->...ai', p.solution_maps, grad) + p.solution_offsets
        feas = np.all(np.abs(cand) <= 1 + 1e-9, axis=-1)
        cost = .5 * np.einsum('...ai,ij,...aj->...a', cand, p.hessian, cand) \
            + np.einsum('...ai,...i->...a', cand, grad)
        cost = np.where(feas, cost, np.inf)
        best = np.argmin(cost, axis=-1)
        return np.clip(np.take_along_axis(cand, best[..., None, None], axis=-2)[..., 0, :], -1, 1)

    def __call__(self, latent, groups):
        n1, n0 = (groups == 1).sum(), (groups == 0).sum()
        s = np.where(groups == 1, 1 / n1, -1 / n0)
        base = latent @ self.base_map + self.base_off
        gapf = lambda U: s @ (base + U @ self.v)
        if self.mode == "constraint":           # find lam with predicted gap = 0
            f = lambda lam: gapf(self._solve(latent, lam * s))
        else:                                   # stationarity lam = 2 beta_gap gap(u(lam))
            f = lambda lam: lam - 2 * self.beta_gap * gapf(self._solve(latent, lam * s))
        lo, hi = -1e4, 1e4
        flo, fhi = f(lo), f(hi)
        self.calls += 1
        if np.sign(flo) == np.sign(fhi):       # infeasible: closest endpoint
            lam = lo if abs(flo) < abs(fhi) else hi
            self.infeasible += 1
        else:
            for _ in range(60):
                mid = .5 * (lo + hi)
                fm = f(mid)
                if np.sign(fm) == np.sign(flo):
                    lo, flo = mid, fm
                else:
                    hi = mid
            lam = .5 * (lo + hi)
            self.lambdas.append(abs(float(lam)))
        U = self._solve(latent, lam * s)
        if self.mode == "constraint":
            self.max_residual = max(self.max_residual, abs(float(gapf(U))))
        return U[:, 0]


def one_config(fa, cfg, seed):
    fa.step = make_step(cfg["gamma"])               # rollouts and training use it
    model = fa.fit_model(seed, cfg["rank"], cfg["depth"])
    init, noise, groups = fa.make_cohort(seed, horizon=cfg["horizon"])
    out = {}
    pols = {
        "mpc_raw": (fa.MPCPolicy(*model.dynamics(False), model.decoder, model.decoder_offset,
                                 beta=cfg["beta"]), False),
        "mpc_proj": (fa.MPCPolicy(*model.dynamics(True), model.decoder, model.decoder_offset,
                                  beta=cfg["beta"]), True),
        "gap_penalty": (CohortMPC(fa, model, 0.0, "penalty", beta_gap=cfg["beta"]), False),
        "gap_constraint": (CohortMPC(fa, model, 0.0, "constraint"), False),
    }
    for name, (pol, projected) in pols.items():
        x = init.copy()
        states, zs, us = [x.copy()], [], []
        for t in range(cfg["horizon"]):
            z = model.encode(x)
            u = pol(z, groups) if isinstance(pol, CohortMPC) else pol(z)
            zs.append(z)
            us.append(u)
            x = fa.step(x, u, groups, noise[:, t])
            states.append(x.copy())
        S = np.stack(states, 1)
        Z, U = np.stack(zs, 1), np.stack(us, 1)
        # coupling diagnostics of the deployed trajectory (before the forecast reuses pol)
        diag = ((pol.infeasible / max(pol.calls, 1), pol.max_residual,
                 float(np.median(pol.lambdas)) if pol.lambdas else 0.0)
                if isinstance(pol, CohortMPC) else (0.0, 0.0, 0.0))
        # one-step model error of the surrogate used by this arm on its own trajectory
        A, B, c = model.dynamics(projected)
        zp = Z @ A.T + U[..., None] * B + c
        xhat = zp @ model.decoder.T + model.decoder_offset
        # free-running terminal forecast of this arm (fa.forecast_terminal)
        fpol = (lambda z: pol(z, groups)) if isinstance(pol, CohortMPC) else pol
        fc = fa.forecast_terminal(model, init, cfg["horizon"], fpol, projected)
        gT = fa.contrast(S[:, -1, fa.D_IDX], groups)
        out[name] = dict(gap=float(gT), knowledge=float(S[:, -1, 0].mean()),
                         kgap=float(fa.contrast(S[:, -1, 0], groups)),
                         onestep_mse=float(((xhat - S[:, 1:]) ** 2).mean()),
                         forecast_gap=float(fa.contrast(fc[:, fa.D_IDX], groups)),
                         mean_abs_control=float(np.abs(U).mean()),
                         p90_abs_control=float(np.percentile(np.abs(U), 90)),
                         infeasible_fraction=float(diag[0]),
                         max_constraint_residual=float(diag[1]),
                         median_abs_lambda=float(diag[2]),
                         k_g0=float(S[groups == 0, -1, 0].mean()), k_g1=float(S[groups == 1, -1, 0].mean()),
                         d_g0=float(S[groups == 0, -1, fa.D_IDX].mean()), d_g1=float(S[groups == 1, -1, fa.D_IDX].mean()))
    pp = dict(out["mpc_raw"])
    pp["forecast_gap"] = 0.0                       # group-mean difference removed
    out["postproc"] = pp
    fa.step = make_step(1.0)
    return out


def summarise(vals):
    v = np.asarray(vals, float)
    h = student_t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return dict(mean=float(v.mean()), ci=[float(v.mean() - h), float(v.mean() + h)],
                min=float(v.min()), max=float(v.max()))


def run_cfg(fa, cfg):
    per = [one_config(fa, cfg, s) for s in SEEDS]
    arms = per[0].keys()
    res = {"config": cfg}
    for a in arms:
        res[a] = {k: summarise([p[a][k] for p in per]) for k in per[0][a]}
    for a in ("mpc_proj", "gap_penalty", "gap_constraint"):
        res[f"effect_{a}_vs_raw"] = {
            "gap": summarise([p[a]["gap"] - p["mpc_raw"]["gap"] for p in per]),
            "abs_gap": summarise([abs(p[a]["gap"]) - abs(p["mpc_raw"]["gap"]) for p in per]),
            "knowledge": summarise([p[a]["knowledge"] - p["mpc_raw"]["knowledge"] for p in per]),
            **{q: summarise([p[a][q] - p["mpc_raw"][q] for p in per]) for q in ("k_g0", "k_g1", "d_g0", "d_g1")}}
    d = np.array([abs(p["mpc_proj"]["gap"]) - abs(p["mpc_raw"]["gap"]) for p in per])
    lo, hi = res["effect_mpc_proj_vs_raw"]["abs_gap"]["ci"]
    res["projection_regime"] = "helps" if hi < 0 else ("hurts" if lo > 0 else "no detectable change")
    res["n_seeds_projection_widens_gap"] = int((d > 0).sum())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="../../Final_Folder_on_DR")
    ap.add_argument("--out", default="out_synthetic")
    args = ap.parse_args()
    fa = load_sim(args.sim)
    os.makedirs(args.out, exist_ok=True)
    grids = {"gamma": [0.0, 0.25, 0.5, 1.0, 2.0, -1.0], "rank": [1, 2, 4, 6, 8],
             "depth": [1, 2, 3, 5], "horizon": [5, 10, 25, 50], "beta": [0.0, 1.0, 8.0, 32.0]}
    results = {"one_factor": {}, "gamma_x_beta": []}
    for factor, values in grids.items():
        results["one_factor"][factor] = []
        for v in values:
            cfg = dict(BASE, **{factor: v})
            r = run_cfg(fa, cfg)
            results["one_factor"][factor].append(r)
            e = r["effect_mpc_proj_vs_raw"]
            print(f"{factor}={v}: proj-raw gap {e['gap']['mean']:+.4f} |gap| {e['abs_gap']['mean']:+.4f} "
                  f"k {e['knowledge']['mean']:+.4f} -> {r['projection_regime']}", flush=True)
    for g, b in itertools.product([0.0, 0.5, 1.0, 2.0], [0.0, 1.0, 8.0, 32.0]):
        r = run_cfg(fa, dict(BASE, gamma=g, beta=b))
        results["gamma_x_beta"].append(r)
        print(f"gamma={g} beta={b}: {r['projection_regime']}", flush=True)
    # Budget check: gap-constrained MPC with a larger control-cost weight rho,
    # so that its mean |u| can be matched to raw MPC (published objective uses rho=1)
    results["budget_check"] = []
    for g in (0.5, 1.0, 2.0):
        for rho in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
            per = []
            for seed in SEEDS:
                cfg = dict(BASE, gamma=g)
                fa.step = make_step(g)
                model = fa.fit_model(seed, cfg["rank"], cfg["depth"])
                init, noise, groups = fa.make_cohort(seed, horizon=cfg["horizon"])
                arms = {"raw": fa.MPCPolicy(*model.dynamics(False), model.decoder, model.decoder_offset, beta=cfg["beta"]),
                        "con": CohortMPC(fa, model, 0.0, "constraint", rho=rho)}
                r = {}
                for name, pol in arms.items():
                    x = init.copy(); us = []
                    for t in range(cfg["horizon"]):
                        z = model.encode(x)
                        u = pol(z, groups) if name == "con" else pol(z)
                        us.append(u); x = fa.step(x, u, groups, noise[:, t])
                    r[name] = (float(fa.contrast(x[:, fa.D_IDX], groups)), float(x[:, 0].mean()),
                               float(np.abs(np.stack(us)).mean()))
                per.append(r)
                fa.step = make_step(1.0)
            results["budget_check"].append(dict(
                gamma=g, rho=rho,
                raw_mean_abs_u=summarise([p["raw"][2] for p in per]),
                con_mean_abs_u=summarise([p["con"][2] for p in per]),
                abs_gap_change=summarise([abs(p["con"][0]) - abs(p["raw"][0]) for p in per]),
                knowledge_change=summarise([p["con"][1] - p["raw"][1] for p in per])))
            b = results["budget_check"][-1]
            print(f"budget gamma={g} rho={rho}: |u| raw {b['raw_mean_abs_u']['mean']:.3f} con {b['con_mean_abs_u']['mean']:.3f} "
                  f"d|gap| {b['abs_gap_change']['mean']:+.3f} dk {b['knowledge_change']['mean']:+.3f}", flush=True)
    with open(os.path.join(args.out, "regimes.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
