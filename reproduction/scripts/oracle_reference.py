"""
A true-model reference for the controllers (review of 13 October 2026, item 6).

Every arm observes the full physical state each week and uses the same objective, horizon (3 weeks) and budget.
Arms
  oracle          MPC on the simulator's true dynamics (expected next state: noise set to 0), knows the group
  oracle_blind    the same true dynamics with the group term replaced by its population mean (no group information)
  group_aware     MPC on the learned group-aware EDMD surrogate (knows the group)
  edmd_raw        MPC on the learned EDMD surrogate (no group input)
  edmd_projected  the same surrogate with the group direction projected out (the intervention of the paper)
  edmd_random     the same surrogate with a random rank-4 subspace removed (compression without the group)
Objectives (the same for every arm; sums over the horizon)
  knowledge       -k + rho u^2                       (alpha = 1, beta_ind = 0)
  paper           -k + beta_ind d^2 + rho u^2        (alpha = 1, beta_ind = 8: the MPC objective of the paper)
Budget and validation.  For every arm, the control weight rho is chosen by bisection on three validation cohorts
(stream 'calibration', two cohorts) so that the mean absolute control (U1) equals that of the oracle at rho = 1 on
the same cohorts; the arm is then run on the evaluation cohort used everywhere in the paper.  The arms at rho = 1
(no calibration) are also reported.  20 replicates (OR_SEEDS), gamma in {0.5, 1, 2}.
Outcomes: final mean knowledge, per group, signed and absolute knowledge gap, absolute disparity gap, U1, U2.

    SIM_DIR=/path/to/sim python oracle_reference.py --workers 4   -> out_reviewer/oracle_reference.json
"""
import argparse
import dataclasses
import itertools
import json
import os
from multiprocessing import Pool

import numpy as np

import pareto_fair as pf
import rank_matched_controls as rm
import reviewer_experiments as rx
import synthetic_regimes as sr

SEEDS = list(range(int(os.environ.get("OR_SEEDS", 20))))
GAMMAS = (0.5, 1.0, 2.0)
OBJECTIVES = {"knowledge": 0.0, "paper": 8.0}
T, H, N_VAL = 25, 3, 2
COARSE = np.array(list(itertools.product(np.linspace(-1, 1, 11), repeat=H)))
FINE = np.array(list(itertools.product(np.linspace(-0.1, 0.1, 5), repeat=H)))
FA = None


def plan_cost(fa, x, g, U, beta, rho):
    """Cost of control sequences U (n, m, H) from states x (n, 4) on the true expected dynamics."""
    n, m, _ = U.shape
    X = np.broadcast_to(x[:, None, :], (n, m, 4))
    G = np.broadcast_to(g[:, None], (n, m))
    zero = np.zeros((n, m, 3))
    J = rho * (U ** 2).sum(-1)
    for j in range(H):
        X = fa.step(X, U[..., j], G, zero)
        J = J - X[..., 0] + beta * X[..., fa.D_IDX] ** 2
    return J


class Oracle:
    def __init__(self, fa, beta, rho, blind):
        self.fa, self.beta, self.rho, self.blind = fa, beta, rho, blind

    def __call__(self, x, groups):
        g = np.full(len(x), groups.mean()) if self.blind else groups.astype(float)
        n = len(x)
        U = np.broadcast_to(COARSE[None], (n,) + COARSE.shape)
        best = U[np.arange(n), plan_cost(self.fa, x, g, U, self.beta, self.rho).argmin(1)]
        U2 = np.clip(best[:, None, :] + FINE[None], -1, 1)
        return U2[np.arange(n), plan_cost(self.fa, x, g, U2, self.beta, self.rho).argmin(1)][:, 0]


def learned(fa, model, kind, beta, rho, extra):
    if kind == "group_aware":
        A, B, a0, aG = extra
        pol = pf.GroupAwareMPC(fa, model, beta, A, B, a0, aG, fair=False, rho=rho)
        return lambda x, g: pol(model.encode(x), g)
    m = model if kind != "edmd_random" else dataclasses.replace(model, projector=extra)
    pol = fa.MPCPolicy(*m.dynamics(kind != "edmd_raw"), m.decoder, m.decoder_offset, beta=beta, rho=rho)
    return lambda x, g: pol(m.encode(x))


def rollout(fa, ctrl, init, noise, groups):
    x = init.copy()
    S, U = [x.copy()], []
    for t in range(noise.shape[1]):
        u = ctrl(x, groups)
        U.append(u)
        x = fa.step(x, u, groups, noise[:, t])
        S.append(x.copy())
    S, U = np.stack(S, 1), np.stack(U, 1)
    return dict(k=float(S[:, -1, 0].mean()), k0=float(S[groups == 0, -1, 0].mean()), k1=float(S[groups == 1, -1, 0].mean()),
                kgap=float(fa.contrast(S[:, -1, 0], groups)), gap=float(fa.contrast(S[:, -1, fa.D_IDX], groups)),
                u1=float(np.abs(U).mean()), u2=float((U ** 2).mean()))


def init_worker(sim):
    global FA
    FA = sr.load_sim(sim)


def task(args):
    gamma, seed, objective = args
    fa = FA
    fa.step = sr.make_step(gamma)
    beta = OBJECTIVES[objective]
    model, extra_ga = rx.ga_models(fa, seed)
    rng = np.random.default_rng(5000 + seed)
    _, Gih = rm.sym_sqrt(model.metric)
    P_rand = rm.removal_projector(Gih @ rng.standard_normal((len(model.metric), 4)), model.metric)
    val = [fa.make_cohort(seed, horizon=T, role="calibration", index=i) for i in range(N_VAL)]
    init, noise, groups = fa.make_cohort(seed, horizon=T)

    def make(arm, rho):
        if arm in ("oracle", "oracle_blind"):
            return Oracle(fa, beta, rho, arm == "oracle_blind")
        return learned(fa, model, arm, beta, rho, extra_ga if arm == "group_aware" else P_rand)

    def val_u1(arm, rho):
        return float(np.mean([rollout(fa, make(arm, rho), *c)["u1"] for c in val]))
    budget = val_u1("oracle", 1.0)
    out = dict(budget_u1=budget)
    out["oracle"] = dict(matched=rollout(fa, make("oracle", 1.0), init, noise, groups), rho=1.0)
    out["oracle"]["rho1"] = out["oracle"]["matched"]          # the oracle at rho = 1 defines the budget
    for arm in ("oracle_blind", "group_aware", "edmd_raw", "edmd_projected", "edmd_random"):
        lo, hi = np.log(0.05), np.log(20.0)
        for _ in range(10):                        # U1 decreases with rho
            mid = 0.5 * (lo + hi)
            if val_u1(arm, np.exp(mid)) > budget:
                lo = mid
            else:
                hi = mid
        rho = float(np.exp(0.5 * (lo + hi)))
        out[arm] = dict(matched=rollout(fa, make(arm, rho), init, noise, groups), rho=rho,
                        rho1=rollout(fa, make(arm, 1.0), init, noise, groups))
    fa.step = sr.make_step(1.0)
    return gamma, seed, objective, out


DIFFS = {"group_information": ("oracle", "oracle_blind"), "model_error_with_group": ("oracle", "group_aware"),
         "model_error_without_group": ("oracle_blind", "edmd_raw"), "erasure": ("edmd_raw", "edmd_projected"),
         "compression": ("edmd_raw", "edmd_random"), "erasure_beyond_compression": ("edmd_random", "edmd_projected"),
         "oracle_minus_projected": ("oracle", "edmd_projected")}
ARMS = ("oracle", "oracle_blind", "group_aware", "edmd_raw", "edmd_projected", "edmd_random")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=os.environ.get("SIM_DIR", rx.SIM))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tasks = [(g, s, o) for o in OBJECTIVES for g in GAMMAS for s in SEEDS]
    res = {}
    with Pool(a.workers, initializer=init_worker, initargs=(a.sim,)) as pool:
        for i, (g, s, o, r) in enumerate(pool.imap_unordered(task, tasks)):
            res.setdefault((o, g), {})[s] = r
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(tasks)}", flush=True)
    S = rx.summ
    summary = {}
    for (o, g), per_seed in sorted(res.items()):
        per = [per_seed[s] for s in SEEDS]
        row = {"budget_u1": S([p["budget_u1"] for p in per])}
        for mode in ("matched", "rho1"):
            row[mode] = {arm: {q: S([p[arm][mode][q] for p in per]) for q in ("k", "k0", "k1", "kgap", "gap", "u1", "u2")}
                         for arm in ARMS}
            for arm in ARMS:
                row[mode][arm]["abs_kgap"] = S([abs(p[arm][mode]["kgap"]) for p in per])
                row[mode][arm]["abs_gap"] = S([abs(p[arm][mode]["gap"]) for p in per])
            row[mode]["differences"] = {name: {q: S([p[x][mode][q] - p[y][mode][q] for p in per]) for q in ("k", "k0", "k1")}
                                        | {"abs_kgap": S([abs(p[x][mode]["kgap"]) - abs(p[y][mode]["kgap"]) for p in per])}
                                        for name, (x, y) in DIFFS.items()}
        row["rho"] = {arm: S([p[arm]["rho"] for p in per]) for arm in ARMS}
        summary[f"{o}|{g}"] = row
        m = row["matched"]
        print(f"== {o} gamma {g}: budget U1 {row['budget_u1']['mean']:.3f} | "
              + " ".join(f"{arm} k {m[arm]['k']['mean']:.3f} (U1 {m[arm]['u1']['mean']:.3f}, |kgap| {m[arm]['abs_kgap']['mean']:.3f})" for arm in ARMS), flush=True)
        print("   differences in k: " + " ".join(f"{n} {m['differences'][n]['k']['mean']:+.4f} [{m['differences'][n]['k']['ci'][0]:+.4f},{m['differences'][n]['k']['ci'][1]:+.4f}]"
                                                for n in DIFFS), flush=True)
    os.makedirs(rx.OUT, exist_ok=True)
    json.dump(dict(seeds=len(SEEDS), objectives=OBJECTIVES, horizon=H, n_val=N_VAL, summary=summary),
              open(os.path.join(rx.OUT, "oracle_reference.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
