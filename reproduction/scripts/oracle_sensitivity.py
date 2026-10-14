"""
Sensitivity of the reference controller that uses the simulator's known equations (reviews of 14 October 2026).

The reference of oracle_reference.py plans with the noise-free next state of the known equations (clipping kept)
and a grid search (11^3 plans, then a local 5^3 grid of step 0.05). Here, on gamma = 1 and both objectives
(10 replicates, rho = 1, evaluation cohort), we compare it with
  fine   a finer search: 21^3 plans, then a local 9^3 grid of step 0.0125;
  noise  the expected cost over 16 draws of the simulator noise (common to all plans), clipping kept, which
         plans with E[clip(a + noise)] instead of clip(a).
Outcomes: final mean knowledge, per group, absolute knowledge gap, mean absolute help.

    SIM_DIR=/path/to/sim python oracle_sensitivity.py --workers 3   -> out_reviewer/oracle_sensitivity.json
"""
import argparse
import itertools
import json
import os
from multiprocessing import Pool

import numpy as np

import oracle_reference as orf
import reviewer_experiments as rx
import synthetic_regimes as sr

SEEDS = list(range(10))
K_NOISE = 16
SD = np.array([0.03, 0.03, 0.02])
FA = None
GRIDS = {"base": (np.array(list(itertools.product(np.linspace(-1, 1, 11), repeat=3))),
                  np.array(list(itertools.product(np.linspace(-0.1, 0.1, 5), repeat=3)))),
         "fine": (np.array(list(itertools.product(np.linspace(-1, 1, 21), repeat=3))),
                  np.array(list(itertools.product(np.linspace(-0.05, 0.05, 9), repeat=3))))}


class Planner:
    def __init__(self, fa, beta, rho, grid, noise, seed):
        self.fa, self.beta, self.rho = fa, beta, rho
        self.coarse, self.fine = GRIDS[grid]
        self.noise = noise
        self.rng = np.random.default_rng(10_000 + seed)

    def cost(self, x, g, U, eps):
        if eps is None:
            return orf.plan_cost(self.fa, x, g, U, self.beta, self.rho)
        n, m, H = U.shape
        J = np.zeros((n, m))
        for k in range(eps.shape[0]):
            X = np.broadcast_to(x[:, None, :], (n, m, 4))
            G = np.broadcast_to(g[:, None], (n, m))
            Jk = self.rho * (U ** 2).sum(-1)
            for j in range(H):
                X = self.fa.step(X, U[..., j], G, np.broadcast_to(eps[k, :, j][:, None, :], (n, m, 3)))
                Jk = Jk - X[..., 0] + self.beta * X[..., self.fa.D_IDX] ** 2
            J += Jk / eps.shape[0]
        return J

    def __call__(self, x, groups):
        g = groups.astype(float)
        n = len(x)
        eps = self.rng.standard_normal((K_NOISE, n, 3, 3)) * SD if self.noise else None
        U = np.broadcast_to(self.coarse[None], (n,) + self.coarse.shape)
        best = U[np.arange(n), self.cost(x, g, U, eps).argmin(1)]
        U2 = np.clip(best[:, None, :] + self.fine[None], -1, 1)
        return U2[np.arange(n), self.cost(x, g, U2, eps).argmin(1)][:, 0]


def init_worker(sim):
    global FA
    FA = sr.load_sim(sim)


def task(args):
    obj, seed = args
    fa = FA
    fa.step = sr.make_step(1.0)
    init, noise, groups = fa.make_cohort(seed, horizon=orf.T)
    out = {}
    for name, grid, nz in (("base", "base", False), ("fine", "fine", False), ("noise", "base", True)):
        out[name] = orf.rollout(fa, Planner(fa, orf.OBJECTIVES[obj], 1.0, grid, nz, seed), init, noise, groups)
    return obj, seed, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=os.environ.get("SIM_DIR", rx.SIM))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    tasks = [(o, s) for o in orf.OBJECTIVES for s in SEEDS]
    res = {}
    with Pool(a.workers, initializer=init_worker, initargs=(a.sim,)) as pool:
        for o, s, r in pool.imap_unordered(task, tasks):
            res.setdefault(o, {})[s] = r
    S = rx.summ
    summary = {}
    for o in orf.OBJECTIVES:
        per = [res[o][s] for s in SEEDS]
        row = {v: {q: S([p[v][q] for p in per]) for q in ("k", "k0", "k1", "kgap", "u1")} for v in ("base", "fine", "noise")}
        for v in ("fine", "noise"):
            row[f"{v}_minus_base"] = {q: S([p[v][q] - p["base"][q] for p in per]) for q in ("k", "u1")}
            row[f"{v}_minus_base"]["abs_kgap"] = S([abs(p[v]["kgap"]) - abs(p["base"]["kgap"]) for p in per])
        summary[o] = row
        print(o, {v: (round(row[v]["k"]["mean"], 4), round(row[v]["u1"]["mean"], 3)) for v in ("base", "fine", "noise")},
              {v: (round(row[f"{v}_minus_base"]["k"]["mean"], 4), [round(x, 4) for x in row[f"{v}_minus_base"]["k"]["ci"]]) for v in ("fine", "noise")}, flush=True)
    os.makedirs(rx.OUT, exist_ok=True)
    json.dump(dict(seeds=len(SEEDS), k_noise=K_NOISE, summary=summary), open(os.path.join(rx.OUT, "oracle_sensitivity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
