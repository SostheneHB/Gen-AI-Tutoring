"""
Controllers aligned with knowledge (referee report 5, item 5): a compact factorial comparison.

Every arm uses the same group-aware surrogate (so every arm has the same access to the group), the same cohorts,
noise and replicates, and the same extra budget (factor c on the absolute control of the applied step):
    ref        the plain group-aware MPC (beta_ind, rho = 1)
    d_floor    narrows the predicted DISPARITY gap, with a floor on each group's predicted knowledge
    d_nofloor  narrows the predicted disparity gap, no floor
    k_floor    narrows the predicted KNOWLEDGE gap, with a floor on each group's predicted knowledge
    k_nofloor  narrows the predicted knowledge gap, no floor
    topup_k    reference plan plus a constant extra control for the group behind in knowledge, with the absolute
               budget (U1) of k_floor
Outcomes, paired with ref: final knowledge of each group, mean knowledge, signed and absolute knowledge gap,
absolute disparity gap, U1 and U2 ratios.  The floors protect predicted group means one step ahead; the outcomes
are measured on the simulated trajectories.

    SIM_DIR=/path/to/sim python knowledge_controller.py --workers 6   -> out_reviewer/knowledge_controller.json
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy.optimize import linprog, minimize

import pareto_fair as pf
import reviewer_experiments as rx
import synthetic_regimes as sr

SEEDS = list(range(int(os.environ.get("KC_SEEDS", 50))))
GAMMAS = (0.5, 1.0, 2.0)
C, BETA, T = 2.0, 8.0, 25
FA = None


class TargetMPC(pf.GroupAwareMPC):
    """Floor-constrained MPC on the group-aware surrogate with a chosen target ('d' or 'k') and optional floors."""

    def __init__(self, *args, target="d", floors=True, **kw):
        super().__init__(*args, **kw)
        self.target, self.use_floors = target, floors
        if target == "k":
            self.v = self.vk.copy()

    def __call__(self, latent, groups):
        gi = groups.astype(int)
        self._goff = np.stack([self.pols[g].gradient_offset for g in (0, 1)])[gi]
        self._kbase = latent @ self.kmap + self.koffs[gi]
        self._dbase = self._kbase if self.target == "k" else latent @ self.base_map + self.base_offs[gi]
        return self._fair_step(latent, groups)

    def _fair_step(self, latent, groups):
        n, H = len(groups), self.H
        g1 = groups == 1
        n1, n0 = g1.sum(), (~g1).sum()
        s = np.where(g1, 1 / n1, -1 / n0)
        wk = np.where(g1, 1 / n1, 0.0), np.where(~g1, 1 / n0, 0.0)
        dbase, kbase = self._dbase, self._kbase
        gap = lambda U: s @ (dbase + U @ self.v)
        kg = lambda U, w: w @ (kbase + U @ self.vk)
        U0, _ = self._solve_w(latent, np.zeros((n, H)))
        w1 = getattr(self, "l1_w", np.ones(H))
        gap0, B0 = gap(U0), self.budget_factor * (np.abs(U0) * w1).sum()
        floors = [kg(U0, w) for w in wk]
        self.steps += 1
        nv = n * H
        cvec = np.zeros(2 * nv + 1); cvec[-1] = 1.0
        gv, Z = np.kron(s, self.v), np.zeros(nv)
        A = [np.r_[gv, Z, -1.0], np.r_[-gv, Z, -1.0]]
        b = [-(s @ dbase), s @ dbase]
        if self.use_floors:
            for w, fl in zip(wk, floors):
                A.append(np.r_[-np.kron(w, self.vk), Z, 0.0]); b.append(-(fl + pf.MARGIN - w @ kbase))
        A.append(np.r_[Z, np.tile(w1, n), 0.0]); b.append(B0 * (1 - pf.MARGIN))
        I = np.eye(nv)
        A2 = np.vstack([np.hstack([I, -I, np.zeros((nv, 1))]), np.hstack([-I, -I, np.zeros((nv, 1))])])
        lp = linprog(cvec, A_ub=np.vstack([np.array(A), A2]), b_ub=np.r_[b, np.zeros(2 * nv)],
                     bounds=[(-1, 1)] * nv + [(0, 1)] * nv + [(0, None)], method="highs")
        if lp.status != 0 or lp.x[-1] >= abs(gap0) - 1e-6:
            return U0[:, 0]
        target = gap0 + pf.THETA * (gap(lp.x[:nv].reshape(n, H)) - gap0)
        self.fair_steps += 1
        fl_on = 1.0 if self.use_floors else 0.0

        def W_of(y):
            lam, mu0, mu1, _ = y
            return lam * s[:, None] * self.v[None, :] - fl_on * (mu0 * wk[0] + mu1 * wk[1])[:, None] * self.vk[None, :]

        def neg_dual(y):
            lam, mu0, mu1, nu = y
            U, cost = self._solve_l1(latent, W_of(y), nu)
            val = cost.sum() - lam * (target - s @ dbase) - nu * B0
            grad = [gap(U) - target, 0.0, 0.0, (np.abs(U) * w1).sum() - B0]
            if self.use_floors:
                val += mu0 * (floors[0] - wk[0] @ kbase) + mu1 * (floors[1] - wk[1] @ kbase)
                grad[1], grad[2] = floors[0] - kg(U, wk[0]), floors[1] - kg(U, wk[1])
            return -val, -np.array(grad)
        bounds = [(-1e5, 1e5), (0, 1e5 if self.use_floors else 0), (0, 1e5 if self.use_floors else 0), (0, 1e5)]
        res = minimize(neg_dual, np.zeros(4), jac=True, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=1000, gtol=1e-10, ftol=1e-15))
        U, _ = self._solve_l1(latent, W_of(res.x), res.x[3])
        if self.use_floors:
            self.floor_violation = max(self.floor_violation, max(0.0, max(fl - kg(U, w) for w, fl in zip(wk, floors))))
        return U[:, 0]


def init_worker(sim):
    global FA
    FA = sr.load_sim(sim)


def task(args):
    gamma, seed = args
    fa = FA
    fa.step = sr.make_step(gamma)
    model, (A, B, a0, aG) = rx.ga_models(fa, seed)
    init, noise, groups = fa.make_cohort(seed, horizon=T)
    mk = lambda **kw: TargetMPC(fa, model, BETA, A, B, a0, aG, fair=True, cH=1, first_budget=True, budget_factor=C, **kw)
    ref = rx.rollout(fa, model, pf.GroupAwareMPC(fa, model, BETA, A, B, a0, aG, fair=False), init, noise, groups)
    out = dict(ref=ref)
    viol = {}
    for name, kw in (("d_floor", dict(target="d", floors=True)), ("d_nofloor", dict(target="d", floors=False)),
                     ("k_floor", dict(target="k", floors=True)), ("k_nofloor", dict(target="k", floors=False))):
        pol = mk(**kw)
        out[name] = rx.rollout(fa, model, pol, init, noise, groups)
        viol[name] = float(pol.floor_violation)
    behind = 0 if ref["k0"] < ref["k1"] else 1
    run_k = lambda kap: rx.rollout(fa, model, rx.TopUp(pf.GroupAwareMPC(fa, model, BETA, A, B, a0, aG, fair=False), kap, behind),
                                   init, noise, groups)
    kap = rx.bisect(lambda k: run_k(k)["u1"], 0.0, 1.0, out["k_floor"]["u1"])
    out["topup_k"] = run_k(kap)
    out["kappa"], out["behind_k"], out["floor_violation"] = float(kap), behind, viol
    fa.step = sr.make_step(1.0)
    return gamma, seed, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=os.environ.get("SIM_DIR", "."))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    tasks = [(g, s) for g in GAMMAS for s in SEEDS]
    res = {}
    with Pool(a.workers, initializer=init_worker, initargs=(a.sim,)) as pool:
        for i, (g, s, o) in enumerate(pool.imap_unordered(task, tasks)):
            res.setdefault(g, {})[s] = o
            if (i + 1) % 15 == 0:
                print(f"{i + 1}/{len(tasks)}", flush=True)
    S = sr.summarise
    arms = ("d_floor", "d_nofloor", "k_floor", "k_nofloor", "topup_k")
    summary = {}
    for g in GAMMAS:
        per = [res[g][s] for s in SEEDS]
        ref = per[0]["ref"]
        row = {"ref_levels": {q: S([p["ref"][q] for p in per]) for q in ("k0", "k1", "k", "kgap")},
               "ref_abs_dgap": S([abs(p["ref"]["gap"]) for p in per])}
        for arm in arms:
            row[arm] = {q: S([p[arm][q] - p["ref"][q] for p in per]) for q in ("k0", "k1", "k", "kgap")}
            row[arm]["abs_kgap"] = S([abs(p[arm]["kgap"]) - abs(p["ref"]["kgap"]) for p in per])
            row[arm]["abs_dgap"] = S([abs(p[arm]["gap"]) - abs(p["ref"]["gap"]) for p in per])
            row[arm]["u1_ratio"] = S([p[arm]["u1"] / p["ref"]["u1"] for p in per])
            row[arm]["u2_ratio"] = S([p[arm]["u2"] / p["ref"]["u2"] for p in per])
            row[arm]["levels"] = {q: S([p[arm][q] for p in per]) for q in ("k0", "k1", "kgap")}
        row["k_floor_minus_topup"] = {q: S([p["k_floor"][q] - p["topup_k"][q] for p in per]) for q in ("k0", "k1", "kgap")}
        row["k_floor_minus_topup"]["abs_kgap"] = S([abs(p["k_floor"]["kgap"]) - abs(p["topup_k"]["kgap"]) for p in per])
        row["max_floor_violation"] = {arm: max(p["floor_violation"][arm] for p in per) for arm in ("d_floor", "k_floor")}
        summary[str(g)] = row
        print(f"== gamma {g}: ref k0 {row['ref_levels']['k0']['mean']:.3f} k1 {row['ref_levels']['k1']['mean']:.3f} kgap {row['ref_levels']['kgap']['mean']:+.3f}")
        for arm in arms:
            r = row[arm]
            print(f"   {arm:10s} dk0 {r['k0']['mean']:+.3f} dk1 {r['k1']['mean']:+.3f} d|kgap| {r['abs_kgap']['mean']:+.3f} "
                  f"d|dgap| {r['abs_dgap']['mean']:+.3f} U1x {r['u1_ratio']['mean']:.2f} U2x {r['u2_ratio']['mean']:.2f}")
    os.makedirs(rx.OUT, exist_ok=True)
    json.dump(dict(seeds=len(SEEDS), c=C, beta=BETA, summary=summary), open(os.path.join(rx.OUT, "knowledge_controller.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
