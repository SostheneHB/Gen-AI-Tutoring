"""Pareto-fair MPC: narrow the predicted group gap without lowering the
predicted knowledge of either group (no levelling down).

    python pareto_fair.py --sim ../../Final_Folder_on_DR --out out_pareto_fair.json

At every turn, with the raw (unprojected) surrogate and the observed states:
  0. the original MPC plan U_raw (beta_ind, rho = 1) is computed for every student;
  1. a linear program finds the smallest predicted |gap| at the end of the MPC
     horizon over all plans in the box whose predicted group-mean knowledge
     (sum over the horizon, the knowledge term of the MPC) is at least that of
     U_raw in each group, plus a margin m.  Its value g* is the largest fair
     reduction (Result 2: a non-levelling-down reduction exists iff g* < |gap_raw|);
  2. the original MPC objective is minimised under the constraints
        predicted gap = gap_raw + theta (g*_signed - gap_raw),   theta = 0.95,
        predicted group knowledge >= that of U_raw  (g = 0, 1),
     exactly, through its concave dual in three multipliers (lambda, mu_0, mu_1):
     for given multipliers every student solves the exact box QP of the original
     MPC with one extra linear term.
If the LP finds no fair reduction, the original MPC plan is kept.
Compared on the same cohorts, noise and fitted models with the original MPC and
the gap-constrained MPC of synthetic_regimes.py; effects are reported for each
group.
"""
import argparse
import json

import numpy as np
from scipy.optimize import linprog, minimize

import synthetic_regimes as sr

SEEDS = list(range(10))
THETA, MARGIN = 0.95, 1e-3


class ParetoFairMPC(sr.CohortMPC):
    def __init__(self, fa, model, beta_student, H=3, rho=1.0):
        super().__init__(fa, model, beta_student, "constraint", H=H, rho=rho)
        prop, ctrl, off = model.dynamics(False)
        dim = len(off)
        infl, smap, drift = np.zeros((dim, H)), np.eye(dim), np.zeros(dim)
        self.vk = np.zeros(H)
        self.kmap, self.koff = np.zeros(dim), 0.0
        for t in range(H):                       # knowledge summed over the horizon
            smap, drift = prop @ smap, prop @ drift + off
            infl = prop @ infl
            infl[:, t] += ctrl
            self.vk += model.decoder[0] @ infl
            self.kmap = self.kmap + model.decoder[0] @ smap
            self.koff += model.decoder[0] @ drift + model.decoder_offset[0]
        self.H = H
        self.steps = self.fair_steps = 0
        self.floor_violation = 0.0
        self.gap_reduction_share = []

    def _solve_w(self, latent, W):
        """Exact per-student box QP with an extra linear term W (n x H)."""
        p = self.pol
        grad = latent @ p.gradient_map.T + getattr(self, '_goff', p.gradient_offset) + W
        cand = np.einsum('aij,...j->...ai', p.solution_maps, grad) + p.solution_offsets
        feas = np.all(np.abs(cand) <= 1 + 1e-9, axis=-1)
        cost = .5 * np.einsum('...ai,ij,...aj->...a', cand, p.hessian, cand) \
            + np.einsum('...ai,...i->...a', cand, grad)
        cost = np.where(feas, cost, np.inf)
        best = np.argmin(cost, axis=-1)
        U = np.clip(np.take_along_axis(cand, best[..., None, None], axis=-2)[..., 0, :], -1, 1)
        return U, np.take_along_axis(cost, best[..., None], axis=-1)[..., 0]

    def __call__(self, latent, groups):
        n, H = len(groups), self.H
        g1 = groups == 1
        n1, n0 = g1.sum(), (~g1).sum()
        s = np.where(g1, 1 / n1, -1 / n0)
        wk = np.where(g1, 1 / n1, 0.0), np.where(~g1, 1 / n0, 0.0)   # group-mean weights
        dbase = latent @ self.base_map + self.base_off
        kbase = latent @ self.kmap + self.koff
        gap = lambda U: s @ (dbase + U @ self.v)
        kg = lambda U, w: w @ (kbase + U @ self.vk)
        U0, _ = self._solve_w(latent, np.zeros((n, H)))                # original MPC plan
        gap0 = gap(U0)
        floors = [kg(U0, w) for w in wk]
        self.steps += 1
        # stage 1: LP  min t  s.t.  -t <= gap(U) <= t,  K_g(U) >= floor_g + m,  |U| <= 1
        nv = n * H
        c = np.zeros(nv + 1)
        c[-1] = 1.0
        gv = np.kron(s, self.v)                  # d gap is linear: gap = s.dbase + gv . vec(U)
        A, b = [np.r_[gv, -1.0], np.r_[-gv, -1.0]], [-(s @ dbase), s @ dbase]
        for w, fl in zip(wk, floors):
            A.append(np.r_[-np.kron(w, self.vk), 0.0])
            b.append(-(fl + MARGIN - w @ kbase))
        lp = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                     bounds=[(-1, 1)] * nv + [(0, None)], method="highs")
        if lp.status != 0 or lp.x[-1] >= abs(gap0) - 1e-6:
            return U0[:, 0]                      # no fair reduction available: keep the MPC plan
        Ulp = lp.x[:-1].reshape(n, H)
        target = gap0 + THETA * (gap(Ulp) - gap0)
        self.fair_steps += 1

        # stage 2: exact dual of  min sum f_i  s.t. gap = target, K_g >= floor_g
        def neg_dual(y):
            lam, mu0, mu1 = y
            W = lam * s[:, None] * self.v[None, :] \
                - (mu0 * wk[0] + mu1 * wk[1])[:, None] * self.vk[None, :]
            U, cost = self._solve_w(latent, W)
            # cost already contains the multiplier terms linear in U
            val = cost.sum() - lam * (target - s @ dbase) + mu0 * (floors[0] - wk[0] @ kbase) \
                + mu1 * (floors[1] - wk[1] @ kbase)
            grad = np.array([gap(U) - target, floors[0] - kg(U, wk[0]), floors[1] - kg(U, wk[1])])
            return -val, -grad
        res = minimize(neg_dual, np.zeros(3), jac=True, method="L-BFGS-B",
                       bounds=[(-1e5, 1e5), (0, 1e5), (0, 1e5)],
                       options=dict(maxiter=500, gtol=1e-10, ftol=1e-14))
        lam, mu0, mu1 = res.x
        W = lam * s[:, None] * self.v[None, :] - (mu0 * wk[0] + mu1 * wk[1])[:, None] * self.vk[None, :]
        U, _ = self._solve_w(latent, W)
        viol = max(0.0, max(fl - kg(U, w) for w, fl in zip(wk, floors)))
        self.floor_violation = max(self.floor_violation, viol)
        self.gap_reduction_share.append(float((abs(gap0) - abs(gap(U))) / max(abs(gap0), 1e-12)))
        return U[:, 0]


class ParetoFairBudgetMPC(ParetoFairMPC):
    """Pareto-fair MPC with, in addition, the control budget of the original
    plan: sum_i ||u_i||_1 <= sum_i ||u_i^raw||_1.  The budget multiplier nu adds
    nu*||u||_1 to every student's QP; the QP stays exact by enumerating, for
    every coordinate, the five regions {-1, free negative, 0, free positive, +1}."""

    def __init__(self, fa, model, beta_student, H=3, rho=1.0):
        super().__init__(fa, model, beta_student, H=H, rho=rho)
        self._tables()
        self.budget_violation = 0.0

    def _tables(self):
        H, Hs = self.H, self.pol.hessian
        maps, offs, sig = [], [], []
        for pat in __import__("itertools").product([-1, -.5, 0, .5, 1], repeat=H):
            pat = np.asarray(pat)
            fixed = np.where(np.abs(pat) != .5, pat, 0.0)
            free = np.flatnonzero(np.abs(pat) == .5)
            M = np.zeros((H, H))
            if len(free):
                M[np.ix_(free, free)] = -np.linalg.inv(Hs[np.ix_(free, free)])
            s = np.where(np.abs(pat) == .5, np.sign(pat), 0.0)
            maps.append(M)
            offs.append(fixed + M @ Hs @ fixed)
            sig.append(s)
        self.l1_maps, self.l1_offs, self.l1_sig = np.asarray(maps), np.asarray(offs), np.asarray(sig)

    def _solve_l1(self, latent, W, nu):
        p = self.pol
        grad = latent @ p.gradient_map.T + getattr(self, '_goff', p.gradient_offset) + W
        w1 = getattr(self, "l1_w", np.ones(self.H))
        cand = np.einsum('aij,...j->...ai', self.l1_maps, grad) \
            + nu * np.einsum('aij,aj->ai', self.l1_maps, self.l1_sig * w1) + self.l1_offs
        ok = np.all(np.abs(cand) <= 1 + 1e-9, axis=-1) & np.all(cand * self.l1_sig >= -1e-12, axis=-1)
        cost = .5 * np.einsum('...ai,ij,...aj->...a', cand, p.hessian, cand) \
            + np.einsum('...ai,...i->...a', cand, grad) + nu * (np.abs(cand) * w1).sum(-1)
        cost = np.where(ok, cost, np.inf)
        best = np.argmin(cost, axis=-1)
        U = np.clip(np.take_along_axis(cand, best[..., None, None], axis=-2)[..., 0, :], -1, 1)
        return U, np.take_along_axis(cost, best[..., None], axis=-1)[..., 0]

    def __call__(self, latent, groups):
        n, H = len(groups), self.H
        g1 = groups == 1
        n1, n0 = g1.sum(), (~g1).sum()
        s = np.where(g1, 1 / n1, -1 / n0)
        wk = np.where(g1, 1 / n1, 0.0), np.where(~g1, 1 / n0, 0.0)
        dbase = self._dbase if hasattr(self, "_dbase") else latent @ self.base_map + self.base_off
        kbase = self._kbase if hasattr(self, "_kbase") else latent @ self.kmap + self.koff
        gap = lambda U: s @ (dbase + U @ self.v)
        kg = lambda U, w: w @ (kbase + U @ self.vk)
        U0, _ = self._solve_w(latent, np.zeros((n, H)))
        w1 = getattr(self, "l1_w", np.ones(H))
        gap0, B0 = gap(U0), getattr(self, "budget_factor", 1.0) * (np.abs(U0) * w1).sum()
        floors = [kg(U0, w) for w in wk]
        self.steps += 1
        nv = n * H                                # LP variables: U, |U|, t
        c = np.zeros(2 * nv + 1)
        c[-1] = 1.0
        gv = np.kron(s, self.v)
        Z = np.zeros(nv)
        A = [np.r_[gv, Z, -1.0], np.r_[-gv, Z, -1.0]]
        b = [-(s @ dbase), s @ dbase]
        for w, fl in zip(wk, floors):
            A.append(np.r_[-np.kron(w, self.vk), Z, 0.0])
            b.append(-(fl + MARGIN - w @ kbase))
        A.append(np.r_[Z, np.tile(w1, n), 0.0])
        b.append(B0 * (1 - MARGIN))
        I = np.eye(nv)
        A2 = np.vstack([np.hstack([I, -I, np.zeros((nv, 1))]), np.hstack([-I, -I, np.zeros((nv, 1))])])
        lp = linprog(c, A_ub=np.vstack([np.array(A), A2]), b_ub=np.r_[b, np.zeros(2 * nv)],
                     bounds=[(-1, 1)] * nv + [(0, 1)] * nv + [(0, None)], method="highs")
        if lp.status != 0 or lp.x[-1] >= abs(gap0) - 1e-6:
            return U0[:, 0]
        target = gap0 + THETA * (gap(lp.x[:nv].reshape(n, H)) - gap0)
        self.fair_steps += 1

        def W_of(y):
            lam, mu0, mu1, _ = y
            return lam * s[:, None] * self.v[None, :] - (mu0 * wk[0] + mu1 * wk[1])[:, None] * self.vk[None, :]

        def neg_dual(y):
            lam, mu0, mu1, nu = y
            U, cost = self._solve_l1(latent, W_of(y), nu)
            val = cost.sum() - lam * (target - s @ dbase) + mu0 * (floors[0] - wk[0] @ kbase) \
                + mu1 * (floors[1] - wk[1] @ kbase) - nu * B0
            grad = np.array([gap(U) - target, floors[0] - kg(U, wk[0]), floors[1] - kg(U, wk[1]),
                             (np.abs(U) * w1).sum() - B0])
            return -val, -grad
        res = minimize(neg_dual, np.zeros(4), jac=True, method="L-BFGS-B",
                       bounds=[(-1e5, 1e5), (0, 1e5), (0, 1e5), (0, 1e5)],
                       options=dict(maxiter=1000, gtol=1e-10, ftol=1e-15))
        U, _ = self._solve_l1(latent, W_of(res.x), res.x[3])
        self.floor_violation = max(self.floor_violation, max(0.0, max(fl - kg(U, w) for w, fl in zip(wk, floors))))
        self.budget_violation = max(self.budget_violation, max(0.0, ((np.abs(U) * w1).sum() - B0) / B0))
        self.gap_reduction_share.append(float((abs(gap0) - abs(gap(U))) / max(abs(gap0), 1e-12)))
        return U[:, 0]

def fit_group_aware(fa, model, seed):
    """Joint least squares [A B a a_G] = Z_+ X^+ with X = [z; u; 1; G], on the
    same training data, lifting and scaling as the original model.  The decoder
    is the original one.  Only the offset depends on the group."""
    states, controls, groups = fa.training_data(seed)
    cur = states[:, :-1].reshape(-1, states.shape[-1])
    nxt = states[:, 1:].reshape(-1, states.shape[-1])
    Z, Zn = model.encode(cur), model.encode(nxt)
    G = np.repeat(groups, states.shape[1] - 1).astype(float)
    X = np.column_stack([Z, controls.reshape(-1), np.ones(len(Z)), G])
    C = np.linalg.lstsq(X, Zn, rcond=None)[0]
    N = Z.shape[1]
    return C[:N].T, C[N], C[N + 1], C[N + 2]


class GroupAwareMPC(ParetoFairBudgetMPC):
    """MPC on the group-aware surrogate.  fair=False: the plain MPC (same
    objective as the original MPC, beta_ind, rho).  fair=True: the Pareto-fair
    budget controller, whose floors and budget come from the plain group-aware
    plan at the same state."""

    def __init__(self, fa, model, beta_student, A, B, a, aG, fair, H=3, rho=1.0, cH=None, first_budget=False,
                 budget_factor=1.0, ref_rho=None):
        """cH: step of the MPC plan at which the gap and the knowledge floors are
        imposed (default H).  cH = 1 constrains the step that the applied
        control decides, which avoids postponing the fair part of the plan."""
        cH = H if cH is None else cH
        dec, doff = model.decoder, model.decoder_offset
        self.pols = [fa.MPCPolicy(A, B, a + g * aG, dec, doff, horizon=H, beta=beta_student, rho=rho) for g in (0, 1)]
        self.pol = self.pols[0]
        self.H, self.fair = H, fair
        dim = len(a)
        infl, smap, drift0, drift1 = np.zeros((dim, H)), np.eye(dim), np.zeros(dim), np.zeros(dim)
        self.vk, self.kmap = np.zeros(H), np.zeros(dim)
        self.koffs = np.zeros(2)
        for t in range(cH):
            smap = A @ smap
            drift0, drift1 = A @ drift0 + a, A @ drift1 + a + aG
            infl = A @ infl
            infl[:, t] += B
            self.vk += dec[0] @ infl
            self.kmap = self.kmap + dec[0] @ smap
            self.koffs += np.array([dec[0] @ drift0, dec[0] @ drift1]) + doff[0]
        self.v = dec[fa.D_IDX] @ infl
        self.base_map = dec[fa.D_IDX] @ smap
        self.base_offs = np.array([dec[fa.D_IDX] @ drift0, dec[fa.D_IDX] @ drift1]) + doff[fa.D_IDX]
        # the five-region tables of the L1 solver (Hessian is the same for both groups)
        ParetoFairBudgetMPC._tables(self)
        self.budget_factor = budget_factor
        if first_budget:                          # budget on the applied control only
            self.l1_w = np.r_[1.0, np.zeros(H - 1)]
        self.steps = self.fair_steps = 0
        self.floor_violation = self.budget_violation = 0.0
        self.gap_reduction_share = []
        self.calls = self.infeasible = 0
        self.lambdas = []
        self.max_residual = 0.0

    def __call__(self, latent, groups):
        gi = groups.astype(int)
        self._goff = np.stack([self.pols[g].gradient_offset for g in (0, 1)])[gi]
        self._dbase = latent @ self.base_map + self.base_offs[gi]
        self._kbase = latent @ self.kmap + self.koffs[gi]
        if not self.fair:
            U, _ = self._solve_w(latent, np.zeros((len(groups), self.H)))
            return U[:, 0]
        return ParetoFairBudgetMPC.__call__(self, latent, groups)


def run_arm(fa, model, pol, init, noise, groups, T):
    x = init.copy()
    S, U = [x.copy()], []
    for t in range(T):
        z = model.encode(x)
        u = pol(z, groups) if isinstance(pol, sr.CohortMPC) else pol(z)
        U.append(u)
        x = fa.step(x, u, groups, noise[:, t])
        S.append(x.copy())
    S, U = np.stack(S, 1), np.stack(U, 1)
    return dict(gap=float(fa.contrast(S[:, -1, fa.D_IDX], groups)), k=float(S[:, -1, 0].mean()),
                k0=float(S[groups == 0, -1, 0].mean()), k1=float(S[groups == 1, -1, 0].mean()),
                d0=float(S[groups == 0, -1, fa.D_IDX].mean()), d1=float(S[groups == 1, -1, fa.D_IDX].mean()),
                u=float(np.abs(U).mean()), u0=float(np.abs(U[groups == 0]).mean()),
                u1=float(np.abs(U[groups == 1]).mean()))


def config(fa, gamma, beta, T=25):
    fa.step = sr.make_step(gamma)
    per = []
    for seed in SEEDS:
        model = fa.fit_model(seed, 4, 3)
        init, noise, groups = fa.make_cohort(seed, horizon=T)
        raw = fa.MPCPolicy(*model.dynamics(False), model.decoder, model.decoder_offset, beta=beta)
        con = sr.CohortMPC(fa, model, 0.0, "constraint")
        par = ParetoFairMPC(fa, model, beta)
        pb = ParetoFairBudgetMPC(fa, model, beta)
        A, B, a0, aG = fit_group_aware(fa, model, seed)
        gam = GroupAwareMPC(fa, model, beta, A, B, a0, aG, fair=False)
        gaf = GroupAwareMPC(fa, model, beta, A, B, a0, aG, fair=True)
        r = {name: run_arm(fa, model, pol, init, noise, groups, T)
             for name, pol in (("raw", raw), ("constraint", con), ("pareto", par), ("pareto_budget", pb),
                               ("ga_mpc", gam), ("ga_fair", gaf))}
        r["fair_step_share"] = gaf.fair_steps / max(gaf.steps, 1)
        r["floor_violation"] = max(par.floor_violation, pb.floor_violation, gaf.floor_violation)
        r["budget_violation"] = max(pb.budget_violation, gaf.budget_violation)
        r["pred_gap_reduction_share"] = float(np.mean(par.gap_reduction_share)) if par.gap_reduction_share else 0.0
        per.append(r)
    fa.step = sr.make_step(1.0)
    out = dict(gamma=gamma, beta_ind=beta, T=T)
    def eff(arm, ref):
        e = {q: sr.summarise([p[arm][q] - p[ref][q] for p in per]) for q in ("k", "k0", "k1", "d0", "d1")}
        e["abs_gap"] = sr.summarise([abs(p[arm]["gap"]) - abs(p[ref]["gap"]) for p in per])
        e["u_ratio"] = sr.summarise([p[arm]["u"] / p[ref]["u"] for p in per])
        e["worst_group"] = sr.summarise([min(p[arm]["k0"] - p[ref]["k0"], p[arm]["k1"] - p[ref]["k1"]) for p in per])
        return e
    for arm in ("constraint", "pareto", "pareto_budget", "ga_mpc", "ga_fair"):
        out[arm] = eff(arm, "raw")
    out["ga_fair_vs_ga_mpc"] = eff("ga_fair", "ga_mpc")
    out["raw_gap"] = sr.summarise([p["raw"]["gap"] for p in per])
    out["raw_u"] = sr.summarise([p["raw"]["u"] for p in per])
    out["fair_step_share"] = sr.summarise([p["fair_step_share"] for p in per])
    out["max_floor_violation"] = float(max(p["floor_violation"] for p in per))
    out["max_budget_violation"] = float(max(p["budget_violation"] for p in per))
    out["pred_gap_reduction_share"] = sr.summarise([p["pred_gap_reduction_share"] for p in per])
    P, C, PB = out["pareto"], out["constraint"], out["pareto_budget"]
    for key in ("ga_fair_vs_ga_mpc", "ga_fair", "ga_mpc"):
        E = out[key]
        print(f"  {key:18s} d|gap| {E['abs_gap']['mean']:+.3f} {np.round(E['abs_gap']['ci'], 3)} dk0 {E['k0']['mean']:+.3f} "
              f"{np.round(E['k0']['ci'], 3)} dk1 {E['k1']['mean']:+.3f} {np.round(E['k1']['ci'], 3)} u x{E['u_ratio']['mean']:.2f}", flush=True)
    print(f"gamma={gamma:g} beta={beta:g} T={T}: BUDGET d|gap| {PB['abs_gap']['mean']:+.3f} {np.round(PB['abs_gap']['ci'], 3)} "
          f"dk0 {PB['k0']['mean']:+.3f} {np.round(PB['k0']['ci'], 3)} dk1 {PB['k1']['mean']:+.3f} {np.round(PB['k1']['ci'], 3)} "
          f"u x{PB['u_ratio']['mean']:.2f} || PARETO d|gap| {P['abs_gap']['mean']:+.3f} {np.round(P['abs_gap']['ci'], 3)} "
          f"dk0 {P['k0']['mean']:+.3f} {np.round(P['k0']['ci'], 3)} dk1 {P['k1']['mean']:+.3f} {np.round(P['k1']['ci'], 3)} "
          f"u x{P['u_ratio']['mean']:.2f} fair-steps {out['fair_step_share']['mean']:.2f} viol {out['max_floor_violation']:.1e} | "
          f"CONSTR d|gap| {C['abs_gap']['mean']:+.3f} dk0 {C['k0']['mean']:+.3f} dk1 {C['k1']['mean']:+.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="../../Final_Folder_on_DR")
    ap.add_argument("--out", default="out_pareto_fair.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    fa = sr.load_sim(a.sim)
    cfgs = [(1.0, 8.0, 25)] if a.quick else \
        [(g, 8.0, 25) for g in (-1.0, 0.25, 0.5, 1.0, 2.0)] + [(1.0, b, 25) for b in (0.0, 1.0, 32.0)] + [(1.0, 8.0, 50)]
    res = [config(fa, g, b, T) for g, b, T in cfgs]
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
