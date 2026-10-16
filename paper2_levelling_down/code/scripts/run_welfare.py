"""Experiment 03: which welfare function chooses the fair plan of Theorem 2?

    python code/scripts/run_welfare.py

Start from the efficient plan (as in experiment 02) and give the planner a small extra
budget eps.  The planner keeps the knowledge floors and maximises a welfare function of
the final disparity of the two groups,
    utilitarian  sum_g n_g D_g,   maximin  min_g D_g,   prioritarian  sum_g n_g phi(D_g),
with phi(x) = -exp(-kappa x) (more concave for larger kappa).  The welfare optimum is
computed exactly (linear programs; Frank-Wolfe for the prioritarian case) and compared with
the knapsack filling.  Proposition 3 predicts the answer from one number of the group
ahead, Lambda_o, and the priority ratio pi = phi'(D_b)/phi'(D_o).
Writes outputs/03_welfare/results.json (aggregate numbers only).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from p2.decomposition import fit_group_models  # noqa: E402
from p2.frontier import (GroupModel, best_return_ahead, efficient_plan, knapsack_plan,  # noqa: E402
                         welfare_plan)
from p2.config import OUT_IDX, OUTCOME, REF_HELP, REF_RULE, TAG  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402

use_paper1_code()
import pareto_fair as pf  # noqa: E402
import synthetic_regimes as sr  # noqa: E402

T, SEEDS, GAMMAS = 25, range(5), (0.5, 1.0, 2.0)
KAPPAS = (0.1, 0.3, 1.0, 3.0, 10.0)
EXTRAS = (0.01, 0.05)                    # extra budget as a share of the reference budget


def lp_welfare(M, v_ref, extra, weights=None, maximin=False):
    """Utilitarian (weights) or maximin optimum under the floors and the budget, as one LP."""
    n, K_ref, B_ref = M.n, M.outcomes(v_ref)[0], M.outcomes(v_ref)[3]
    z = np.zeros(T)
    A_ub = [np.r_[-M.hk[0], z], np.r_[z, -M.hk[1]], np.r_[np.full(T, n[0]), np.full(T, n[1])]]
    b_ub = [M.ak[0] - K_ref[0], M.ak[1] - K_ref[1], B_ref + extra]
    bounds = [(0, 1)] * (2 * T)
    if maximin:                           # variables (v0, v1, m): max m, m <= D_g
        A_ub = [np.r_[a, 0.0] for a in A_ub] + [np.r_[-M.hd[0], z, 1.0], np.r_[z, -M.hd[1], 1.0]]
        b_ub = b_ub + [M.ad[0], M.ad[1]]
        c = np.r_[np.zeros(2 * T), -1.0]
        bounds = bounds + [(None, None)]
    else:
        c = -np.r_[weights[0] * n[0] * M.hd[0], weights[1] * n[1] * M.hd[1]]
    x = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, bounds=bounds, method="highs").x
    return {0: x[:T], 1: x[T:2 * T]}


def is_knapsack(M, w, v_ks, b):
    """Same final outcomes as the knapsack plan and the group ahead unchanged."""
    Kw, Dw, _, _ = M.outcomes(w)
    Kk, Dk, _, _ = M.outcomes(v_ks)
    scale = abs(Dk[b] - M.outcomes({0: w[0] * 0 + v_ks[0], 1: v_ks[1]})[1][b]) + 1e-3
    return bool(abs(Dw[b] - Dk[b]) < 2e-3 * max(abs(Dk[b]), 1e-3) and abs(Dw[1 - b] - Dk[1 - b]) < 1e-3 * scale)


def one(fa, gamma, seed, kind):
    fa.step = sr.make_step(gamma)
    model = fa.fit_model(seed, 4, 3)
    init, _, groups = fa.make_cohort(seed, horizon=T)
    n = {g: int((groups == g).sum()) for g in (0, 1)}
    m0 = {g: model.encode(init[groups == g]).mean(0) for g in (0, 1)}
    args = (m0, model.decoder[0], model.decoder_offset[0], model.decoder[OUT_IDX], model.decoder_offset[OUT_IDX], n, T)
    if kind == "group_aware":
        M = GroupModel(*pf.fit_group_aware(fa, model, seed), *args)
    else:
        st, ctrl, grp = fa.training_data(seed)
        Z, Zn = model.encode(st[:, :-1].reshape(-1, 4)), model.encode(st[:, 1:].reshape(-1, 4))
        M = GroupModel.separate(fit_group_models(Z, ctrl.reshape(-1), Zn, np.repeat(grp, st.shape[1] - 1)), *args)
    fa.step = sr.make_step(1.0)
    v = efficient_plan(M, REF_HELP * T * sum(n.values()), REF_RULE)
    _, D, gap, B = M.outcomes(v)
    b = 1 if gap < 0 else 0
    lam = best_return_ahead(M, v)
    rows = []
    for share in EXTRAS:
        eps = share * B
        v_ks, last = knapsack_plan(M, v, eps)
        D_ks = M.outcomes(v_ks)[1]
        hb = M.hd[b][last]
        if D_ks[b] > D[1 - b]:            # the filling would overshoot the gap: outside the proposition
            continue
        util = lp_welfare(M, v, eps, weights={0: 1.0, 1: 1.0})
        mm = lp_welfare(M, v, eps, maximin=True)
        r = dict(share=share, lam_over_h=lam / hb,
                 util_is_knapsack=is_knapsack(M, util, v_ks, b), util_predicted=bool(hb >= lam),
                 util_gap_change=float(abs(M.outcomes(util)[2]) - abs(gap)),
                 maximin_Db_diff=float(abs(M.outcomes(mm)[1][b] - D_ks[b])), prio=[])
        for kappa in KAPPAS:
            dphi = lambda x, k=kappa: k * np.exp(-k * x)
            pi_eps = dphi(D_ks[b]) / dphi(D[1 - b])      # priority ratio at the knapsack plan
            w = welfare_plan(M, v, eps, dphi)
            r["prio"].append(dict(kappa=kappa, pi=float(pi_eps), predicted=bool(pi_eps * hb >= lam),
                                  observed=is_knapsack(M, w, v_ks, b)))
        rows.append(r)
    return rows


def main():
    out = {}
    for kind in ("group_aware", "separate"):
        fa = sr.load_sim(str(PATHS["simulator"]))
        rows = [r for gamma in GAMMAS for s in SEEDS for r in one(fa, gamma, s, kind)]
        prio = [p for r in rows for p in r["prio"]]
        agree = sum(p["predicted"] == p["observed"] for p in prio)
        util_agree = sum(r["util_is_knapsack"] == r["util_predicted"] for r in rows)
        out[kind] = dict(
            cases=len(rows), lam_over_h=[float(min(r["lam_over_h"] for r in rows)), float(max(r["lam_over_h"] for r in rows))],
            maximin_max_diff=float(max(r["maximin_Db_diff"] for r in rows)),
            utilitarian_is_knapsack=sum(r["util_is_knapsack"] for r in rows), utilitarian_agree=util_agree,
            utilitarian_gap_change=float(np.mean([r["util_gap_change"] for r in rows])),
            prioritarian_agree=agree, prioritarian_cases=len(prio),
            by_kappa={str(k): dict(knapsack=sum(p["observed"] for p in prio if p["kappa"] == k),
                                   predicted=sum(p["predicted"] for p in prio if p["kappa"] == k),
                                   cases=sum(p["kappa"] == k for p in prio),
                                   pi=[float(min(p["pi"] for p in prio if p["kappa"] == k)),
                                       float(max(p["pi"] for p in prio if p["kappa"] == k))]) for k in KAPPAS})
        o = out[kind]
        print(f"\n[{kind}] {o['cases']} cases; Lambda_o / h_b in {o['lam_over_h'][0]:.3f}-{o['lam_over_h'][1]:.3f}")
        print(f"  maximin: D of the group behind equals the knapsack to {o['maximin_max_diff']:.1e}")
        print(f"  utilitarian: knapsack in {o['utilitarian_is_knapsack']}/{o['cases']} (prediction right in {util_agree}/{o['cases']}); "
              f"mean change of |gap| {o['utilitarian_gap_change']:+.4f}")
        print(f"  prioritarian: prediction right in {agree}/{len(prio)}")
        for k, d in o["by_kappa"].items():
            print(f"    kappa={k:>4}: pi {d['pi'][0]:.2f}-{d['pi'][1]:.2f}; knapsack {d['knapsack']}/{d['cases']}, predicted {d['predicted']}/{d['cases']}")
    d = ROOT / "outputs" / "03_welfare"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f"results{TAG}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
