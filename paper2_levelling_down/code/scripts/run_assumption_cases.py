"""Experiment 10: what happens to the price of fairness when the assumptions of Theorem 2 hold or fail.

    python code/scripts/run_assumption_cases.py

Random instances of the affine set-up: two groups, T steps, positive Markov parameters with random
time profiles (knowledge effects h^k_{g,t} and disparity effects h^d_{g,t}, each a positive base level
times a random profile, so that the ratio h^d/h^k changes over time), group sizes n_g, an efficient
reference plan for a random budget, and a small target narrowing rho (5% of the capacity).  For each
instance we check (A1)-(A5), compute the price beta(rho) by the linear program, and compare it with the
closed form of Theorem 2.  We also include the two-step example of Remark 1 (A2 fails, fairness free).
Writes outputs/10_assumptions/results.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from p2.frontier import efficient_plan, fair_frontier, knapsack_price  # noqa: E402


class Inst:
    """Minimal stand-in for GroupModel: Markov parameters and offsets given directly."""

    def __init__(self, hk, hd, ak, ad, n):
        self.hk, self.hd, self.ak, self.ad, self.n, self.T = hk, hd, ak, ad, n, len(hk[0])

    def outcomes(self, v):
        K = {g: self.ak[g] + self.hk[g] @ v[g] for g in (0, 1)}
        D = {g: self.ad[g] + self.hd[g] @ v[g] for g in (0, 1)}
        return K, D, D[1] - D[0], sum(self.n[g] * v[g].sum() for g in (0, 1))


def assumptions(M, v):
    _, _, gap, _ = M.outcomes(v)
    b = 1 if gap < 0 else 0
    o = 1 - b
    F = {g: np.where(v[g] < 1 - 1e-9)[0] for g in (0, 1)}
    S = {g: np.where(v[g] > 1e-9)[0] for g in (0, 1)}
    hk, hd = M.hk, M.hd
    A1 = bool((hk[o] > 0).all() and (hd[o] > 0).all() and (hd[b] > 0).all())
    A2 = all(hd[o][t] / hk[o][t] >= hd[o][u] / hk[o][u] - 1e-12 for t in F[o] for u in S[o])
    A3 = bool((hk[b] >= 0).all())
    A4 = all(hk[g][t] <= hk[g][u] + 1e-12 for g in (0, 1) for t in F[g] for u in S[g])
    A5 = all(hd[b][u] >= hd[b][t] - 1e-12 for u in S[b] for t in F[b])
    return dict(A1=A1, A2=A2, A3=A3, A4=A4, A5=A5)


def random_instance(rng, T):
    prof = lambda: np.exp(rng.normal(0, 0.6, T))
    hk = {g: 0.1 * prof() for g in (0, 1)}
    hd = {g: 0.05 * prof() for g in (0, 1)}
    if rng.random() < 0.5:                       # half of the instances: aligned profiles (A2 likely)
        for g in (0, 1):
            hd[g] = hk[g] * 0.5 * np.exp(rng.normal(0, 0.05, T))
    n = {0: 100, 1: int(rng.integers(30, 100))}
    return Inst(hk, hd, {0: 0.6, 1: 0.5}, {0: 0.0, 1: -0.3}, n)


def main():
    rng = np.random.default_rng(20261015)
    rows = []
    for i in range(2000):
        T = int(rng.integers(3, 12))
        M = random_instance(rng, T)
        v = efficient_plan(M, float(rng.uniform(0.2, 0.7)) * T * sum(M.n.values()))
        cap = knapsack_price(M, v, 0.0)[1]
        if cap <= 1e-9:
            continue
        rho = 0.05 * cap
        lp = fair_frontier(M, v, [rho])[0]
        if not lp["feasible"]:
            continue
        cf = knapsack_price(M, v, rho)[0]
        a = assumptions(M, v)
        rows.append(dict(**a, all=all(a.values()), lp=lp["extra_budget"], cf=cf,
                         match=bool(abs(lp["extra_budget"] - cf) <= 1e-7 * max(1.0, abs(cf))), free=bool(lp["extra_budget"] <= 1e-9)))
    # A4 fails: the reference is not efficient (a random feasible plan with the same budget share)
    rows4 = []
    for i in range(1000):
        T = int(rng.integers(3, 12))
        M = random_instance(rng, T)
        v = {g: rng.uniform(0, 1, T) * rng.uniform(0.3, 1.0) for g in (0, 1)}
        cap = knapsack_price(M, v, 0.0)[1]
        if cap <= 1e-9:
            continue
        rho = 0.05 * cap
        lp0 = fair_frontier(M, v, [0.0])[0]
        lp = fair_frontier(M, v, [rho])[0]
        if not (lp["feasible"] and lp0["feasible"]):
            continue
        a = assumptions(M, v)
        cf = knapsack_price(M, v, rho)[0]
        rows4.append(dict(A4=a["A4"], beta0=lp0["extra_budget"], lp=lp["extra_budget"], cf=cf,
                          match=bool(abs(lp["extra_budget"] - cf) <= 1e-7 * max(1.0, abs(cf)))))
    s4 = [r for r in rows4 if not r["A4"]]
    a4 = dict(n=len(s4), beta0_negative=float(np.mean([r["beta0"] < -1e-9 for r in s4])),
              price_at_most_zero=float(np.mean([r["lp"] <= 1e-9 for r in s4])),
              closed_form_equals_lp=float(np.mean([r["match"] for r in s4])),
              median_beta0_share=float(np.median([r["beta0"] for r in s4])))
    print(f"A4 fails (inefficient reference) n={a4['n']}: beta(0) < 0 in {a4['beta0_negative']:.2f}, price <= 0 at the target in "
          f"{a4['price_at_most_zero']:.2f}, closed form = LP in {a4['closed_form_equals_lp']:.2f}", flush=True)
    summary = {"A4 fails (inefficient reference)": a4}
    for name, sel in (("all assumptions hold", lambda r: r["all"]),
                      ("A2 fails (others hold)", lambda r: not r["A2"] and r["A1"] and r["A3"] and r["A4"] and r["A5"]),
                      ("A5 fails (others hold)", lambda r: not r["A5"] and r["A1"] and r["A2"] and r["A3"] and r["A4"]),
                      ("A2 and A5 fail", lambda r: not r["A2"] and not r["A5"])):
        s = [r for r in rows if sel(r)]
        summary[name] = dict(n=len(s), closed_form_equals_lp=float(np.mean([r["match"] for r in s])) if s else None,
                             price_zero=float(np.mean([r["free"] for r in s])) if s else None,
                             lp_below_closed_form=float(np.mean([r["lp"] < r["cf"] - 1e-9 for r in s])) if s else None)
        print(f"{name:26s} n={len(s):4d} | closed form = LP {summary[name]['closed_form_equals_lp']} | price zero (free) {summary[name]['price_zero']} "
              f"| LP below closed form {summary[name]['lp_below_closed_form']}", flush=True)
    # Remark 1: two steps, group ahead (group 0) h^k=(1,1), h^d=(1,2), reference (0,1)
    M = Inst({0: np.array([1.0, 1.0]), 1: np.array([1.0, 1.0])}, {0: np.array([1.0, 2.0]), 1: np.array([1.0, 1.0])},
             {0: 0.0, 1: 0.0}, {0: 0.0, 1: -5.0}, {0: 1, 1: 1})
    v = {0: np.array([0.0, 1.0]), 1: np.array([0.5, 0.5])}
    ex = dict(assumptions=assumptions(M, v), price_lp=fair_frontier(M, v, [0.1])[0]["extra_budget"])
    print("Remark 1 example:", ex)
    d = ROOT / "outputs" / "10_assumptions"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(dict(summary=summary, remark1=ex, n_instances=len(rows)), open(d / "results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
