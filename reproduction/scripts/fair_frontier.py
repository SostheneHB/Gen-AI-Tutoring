"""Gap reduction without levelling down: budget frontier of the Pareto-fair MPC.

    python fair_frontier.py --gamma 1 --out out_frontier_g1.json

All arms use the group-aware surrogate (fit_group_aware in pareto_fair.py) on
the same cohorts, noise and training data (ten replicates):
  plain(rho)   the MPC of Eq. (mpc) with control cost rho (rho = 1 is the reference);
               a smaller rho buys more control;
  fair(c)      the Pareto-fair MPC: at every turn it narrows the predicted gap
               of the next step while (i) the predicted next-step knowledge of
               each group is at least that of the reference plan and (ii) the
               applied control budget sum_i |u_i| is at most c times that of the
               reference plan (LP for the largest fair reduction, then the exact
               dual QP, see pareto_fair.py).
Effects are paired with plain(1).  For c = 1 no fair reduction exists (the LP
is infeasible at every step), as predicted by the Farkas condition.
"""
import argparse
import json

import numpy as np

import pareto_fair as pf
import synthetic_regimes as sr

RHOS = (1.0, 0.6, 0.4, 0.25, 0.15, 0.1)
FACTORS = (1.0, 1.1, 1.25, 1.5, 1.75, 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="../../Final_Folder_on_DR")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--out", default="out_frontier.json")
    a = ap.parse_args()
    fa = sr.load_sim(a.sim)
    fa.step = sr.make_step(a.gamma)
    arms = [("plain", r) for r in RHOS] + [("fair", c) for c in FACTORS]
    per = {f"{k}_{x:g}": [] for k, x in arms}
    for seed in sr.SEEDS:
        model = fa.fit_model(seed, 4, 3)
        A, B, a0, aG = pf.fit_group_aware(fa, model, seed)
        init, noise, groups = fa.make_cohort(seed, horizon=25)
        ref = pf.run_arm(fa, model, pf.GroupAwareMPC(fa, model, a.beta, A, B, a0, aG, False), init, noise, groups, 25)
        for kind, x in arms:
            if kind == "plain":
                pol = pf.GroupAwareMPC(fa, model, a.beta, A, B, a0, aG, False, rho=x)
            else:
                pol = pf.GroupAwareMPC(fa, model, a.beta, A, B, a0, aG, True, cH=1, first_budget=True, budget_factor=x)
            r = pf.run_arm(fa, model, pol, init, noise, groups, 25)
            per[f"{kind}_{x:g}"].append(dict(
                u_ratio=r["u"] / ref["u"], abs_gap=abs(r["gap"]) - abs(ref["gap"]),
                k=r["k"] - ref["k"], k0=r["k0"] - ref["k0"], k1=r["k1"] - ref["k1"],
                d0=r["d0"] - ref["d0"], d1=r["d1"] - ref["d1"],
                worst=min(r["k0"] - ref["k0"], r["k1"] - ref["k1"]),
                fair_steps=getattr(pol, "fair_steps", 0) / max(getattr(pol, "steps", 1), 1),
                floor_violation=getattr(pol, "floor_violation", 0.0),
                budget_violation=getattr(pol, "budget_violation", 0.0),
                ref_gap=ref["gap"]))
    out = {"gamma": a.gamma, "beta_ind": a.beta, "arms": {}}
    for name, rows in per.items():
        out["arms"][name] = {q: sr.summarise([r[q] for r in rows]) for q in rows[0]}
        e = out["arms"][name]
        print(f"gamma={a.gamma:g} {name:10s} u x{e['u_ratio']['mean']:.2f}  d|gap| {e['abs_gap']['mean']:+.3f} "
              f"[{e['abs_gap']['ci'][0]:+.3f},{e['abs_gap']['ci'][1]:+.3f}]  dk0 {e['k0']['mean']:+.4f} (min {e['k0']['min']:+.4f})  "
              f"dk1 {e['k1']['mean']:+.3f} (min {e['k1']['min']:+.4f})  worst-min {e['worst']['min']:+.4f}  "
              f"fair-steps {e['fair_steps']['mean']:.2f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
