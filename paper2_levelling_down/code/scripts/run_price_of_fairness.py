"""Experiment 02: the price of fairness over a horizon (simulator).

    python code/scripts/run_price_of_fairness.py

For each gamma and replicate:
  1. run the group-aware reference MPC of the companion paper [A] on the true simulator; its total
     control is the budget B;
  2. compute the efficient open-loop plan (max total knowledge at budget B) on the
     group-aware model;
  3. trace the fairness frontier from the efficient plan: minimum budget to narrow the
     gap in disparity by rho with no group losing final knowledge, and the shadow price;
  4. apply the efficient plan and the fair plans open-loop on the TRUE simulator (every
     student of group g receives v_{g,t}; same noise) and measure what really happens.
Writes outputs/02_price_of_fairness/results.json (aggregate numbers only).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from p2.frontier import GroupModel, efficient_plan, fair_frontier, knapsack_price  # noqa: E402
from p2.config import OUT_IDX, OUTCOME, REF_HELP, REF_RULE, TAG  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402

use_paper1_code()
import pareto_fair as pf  # noqa: E402
import synthetic_regimes as sr  # noqa: E402

T, SEEDS, GAMMAS = 25, range(10), (0.5, 1.0, 2.0)
FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)          # rho as a fraction of |gap| of the efficient plan


def simulate(fa, init, noise, groups, u_of_t):
    x = init.copy()
    for t in range(T):
        x = fa.step(x, u_of_t(t, x), groups, noise[:, t])
    return x


def observe(x, groups, U):
    k = {g: float(x[groups == g, 0].mean()) for g in (0, 1)}
    d = {g: float(x[groups == g, OUT_IDX].mean()) for g in (0, 1)}
    return dict(k0=k[0], k1=k[1], gap=d[1] - d[0], budget=float(np.abs(U).sum()))


def one(fa, gamma, seed):
    fa.step = sr.make_step(gamma)
    model = fa.fit_model(seed, 4, 3)
    A, B, a, aG = pf.fit_group_aware(fa, model, seed)
    init, noise, groups = fa.make_cohort(seed, horizon=T)
    n = {g: int((groups == g).sum()) for g in (0, 1)}
    # 1. reference MPC (closed loop, true simulator)
    ref = pf.GroupAwareMPC(fa, model, 8.0, A, B, a, aG, fair=False)
    U_ref = []

    def u_ref(t, x):
        u = ref(model.encode(x), groups)
        U_ref.append(u)
        return u
    x_ref = simulate(fa, init, noise, groups, u_ref)
    B_ref = float(np.abs(np.array(U_ref)).sum())
    # 2. efficient plan on the group-aware model
    m0 = {g: model.encode(init[groups == g]).mean(0) for g in (0, 1)}
    M = GroupModel(A, B, a, aG, m0, model.decoder[0], model.decoder_offset[0],
                   model.decoder[OUT_IDX], model.decoder_offset[OUT_IDX], n, T)
    if OUTCOME == "knowledge":         # fixed budget below the knowledge ceiling (see p2/config.py)
        B_ref = REF_HELP * T * sum(n.values())
    v_eff = efficient_plan(M, B_ref, REF_RULE)
    K_eff, _, gap_eff, _ = M.outcomes(v_eff)
    # 3. frontier from the efficient plan
    front = fair_frontier(M, v_eff, [f * abs(gap_eff) for f in FRACTIONS])
    # 4. open-loop on the true simulator
    def open_loop(v):
        U = []
        def u(t, x):
            ut = np.where(groups == 1, v[1][t], v[0][t])
            U.append(ut)
            return ut
        x = simulate(fa, init, noise, groups, u)
        return observe(x, groups, np.array(U))
    obs_ref = observe(x_ref, groups, np.array(U_ref))
    obs_eff = open_loop(v_eff)
    rows = []
    for f, fr in zip(FRACTIONS, front):
        if not fr["feasible"]:
            rows.append(dict(fraction=f, feasible=False))
            continue
        o = open_loop(fr["plan"])
        rows.append(dict(fraction=f, feasible=True, extra_budget_model=fr["extra_budget"] / B_ref,
                         promised_reduction=f * abs(gap_eff), true_reduction=abs(obs_eff["gap"]) - abs(o["gap"]),
                         shadow_price=fr["shadow_price"] / B_ref * abs(gap_eff),
                         true_gap_change=abs(o["gap"]) - abs(obs_eff["gap"]),
                         true_dk0=o["k0"] - obs_eff["k0"], true_dk1=o["k1"] - obs_eff["k1"],
                         true_budget_ratio=o["budget"] / obs_eff["budget"]))
    fa.step = sr.make_step(1.0)
    capacity = knapsack_price(M, v_eff, 0.0)[1]
    return dict(ref=obs_ref, eff=obs_eff, model_gap_eff=gap_eff, capacity=capacity, capacity_share=capacity / abs(gap_eff) if abs(gap_eff) > 1e-9 else None, rows=rows,
                eff_vs_ref=dict(dk0=obs_eff["k0"] - obs_ref["k0"], dk1=obs_eff["k1"] - obs_ref["k1"],
                                gap=abs(obs_eff["gap"]) - abs(obs_ref["gap"]), budget=obs_eff["budget"] / obs_ref["budget"]))


def main():
    fa = sr.load_sim(str(PATHS["simulator"]))
    out = {}
    for gamma in GAMMAS:
        runs = [one(fa, gamma, s) for s in SEEDS]
        mean = lambda xs: float(np.mean(xs))
        e = {k: mean([r["eff_vs_ref"][k] for r in runs]) for k in ("dk0", "dk1", "gap", "budget")}
        print(f"\ngamma={gamma}: efficient plan vs reference MPC (true simulator): dk0 {e['dk0']:+.3f} dk1 {e['dk1']:+.3f} "
              f"d|gap| {e['gap']:+.3f} budget x{e['budget']:.2f}")
        rows = []
        for i, f in enumerate(FRACTIONS):
            ok = [r["rows"][i] for r in runs if r["rows"][i]["feasible"]]
            if not ok:
                print(f"  rho={f:.2f}|gap|: infeasible"); rows.append(dict(fraction=f, feasible=0)); continue
            agg = {k: mean([x[k] for x in ok]) for k in ok[0] if k not in ("fraction", "feasible")}
            agg["min_true_dk"] = float(min(min(x["true_dk0"], x["true_dk1"]) for x in ok))
            agg.update(fraction=f, feasible=len(ok))
            rows.append(agg)
            print(f"  rho={f:.2f}|gap| ({len(ok)}/10 feasible): extra budget (model) {agg['extra_budget_model']:+.3f} of B | "
                  f"shadow price {agg['shadow_price']:.3f} | TRUE: d|gap| {agg['true_gap_change']:+.3f}, "
                  f"dk0 {agg['true_dk0']:+.3f}, dk1 {agg['true_dk1']:+.3f} (worst {agg['min_true_dk']:+.3f}), budget x{agg['true_budget_ratio']:.2f}", flush=True)
        cap = [r["capacity_share"] for r in runs if r["capacity_share"] is not None]
        print(f"  fairness capacity: {np.mean(cap):.3f} of the gap (range {min(cap):.3f}-{max(cap):.3f}); model gap {np.mean([abs(r['model_gap_eff']) for r in runs]):.3f}, true gap {np.mean([abs(r['eff']['gap']) for r in runs]):.3f}")
        out[str(gamma)] = dict(eff_vs_ref=e, frontier=rows, capacity_share=[float(x) for x in cap],
                               model_gap=float(np.mean([abs(r["model_gap_eff"]) for r in runs])),
                               true_gap=float(np.mean([abs(r["eff"]["gap"]) for r in runs])))
    d = ROOT / "outputs" / "02_price_of_fairness"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f"results{TAG}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
