"""Experiment 08: is the fairness capacity a limit of the simulated system, of the affine model, or of
the class of plans?

    python code/scripts/run_capacity_simulator.py [--seeds 3]

For each group strength gamma and fitted model (same set-up as experiments 02 and 04):
  * model capacity rho_max (Theorem 2, affine model, open-loop plans);
  * full filling: the plan that reaches rho_max in the model, applied to the true simulator;
  * direct search on the true simulator, open-loop: maximise the narrowing of the gap over plans
    (v_0, v_1) in [0,1]^{2T}, subject to the same floors (true mean final knowledge of each group not
    below the reference), no budget limit (SLSQP, finite differences, common random numbers on a
    design cohort);
  * the same search over a simple adaptive class: u_i,t = clip(v_{g,t} + kappa_g (kbar_{g,t} - k_{i,t}), 0, 1),
    where kbar is the mean knowledge of the student's group at step t (help moves towards the weaker
    students of each group);
  * equal budget: at the extra budget of the filling that narrows the gap by half of min(capacity, |gap|), the
    filling, a uniform top-up for the group behind, and the same extra help spread over everyone.
All final numbers are measured on independent evaluation cohorts, not on the design cohort.
Writes outputs/08_capacity/results.json (aggregate numbers only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from p2.frontier import GroupModel, efficient_plan, knapsack_plan, knapsack_price  # noqa: E402
from p2.config import OUT_IDX, OUTCOME, REF_HELP, TAG  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402

use_paper1_code()
import pareto_fair as pf  # noqa: E402
import synthetic_regimes as sr  # noqa: E402

T, GAMMAS = 25, (0.5, 1.0, 2.0)
N_DESIGN, N_EVAL, EVAL_COHORTS = 1500, 1500, 10


def cohort(fa, seed, n, index, role="evaluation"):
    return fa.make_cohort(seed, students=n, horizon=T, role=role, index=index)


def rollout(step, co, plan, kappa=None):
    """Mean final knowledge and disparity of each group, and total control, for an open-loop plan
    (dict g -> v_g) with optional feedback gains kappa (dict g -> float)."""
    init, noise, groups = co
    x = init.copy()
    total = 0.0
    for t in range(T):
        u = np.where(groups == 1, plan[1][t], plan[0][t])
        if kappa is not None:
            for g in (0, 1):
                m = groups == g
                u[m] = u[m] + kappa[g] * (x[m, 0].mean() - x[m, 0])
            u = np.clip(u, 0, 1)
        total += u.sum()
        x = step(x, u, groups, noise[:, t])
    K = {g: float(x[groups == g, 0].mean()) for g in (0, 1)}
    D = {g: float(x[groups == g, OUT_IDX].mean()) for g in (0, 1)}
    return K, D, D[1] - D[0], total


def evaluate(step, cohorts, plan, ref, kappa=None):
    """Mean over evaluation cohorts of the change against the reference (same students and noise)."""
    out = []
    for co in cohorts:
        K, _, gap, B = rollout(step, co, plan, kappa)
        Kr, _, gr, Br = rollout(step, co, ref)
        out.append((K[0] - Kr[0], K[1] - Kr[1], abs(gr) - abs(gap), (B - Br) / Br))
    o = np.array(out)
    return dict(dK0=float(o[:, 0].mean()), dK1=float(o[:, 1].mean()), narrowing=float(o[:, 2].mean()),
                narrowing_sd=float(o[:, 2].std()), extra_budget=float(o[:, 3].mean()),
                floors_kept=float(np.mean((o[:, 0] >= -1e-9) & (o[:, 1] >= -1e-9))))


def search(step, design, ref, sgn, adaptive, start):
    """max narrowing  s.t.  true mean knowledge of each group >= reference, box (design cohort).
    start: a feasible plan to start from (the full filling, or the open-loop optimum)."""
    K_ref, _, gap_ref, _ = rollout(step, design, ref)
    x0 = np.r_[start[0], start[1]] if not adaptive else np.r_[start[0], start[1], 0.0, 0.0]

    def unpack(x):
        plan = {0: x[:T], 1: x[T:2 * T]}
        return plan, ({0: x[2 * T], 1: x[2 * T + 1]} if adaptive else None)

    def obj(x):
        plan, kap = unpack(x)
        return -sgn * (rollout(step, design, plan, kap)[2] - gap_ref)

    cons = [{"type": "ineq", "fun": (lambda x, g=g: rollout(step, design, *unpack(x))[0][g] - K_ref[g])} for g in (0, 1)]
    # no overshoot: the oriented change of the gap stays below |gap_ref|
    cons.append({"type": "ineq", "fun": lambda x: abs(gap_ref) - sgn * (rollout(step, design, *unpack(x))[2] - gap_ref)})
    bounds = [(0, 1)] * (2 * T) + ([(-3, 3)] * 2 if adaptive else [])
    r = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                 options=dict(maxiter=200, ftol=1e-9, eps=1e-4))
    x = np.clip(r.x, [lo for lo, _ in bounds], [hi for _, hi in bounds])
    feasible = min(c["fun"](x) for c in cons) >= -1e-6
    kept_start = (not feasible) or obj(x) > obj(x0)          # SLSQP can end worse than its feasible start
    if kept_start:
        x = x0
    plan, kap = unpack(x)
    return plan, kap, dict(success=bool(r.success), iterations=int(r.nit), message=str(r.message), kept_start=bool(kept_start))


def one(fa, gamma, seed):
    step = sr.make_step(gamma)
    fa.step = step
    model = fa.fit_model(seed, 4, 3)
    A, B, a, aG = pf.fit_group_aware(fa, model, seed)
    init, _, groups = fa.make_cohort(seed, horizon=T)
    n = {g: int((groups == g).sum()) for g in (0, 1)}
    m0 = {g: model.encode(init[groups == g]).mean(0) for g in (0, 1)}
    M = GroupModel(A, B, a, aG, m0, model.decoder[0], model.decoder_offset[0],
                   model.decoder[OUT_IDX], model.decoder_offset[OUT_IDX], n, T)
    v_ref = efficient_plan(M, REF_HELP * T * sum(n.values()))
    _, _, gap_ref_model, B_ref_model = M.outcomes(v_ref)
    sgn = -np.sign(gap_ref_model)
    cap = knapsack_price(M, v_ref, 0.0)[1]
    b = 1 if gap_ref_model < 0 else 0
    # the plan that closes the gap as far as the model allows, without overshooting: min(capacity, |gap|)
    close_target = min(cap, abs(gap_ref_model))
    full, _ = knapsack_plan(M, v_ref, knapsack_price(M, v_ref, close_target)[0])
    design = cohort(fa, seed, N_DESIGN, 0, role="calibration")
    evals = [cohort(fa, seed, N_EVAL, 100 + j) for j in range(EVAL_COHORTS)]
    gap_true_ref = float(np.mean([abs(rollout(step, co, v_ref)[2]) for co in evals]))
    res = dict(gamma=gamma, seed=seed, gap_ref_true=gap_true_ref, gap_ref_model=float(abs(gap_ref_model)),
               capacity_model=float(cap), capacity_model_share=float(cap / abs(gap_ref_model)),
               closing_target_model=float(close_target))
    res["full_filling_true"] = evaluate(step, evals, full, v_ref)
    start = full
    for name, adaptive in (("search_open_loop", False), ("search_adaptive", True)):
        plan, kap, info = search(step, design, v_ref, sgn, adaptive, start)
        if not adaptive:
            start = plan
        r = evaluate(step, evals, plan, v_ref, kap)
        r.update(info)
        if kap is not None:
            r["kappa"] = {str(g): float(kap[g]) for g in (0, 1)}
        r["group_ahead_change"] = float(np.abs(plan[1 - b] - v_ref[1 - b]).sum())
        res[name] = r
    # equal budget at half the capacity
    half_extra = knapsack_price(M, v_ref, 0.5 * close_target)[0]     # half of what the model can close, no overshoot
    fill, _ = knapsack_plan(M, v_ref, half_extra)
    top = {g: v_ref[g].copy() for g in (0, 1)}
    room = 1 - v_ref[b]
    top[b] = v_ref[b] + room * min(1.0, (half_extra / n[b]) / max(room.sum(), 1e-12))
    spread = {g: np.minimum(1.0, v_ref[g] + half_extra / sum(n.values()) / T) for g in (0, 1)}
    res["equal_budget"] = {k: evaluate(step, evals, p, v_ref) for k, p in (("filling", fill), ("topup_behind", top), ("spread_all", spread))}
    res["equal_budget"]["extra_budget_model"] = float(half_extra / B_ref_model)
    fa.step = sr.make_step(1.0)
    return res


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--seeds", type=int, default=3)
    a = a_.parse_args()
    fa = sr.load_sim(str(PATHS["simulator"]))
    out = {}
    for gamma in GAMMAS:
        runs = []
        for s in range(a.seeds):
            r = one(fa, gamma, s)
            runs.append(r)
            print(f"gamma={gamma} seed={s}: gap {r['gap_ref_true']:.3f} | model capacity {r['capacity_model']:.3f} ({r['capacity_model_share']:.0%}) "
                  f"| full filling true {r['full_filling_true']['narrowing']:.3f} floors {r['full_filling_true']['floors_kept']:.2f} "
                  f"| search open-loop {r['search_open_loop']['narrowing']:.3f} (dK {r['search_open_loop']['dK0']:+.3f}/{r['search_open_loop']['dK1']:+.3f}, ahead moved {r['search_open_loop']['group_ahead_change']:.2f}) "
                  f"| adaptive {r['search_adaptive']['narrowing']:.3f} kappa {r['search_adaptive']['kappa']} "
                  f"| equal budget: fill {r['equal_budget']['filling']['narrowing']:.3f} top-up {r['equal_budget']['topup_behind']['narrowing']:.3f} "
                  f"spread {r['equal_budget']['spread_all']['narrowing']:.3f}", flush=True)
            out[str(gamma)] = runs
            d = ROOT / "outputs" / "08_capacity"
            d.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(d / f"results{TAG}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
