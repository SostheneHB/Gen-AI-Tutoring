"""Experiment 23: is the value of the filling the model, or only the timing?  (fifth report, point 8)

    P2_OUTCOME=knowledge python code/scripts/run_late_help_baseline.py

At the extra budget that the filling needs to narrow the model gap by half of what the model can close (as in the
equal-budget comparison of the paper), four ways of spending the same extra budget on the group behind or on all:
  filling      the model's plan: the steps with the largest effect h^k first;
  late rule    no model: the last steps with room first (T-1, T-2, ...), filled up to 1;
  top-up       the extra help spread over all steps with room of the group behind, in proportion to the room;
  spread all   the same extra help spread over everyone and every step.
Main simulator (gamma in {0.5, 1, 2}, seeds 0-2) and the Eedi-calibrated semi-synthetic system (kappa = 0.5, seeds 0-2).
True changes on 10 evaluation cohorts (same students and noise for all arms).
Writes outputs/23_late_help/results_knowledge.json (aggregate numbers only).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from p2 import semisim  # noqa: E402
from p2.config import REF_HELP  # noqa: E402
from p2.frontier import GroupModel, efficient_plan, knapsack_plan, knapsack_price  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402

use_paper1_code()
import pareto_fair as pf  # noqa: E402
import synthetic_regimes as sr  # noqa: E402

T = 25


def late_rule(v_ref, b, extra_per_student):
    v = {g: v_ref[g].copy() for g in (0, 1)}
    left = extra_per_student
    for t in range(T - 1, -1, -1):
        take = min(1 - v[b][t], left)
        v[b][t] += take
        left -= take
        if left <= 1e-12:
            break
    return v


def arms(M, v_ref):
    _, _, gap, _ = M.outcomes(v_ref)
    b = 1 if gap < 0 else 0
    cap = knapsack_price(M, v_ref, 0.0)[1]
    extra = knapsack_price(M, v_ref, 0.5 * min(cap, abs(gap)))[0]
    fill, _ = knapsack_plan(M, v_ref, extra)
    room = 1 - v_ref[b]
    top = {g: v_ref[g].copy() for g in (0, 1)}
    top[b] = v_ref[b] + room * min(1.0, (extra / M.n[b]) / max(room.sum(), 1e-12))
    spread = {g: np.minimum(1.0, v_ref[g] + extra / sum(M.n.values()) / T) for g in (0, 1)}
    return dict(filling=fill, late_rule=late_rule(v_ref, b, extra / M.n[b]), topup=top, spread_all=spread), extra


def evaluate(cohort_steps, v_ref, plans):
    out = {k: [] for k in plans}
    for (init, noise, groups), step in cohort_steps:
        def run(plan):
            x = init.copy()
            for t in range(T):
                x = step(x, np.where(groups == 1, plan[1][t], plan[0][t]), groups, noise[:, t])
            k = x[:, 0]
            return np.array([k[groups == 0].mean(), k[groups == 1].mean()])
        r = run(v_ref)
        for name, p in plans.items():
            q = run(p)
            out[name].append(abs(r[1] - r[0]) - abs(q[1] - q[0]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def model_of(fa, seed):
    model = fa.fit_model(seed, 4, 3)
    A, B, a, aG = pf.fit_group_aware(fa, model, seed)
    init, _, groups = fa.make_cohort(seed, horizon=T)
    n = {g: int((groups == g).sum()) for g in (0, 1)}
    m0 = {g: model.encode(init[groups == g]).mean(0) for g in (0, 1)}
    return GroupModel(A, B, a, aG, m0, model.decoder[0], model.decoder_offset[0], model.decoder[0], model.decoder_offset[0], n, T)


def main():
    fa = sr.load_sim(str(PATHS["simulator"]))
    orig_cohort = fa.make_cohort
    rows = []
    for gamma in (0.5, 1.0, 2.0):
        for seed in range(3):
            fa.make_cohort = orig_cohort
            fa.step = sr.make_step(gamma)
            M = model_of(fa, seed)
            v_ref = efficient_plan(M, REF_HELP * T * sum(M.n.values()), "min_gap")
            if abs(M.outcomes(v_ref)[2]) <= 1e-9:
                continue
            plans, extra = arms(M, v_ref)
            cs = [(fa.make_cohort(seed, horizon=T, role="evaluation", index=200 + j), sr.make_step(gamma)) for j in range(10)]
            r = dict(system="simulator", gamma=gamma, seed=seed, extra=float(extra), **evaluate(cs, v_ref, plans))
            rows.append(r)
            print(r, flush=True)
    par = semisim.load_params()
    for seed in range(3):
        semisim.patch(fa, par, 0.5, seed)
        M = model_of(fa, seed)
        v_ref = efficient_plan(M, REF_HELP * T * sum(M.n.values()), "min_gap")
        plans, extra = arms(M, v_ref)
        cs = []
        for j in range(10):
            co = fa.make_cohort(seed, horizon=T, role="evaluation", index=200 + j)
            cs.append((co, semisim.cohort_step(fa, par, 0.5, seed, "evaluation", 200 + j, len(co[2]))))
        r = dict(system="semi-synthetic", kappa=0.5, seed=seed, extra=float(extra), **evaluate(cs, v_ref, plans))
        rows.append(r)
        print(r, flush=True)
    d = ROOT / "outputs" / "23_late_help"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(dict(rows=rows), open(d / "results_knowledge.json", "w"), indent=1)


if __name__ == "__main__":
    main()
