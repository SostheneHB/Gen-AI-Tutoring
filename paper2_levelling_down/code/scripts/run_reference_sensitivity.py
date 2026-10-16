"""Experiment 18: which efficient plan is the reference?  (review of 16 October 2026, third round, point 6)

    P2_OUTCOME=knowledge python code/scripts/run_reference_sensitivity.py

In the group-aware model both groups have the same Markov parameters h^k, so the efficient plan is not unique:
moving help between the groups at equal budget leaves the total knowledge unchanged and changes the gap.  The
rule stated in the manuscript is lexicographic: maximise total knowledge, then minimise |gap| ("min_gap").
Part 1 checks, for every model fit of experiments 02, 12 and 13, whether the plan returned by the solver (used in
those experiments) satisfies this rule (its |gap| equals the minimum over efficient plans, to 1e-9).
Part 2 (fits of experiment 02) compares the two ends of the efficient set, "min_gap" and "max_gap":
model gap and true gap, the extra budget (linear program) to narrow the model gap by half of what the model can
close, and what this plan does on the simulator (ten evaluation cohorts).
Writes outputs/18_reference/results_knowledge.json (aggregate numbers only).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from p2.config import OUT_IDX, OUTCOME, REF_HELP  # noqa: E402
from p2.frontier import GroupModel, efficient_plan, fair_frontier, knapsack_price  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402
from run_certificate_v3 import hetero_step, response  # noqa: E402
from run_model_sensitivity import fit as fit_sized, true_change  # noqa: E402

use_paper1_code()
import pareto_fair as pf  # noqa: E402
import synthetic_regimes as sr  # noqa: E402

T, GAMMAS = 25, (0.5, 1.0, 2.0)


def model_02(fa, gamma, seed, step=None):
    fa.step = step or sr.make_step(gamma)
    model = fa.fit_model(seed, 4, 3)
    A, B, a, aG = pf.fit_group_aware(fa, model, seed)
    init, _, groups = fa.make_cohort(seed, horizon=T)
    n = {g: int((groups == g).sum()) for g in (0, 1)}
    m0 = {g: model.encode(init[groups == g]).mean(0) for g in (0, 1)}
    return GroupModel(A, B, a, aG, m0, model.decoder[0], model.decoder_offset[0], model.decoder[OUT_IDX], model.decoder_offset[OUT_IDX], n, T)


def model_13(fa, gamma, seed, n_train):
    fa.step = sr.make_step(gamma)
    A, B, a, aG, dec, dec0, enc = fit_sized(fa, seed, n_train)
    init, _, groups = fa.make_cohort(seed, horizon=T)
    n = {g: int((groups == g).sum()) for g in (0, 1)}
    m0 = {g: enc(init[groups == g]).mean(0) for g in (0, 1)}
    return GroupModel(A, B, a, aG, m0, dec[0], dec0[0], dec[OUT_IDX], dec0[OUT_IDX], n, T)


def check(M):
    bud = REF_HELP * T * sum(M.n.values())
    g_s = abs(M.outcomes(efficient_plan(M, bud))[2])
    g_min = abs(M.outcomes(efficient_plan(M, bud, "min_gap"))[2])
    g_max = abs(M.outcomes(efficient_plan(M, bud, "max_gap"))[2])
    return dict(solver=g_s, min=g_min, max=g_max, solver_is_min=bool(g_s <= g_min + 1e-9), same_hk=bool(np.allclose(M.hk[0], M.hk[1])))


def main():
    fa = sr.load_sim(str(PATHS["simulator"]))
    checks = []
    for gamma in GAMMAS:
        for seed in range(10):
            checks.append(dict(exp="02", gamma=gamma, seed=seed, **check(model_02(fa, gamma, seed))))
        for regime in ("standard", "heterogeneous"):
            for seed in range(5):
                checks.append(dict(exp=f"12_{regime}", gamma=gamma, seed=seed,
                                   **check(model_02(fa, gamma, seed, hetero_step(gamma, response(seed, 999999, 150, regime))))))
        for n_train in (50, 150, 600):
            for seed in range(10):
                checks.append(dict(exp="13", gamma=gamma, seed=seed, n_train=n_train, **check(model_13(fa, gamma, seed, n_train))))
        print(f"gamma={gamma}: solver = min_gap in {sum(c['solver_is_min'] for c in checks if c['gamma'] == gamma)} of "
              f"{sum(c['gamma'] == gamma for c in checks)} fits", flush=True)
    rows = []
    for gamma in GAMMAS:
        for seed in range(10):
            M = model_02(fa, gamma, seed)
            step = sr.make_step(gamma)
            cohorts = [fa.make_cohort(seed, horizon=T, role="evaluation", index=200 + j) for j in range(10)]
            row = dict(gamma=gamma, seed=seed)
            for rule in ("min_gap", "max_gap"):
                v_ref = efficient_plan(M, REF_HELP * T * sum(M.n.values()), rule)
                K, _, gap, B_ref = M.outcomes(v_ref)
                true_gap = []
                for init, noise, groups in cohorts:
                    x = init.copy()
                    for t in range(T):
                        x = step(x, np.where(groups == 1, v_ref[1][t], v_ref[0][t]), groups, noise[:, t])
                    true_gap.append(x[groups == 1, OUT_IDX].mean() - x[groups == 0, OUT_IDX].mean())
                cap = knapsack_price(M, v_ref, 0.0)[1]
                target = 0.5 * min(cap, abs(gap))
                fr = fair_frontier(M, v_ref, [target])[0]
                r = dict(model_gap=float(gap), true_gap=float(np.mean(true_gap)), K=[float(K[0]), float(K[1])], target=float(target))
                if fr["feasible"]:
                    tc = true_change(step, cohorts, v_ref, fr["plan"])
                    r.update(extra_budget=fr["extra_budget"] / B_ref, true_narrowing=float(tc[2]), true_dK=[float(tc[0]), float(tc[1])])
                row[rule] = r
            rows.append(row)
            print(f"gamma={gamma} seed={seed}: " + " | ".join(
                f"{k} model gap {row[k]['model_gap']:+.3f} true {row[k]['true_gap']:+.3f} target {row[k]['target']:.3f} "
                f"extra {row[k].get('extra_budget', float('nan')):.3f} true narrowing {row[k].get('true_narrowing', float('nan')):+.3f}"
                for k in ("min_gap", "max_gap")), flush=True)
    fa.step = sr.make_step(1.0)
    d = ROOT / "outputs" / "18_reference"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(dict(outcome=OUTCOME, checks=checks, rows=rows), open(d / "results_knowledge.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
