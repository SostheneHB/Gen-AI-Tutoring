"""Experiment 21: losses of individual students and temporary drops of a group mean (review of 16 October 2026,
reviewer 1, point 7).

    P2_OUTCOME=knowledge python code/scripts/run_individual_losses.py

The floors protect the mean FINAL knowledge of each group.  For the plans of the certificate experiment (margins
0 and 0.3, regimes standard, heterogeneous and hard; three fitted models per gamma) we apply the reference and the
plan to the same students with the same noise and the same response factors, and report on 20 evaluation cohorts:
  * the share of students of each group whose final knowledge is lower under the plan than under the reference
    (by more than 1e-9), and the mean size of these losses;
  * the largest drop, over the steps, of each group's mean knowledge under the plan compared with the reference.
Writes outputs/21_individual_losses/results_knowledge.json (aggregate numbers only).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from p2.certificate import margin_plan  # noqa: E402
from p2.config import OUTCOME, REF_HELP, REF_RULE  # noqa: E402
from p2.frontier import GroupModel, efficient_plan, knapsack_price  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402
from run_certificate_v3 import hetero_step, max_narrowing, response  # noqa: E402

use_paper1_code()
import pareto_fair as pf  # noqa: E402
import synthetic_regimes as sr  # noqa: E402

T, GAMMAS, SEEDS, N_EVAL = 25, (0.5, 1.0, 2.0), range(3), 20


def trajectories(step, co, plan):
    init, noise, groups = co
    x, ks = init.copy(), [init[:, 0].copy()]
    for t in range(T):
        x = step(x, np.where(groups == 1, plan[1][t], plan[0][t]), groups, noise[:, t])
        ks.append(x[:, 0].copy())
    return np.array(ks)                                   # (T+1, students)


def one(fa, gamma, seed, regime):
    fa.step = hetero_step(gamma, response(seed, 999999, 150, regime))
    model = fa.fit_model(seed, 4, 3)
    A, B, a, aG = pf.fit_group_aware(fa, model, seed)
    init, _, groups = fa.make_cohort(seed, horizon=T)
    n = {g: int((groups == g).sum()) for g in (0, 1)}
    m0 = {g: model.encode(init[groups == g]).mean(0) for g in (0, 1)}
    M = GroupModel(A, B, a, aG, m0, model.decoder[0], model.decoder_offset[0], model.decoder[0], model.decoder_offset[0], n, T)
    if regime == "hard":
        v_ref = {g: np.full(T, REF_HELP) for g in (0, 1)}
        cap = max_narrowing(M, v_ref)
    else:
        v_ref = efficient_plan(M, REF_HELP * T * sum(n.values()), REF_RULE)
        cap = min(knapsack_price(M, v_ref, 0.0)[1], abs(M.outcomes(v_ref)[2]))
    rho = 0.5 * cap
    fa.step = sr.make_step(1.0)
    if rho <= 1e-9:
        return None
    out = {}
    for d in (0.0, 0.3):
        plan = margin_plan(M, v_ref, rho, d)
        if plan is None or rho / (1 - d) > 0.9 * cap:
            continue
        share, size, drop = {0: [], 1: []}, {0: [], 1: []}, {0: [], 1: []}
        for j in range(N_EVAL):
            co = fa.make_cohort(seed, horizon=T, role="evaluation", index=3000 + j)
            step = hetero_step(gamma, response(seed, 3000 + j, len(co[2]), regime))
            kr, kp = trajectories(step, co, v_ref), trajectories(step, co, plan)
            g_ = co[2]
            for g in (0, 1):
                diff = kp[-1, g_ == g] - kr[-1, g_ == g]
                lost = diff < -1e-9
                share[g].append(lost.mean())
                size[g].append(float(-diff[lost].mean()) if lost.any() else 0.0)
                drop[g].append(float(np.max(kr[:, g_ == g].mean(1) - kp[:, g_ == g].mean(1))))
        out[str(d)] = {str(g): dict(share_students_losing=float(np.mean(share[g])), mean_loss_if_losing=float(np.mean(size[g])),
                                    largest_temporary_drop_of_mean=float(np.max(drop[g]))) for g in (0, 1)}
    return out


def main():
    fa = sr.load_sim(str(PATHS["simulator"]))
    res = {}
    for regime in ("standard", "heterogeneous", "hard"):
        for gamma in GAMMAS:
            for seed in SEEDS:
                r = one(fa, gamma, seed, regime)
                res[f"{regime}|{gamma}|{seed}"] = r
                if r:
                    print(regime, gamma, seed, {d: {g: (round(v["share_students_losing"], 3), round(v["mean_loss_if_losing"], 4),
                                                         round(v["largest_temporary_drop_of_mean"], 4)) for g, v in x.items()} for d, x in r.items()}, flush=True)
    d = ROOT / "outputs" / "21_individual_losses"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(dict(outcome=OUTCOME, ref_rule=REF_RULE, results=res), open(d / "results_knowledge.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
