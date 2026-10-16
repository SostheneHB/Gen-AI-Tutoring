"""Experiment 09: the conformal certificate under repeated calibration, reported decision by decision.

    python code/scripts/run_certificate_v2.py [--repeats 200]

Two regimes, three group strengths, five fitted models each.
  standard  reference = the efficient plan (as in experiment 04): the fair plans only add teaching to the
            group behind, so the group ahead cannot lose and the floor of the group behind is not binding;
  hard_slack  as hard, with plans that promise a knowledge gain of at least 0.01 in each group (a margin
            on the floors, fixed in advance);
  hard      reference = a uniform plan (0.3 at every step, not efficient): the cheapest fair plans move
            teaching between steps inside the groups, the knowledge floors are binding in the model, and a
            group mean can really go down on the simulator.
For each model, the family of plans w^delta is fixed from the model only.  A pool of cohorts is drawn
once; for each cohort the reference and every plan are applied to the same students (common random
numbers).  The first 2,000 cohorts form the calibration pool and the last 1,000 the test pool.  Each
repeat draws m = 100 calibration cohorts from the calibration pool, computes the margin, chooses the
plan (or keeps the reference), and is evaluated on the whole test pool.  Reported, over repeats:
  * marginal coverage of the score, and the distribution of the coverage given the calibration set;
  * how often the plan is changed (intervention) or the reference kept (fallback);
  * success of the guarantee event (no group mean loses knowledge; if changed, narrowing >= rho) over all
    test cohorts, and among the repeats that intervene;
  * mean narrowing and extra budget of the plans used.
Also: 20 disjoint calibration sets of m cohorts (no shared cohort), each evaluated on the test pool.
Writes outputs/09_certificate_v2/results.json (aggregate numbers only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from p2.certificate import conformal_margin, margin_plan, predicted_change, score  # noqa: E402
from p2.frontier import GroupModel, efficient_plan, knapsack_price  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402

use_paper1_code()
import pareto_fair as pf  # noqa: E402
import synthetic_regimes as sr  # noqa: E402

T, SEEDS, GAMMAS = 25, range(5), (0.5, 1.0, 2.0)
M_CAL, POOL_CAL, POOL_TEST, ALPHA = 100, 2000, 1000, 0.1
DELTAS = np.round(np.arange(0, 0.301, 0.02), 4)


def realised(step, co, v_ref, plans):
    """For one cohort: (dK0, dK1, narrowing) of each plan against the reference, same students and noise."""
    init, noise, groups = co

    def run(plan):
        x = init.copy()
        for t in range(T):
            x = step(x, np.where(groups == 1, plan[1][t], plan[0][t]), groups, noise[:, t])
        return {g: x[groups == g, 0].mean() for g in (0, 1)}, x[groups == 1, 3].mean() - x[groups == 0, 3].mean()

    k_ref, gap_ref = run(v_ref)
    out = []
    for p in plans:
        k, gap = run(p)
        out.append((k[0] - k_ref[0], k[1] - k_ref[1], abs(gap_ref) - abs(gap)))
    return np.array(out)


def max_narrowing(M, v_ref):
    """Largest narrowing in the model with both floors and the box, any budget (LP)."""
    T_ = M.T
    _, _, gap_ref, _ = M.outcomes(v_ref)
    sgn = -np.sign(gap_ref)
    c = -sgn * np.r_[-M.hd[0], M.hd[1]]
    z = np.zeros(T_)
    A_ub = [np.r_[-M.hk[0], z], np.r_[z, -M.hk[1]]]
    bounds = [(-v_ref[0][t], 1 - v_ref[0][t]) for t in range(T_)] + [(-v_ref[1][t], 1 - v_ref[1][t]) for t in range(T_)]
    r = linprog(c, A_ub=np.array(A_ub), b_ub=[0.0, 0.0], bounds=bounds, method="highs")
    return float(min(-r.fun, abs(gap_ref)))


def one(fa, gamma, seed, regime, repeats, rng):
    step = sr.make_step(gamma)
    fa.step = step
    model = fa.fit_model(seed, 4, 3)
    A, B, a, aG = pf.fit_group_aware(fa, model, seed)
    init, _, groups = fa.make_cohort(seed, horizon=T)
    n = {g: int((groups == g).sum()) for g in (0, 1)}
    m0 = {g: model.encode(init[groups == g]).mean(0) for g in (0, 1)}
    M = GroupModel(A, B, a, aG, m0, model.decoder[0], model.decoder_offset[0],
                   model.decoder[fa.D_IDX], model.decoder_offset[fa.D_IDX], n, T)
    if regime == "standard":
        v_ref = efficient_plan(M, 0.3 * T * sum(n.values()))
        cap = knapsack_price(M, v_ref, 0.0)[1]
    else:                                   # "hard" and "hard_slack": uniform, inefficient reference
        v_ref = {g: np.full(T, 0.3) for g in (0, 1)}
        cap = max_narrowing(M, v_ref)
    slack = 0.01 if regime == "hard_slack" else 0.0
    rho = 0.5 * cap
    deltas, plans, preds = [], [], []
    for d in DELTAS:
        p = margin_plan(M, v_ref, rho, d, slack)
        if p is not None and rho / (1 - d) <= 0.9 * cap:
            deltas.append(float(d))
            plans.append(p)
            preds.append(predicted_change(M, v_ref, p))
    B_ref = M.outcomes(v_ref)[3]
    extra = np.array([(M.outcomes(p)[3] - B_ref) / B_ref for p in plans])
    # a model floor is exactly met in a group whose plan changes
    binding = [any(np.abs(p[g] - v_ref[g]).sum() > 1e-9 and pr[g] < slack + 1e-9 for g in (0, 1)) for p, pr in zip(plans, preds)]
    truth = np.array([realised(step, fa.make_cohort(seed, horizon=T, role="evaluation", index=1000 + j), v_ref, plans)
                      for j in range(POOL_CAL + POOL_TEST)])      # (cohorts, plans, 3)
    S = np.array([max(score(preds[i], truth[c, i]) for i in range(len(plans))) for c in range(len(truth))])
    cal_pool, test = np.arange(POOL_CAL), np.arange(POOL_CAL, POOL_CAL + POOL_TEST)
    ok = (truth[test, :, 0] >= 0) & (truth[test, :, 1] >= 0)                   # no group mean loses
    reach = truth[test, :, 2] >= rho
    reps = []
    for _ in range(repeats):
        cal = rng.choice(cal_pool, M_CAL, replace=False)
        q = conformal_margin(S[cal], ALPHA)
        idx = next((i for i, d in enumerate(deltas) if d >= q), None)
        cover = float((S[test] <= q).mean())
        if idx is None:
            reps.append(dict(intervene=False, cover=cover, success=1.0))
        else:
            reps.append(dict(intervene=True, cover=cover, delta=deltas[idx], success=float((ok[:, idx] & reach[:, idx]).mean()),
                             no_loss=float(ok[:, idx].mean()), narrowing=float(truth[test, idx, 2].mean()), extra=float(extra[idx])))
    # truly independent calibrations: 20 disjoint sets of m cohorts from the calibration pool
    disj = []
    for k in range(POOL_CAL // M_CAL):
        cal = cal_pool[k * M_CAL:(k + 1) * M_CAL]
        q = conformal_margin(S[cal], ALPHA)
        idx = next((i for i, d in enumerate(deltas) if d >= q), None)
        disj.append(dict(cover=float((S[test] <= q).mean()), intervene=idx is not None,
                         success=1.0 if idx is None else float((ok[:, idx] & reach[:, idx]).mean())))
    iv = [r for r in reps if r["intervene"]]
    naive = 0 if deltas and deltas[0] == 0.0 else None
    res = dict(regime=regime, gamma=gamma, seed=seed, rho=rho, capacity=cap, family=len(plans),
               floors_binding_in_model=int(sum(binding)),
               marginal_cover=float(np.mean([r["cover"] for r in reps])),
               cover_p05=float(np.percentile([r["cover"] for r in reps], 5)),
               share_cal_sets_below_target=float(np.mean([r["cover"] < 1 - ALPHA for r in reps])),
               intervention_rate=float(len(iv) / repeats), fallback_rate=float(1 - len(iv) / repeats),
               disjoint=dict(n_sets=len(disj), cover_mean=float(np.mean([d["cover"] for d in disj])),
                             cover_min=float(np.min([d["cover"] for d in disj])),
                             share_below_target=float(np.mean([d["cover"] < 1 - ALPHA for d in disj])),
                             intervention_rate=float(np.mean([d["intervene"] for d in disj])),
                             success_min=float(np.min([d["success"] for d in disj]))),
               success_overall=float(np.mean([r["success"] for r in reps])),
               success_given_intervention=float(np.mean([r["success"] for r in iv])) if iv else None,
               no_loss_given_intervention=float(np.mean([r["no_loss"] for r in iv])) if iv else None,
               narrowing_given_intervention=float(np.mean([r["narrowing"] for r in iv])) if iv else None,
               extra_budget_given_intervention=float(np.mean([r["extra"] for r in iv])) if iv else None,
               naive=None if naive is None else dict(success=float((ok[:, 0] & reach[:, 0]).mean()), no_loss=float(ok[:, 0].mean()),
                                                     dK_true=[float(truth[test, 0, 0].mean()), float(truth[test, 0, 1].mean())],
                                                     dK_predicted=[float(preds[0][0]), float(preds[0][1])],
                                                     narrowing=float(truth[test, 0, 2].mean()), extra=float(extra[0])))
    fa.step = sr.make_step(1.0)
    return res


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--repeats", type=int, default=200)
    a = a_.parse_args()
    fa = sr.load_sim(str(PATHS["simulator"]))
    rng = np.random.default_rng(20261015)
    out = {}
    for regime in ("standard", "hard"):
        for gamma in GAMMAS:
            runs = []
            for s in SEEDS:
                r = one(fa, gamma, s, regime, a.repeats, rng)
                runs.append(r)
                fmt = lambda v: "--" if v is None else f"{v:.3f}"
                print(f"{regime:8s} gamma={gamma} seed={s}: rho {r['rho']:.3f} family {r['family']} binding {r['floors_binding_in_model']} | "
                      f"cover {r['marginal_cover']:.3f} (p05 {r['cover_p05']:.3f}, sets below {r['share_cal_sets_below_target']:.2f}) | "
                      f"intervene {r['intervention_rate']:.2f} | success all {r['success_overall']:.3f} | given intervention "
                      f"{fmt(r['success_given_intervention'])} no-loss {fmt(r['no_loss_given_intervention'])} narrowing "
                      f"{fmt(r['narrowing_given_intervention'])} extra {fmt(r['extra_budget_given_intervention'])} | naive "
                      f"{'--' if r['naive'] is None else f'success {r['naive']['success']:.3f} no-loss {r['naive']['no_loss']:.3f}'}", flush=True)
            out[f"{regime}|{gamma}"] = runs
            d = ROOT / "outputs" / "09_certificate_v2"
            d.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(d / "results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
