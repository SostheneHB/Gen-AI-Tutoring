"""Experiment 22, step 2: the method on a semi-synthetic system calibrated on Eedi (reviewer 1, point 2).

    P2_OUTCOME=knowledge python code/scripts/run_semisynthetic.py [--kappas 0.25 0.5 1.0] [--seeds 0 1 2 3 4]

The system is p2/semisim.py: group-specific BKT with forgetting fitted on Eedi (PremiumPupil), an ASSUMED effect of
help of strength kappa and an assumed spread of learning rates between students.  For each kappa and fitted model
(the lifting, the decoder and the group-aware model are fitted exactly as in the main experiments, on 150 training
students under random controls), with knowledge as the target and the efficient reference with the smallest gap:
  * model gap and true gap of the reference (20 evaluation cohorts), and whether (A1)-(A5) hold;
  * price: the cheapest fair plans for 25, 50 and 100% of the gap that the model can close; their true narrowing,
    the true change of each group mean, and the share of students who end with less knowledge (same students);
  * certificate (Theorem 3) and Learn then Test as in experiment 12: margins 0, 0.04, ..., 0.60 (wider than in experiment 12); 1,500 cohorts
    (1,000 for calibration, 500 for test), 200 calibrations of m = 100, alpha = eta = 0.1.
Writes outputs/22_semisynthetic/results_knowledge.json (aggregate numbers only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from p2 import semisim  # noqa: E402
from p2.certificate import conformal_margin, margin_plan, predicted_change  # noqa: E402
from p2.config import OUTCOME, REF_HELP  # noqa: E402
from p2.frontier import GroupModel, efficient_plan, fair_frontier, knapsack_price  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402
from run_assumption_cases import assumptions  # noqa: E402

use_paper1_code()
import pareto_fair as pf  # noqa: E402
import synthetic_regimes as sr  # noqa: E402

T = 25
DELTAS = np.round(np.arange(0, 0.601, 0.04), 4)          # wider than in experiment 12: the model can be optimistic here
POOL_CAL, POOL_TEST, M_CAL, REPEATS, ALPHA, ETA = 1000, 500, 100, 200, 0.1, 0.1


def run_plans(fa, par, kappa, seed, index, plans):
    co = fa.make_cohort(seed, horizon=T, role="evaluation", index=index)
    step = semisim.cohort_step(fa, par, kappa, seed, "evaluation", index, len(co[2]))
    init, noise, groups = co
    finals = []
    for plan in plans:
        x = init.copy()
        for t in range(T):
            x = step(x, np.where(groups == 1, plan[1][t], plan[0][t]), groups, noise[:, t])
        finals.append(x[:, 0])
    return np.array(finals), groups


def changes(finals, groups):
    """(dK0, dK1, narrowing, share of students losing) of each plan against plan 0 (the reference)."""
    m = np.array([[f[groups == g].mean() for g in (0, 1)] for f in finals])
    gap = m[:, 1] - m[:, 0]
    out = []
    for a in range(1, len(finals)):
        lost = (finals[a] - finals[0]) < -1e-9
        out.append((m[a, 0] - m[0, 0], m[a, 1] - m[0, 1], abs(gap[0]) - abs(gap[a]), float(lost.mean())))
    return np.array(out), gap[0]


def ltt(fail, deltas):
    m, chosen = fail.shape[0], None
    for i in sorted(range(len(deltas)), key=lambda i: -deltas[i]):
        if binom.cdf(int(fail[:, i].sum()), m, ALPHA) <= ETA:
            chosen = i
        else:
            break
    return chosen


def one(fa, par, kappa, seed, rng):
    semisim.patch(fa, par, kappa, seed)
    model = fa.fit_model(seed, 4, 3)
    A, B, a, aG = pf.fit_group_aware(fa, model, seed)
    init, _, groups = fa.make_cohort(seed, horizon=T)
    n = {g: int((groups == g).sum()) for g in (0, 1)}
    m0 = {g: model.encode(init[groups == g]).mean(0) for g in (0, 1)}
    M = GroupModel(A, B, a, aG, m0, model.decoder[0], model.decoder_offset[0], model.decoder[0], model.decoder_offset[0], n, T)
    v_ref = efficient_plan(M, REF_HELP * T * sum(n.values()), "min_gap")
    _, _, gap, B_ref = M.outcomes(v_ref)
    res = dict(kappa=kappa, seed=seed, model_gap=float(gap), assumptions=assumptions(M, v_ref),
               hk_min_over_max=float(min(M.hk[0].min(), M.hk[1].min()) / max(M.hk[0].max(), M.hk[1].max())))
    if abs(gap) <= 1e-9:
        res["skipped"] = "no model gap"
        return res
    cap = knapsack_price(M, v_ref, 0.0)[1]
    close = min(cap, abs(gap))
    # price: three targets, evaluated on 20 cohorts
    fr = [f for f in fair_frontier(M, v_ref, [0.25 * close, 0.5 * close, close])]
    plans = [v_ref] + [f["plan"] for f in fr if f["feasible"]]
    ev = [changes(*run_plans(fa, par, kappa, seed, 200 + j, plans)) for j in range(20)]
    ch = np.array([e[0] for e in ev])
    res["true_gap"] = float(np.mean([e[1] for e in ev]))
    res["price"] = [dict(fraction=f, promised=float(f * close), extra_budget=float(fr[i]["extra_budget"] / B_ref),
                         true_narrowing=float(ch[:, i, 2].mean()), true_dK=[float(ch[:, i, 0].mean()), float(ch[:, i, 1].mean())],
                         min_dK=float(ch[:, i, :2].min()), share_students_losing=float(ch[:, i, 3].mean()))
                    for i, f in enumerate([0.25, 0.5, 1.0]) if fr[i]["feasible"]]
    # certificate
    rho = 0.5 * close
    deltas, fam, preds = [], [], []
    for d in DELTAS:
        p = margin_plan(M, v_ref, rho, d)
        if p is None or rho / (1 - d) > 0.9 * cap:
            continue
        deltas.append(float(d)); fam.append(p); preds.append(predicted_change(M, v_ref, p))
    if not fam:
        res["certificate"] = None
        return res
    truth = np.array([changes(*run_plans(fa, par, kappa, seed, 1000 + j, [v_ref] + fam))[0] for j in range(POOL_CAL + POOL_TEST)])
    ok = (truth[:, :, 0] >= 0) & (truth[:, :, 1] >= 0) & (truth[:, :, 2] >= rho)
    S = np.array([np.inf if (tr[:, :2] < 0).any() else float(np.round(np.max(1 - tr[:, 2] / np.array([p[2] for p in preds])), 12))
                  for tr in truth])
    cal_pool, test = np.arange(POOL_CAL), np.arange(POOL_CAL, POOL_CAL + POOL_TEST)
    conf, cov, lt = [], [], []
    for _ in range(REPEATS):
        cal = rng.choice(cal_pool, M_CAL, replace=False)
        q = conformal_margin(S[cal], ALPHA)
        j = next((k for k, d in enumerate(deltas) if d >= q), None)
        cov.append(float((S[test] <= q).mean()))
        conf.append((0.0, 0.0) if j is None else (1.0, float((~ok[test, j]).mean())))
        jl = ltt((~ok[cal]).astype(int), deltas)
        lt.append((0.0, 0.0) if jl is None else (1.0, float((~ok[test, jl]).mean())))
    conf, lt = np.array(conf), np.array(lt)
    iv = conf[:, 0] > 0
    res["certificate"] = dict(rho=float(rho), family=len(fam), naive_success=float(ok[test, 0].mean()), score_finite=float(np.isfinite(S).mean()),
                              cover_mean=float(np.mean(cov)), cover_below=float(np.mean(np.array(cov) < 1 - ALPHA)),
                              intervene=float(iv.mean()), risk=float(conf[:, 1].mean()),
                              success_given_intervention=float(1 - conf[iv, 1].mean()) if iv.any() else None,
                              ltt_intervene=float((lt[:, 0] > 0).mean()), ltt_risk=float(lt[:, 1].mean()))
    return res


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--kappas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    a_.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a_.add_argument("--tag", default="")
    a = a_.parse_args()
    fa = sr.load_sim(str(PATHS["simulator"]))
    par = semisim.load_params()
    rows = []
    d = ROOT / "outputs" / "22_semisynthetic"
    for kappa in a.kappas:
        for seed in a.seeds:
            rng = np.random.default_rng([20261016, int(kappa * 100), seed])
            r = one(fa, par, kappa, seed, rng)
            rows.append(r)
            c = r.get("certificate") or {}
            print(f"kappa={kappa} seed={seed}: model gap {r['model_gap']:+.3f} true gap {r.get('true_gap', float('nan')):+.3f} | "
                  f"A1-A5 {all(r['assumptions'].values())} | price " + " ".join(f"{p['fraction']}: promised {p['promised']:.3f} true {p['true_narrowing']:.3f} "
                                                                                  f"minK {p['min_dK']:+.4f} losing {p['share_students_losing']:.3f}" for p in r.get("price", [])) +
                  f" | cert iv {c.get('intervene', float('nan')):.2f} risk {c.get('risk', float('nan')):.3f} cover {c.get('cover_mean', float('nan')):.3f} "
                  f"naive {c.get('naive_success', float('nan')):.2f} | ltt iv {c.get('ltt_intervene', float('nan')):.2f}", flush=True)
            json.dump(dict(outcome=OUTCOME, params=par, rows=rows), open(d / f"results_knowledge{('_' + a.tag) if a.tag else ''}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
