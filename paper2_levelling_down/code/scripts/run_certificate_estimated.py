"""Experiment 17: the certificate when the calibration effects are estimated from a randomized design (Theorem 4).

    P2_OUTCOME=knowledge python code/scripts/run_certificate_estimated.py --regime heterogeneous|hard [--sizes 10000 50000 250000]

Same models, regimes, references and target as experiment 12, with the small family (margins 0, 0.1, 0.2, 0.3).
In each calibration cohort of n students, the students of each group are split at random into equal arms (the
reference and each plan): each student follows one plan only, so the changes are estimated, not observed.  The
test cohorts are evaluated on their TRUE changes (all plans on the same students, as in experiment 12).
Procedures (alpha + beta = 0.1 for the procedures with a guarantee):
  oracle       Theorem 3 with the true calibration scores (alpha = 0.1), as in experiment 12;
  plugin       Theorem 3 applied to scores computed from the estimates (alpha = 0.1): no guarantee;
  hs, eb       Theorem 4: conservative scores from simultaneous bounds (Hoeffding-Serfling or empirical
               Bernstein-Serfling, beta = 0.02) and alpha = 0.08;
  ltt_oracle, ltt_eb   Learn then Test with true / conservative failures (eta = 0.1; level 0.1 / 0.08).
For each model and size: a pool of 150 calibration and 150 test cohorts, 100 calibrations of m = 100 drawn from
the calibration pool.  Risk = P(intervene and fail) over calibrations and test cohorts, split into loss and
shortfall; we also report how often the bounds held (simultaneous coverage, target >= 1 - beta).
Writes outputs/17_certificate_estimated/results_<regime>.json (aggregate numbers only).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from p2.certificate import conformal_margin, margin_plan, predicted_change  # noqa: E402
from p2.config import OUTCOME, REF_HELP, REF_RULE  # noqa: E402
from p2.estimated import arm_estimates, assign, conservative_score  # noqa: E402
from p2.frontier import GroupModel, efficient_plan, knapsack_price  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402
from run_certificate_v3 import hetero_step, max_narrowing, response  # noqa: E402

use_paper1_code()
import pareto_fair as pf  # noqa: E402
import synthetic_regimes as sr  # noqa: E402

T, SEEDS, GAMMAS = 25, range(3), (0.5, 1.0, 2.0)
DELTAS = (0.0, 0.1, 0.2, 0.3)
M_CAL, POOL, REPEATS, ALPHA, BETA, ETA = 100, 150, 100, 0.1, 0.02, 0.1


def final_k(step, co, plans):
    init, noise, groups = co
    out = []
    for plan in plans:
        x = init.copy()
        for t in range(T):
            x = step(x, np.where(groups == 1, plan[1][t], plan[0][t]), groups, noise[:, t])
        out.append(x[:, 0])
    return np.array(out)                                    # (arms, students)


def truth_of(Y, groups):
    mu = np.array([[Y[a, groups == g].mean() for a in range(Y.shape[0])] for g in (0, 1)])
    gap = mu[1] - mu[0]
    return np.array([(mu[0, a] - mu[0, 0], mu[1, a] - mu[1, 0], abs(gap[0]) - abs(gap[a])) for a in range(1, Y.shape[0])])


def true_score(pred, tr):
    if (tr[:, :2] < 0).any():
        return np.inf
    return float(np.round(np.max(1 - tr[:, 2] / np.array([p[2] for p in pred])), 12))


def ltt(fail, deltas, level):
    m, chosen = fail.shape[0], None
    for i in sorted(range(len(deltas)), key=lambda i: -deltas[i]):
        if binom.cdf(int(fail[:, i].sum()), m, level) <= ETA:
            chosen = i
        else:
            break
    return chosen


def setup(fa, gamma, seed, regime):
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
    deltas, plans, preds = [], [], []
    for d in DELTAS:
        p = margin_plan(M, v_ref, rho, d)
        if p is None or rho / (1 - d) > 0.9 * cap:
            continue
        deltas.append(d); plans.append(p); preds.append(predicted_change(M, v_ref, p))
    unchanged = [[bool(np.allclose(p[g], v_ref[g])) for g in (0, 1)] for p in plans]
    B_ref = M.outcomes(v_ref)[3]
    extra = [(M.outcomes(p)[3] - B_ref) / B_ref for p in plans]
    return v_ref, rho, deltas, plans, preds, unchanged, extra


def one(fa, gamma, seed, regime, n, rng):
    v_ref, rho, deltas, plans, preds, unchanged, extra = setup(fa, gamma, seed, regime)
    if rho <= 1e-9 or not plans:              # the reference has no model gap: nothing to narrow
        return None
    L = len(plans)
    methods = ("plugin", "hoeffding_serfling", "empirical_bernstein")
    S = {k: np.zeros(2 * POOL) for k in ("oracle",) + methods}
    FAILB = np.zeros((2 * POOL, L), bool)                    # conservative failures (empirical Bernstein)
    truth, held = np.zeros((2 * POOL, L, 3)), {m: np.zeros(POOL, bool) for m in methods[1:]}
    for j in range(2 * POOL):
        co = fa.make_cohort(seed, students=n, horizon=T, role="evaluation", index=50000 + j)
        groups = co[2]
        Y = final_k(hetero_step(gamma, response(seed, 50000 + j, n, regime)), co, [v_ref] + plans)
        tr = truth_of(Y, groups)
        truth[j] = tr
        S["oracle"][j] = true_score(preds, tr)
        if j >= POOL:
            continue                                         # test cohorts: only the truth is used
        arm = assign(rng, groups, L + 1)
        mu = np.array([[Y[a, groups == g].mean() for a in range(L + 1)] for g in (0, 1)])
        for meth in methods:
            _, lo, hi = arm_estimates(Y, groups, arm, BETA, meth)
            S[meth][j] = conservative_score(lo, hi, preds, unchanged)
            if meth != "plugin":
                held[meth][j] = bool(((lo <= mu + 1e-15) & (mu <= hi + 1e-15)).all())
            if meth == "empirical_bernstein":
                for l in range(L):
                    sc = conservative_score(lo[:, [0, l + 1]], hi[:, [0, l + 1]], [preds[l]], [unchanged[l]])
                    FAILB[j, l] = not (sc <= 1 - rho / preds[l][2] + 1e-12)       # bound: loss, or narrowing < rho
    ok = (truth[:, :, 0] >= 0) & (truth[:, :, 1] >= 0) & (truth[:, :, 2] >= rho)
    loss = (truth[:, :, 0] < 0) | (truth[:, :, 1] < 0)
    cal_pool, test = np.arange(POOL), np.arange(POOL, 2 * POOL)

    def evaluate(ch):
        if ch is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        fail = ~ok[test, ch]
        return (1.0, float(fail.mean()), float(loss[test, ch].mean()), float((fail & ~loss[test, ch]).mean()),
                float(truth[test, ch, 2].mean()), float(extra[ch]))

    level = {"oracle": ALPHA, "plugin": ALPHA, "hoeffding_serfling": ALPHA - BETA, "empirical_bernstein": ALPHA - BETA}
    res = {}
    for k in S:
        rows = []
        for _ in range(REPEATS):
            cal = rng.choice(cal_pool, M_CAL, replace=False)
            q = conformal_margin(S[k][cal], level[k])
            rows.append(evaluate(next((i for i, d in enumerate(deltas) if d >= q), None)))
        res[k] = np.array(rows)
    for k, fail, lev in (("ltt_oracle", ~ok, ALPHA), ("ltt_eb", FAILB, ALPHA - BETA)):
        rows = []
        for _ in range(REPEATS):
            cal = rng.choice(cal_pool, M_CAL, replace=False)
            rows.append(evaluate(ltt(fail[cal].astype(int), deltas, lev)))
        res[k] = np.array(rows)
    out = dict(gamma=gamma, seed=seed, n=n, rho=rho, family=L, deltas=deltas,
               oracle_finite=float(np.isfinite(S["oracle"]).mean()),
               bounds_held={m: float(held[m].mean()) for m in held},
               naive_success=float(ok[test, 0].mean()))
    for k, r in res.items():
        iv = r[:, 0] > 0
        out[k] = dict(intervention=float(iv.mean()), risk=float(r[:, 1].mean()), risk_loss=float(r[:, 2].mean()),
                      risk_shortfall=float(r[:, 3].mean()), risk_max_per_calibration=float(r[:, 1].max()),
                      narrowing_given_intervention=float(r[iv, 4].mean()) if iv.any() else None,
                      extra_given_intervention=float(r[iv, 5].mean()) if iv.any() else None)
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--regime", default="heterogeneous")
    a_.add_argument("--sizes", type=int, nargs="+", default=[10000, 50000, 250000])
    a_.add_argument("--gammas", type=float, nargs="+", default=list(GAMMAS))
    a_.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a_.add_argument("--tag", default="", help="run only these models, with a random stream of their own, into results_<regime>_<tag>.json")
    a = a_.parse_args()
    fa = sr.load_sim(str(PATHS["simulator"]))
    rng = np.random.default_rng(20261016)
    rows = []
    d = ROOT / "outputs" / "17_certificate_estimated"
    d.mkdir(parents=True, exist_ok=True)
    for n in a.sizes:
        for gamma in a.gammas:
            for seed in a.seeds:
                t0 = time.time()
                if a.tag:                          # parallel runs: one random stream per model, fixed by (n, gamma, seed)
                    rng = np.random.default_rng([20261016, n, int(round(gamma * 10)), seed])
                r = one(fa, gamma, seed, a.regime, n, rng)
                fa.step = sr.make_step(1.0)
                if r is None:
                    continue
                r["regime"] = a.regime
                rows.append(r)
                print(f"n={n} gamma={gamma} seed={seed} ({time.time() - t0:.0f}s): rho {r['rho']:.3f} naive {r['naive_success']:.2f} "
                      f"held {r['bounds_held']} | " + " | ".join(f"{k} iv {r[k]['intervention']:.2f} risk {r[k]['risk']:.3f}"
                                                                 for k in ("oracle", "plugin", "hoeffding_serfling", "empirical_bernstein", "ltt_oracle", "ltt_eb")), flush=True)
                name = f"results_{a.regime}_{a.tag}.json" if a.tag else f"results_{a.regime}.json"
                json.dump(dict(outcome=OUTCOME, alpha=ALPHA, beta=BETA, rows=rows), open(d / name, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
