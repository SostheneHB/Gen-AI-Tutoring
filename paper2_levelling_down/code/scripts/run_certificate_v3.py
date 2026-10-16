"""Experiment 12: the conformal certificate, with knowledge as the fairness outcome, in three regimes, compared
with a Learn-then-Test selection (review of 16 October 2026, points 3-5).

    P2_OUTCOME=knowledge python code/scripts/run_certificate_v3.py --regime standard|heterogeneous|hard [--repeats 200]

Regimes (three group strengths, five fitted models each):
  standard       reference = the efficient plan; every student responds to help in the same way;
  heterogeneous  reference = the efficient plan; each student has a response strength r_i (log-normal, population
                 mean 1, drawn anew for each cohort), which multiplies the effect of help on knowledge, engagement and
                 disparity; the mean effect of a plan therefore varies between cohorts of the same population;
  hard           reference = a uniform plan (0.3 at every step, not efficient); fair plans move teaching between
                 steps, the floors bind in the model, and a group mean can really go down.
Family: plans w^delta for delta in {0, 0.02, ..., 0.30} that exist (the linear program is feasible) and promise at
most 90% of the capacity; a smaller family (delta in {0, 0.1, 0.2, 0.3}) is also evaluated.  Every linear program
is solved by HiGHS; when it has several solutions, the one returned by the solver is used, which is fixed by the
model and does not depend on the cohorts.
Scores are rounded to 1e-12 before ranking, so that numerically equal scores are tied (in the standard regime
they are equal for every cohort: the disparity is linear and help is monotone, so the score is constant).
For each model: a pool of 3,000 cohorts (2,000 for calibration, 1,000 for test); 200 calibrations of m = 100
cohorts drawn from the calibration pool and 20 disjoint ones; each evaluated on the test pool.
Learn then Test (fixed-sequence testing, from the largest margin to the smallest): plan delta is declared safe
if the binomial test of "failure probability >= alpha" rejects at level eta = 0.1; the smallest safe margin is
used, or the reference is kept.  Failure = a group mean loses knowledge, or the narrowing is below rho.
Writes outputs/12_certificate_v3/results_<regime>.json (aggregate numbers only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.stats import binom
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from p2.certificate import conformal_margin, margin_plan, predicted_change  # noqa: E402
from p2.config import OUT_IDX, OUTCOME, REF_HELP, REF_RULE  # noqa: E402
from p2.frontier import GroupModel, efficient_plan, knapsack_price  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402

use_paper1_code()
import pareto_fair as pf  # noqa: E402
import synthetic_regimes as sr  # noqa: E402

T, SEEDS, GAMMAS = 25, range(5), (0.5, 1.0, 2.0)
M_CAL, POOL_CAL, POOL_TEST, ALPHA, ETA = 100, 2000, 1000, 0.1, 0.1
DELTAS = np.round(np.arange(0, 0.301, 0.02), 4)
SMALL = (0.0, 0.1, 0.2, 0.3)
SIGMA_R = 0.5                                  # spread of the individual response to help


def hetero_step(gamma, r):
    """The simulator step with an individual response r_i to help (r = 1: the standard simulator)."""
    def step(states, controls, groups, noise):
        k, e, p, d = np.moveaxis(np.asarray(states), -1, 0)
        nk, ne, nd = np.moveaxis(np.asarray(noise), -1, 0)
        u = r * controls
        k1 = np.clip(.85 * k + .15 * expit(3 * e - 1.5) + .20 * u - .06 * gamma * groups + nk, 0, 1)
        e1 = np.clip(.80 * e + .10 * np.tanh(p) + .15 * u + ne, 0, 1)
        return np.stack([k1, e1, expit(6 * (k1 - .5)), .90 * d - .05 * gamma * groups + .05 * u + nd], axis=-1)
    return step


def response(seed, index, n, regime):
    if regime != "heterogeneous":
        return np.ones(n)
    g = np.random.default_rng([seed, index, 7])
    return g.lognormal(-SIGMA_R ** 2 / 2, SIGMA_R, n)          # population mean 1


def realised(step, co, v_ref, plans):
    init, noise, groups = co

    def run(plan):
        x = init.copy()
        for t in range(T):
            x = step(x, np.where(groups == 1, plan[1][t], plan[0][t]), groups, noise[:, t])
        return {g: x[groups == g, 0].mean() for g in (0, 1)}, x[groups == 1, OUT_IDX].mean() - x[groups == 0, OUT_IDX].mean()

    k_ref, gap_ref = run(v_ref)
    out = []
    for p in plans:
        k, gap = run(p)
        out.append((k[0] - k_ref[0], k[1] - k_ref[1], abs(gap_ref) - abs(gap)))
    return np.array(out)


def score_of(pred, tr):
    """Worst relative shortfall over the family; +inf if a group mean loses knowledge under some plan."""
    s = -np.inf
    for p, (dk0, dk1, nar) in zip(pred, tr):
        if min(dk0, dk1) < 0:
            return np.inf
        s = max(s, 1 - nar / p[2])
    return float(np.round(s, 12))


def max_narrowing(M, v_ref):
    _, _, gap_ref, _ = M.outcomes(v_ref)
    sgn = -np.sign(gap_ref)
    z = np.zeros(M.T)
    bounds = [(-v_ref[0][t], 1 - v_ref[0][t]) for t in range(M.T)] + [(-v_ref[1][t], 1 - v_ref[1][t]) for t in range(M.T)]
    r = linprog(-sgn * np.r_[-M.hd[0], M.hd[1]], A_ub=np.array([np.r_[-M.hk[0], z], np.r_[z, -M.hk[1]], sgn * np.r_[-M.hd[0], M.hd[1]]]),
                b_ub=[0.0, 0.0, abs(gap_ref)], bounds=bounds, method="highs")
    return float(-r.fun)


def ltt_choice(fail_cal, deltas):
    """Fixed-sequence Learn then Test from the largest margin down; returns the index of the plan used or None."""
    m = fail_cal.shape[0]
    chosen = None
    for i in sorted(range(len(deltas)), key=lambda i: -deltas[i]):
        p = binom.cdf(int(fail_cal[:, i].sum()), m, ALPHA)       # H0: failure probability >= alpha
        if p <= ETA:
            chosen = i
        else:
            break
    return chosen


def one(fa, gamma, seed, regime, repeats, rng):
    r_train = response(seed, 999999, 150, regime)
    fa.step = hetero_step(gamma, r_train)
    model = fa.fit_model(seed, 4, 3)
    A, B, a, aG = pf.fit_group_aware(fa, model, seed)
    init, _, groups = fa.make_cohort(seed, horizon=T)
    n = {g: int((groups == g).sum()) for g in (0, 1)}
    m0 = {g: model.encode(init[groups == g]).mean(0) for g in (0, 1)}
    M = GroupModel(A, B, a, aG, m0, model.decoder[0], model.decoder_offset[0], model.decoder[OUT_IDX], model.decoder_offset[OUT_IDX], n, T)
    if regime == "hard":
        v_ref = {g: np.full(T, REF_HELP) for g in (0, 1)}
        cap = max_narrowing(M, v_ref)
    else:
        v_ref = efficient_plan(M, REF_HELP * T * sum(n.values()), REF_RULE)
        cap = min(knapsack_price(M, v_ref, 0.0)[1], abs(M.outcomes(v_ref)[2]))
    rho = 0.5 * cap
    if rho <= 1e-9:                            # the reference has no model gap: nothing to narrow
        fa.step = sr.make_step(1.0)
        return dict(regime=regime, outcome=OUTCOME, gamma=gamma, seed=seed, rho=0.0, skipped="no model gap")
    deltas, plans, preds, infeasible = [], [], [], []
    for d in DELTAS:
        p = margin_plan(M, v_ref, rho, d)
        if p is None or rho / (1 - d) > 0.9 * cap:
            infeasible.append(float(d))
            continue
        deltas.append(float(d)); plans.append(p); preds.append(predicted_change(M, v_ref, p))
    B_ref = M.outcomes(v_ref)[3]
    extra = np.array([(M.outcomes(p)[3] - B_ref) / B_ref for p in plans])
    truth = []
    for j in range(POOL_CAL + POOL_TEST):
        co = fa.make_cohort(seed, horizon=T, role="evaluation", index=1000 + j)
        truth.append(realised(hetero_step(gamma, response(seed, 1000 + j, len(co[2]), regime)), co, v_ref, plans))
    truth = np.array(truth)                                                    # (cohorts, plans, 3)
    ok = (truth[:, :, 0] >= 0) & (truth[:, :, 1] >= 0) & (truth[:, :, 2] >= rho)  # success of each plan on each cohort
    cal_pool, test = np.arange(POOL_CAL), np.arange(POOL_CAL, POOL_CAL + POOL_TEST)
    res = dict(regime=regime, outcome=OUTCOME, gamma=gamma, seed=seed, rho=rho, capacity=cap, family=len(plans),
               infeasible_margins=infeasible)
    for fam_name, idx in (("full", list(range(len(plans)))), ("small", [i for i, d in enumerate(deltas) if d in SMALL])):
        if not idx:
            continue
        dl = [deltas[i] for i in idx]
        S = np.array([score_of([preds[i] for i in idx], truth[c, idx]) for c in range(len(truth))])
        finite = S[np.isfinite(S)]
        stats = dict(score_distinct=int(len(np.unique(S))), score_finite_share=float(np.isfinite(S).mean()),
                     score_sd=float(finite.std()) if len(finite) else None)

        def decide_conformal(cal):
            q = conformal_margin(S[cal], ALPHA)
            j = next((k for k, d in enumerate(dl) if d >= q), None)
            return q, (None if j is None else idx[j])

        def evaluate(choice):
            if choice is None:
                return dict(intervene=False, success=1.0, fail_loss=0.0, fail_shortfall=0.0)
            lc = (truth[test, choice, 0] < 0) | (truth[test, choice, 1] < 0)
            return dict(intervene=True, success=float(ok[test, choice].mean()), fail_loss=float(lc.mean()),
                        fail_shortfall=float((~ok[test, choice] & ~lc).mean()), no_loss=float(((truth[test, choice, 0] >= 0) & (truth[test, choice, 1] >= 0)).mean()),
                        narrowing=float(truth[test, choice, 2].mean()), extra=float(extra[choice]))

        def summarise(rows, covers=None):
            iv = [r for r in rows if r["intervene"]]
            out = dict(intervention_rate=float(len(iv) / len(rows)), success_overall=float(np.mean([r["success"] for r in rows])),
                       risk=float(1 - np.mean([r["success"] for r in rows])), risk_loss=float(np.mean([r["fail_loss"] for r in rows])),
                       risk_shortfall=float(np.mean([r["fail_shortfall"] for r in rows])),
                       risk_max_per_calibration=float(1 - np.min([r["success"] for r in rows])),
                       success_given_intervention=float(np.mean([r["success"] for r in iv])) if iv else None,
                       success_given_intervention_min=float(np.min([r["success"] for r in iv])) if iv else None,
                       narrowing_given_intervention=float(np.mean([r["narrowing"] for r in iv])) if iv else None,
                       extra_given_intervention=float(np.mean([r["extra"] for r in iv])) if iv else None)
            if covers is not None:
                out.update(cover_mean=float(np.mean(covers)), cover_min=float(np.min(covers)),
                           share_sets_below_target=float(np.mean(np.array(covers) < 1 - ALPHA)))
            return out

        conf_rows, conf_cov, ltt_rows = [], [], []
        for _ in range(repeats):
            cal = rng.choice(cal_pool, M_CAL, replace=False)
            q, ch = decide_conformal(cal)
            conf_rows.append(evaluate(ch)); conf_cov.append(float((S[test] <= q).mean()))
            j = ltt_choice((~ok[cal][:, idx]).astype(int), dl)        # j indexes the family; map back to the plan
            ltt_rows.append(evaluate(None if j is None else idx[j]))
        disj_rows, disj_cov = [], []
        for k in range(POOL_CAL // M_CAL):
            cal = cal_pool[k * M_CAL:(k + 1) * M_CAL]
            q, ch = decide_conformal(cal)
            disj_rows.append(evaluate(ch)); disj_cov.append(float((S[test] <= q).mean()))
        # Monte Carlo uncertainty: resample the calibration pool and the test pool (cohorts), 200 times, with 50
        # calibrations each; 95% intervals of the mean coverage and of the risk P(intervene and fail).
        bc, br = [], []
        for _ in range(200):
            cp, tp = rng.choice(cal_pool, len(cal_pool)), rng.choice(test, len(test))
            cv, rk = [], []
            for _ in range(50):
                cal = cp[rng.choice(len(cp), M_CAL, replace=False)]
                q, ch = decide_conformal(cal)
                cv.append(float((S[tp] <= q).mean())); rk.append(0.0 if ch is None else float((~ok[tp, ch]).mean()))
            bc.append(np.mean(cv)); br.append(np.mean(rk))
        mc = dict(cover_mean_ci=[float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))],
                  risk_ci=[float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))])
        res[fam_name] = dict(margins=dl, score=stats, conformal=dict(summarise(conf_rows, conf_cov), **mc),
                             conformal_disjoint=summarise(disj_rows, disj_cov), ltt=summarise(ltt_rows),
                             plan_success_on_test={str(d): float(ok[test, i].mean()) for d, i in zip(dl, idx)})
    res["naive"] = dict(success=float(ok[test, 0].mean()), dK_true=[float(truth[test, 0, 0].mean()), float(truth[test, 0, 1].mean())],
                        dK_predicted=[float(preds[0][0]), float(preds[0][1])], extra=float(extra[0])) if deltas and deltas[0] == 0.0 else None
    fa.step = sr.make_step(1.0)
    return res


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--regime", choices=["standard", "heterogeneous", "hard"], required=True)
    a_.add_argument("--repeats", type=int, default=200)
    a = a_.parse_args()
    fa = sr.load_sim(str(PATHS["simulator"]))
    rng = np.random.default_rng([20261016, ["standard", "heterogeneous", "hard"].index(a.regime)])
    out = {}
    d = ROOT / "outputs" / "12_certificate_v3"
    d.mkdir(parents=True, exist_ok=True)
    for gamma in GAMMAS:
        runs = []
        for s in SEEDS:
            r = one(fa, gamma, s, a.regime, a.repeats, rng)
            runs.append(r)
            if r.get("skipped"):
                print(f"{a.regime} gamma={gamma} seed={s}: skipped ({r['skipped']})", flush=True)
                out[str(gamma)] = runs
                continue
            f = r["full"]
            fmt = lambda v: "--" if v is None else f"{v:.3f}"
            print(f"{a.regime} gamma={gamma} seed={s}: rho {r['rho']:.3f} family {r['family']} | score distinct {f['score']['score_distinct']} "
                  f"finite {f['score']['score_finite_share']:.2f} | conformal intervene {f['conformal']['intervention_rate']:.2f} "
                  f"success|int {fmt(f['conformal']['success_given_intervention'])} cover {fmt(f['conformal'].get('cover_mean'))} "
                  f"(sets below {fmt(f['conformal'].get('share_sets_below_target'))}) | LTT intervene {f['ltt']['intervention_rate']:.2f} "
                  f"success|int {fmt(f['ltt']['success_given_intervention'])} | small family intervene "
                  f"{fmt(r.get('small', {}).get('conformal', {}).get('intervention_rate'))} | naive "
                  f"{'--' if r['naive'] is None else f'{r['naive']['success']:.2f}'}", flush=True)
            out[str(gamma)] = runs
            json.dump(out, open(d / f"results_{a.regime}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
