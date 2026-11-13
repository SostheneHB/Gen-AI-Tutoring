"""Experiment 24: a family with safe, harmful and uncertain plans (review of 16 October 2026, fifth report, point 5).

    P2_OUTCOME=knowledge python code/scripts/run_mixed_family.py

Reference: the uniform plan (0.1 at every step), as in the hard regime; students respond differently to help (the
heterogeneous regime).  Target rho = half of what the model can close by adding help to the group behind.  The family
mixes two kinds of plans that the MODEL predicts to be safe and to narrow the gap by rho / (1 - delta):
  add-only   the filling: extra help for the group behind only (the group ahead is unchanged, so it cannot lose);
             costly; whether its narrowing reaches rho is uncertain (heterogeneous responses);
  move       the cheapest plan of the linear program with floors: it moves teaching between steps; on the
             simulator it lowers the knowledge of the group ahead (harmful);
for delta in {0, 0.1, 0.2, 0.3}: eight plans.  Procedures, each choosing the cheapest plan it accepts (alpha = eta = 0.1):
  max-score   Theorem 3: one score, the maximum over the whole family; a plan j is accepted if q <= 1 - rho / rho_hat_j;
  per-plan    a union bound: plan j has its own conformal quantile q_j at level alpha / L and is accepted if
              q_j <= 1 - rho / rho_hat_j; P(an accepted plan fails) <= alpha;
  LTT-FS      Learn then Test, fixed sequence from the largest margin down, add-only before move at each margin;
  LTT-Bonf    Learn then Test with Bonferroni tests at eta / L.
A plan is "truly safe" if it succeeds (no group loses, narrowing >= rho) in at least 1 - alpha of the test cohorts.
Reported per procedure: intervention rate, risk (changed and failed), number of truly safe plans not accepted (for the
two procedures that accept sets of plans), and the extra budget of the chosen plan compared with the cheapest truly safe plan.
Three fitted models per gamma; 1,000 calibration and 500 test cohorts; 200 calibrations of m = 100.
Writes outputs/24_mixed_family/results_knowledge.json (aggregate numbers only).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from p2.certificate import margin_plan, predicted_change  # noqa: E402
from p2.config import OUTCOME, REF_HELP  # noqa: E402
from p2.frontier import GroupModel, knapsack_plan, knapsack_price  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402
from run_certificate_v3 import hetero_step, realised, response  # noqa: E402

use_paper1_code()
import pareto_fair as pf  # noqa: E402
import synthetic_regimes as sr  # noqa: E402

T, GAMMAS, SEEDS = 25, (0.5, 1.0, 2.0), range(3)
DELTAS = (0.0, 0.1, 0.2, 0.3)
POOL_CAL, POOL_TEST, M_CAL, REPEATS, ALPHA, ETA = 1000, 500, 100, 200, 0.1, 0.1


def quantile(scores, level):
    m = len(scores)
    k = math.ceil((m + 1) * (1 - level))
    return float("inf") if k > m else float(np.sort(scores)[k - 1])


def one(fa, gamma, seed, rng):
    fa.step = hetero_step(gamma, response(seed, 999999, 150, "heterogeneous"))
    model = fa.fit_model(seed, 4, 3)
    A, B, a, aG = pf.fit_group_aware(fa, model, seed)
    init, _, groups = fa.make_cohort(seed, horizon=T)
    n = {g: int((groups == g).sum()) for g in (0, 1)}
    m0 = {g: model.encode(init[groups == g]).mean(0) for g in (0, 1)}
    M = GroupModel(A, B, a, aG, m0, model.decoder[0], model.decoder_offset[0], model.decoder[0], model.decoder_offset[0], n, T)
    fa.step = sr.make_step(1.0)
    v_ref = {g: np.full(T, REF_HELP) for g in (0, 1)}
    _, _, gap, B_ref = M.outcomes(v_ref)
    cap = knapsack_price(M, v_ref, 0.0)[1]
    rho = 0.5 * min(cap, abs(gap))
    plans, kinds, deltas = [], [], []
    for d in DELTAS:
        if rho / (1 - d) > 0.9 * min(cap, abs(gap)):
            continue
        extra = knapsack_price(M, v_ref, rho / (1 - d))[0]
        add, _ = knapsack_plan(M, v_ref, extra)
        mv = margin_plan(M, v_ref, rho, d)
        for kind, p in (("add-only", add), ("move", mv)):
            if p is not None:
                plans.append(p); kinds.append(kind); deltas.append(d)
    preds = np.array([predicted_change(M, v_ref, p) for p in plans])
    budget = np.array([(M.outcomes(p)[3] - B_ref) / B_ref for p in plans])
    L = len(plans)
    truth = []
    for j in range(POOL_CAL + POOL_TEST):
        co = fa.make_cohort(seed, horizon=T, role="evaluation", index=4000 + j)
        truth.append(realised(hetero_step(gamma, response(seed, 4000 + j, len(co[2]), "heterogeneous")), co, v_ref, plans))
    truth = np.array(truth)                                                       # (cohorts, plans, 3)
    loss = (truth[:, :, 0] < 0) | (truth[:, :, 1] < 0)
    ok = ~loss & (truth[:, :, 2] >= rho)
    S = np.where(loss, np.inf, np.round(1 - truth[:, :, 2] / preds[None, :, 2], 12))   # per-plan scores
    need = 1 - rho / preds[:, 2]                                                  # plan j succeeds iff s_j <= need_j
    cal_pool, test = np.arange(POOL_CAL), np.arange(POOL_CAL, POOL_CAL + POOL_TEST)
    safe = ok[test].mean(0) >= 1 - ALPHA
    cheapest_safe = float(budget[safe].min()) if safe.any() else None
    order_fs = sorted(range(L), key=lambda j: (-deltas[j], kinds[j] != "add-only"))

    def choose(accepted):
        acc = [j for j in range(L) if accepted[j]]
        return min(acc, key=lambda j: budget[j]) if acc else None

    rows = {k: [] for k in ("max-score", "per-plan", "LTT-FS", "LTT-Bonf")}
    for _ in range(REPEATS):
        cal = rng.choice(cal_pool, M_CAL, replace=False)
        q = quantile(S[cal].max(1), ALPHA)
        acc_max = q <= need
        acc_pp = np.array([quantile(S[cal, j], ALPHA / L) <= need[j] for j in range(L)])
        fails = (~ok[cal]).sum(0)
        acc_fs = np.zeros(L, bool)
        for j in order_fs:
            if binom.cdf(int(fails[j]), M_CAL, ALPHA) <= ETA:
                acc_fs[j] = True
            else:
                break
        acc_bf = np.array([binom.cdf(int(fails[j]), M_CAL, ALPHA) <= ETA / L for j in range(L)])
        for name, acc in (("max-score", acc_max), ("per-plan", acc_pp), ("LTT-FS", acc_fs), ("LTT-Bonf", acc_bf)):
            ch = choose(acc)
            rows[name].append(dict(intervene=ch is not None, fail=0.0 if ch is None else float((~ok[test, ch]).mean()),
                                   safe_rejected=int((safe & ~acc).sum()), chosen_kind=None if ch is None else kinds[ch],
                                   extra=None if ch is None else float(budget[ch])))
    out = dict(gamma=gamma, seed=seed, rho=float(rho), plans=[dict(kind=k, delta=d, extra_budget=float(b), success=float(ok[test, j].mean()),
                                                                  loss_rate=float(loss[test, j].mean()), truly_safe=bool(safe[j]))
                                                             for j, (k, d, b) in enumerate(zip(kinds, deltas, budget))],
               cheapest_safe_extra=cheapest_safe)
    for name, rs in rows.items():
        iv = [r for r in rs if r["intervene"]]
        out[name] = dict(intervene=float(len(iv) / len(rs)), risk=float(np.mean([r["fail"] for r in rs])),
                         safe_rejected=float(np.mean([r["safe_rejected"] for r in rs])),
                         chose_move=float(np.mean([r["chosen_kind"] == "move" for r in iv])) if iv else None,
                         extra=float(np.mean([r["extra"] for r in iv])) if iv else None,
                         extra_over_cheapest_safe=(float(np.mean([r["extra"] for r in iv])) - cheapest_safe) if iv and cheapest_safe is not None else None)
    return out


def main():
    fa = sr.load_sim(str(PATHS["simulator"]))
    res = []
    d = ROOT / "outputs" / "24_mixed_family"
    d.mkdir(parents=True, exist_ok=True)
    for gamma in GAMMAS:
        for seed in SEEDS:
            r = one(fa, gamma, seed, np.random.default_rng([20261016, int(gamma * 10), seed]))
            res.append(r)
            print(f"gamma={gamma} seed={seed}: rho {r['rho']:.3f} | plans " +
                  " ".join(f"{p['kind'][0]}{p['delta']}:{'S' if p['truly_safe'] else ('H' if p['loss_rate'] > 0.5 else 'u')}" for p in r["plans"]) +
                  " | " + " | ".join(f"{k} iv {r[k]['intervene']:.2f} risk {r[k]['risk']:.3f} rej {r[k]['safe_rejected']:.1f} move {r[k]['chose_move']}"
                                     for k in ("max-score", "per-plan", "LTT-FS", "LTT-Bonf")), flush=True)
            json.dump(dict(outcome=OUTCOME, results=res), open(d / "results_knowledge.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
