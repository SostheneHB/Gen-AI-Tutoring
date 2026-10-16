"""Experiment 19: does training with controls U(-1, 1) bias the model?  (review of 16 October 2026, third round,
point 12D)

    P2_OUTCOME=knowledge python code/scripts/run_training_controls.py

The models are trained on 150 students observed under random controls u ~ U(-1, 1), while decisions are in [0, 1].
Here the same fit is repeated with training controls u ~ U(0, 1) (same seeds, same students, same noise), and we
compare, for each gamma and ten seeds: the model gap and the true gap of the efficient reference ("min_gap" rule),
whether (A1)-(A5) hold, and what the plan that closes the model gap (as far as the model allows, no overshoot)
promises and really does on the simulator (ten evaluation cohorts).
Writes outputs/19_training_controls/results_knowledge.json (aggregate numbers only).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from p2.config import OUT_IDX, OUTCOME, REF_HELP, REF_RULE  # noqa: E402
from p2.frontier import GroupModel, efficient_plan, knapsack_plan, knapsack_price  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402
from run_assumption_cases import assumptions  # noqa: E402
from run_model_sensitivity import fit as fit_sized, true_change  # noqa: E402

use_paper1_code()
import synthetic_regimes as sr  # noqa: E402

T, GAMMAS, SEEDS = 25, (0.5, 1.0, 2.0), range(10)


def patched_training_data(fa, low):
    """fa.training_data with controls U(low, 1) instead of U(-1, 1); everything else unchanged."""
    def training_data(seed, students=150, horizon=25):
        initial, noise, groups = fa.make_cohort(seed, students, horizon, role="training")
        u = fa.random_stream(seed, "training_controls").uniform(-1, 1, (students, horizon))
        controls = (u + 1) / 2 * (1 - low) + low          # same uniform draws, mapped to [low, 1]
        states = np.empty((students, horizon + 1, fa.STATE_DIM))
        states[:, 0] = initial
        for t in range(horizon):
            states[:, t + 1] = fa.step(states[:, t], controls[:, t], groups, noise[:, t])
        return states, controls, groups
    return training_data


def one(fa, gamma, seed):
    step = sr.make_step(gamma)
    fa.step = step
    A, B, a, aG, dec, dec0, enc = fit_sized(fa, seed, 150)
    init, _, groups = fa.make_cohort(seed, horizon=T)
    n = {g: int((groups == g).sum()) for g in (0, 1)}
    m0 = {g: enc(init[groups == g]).mean(0) for g in (0, 1)}
    M = GroupModel(A, B, a, aG, m0, dec[0], dec0[0], dec[OUT_IDX], dec0[OUT_IDX], n, T)
    v_ref = efficient_plan(M, REF_HELP * T * sum(n.values()), REF_RULE)
    _, _, gap, B_ref = M.outcomes(v_ref)
    cohorts = [fa.make_cohort(seed, horizon=T, role="evaluation", index=200 + j) for j in range(10)]
    tg = []
    for init_, noise, grp in cohorts:
        x = init_.copy()
        for t in range(T):
            x = step(x, np.where(grp == 1, v_ref[1][t], v_ref[0][t]), grp, noise[:, t])
        tg.append(x[grp == 1, OUT_IDX].mean() - x[grp == 0, OUT_IDX].mean())
    cap = knapsack_price(M, v_ref, 0.0)[1]
    close = min(cap, abs(gap))
    out = dict(gamma=gamma, seed=seed, model_gap=float(gap), true_gap=float(np.mean(tg)), capacity=float(cap),
               assumptions=assumptions(M, v_ref), hk_max=float(max(M.hk[0].max(), M.hk[1].max())))
    if close > 1e-9:
        plan, _ = knapsack_plan(M, v_ref, knapsack_price(M, v_ref, close)[0])
        tc = true_change(step, cohorts, v_ref, plan)
        out.update(promised=float(close), true_narrowing=float(tc[2]), true_dK=[float(tc[0]), float(tc[1])])
    fa.step = sr.make_step(1.0)
    return out


def main():
    fa = sr.load_sim(str(PATHS["simulator"]))
    original = fa.training_data
    res = {}
    for name, low in (("U(-1,1)", -1.0), ("U(0,1)", 0.0)):
        fa.training_data = patched_training_data(fa, low)
        rows = [one(fa, g, s) for g in GAMMAS for s in SEEDS]
        res[name] = rows
        for g in GAMMAS:
            rs = [r for r in rows if r["gamma"] == g]
            def q(k):
                v = [r[k] for r in rs if k in r]
                return np.round([np.min(v), np.median(v), np.max(v)], 3) if v else "none"
            print(f"{name} gamma={g}: model gap {q('model_gap')} true gap {q('true_gap')} | A1-A5 all "
                  f"{np.mean([all(r['assumptions'].values()) for r in rs]):.1f} | promised {q('promised')} true narrowing {q('true_narrowing')} | "
                  f"h^k max {q('hk_max')}", flush=True)
    fa.training_data = original
    d = ROOT / "outputs" / "19_training_controls"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(dict(outcome=OUTCOME, ref_rule=REF_RULE, results=res), open(d / "results_knowledge.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
