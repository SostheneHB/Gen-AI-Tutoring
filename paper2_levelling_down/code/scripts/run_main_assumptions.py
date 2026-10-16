"""Experiment 20: do the assumptions of Theorem 2 hold in the fitted models of the main experiments?

    P2_OUTCOME=knowledge python code/scripts/run_main_assumptions.py

For gamma in {0.5, 1, 2} and ten seeds, with the group-aware model and with separate group models (knowledge as the
target, efficient reference with the smallest gap): (A1)-(A5), the smallest Markov parameter relative to the
largest, and the largest difference between the closed form of Theorem 2 and the linear program, at targets of
10, 25, 50, 75 and 100% of the closable gap.  Fits whose reference has no model gap are counted and skipped.
Writes outputs/20_main_assumptions/results_knowledge.json (aggregate numbers only).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from p2.config import OUT_IDX, OUTCOME, REF_HELP, REF_RULE  # noqa: E402
from p2.decomposition import fit_group_models  # noqa: E402
from p2.frontier import GroupModel, efficient_plan, fair_frontier, knapsack_price  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402
from run_assumption_cases import assumptions  # noqa: E402

use_paper1_code()
import pareto_fair as pf  # noqa: E402
import synthetic_regimes as sr  # noqa: E402

T = 25


def model(fa, gamma, seed, kind):
    fa.step = sr.make_step(gamma)
    m = fa.fit_model(seed, 4, 3)
    init, _, groups = fa.make_cohort(seed, horizon=T)
    n = {g: int((groups == g).sum()) for g in (0, 1)}
    m0 = {g: m.encode(init[groups == g]).mean(0) for g in (0, 1)}
    args = (m0, m.decoder[0], m.decoder_offset[0], m.decoder[OUT_IDX], m.decoder_offset[OUT_IDX], n, T)
    if kind == "group_aware":
        M = GroupModel(*pf.fit_group_aware(fa, m, seed), *args)
    else:
        st, ctrl, grp = fa.training_data(seed)
        Z, Zn = m.encode(st[:, :-1].reshape(-1, 4)), m.encode(st[:, 1:].reshape(-1, 4))
        M = GroupModel.separate(fit_group_models(Z, ctrl.reshape(-1), Zn, np.repeat(grp, st.shape[1] - 1)), *args)
    fa.step = sr.make_step(1.0)
    return M


def main():
    fa = sr.load_sim(str(PATHS["simulator"]))
    out = {}
    for kind in ("group_aware", "separate"):
        rows = []
        for gamma in (0.5, 1.0, 2.0):
            for seed in range(10):
                M = model(fa, gamma, seed, kind)
                v = efficient_plan(M, REF_HELP * T * sum(M.n.values()), REF_RULE)
                gap = M.outcomes(v)[2]
                if abs(gap) <= 1e-9:
                    rows.append(dict(gamma=gamma, seed=seed, no_gap=True))
                    continue
                a = assumptions(M, v)
                hk = np.r_[M.hk[0], M.hk[1]]
                cap = knapsack_price(M, v, 0.0)[1]
                close = min(cap, abs(gap))
                diffs = []
                for f in (0.1, 0.25, 0.5, 0.75, 1.0):
                    lp = fair_frontier(M, v, [f * close])[0]
                    cf = knapsack_price(M, v, f * close)[0]
                    if lp["feasible"] and np.isfinite(cf):
                        diffs.append(abs(lp["extra_budget"] - cf))
                rows.append(dict(gamma=gamma, seed=seed, no_gap=False, assumptions=a, all=bool(all(a.values())),
                                 min_over_max_hk=float(hk.min() / hk.max()), max_cf_lp_diff=float(max(diffs)) if diffs else None, cases=len(diffs)))
        ok = [r for r in rows if not r["no_gap"]]
        out[kind] = dict(fits=len(rows), no_gap=len(rows) - len(ok), all_assumptions=sum(r["all"] for r in ok),
                         each={k: sum(r["assumptions"][k] for r in ok) for k in ("A1", "A2", "A3", "A4", "A5")},
                         min_over_max_hk=float(min(r["min_over_max_hk"] for r in ok)),
                         cases=sum(r["cases"] for r in ok), max_cf_lp_diff=float(max(r["max_cf_lp_diff"] for r in ok if r["max_cf_lp_diff"] is not None)),
                         rows=rows)
        print(kind, {k: v for k, v in out[kind].items() if k != "rows"}, flush=True)
    d = ROOT / "outputs" / "20_main_assumptions"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(dict(outcome=OUTCOME, ref_rule=REF_RULE, results=out), open(d / "results_knowledge.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
