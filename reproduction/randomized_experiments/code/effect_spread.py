"""
Post-hoc descriptive addition (listed in the plan's Deviations, 13 October 2026): spread of the estimated
treatment effects across the 30 experiments, by prior-achievement group, and the largest change of the gap
that any allocation of the two existing conditions by group could give (an in-sample bound, inflated by noise).
Uses only outputs/policy_real.json (cross-fitted doubly robust values of P0 = control for all, P1 = treatment
for all, P2 = reference); no new model is fitted.

    python effect_spread.py   -> outputs/effect_spread.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
d = json.load(open(os.path.join(HERE, "..", "outputs", "policy_real.json")))
rows = []
for name, e in d["per_experiment"].items():
    v = e["values"]
    tl, th = v["P1"]["low"] - v["P0"]["low"], v["P1"]["high"] - v["P0"]["high"]
    gaps = [v[a]["high"] - v[b]["low"] for a in ("P0", "P1") for b in ("P0", "P1")]   # arm for high, arm for low
    ref = v["P2"]["gap"]
    rows.append(dict(exp=name, n=e["n"], effect_low=tl, effect_high=th, best_gap_change=min(gaps) - ref))
n = np.array([r["n"] for r in rows], float)
w = n / n.sum()
el, eh = np.array([r["effect_low"] for r in rows]), np.array([r["effect_high"] for r in rows])
bg = np.array([r["best_gap_change"] for r in rows])
q = lambda x: [float(np.percentile(x, p)) for p in (10, 25, 50, 75, 90)]
out = dict(n_experiments=len(rows),
           effect_low_percentiles_10_25_50_75_90=q(el), effect_high_percentiles_10_25_50_75_90=q(eh),
           share_abs_effect_above_2pts_low=float((np.abs(el) > 0.02).mean()), share_abs_effect_above_2pts_high=float((np.abs(eh) > 0.02).mean()),
           share_abs_effect_above_5pts_low=float((np.abs(el) > 0.05).mean()), share_abs_effect_above_5pts_high=float((np.abs(eh) > 0.05).mean()),
           effect_low_weighted_mean=float(w @ el), effect_high_weighted_mean=float(w @ eh),
           corr_effect_low_high=float(np.corrcoef(el, eh)[0, 1]),
           best_allocation_gap_change_weighted=float(w @ bg), best_allocation_gap_change_percentiles=q(bg),
           rows=rows)
json.dump(out, open(os.path.join(HERE, "..", "outputs", "effect_spread.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
