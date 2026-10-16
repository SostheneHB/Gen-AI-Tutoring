"""Audit of the certificate scores (review of 16 October 2026, point 3).

    python code/scripts/audit_certificate_scores.py

For one fitted model (gamma = 1, seed 0), standard regime: compute the score of every cohort, report its
minimum, maximum, standard deviation and number of distinct values, and compare the narrowing of the gap
measured on the simulator with the analytic formula of the linear disparity equation,
    d_T(v) - d_T(ref) = 0.05 sum_t 0.9^(T-1-t) (v_t - v_ref_t)   (same students, same noise).
Writes outputs/11_certificate_audit/results.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
import run_certificate_v2 as rc  # noqa: E402
from p2.paths import PATHS  # noqa: E402

fa = rc.sr.load_sim(str(PATHS["simulator"]))
gamma, seed, T = 1.0, 0, rc.T
step = rc.sr.make_step(gamma)
fa.step = step
model = fa.fit_model(seed, 4, 3)
A, B, a, aG = rc.pf.fit_group_aware(fa, model, seed)
init, _, groups = fa.make_cohort(seed, horizon=T)
n = {g: int((groups == g).sum()) for g in (0, 1)}
m0 = {g: model.encode(init[groups == g]).mean(0) for g in (0, 1)}
M = rc.GroupModel(A, B, a, aG, m0, model.decoder[0], model.decoder_offset[0], model.decoder[fa.D_IDX], model.decoder_offset[fa.D_IDX], n, T)
v_ref = rc.efficient_plan(M, 0.3 * T * sum(n.values()))
cap = rc.knapsack_price(M, v_ref, 0.0)[1]
rho = 0.5 * cap
deltas, plans, preds = [], [], []
for d in rc.DELTAS:
    p = rc.margin_plan(M, v_ref, rho, d)
    if p is not None and rho / (1 - d) <= 0.9 * cap:
        deltas.append(float(d)); plans.append(p); preds.append(rc.predicted_change(M, v_ref, p))
truth = np.array([rc.realised(step, fa.make_cohort(seed, horizon=T, role="evaluation", index=1000 + j), v_ref, plans) for j in range(300)])
S = np.array([max(rc.score(preds[i], truth[c, i]) for i in range(len(plans))) for c in range(len(truth))])
w = 0.05 * 0.9 ** (T - 1 - np.arange(T))
analytic = [float(w @ (p[1] - v_ref[1]) - w @ (p[0] - v_ref[0])) for p in plans]   # change of D1 - D0
sim = truth[:, :, 2]
out = dict(n_cohorts=len(S), score_min=float(S.min()), score_max=float(S.max()), score_sd=float(S.std()),
           distinct_exact=int(len(np.unique(S))), distinct_rounded_1e12=int(len(np.unique(np.round(S, 12)))),
           narrowing_sd_over_cohorts=float(sim.std(0).max()),
           max_abs_sim_minus_analytic=float(np.abs(sim - np.array(analytic)[None, :]).max()),
           dK_min=[float(truth[:, :, 0].min()), float(truth[:, :, 1].min())], plans=len(plans))
print(json.dumps(out, indent=1))
d = ROOT / "outputs" / "11_certificate_audit"
d.mkdir(parents=True, exist_ok=True)
json.dump(out, open(d / "results.json", "w"), indent=1)
