"""Experiment 13: how much do the capacity, the price, the assumptions and the welfare threshold depend on the
learned model?  (review of 16 October 2026, point 12)

    P2_OUTCOME=knowledge python code/scripts/run_model_sensitivity.py

For each group strength gamma, training size n (number of training students: 50, 150, 600) and ten training
seeds: fit the lifting, the decoder and the group-aware model on the training data only; take the efficient
reference plan; then compute
  * the model capacity and the model price of narrowing the gap by half of it,
  * what the full filling and the half filling really do on the simulator (ten evaluation cohorts),
  * whether (A1)-(A5) hold,
  * the welfare threshold Lambda_o / h^d_b at an extra budget of 1% (the utilitarian planner chooses the fair
    plan if and only if this ratio is <= 1).
Refits whose efficient reference has no model gap (nothing to narrow) are counted and left out of the summaries.
Writes outputs/13_model_sensitivity/results<TAG>.json (aggregate numbers only).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from p2.config import OUT_IDX, OUTCOME, REF_HELP, REF_RULE, TAG  # noqa: E402
from p2.frontier import GroupModel, best_return_ahead, efficient_plan, knapsack_plan, knapsack_price  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402
from run_assumption_cases import assumptions  # noqa: E402

use_paper1_code()
import synthetic_regimes as sr  # noqa: E402

T, GAMMAS, SIZES, SEEDS, N_EVAL = 25, (0.5, 1.0, 2.0), (50, 150, 600), range(10), 10


def fit(fa, seed, n_train):
    """Lifting, decoder and group-aware model fitted on n_train training students (as fa.fit_model)."""
    st, ctrl, grp = fa.training_data(seed, students=n_train)
    cur, nxt = st[:, :-1].reshape(-1, 4), st[:, 1:].reshape(-1, 4)
    raw = fa.psi_raw(cur)
    mu, sd = raw.mean(0), raw.std(0)
    sd[sd < 1e-8] = 1
    enc = lambda x: (fa.psi_raw(x) - mu) / sd
    Z, Zn = enc(cur), enc(nxt)
    G = np.repeat(grp, st.shape[1] - 1).astype(float)
    C = np.linalg.lstsq(np.column_stack([Z, ctrl.reshape(-1), np.ones(len(Z)), G]), Zn, rcond=None)[0]
    N = Z.shape[1]
    dec = np.linalg.lstsq(np.column_stack([Z, np.ones(len(Z))]), cur, rcond=None)[0]
    return C[:N].T, C[N], C[N + 1], C[N + 2], dec[:-1].T, dec[-1], enc


def true_change(step, cohorts, v_ref, v):
    out = []
    for init, noise, groups in cohorts:
        res = []
        for p in (v_ref, v):
            x = init.copy()
            for t in range(T):
                x = step(x, np.where(groups == 1, p[1][t], p[0][t]), groups, noise[:, t])
            res.append(({g: x[groups == g, 0].mean() for g in (0, 1)}, x[groups == 1, OUT_IDX].mean() - x[groups == 0, OUT_IDX].mean()))
        (kr, gr), (k, g) = res
        out.append((k[0] - kr[0], k[1] - kr[1], abs(gr) - abs(g)))
    return np.array(out).mean(0)


def one(fa, gamma, n_train, seed):
    step = sr.make_step(gamma)
    fa.step = step
    A, B, a, aG, dec, dec0, enc = fit(fa, seed, n_train)
    init, _, groups = fa.make_cohort(seed, horizon=T)
    n = {g: int((groups == g).sum()) for g in (0, 1)}
    m0 = {g: enc(init[groups == g]).mean(0) for g in (0, 1)}
    M = GroupModel(A, B, a, aG, m0, dec[0], dec0[0], dec[OUT_IDX], dec0[OUT_IDX], n, T)
    v_ref = efficient_plan(M, REF_HELP * T * sum(n.values()), REF_RULE)
    _, _, gap, B_ref = M.outcomes(v_ref)
    b = 1 if gap < 0 else 0
    cap = knapsack_price(M, v_ref, 0.0)[1]
    target = 0.5 * min(cap, abs(gap))
    price = knapsack_price(M, v_ref, target)[0] / B_ref
    close_target = min(cap, abs(gap))                       # close the gap as far as the model allows, no overshoot
    full, _ = knapsack_plan(M, v_ref, knapsack_price(M, v_ref, close_target)[0])
    half, _ = knapsack_plan(M, v_ref, price * B_ref)
    cohorts = [fa.make_cohort(seed, horizon=T, role="evaluation", index=200 + j) for j in range(N_EVAL)]
    tf, th = true_change(step, cohorts, v_ref, full), true_change(step, cohorts, v_ref, half)
    eps = 0.01 * B_ref
    _, last = knapsack_plan(M, v_ref, eps)
    lam = best_return_ahead(M, v_ref)
    thr = float(lam / M.hd[b][last]) if last is not None and M.hd[b][last] > 0 else None
    a_ = assumptions(M, v_ref)
    fa.step = sr.make_step(1.0)
    return dict(gamma=gamma, n_train=n_train, seed=seed, model_gap=float(gap), capacity=float(cap), close_target=float(close_target), price_half=float(price),
                target=float(target), true_full=float(tf[2]), true_half=float(th[2]), true_half_dK=[float(th[0]), float(th[1])],
                assumptions=a_, all_assumptions=bool(all(a_.values())), threshold=thr)


def main():
    fa = sr.load_sim(str(PATHS["simulator"]))
    rows = []
    q = lambda v: [float(np.percentile(v, 5)), float(np.percentile(v, 50)), float(np.percentile(v, 95))]
    summary = {}
    for gamma in GAMMAS:
        for n_train in SIZES:
            allr = [one(fa, gamma, n_train, s) for s in SEEDS]
            rows += allr
            rs = [r for r in allr if abs(r["model_gap"]) > 1e-9]       # refits whose reference has no model gap: nothing to narrow
            thr = [r["threshold"] for r in rs if r["threshold"] is not None]
            s = dict(refits=len(allr), refits_without_model_gap=len(allr) - len(rs), capacity=q([r["capacity"] for r in rs]), close_target=q([r["close_target"] for r in rs]), true_full=q([r["true_full"] for r in rs]),
                     price_half=q([r["price_half"] for r in rs]), target=q([r["target"] for r in rs]), true_half=q([r["true_half"] for r in rs]),
                     half_reached=float(np.mean([r["true_half"] >= r["target"] for r in rs])),
                     half_no_loss=float(np.mean([min(r["true_half_dK"]) >= 0 for r in rs])),
                     assumptions_hold=float(np.mean([r["all_assumptions"] for r in rs])),
                     threshold=q(thr) if thr else None, utilitarian_fair_share=float(np.mean([t <= 1 for t in thr])) if thr else None)
            summary[f"{gamma}|{n_train}"] = s
            print(f"gamma={gamma} n={n_train:4d}: capacity {np.round(s['capacity'], 3)} true full {np.round(s['true_full'], 3)} | price(half) "
                  f"{np.round(s['price_half'], 3)} reached {s['half_reached']:.1f} no-loss {s['half_no_loss']:.1f} | A1-A5 {s['assumptions_hold']:.1f} | "
                  f"threshold {None if s['threshold'] is None else np.round(s['threshold'], 2)} utilitarian fair {s['utilitarian_fair_share']}", flush=True)
            d = ROOT / "outputs" / "13_model_sensitivity"
            d.mkdir(parents=True, exist_ok=True)
            json.dump(dict(outcome=OUTCOME, summary=summary, rows=rows), open(d / f"results{TAG}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
